#!/bin/bash
# Sequential sweep of untested levers on top of run #11 best config
# Each: kill workers, launch with extra flag, wait ready (or crash), bench, save result

set -uo pipefail

RUN_DIR=/sgl-workspace/sglang/runs/20260525_dsv32_2rep_sota_loop
SWEEP_LOG="$RUN_DIR/sweep_log.txt"
WORKER_LOG_DIR=/sgl-workspace/sglang/development/logs
NODE1_LOG_DIR=/tmp/sgl_logs
NODE1_RX='rx devbox run double-sparsity --rank 1 -- bash -lc'

MODEL=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2
PORT=30001
ROUTER_PORT=30000

BASE_FLAGS=(
  --model-path "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --tp-size 8 --ep-size 8
  --dp-size 8 --enable-dp-attention
  --kv-cache-dtype fp8_e4m3
  --page-size 64
  --enable-mixed-chunk
  --schedule-policy lpm
  --mem-fraction-static 0.88
  --enable-cache-report
  --trust-remote-code
)

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "$SWEEP_LOG"
}

kill_workers() {
  pkill -9 -f sglang.launch_server 2>/dev/null || true
  $NODE1_RX 'pkill -9 -f sglang.launch_server 2>/dev/null || true; sleep 3' >/dev/null 2>&1 || true
  sleep 4
}

kill_router() {
  pkill -9 -f sglang_router 2>/dev/null || true
  sleep 3
}

# Args: tag, extra_flags...
launch_workers() {
  local tag="$1"
  shift
  local extra=("$@")
  local TS=$(date +%H%M%S)
  local N0_LOG="$WORKER_LOG_DIR/worker_node0_sw_${tag}_${TS}.log"
  local N1_LOG="$NODE1_LOG_DIR/worker_node1_sw_${tag}_${TS}.log"
  log "  launching tag=$tag flags=${extra[*]}"
  nohup python3 -m sglang.launch_server "${BASE_FLAGS[@]}" "${extra[@]}" > "$N0_LOG" 2>&1 &
  local NODE0_PID=$!
  echo "$NODE0_PID" > /tmp/sweep_node0_pid
  echo "$N0_LOG" > /tmp/sweep_node0_log
  sleep 2
  local cmd="LOG=$N1_LOG; nohup python3 -m sglang.launch_server"
  for f in "${BASE_FLAGS[@]}" "${extra[@]}"; do
    cmd="$cmd $(printf '%q' "$f")"
  done
  cmd="$cmd > \$LOG 2>&1 < /dev/null & disown; sleep 3"
  $NODE1_RX "$cmd" >/dev/null 2>&1 || true
  echo "$N1_LOG" > /tmp/sweep_node1_log
}

wait_ready() {
  local timeout_min=15
  local elapsed=0
  local poll=15
  local last_n0=0
  local last_n1=0
  local stale_n0=0
  local stale_n1=0
  while [ $elapsed -lt $((timeout_min*60)) ]; do
    local h0
    local h1
    h0=$(curl -sf -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health_generate 2>/dev/null || echo "000")
    h1=$($NODE1_RX "curl -sf -o /dev/null -w \"%{http_code}\" http://127.0.0.1:$PORT/health_generate 2>/dev/null || echo 000" 2>/dev/null | tail -1)
    if [ "$h0" = "200" ] && [ "$h1" = "200" ]; then
      log "  BOTH_READY in ${elapsed}s"
      return 0
    fi
    # Check for crash via Traceback
    local n0log
    n0log=$(cat /tmp/sweep_node0_log)
    if [ -f "$n0log" ]; then
      if grep -q "Traceback\|raise [A-Z]" "$n0log" 2>/dev/null; then
        local sz=$(stat -c%s "$n0log")
        if [ "$sz" = "$last_n0" ]; then stale_n0=$((stale_n0+1)); else stale_n0=0; last_n0=$sz; fi
        if [ $stale_n0 -ge 2 ]; then
          log "  CRASH_N0 detected at ${elapsed}s"
          tail -40 "$n0log" >> "$SWEEP_LOG"
          return 1
        fi
      fi
    fi
    sleep $poll
    elapsed=$((elapsed+poll))
  done
  log "  TIMEOUT after ${timeout_min}min"
  return 2
}

setup_router_regular() {
  kill_router
  nohup python3 -m sglang_router.launch_router \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    --worker-urls http://10.220.51.16:$PORT http://10.220.51.5:$PORT \
    --policy round_robin \
    --worker-startup-timeout-secs 60 \
    --disable-circuit-breaker \
    > "$WORKER_LOG_DIR/router_sw_$(date +%H%M%S).log" 2>&1 &
  sleep 6
  local h=$(curl -sf -o /dev/null -w "%{http_code}" http://127.0.0.1:$ROUTER_PORT/health || echo "000")
  if [ "$h" != "200" ]; then
    log "  ROUTER FAILED ($h)"
    return 1
  fi
  return 0
}

run_bench() {
  local mode="$1"
  curl -s -X POST http://127.0.0.1:$ROUTER_PORT/flush_cache > /dev/null 2>&1
  PORT=$ROUTER_PORT MODE="$mode" bash /sgl-workspace/sglang/development/benchmark.sh 2>&1 | tail -28
}

# Single test: tag + extra flags
test_one() {
  local tag="$1"
  shift
  log "=== TEST: $tag ==="
  kill_workers
  launch_workers "$tag" "$@"
  if ! wait_ready; then
    log "  SKIP (load failed/timeout)"
    return
  fi
  if ! setup_router_regular; then
    log "  SKIP (router failed)"
    return
  fi
  local out
  out=$(run_bench "sw_$tag")
  echo "$out" | tee -a "$SWEEP_LOG"
  # Extract key metrics
  local tpot=$(echo "$out" | grep "Mean TPOT" | awk '{print $4}')
  local ttft=$(echo "$out" | grep "P99 TTFT" | awk '{print $4}')
  local out_thru=$(echo "$out" | grep "Output token throughput" | awk '{print $5}')
  log "  RESULT $tag: mean_TPOT=$tpot P99_TTFT=$ttft out_thru=$out_thru"
  cp "/sgl-workspace/sglang/sw_${tag}_gsp_isl4096_osl512_c64.jsonl" "$RUN_DIR/benchmark/sw_${tag}.jsonl" 2>/dev/null || true
}

# === SWEEP STARTS ===
log "===== SWEEP START $(date) ====="
log "Base config: run #11 = mixed-chunk + lpm + mem 0.88 + ep=8 + tp=8 + dp-attention dp=8 + fp8 KV + page 64"
log ""

# 1. torch.compile
test_one torchcompile --enable-torch-compile --torch-compile-max-bs 32

# 2. NGRAM speculative dec
test_one ngram --speculative-algorithm NGRAM --speculative-num-draft-tokens 4

# 3. mscclpp
test_one mscclpp --enable-mscclpp

# 4. moe-a2a-backend mooncake
test_one a2a_mooncake --moe-a2a-backend mooncake

# 5. moe-a2a-backend nixl
test_one a2a_nixl --moe-a2a-backend nixl

# 6. moe-a2a-backend flashinfer
test_one a2a_flashinfer --moe-a2a-backend flashinfer

# 7. moe-a2a-backend megamoe
test_one a2a_megamoe --moe-a2a-backend megamoe

# 8. moe-a2a-backend mori
test_one a2a_mori --moe-a2a-backend mori

# 9. moe-runner-backend triton_kernel
test_one moe_triton_kernel --moe-runner-backend triton_kernel

# 10. moe-runner-backend flashinfer_trtllm
test_one moe_finfer_trtllm --moe-runner-backend flashinfer_trtllm

# 11. moe-runner-backend flashinfer_trtllm_routed
test_one moe_finfer_trtllm_routed --moe-runner-backend flashinfer_trtllm_routed

# 12. enable-pdmux (in-server PD)
test_one pdmux --enable-pdmux

# 13. Custom cuda-graph-bs matching our 4-rec/rank shape
test_one cgbs_custom --cuda-graph-bs 1 2 3 4 5 6 7 8 10 12 16 20 24 32 48 64 96 128 256 512

# 14. max-running-requests=32
test_one maxrr32 --max-running-requests 32

# 15. enable-multi-layer-eagle (without spec algo, just see if it accepts)
# Skip — needs algorithm

# 16. tokenizer batch encode (helps prefill)
test_one tok_batch_encode --enable-tokenizer-batch-encode

log ""
log "===== SWEEP COMPLETE $(date) ====="
