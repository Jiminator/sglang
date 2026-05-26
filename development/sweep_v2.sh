#!/bin/bash
# Trimmed sweep — skips configs already tested and configs predicted to crash via static analysis.
# Pulls helpers from sweep.sh; same BASE_FLAGS = run #11 best config.
#
# Skipped (already tested, see sweep_log.txt):
#   torchcompile, ngram, a2a_mooncake, a2a_nixl, a2a_flashinfer, a2a_megamoe (in-flight when killed)
# Skipped (predicted CRASH via code analysis, no GPU run needed):
#   a2a_mori (ImportError mori not installed)
#   moe_triton_kernel (TopKOutputChecker.format_is_triton_kernels assert)
#   moe_finfer_trtllm / moe_finfer_trtllm_routed (block_k assert + SM100-only path)
#   pdmux (asserts on chunked_prefill_size==-1 + disable_overlap_schedule==True)
#
# Re-test:
#   mscclpp — workers DID come up last time; bench failed only because of stale router circuit breaker
# Net-new:
#   cgbs_custom, maxrr32, tok_batch_encode

set -uo pipefail

RUN_DIR=/sgl-workspace/sglang/runs/20260525_dsv32_2rep_sota_loop
SWEEP_LOG="$RUN_DIR/sweep_v2_log.txt"
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
  # Rust router renames itself to `sglang::router` (double-colon), so
  # `pkill -f sglang_router` (underscore) misses it. Match the python
  # entrypoint AND kill anything holding the router/prom ports.
  pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
  pkill -9 -f "sglang::router" 2>/dev/null || true
  for p in 30000 29000; do
    pids=$(lsof -ti:$p 2>/dev/null || true)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  done
  sleep 3
}

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
  local stale_n0=0
  while [ $elapsed -lt $((timeout_min*60)) ]; do
    local h0
    local h1
    h0=$(curl -sf -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health_generate 2>/dev/null || echo "000")
    h1=$($NODE1_RX "curl -sf -o /dev/null -w \"%{http_code}\" http://127.0.0.1:$PORT/health_generate 2>/dev/null || echo 000" 2>/dev/null | tail -1)
    if [ "$h0" = "200" ] && [ "$h1" = "200" ]; then
      log "  BOTH_READY in ${elapsed}s"
      return 0
    fi
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
  local tpot=$(echo "$out" | grep "Mean TPOT" | awk '{print $4}')
  local ttft=$(echo "$out" | grep "P99 TTFT" | awk '{print $4}')
  local out_thru=$(echo "$out" | grep "Output token throughput" | awk '{print $5}')
  log "  RESULT $tag: mean_TPOT=$tpot P99_TTFT=$ttft out_thru=$out_thru"
  cp "/sgl-workspace/sglang/sw_${tag}_gsp_isl4096_osl512_c64.jsonl" "$RUN_DIR/benchmark/sw_${tag}.jsonl" 2>/dev/null || true
}

# === TRIMMED SWEEP ===
log "===== SWEEP_V2 START $(date) ====="
log "Base = run #11 best (mixed-chunk + lpm + mem 0.88 + ep=8 + tp=8 + DPA dp=8 + fp8 KV + page 64)"
log ""

# 1. mscclpp — re-test (workers OK last time, router CB was stale)
test_one mscclpp --enable-mscclpp

# 2. Custom cuda-graph-bs matching the 4-rec/rank shape we saw in decode
test_one cgbs_custom --cuda-graph-bs 1 2 3 4 5 6 7 8 10 12 16 20 24 32 48 64 96 128 256 512

# 3. max-running-requests=32 (cap batch at concurrency to reduce KV churn)
test_one maxrr32 --max-running-requests 32

# 4. tokenizer batch encode (prefill optimization)
test_one tok_batch_encode --enable-tokenizer-batch-encode

log ""
log "===== SWEEP_V2 COMPLETE $(date) ====="
