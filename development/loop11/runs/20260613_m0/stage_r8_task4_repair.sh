#!/usr/bin/env bash
# Loop 11 R8 task4 repair (R7-review gaps):
#   phase durable_int8 : boot the DURABLE launcher development/serve_double_sparsity.sh (GLM env +
#                        its new int8/0.8/rs defaults) -> record effective args + cap + capture +
#                        smoke; then conc-64 serve smoke at the RECIPE OF RECORD (60s warmup /
#                        180s window / NUM_PROMPTS=64) with an explicit gate: achieved >= 60.8.
#   phase case2        : AC-7 frozen Case-2 (DSA-native @0.7, dsa07) re-validation (NEW artifact).
#   phase dsa_radixon  : AC-7 DSA @0.8 radix-ON serving smoke (NEW artifact).
# One TP=8 server at a time; foreground per phase; never expandable_segments (launcher strips it).
set -uo pipefail
PHASE="${1:?usage: stage_r8_task4_repair.sh <durable_int8|case2|dsa_radixon>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_r8_${PHASE}.log"; exec > >(tee "$LOG") 2>&1
unset SGLANG_DS_PROBE_TABLE_TOKENS SGLANG_DS_PROBE_SKIP_INDEXER || true
echo "=== R8 task4-repair phase=$PHASE start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
git diff --quiet -- python/ development/serve_double_sparsity.sh && echo "tree=committed" || echo "NOTE: uncommitted diff present"

ready_wait() {
  local slog="$1"
  for i in $(seq 1 80); do
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory|ValueError" "$slog" 2>/dev/null; then return 1; fi
    if grep -aq "The server is fired up" "$slog" 2>/dev/null; then
      curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "ready ~$((i*10))s"; return 0; }
    fi
    sleep 10
  done
  return 1
}
smoke() {
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null \
  | python3 -c "import json,sys
try:
 d=json.load(sys.stdin); t=(d.get('text') or '').strip().replace(chr(10),' ')
 print('OK:'+t[:60] if t else 'FAIL:empty')
except Exception as e: print('FAIL:'+str(e)[:60])"
}

teardown
if [[ "$PHASE" == "durable_int8" ]]; then
  echo "=== boot DURABLE launcher serve_double_sparsity.sh (GLM, int8/0.8/rs defaults) $(date -u +%H:%M:%SZ) ==="
  slog="$HERE/r8_durable_int8_serve.log"
  MODEL_PATH="$GLM" CHANNEL_MASK_PATH="$MASK" PORT="$PORT" HOST=127.0.0.1 TP_SIZE=8 \
    LOG_DIR="$HERE/r8_launcher_logs" \
    bash /sgl-workspace/sglang/development/serve_double_sparsity.sh > "$slog" 2>&1 &
  if ! ready_wait "$slog"; then echo "!! durable boot FAIL"; grep -aiE "error|valueerror|oom|exception" "$slog" | head -5; teardown; exit 1; fi
  CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
  SMK=$(smoke)
  echo ">>> durable boot cap=$CAP bs=$(( CAP / 4608 )) smoke=$SMK"
  echo "=== conc-64 serve smoke @ recipe of record (60s warmup / 180s window / 64 prompts) ==="
  set +e
  PORT=30000 HOST=127.0.0.1 MODE=ds_int8_durable RESULTS_DIR="$HERE/serving_r8_int8" \
    CONCURRENCIES="64" TRIALS=1 WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
    bash development/benchmark_baseline.sh
  echo ">>> conc64 rc=$?"
  set -e
  f=$(ls "$HERE/serving_r8_int8/"*"c64_t1.jsonl" 2>/dev/null | head -1)
  ACH=$(python3 -c "import json;print('%.2f'%json.loads(open('$f').readline())['concurrency'])" 2>/dev/null || echo 0)
  GATE=$(python3 -c "print('PASS' if float('$ACH') >= 60.8 else 'FAIL')")
  {
    echo "probe=durable_int8 (serve_double_sparsity.sh GLM int8/0.8/rs) cap=$CAP bs_cap=$(( CAP / 4608 )) smoke=$SMK"
    echo "effective launcher args:"; grep -aE "signature_dtype|mem_fraction|max_running_reqs|cuda_graph_max_bs|effective double_sparsity_config" "$slog" | head -6
    grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -3
    grep -aiE "token_label_table" "$slog" | head -1
    echo "--- AC-1.1 conc-64 serve smoke (recipe of record) ---"
    [[ -n "$f" ]] && python3 -c "import json;r=json.loads(open('$f').readline());print('achieved_conc=%.2f (gate>=60.8 -> $GATE)  decTPS_p50=%.2f agg_tok/s=%.1f ttft_p99=%.2fs completed=%d'%(r['concurrency'],r['median_decode_throughput_tps'],r['output_throughput'],r['p99_ttft_ms']/1000,r['completed']))"
  } > "$HERE/r8_durable_int8_evidence.txt"
  cat "$HERE/r8_durable_int8_evidence.txt"
  echo ">>> AC-1.1 conc-64 achieved=$ACH gate=$GATE"
  teardown

elif [[ "$PHASE" == "dsa_radixoff_conc64" ]]; then
  echo "=== isolation + AC-7 DS-off: DSA-native @0.8 radix-OFF conc-64 (same workload) $(date -u +%H:%M:%SZ) ==="
  build_server_args dsa08
  slog="$HERE/r8_dsa08_radixoff_serve.log"
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$slog" 2>&1 &
  if ! ready_wait "$slog"; then echo "!! boot FAIL"; teardown; exit 1; fi
  CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
  SMK=$(smoke)
  set +e
  PORT=30000 HOST=127.0.0.1 MODE=dsa_radixoff RESULTS_DIR="$HERE/serving_r8_dsa_off" \
    CONCURRENCIES="64" TRIALS=1 WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
    bash development/benchmark_baseline.sh
  echo ">>> conc64 rc=$?"
  set -e
  f=$(ls "$HERE/serving_r8_dsa_off/"*"c64_t1.jsonl" 2>/dev/null | head -1)
  RUN=$(grep -aoE "Decode batch. #running-req: [0-9]+" "$slog" | grep -oE "[0-9]+$" | sort -n | tail -1)
  { echo "probe=dsa08_radixoff_conc64 (isolation + AC-7 DS-off @0.8) cap=$CAP bs_cap=$(( CAP / 4608 )) smoke=$SMK max_decode_running_req=$RUN";
    [[ -n "$f" ]] && python3 -c "import json;r=json.loads(open('$f').readline());print('achieved_conc=%.2f decTPS_p50=%.2f agg_tok/s=%.1f ttft_p99=%.2fs completed=%d'%(r['concurrency'],r['median_decode_throughput_tps'],r['output_throughput'],r['p99_ttft_ms']/1000,r['completed']))"; } > "$HERE/r8_dsa08_radixoff_conc64_evidence.txt"
  cat "$HERE/r8_dsa08_radixoff_conc64_evidence.txt"; teardown

elif [[ "$PHASE" == "case2" ]]; then
  echo "=== AC-7 Case-2 re-validation: DSA-native @0.7 (dsa07) $(date -u +%H:%M:%SZ) ==="
  build_server_args dsa07
  slog="$HERE/r8_case2_dsa07_serve.log"
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$slog" 2>&1 &
  if ready_wait "$slog"; then CAP=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); SMK=$(smoke); NOTE=OK; else NOTE="BOOT_FAIL"; CAP=NA; SMK=-; fi
  { echo "probe=case2_dsa07 status=$NOTE max_total_num_tokens=${CAP} smoke=$SMK";
    grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -3; } > "$HERE/r8_case2_dsa07_evidence.txt"
  cat "$HERE/r8_case2_dsa07_evidence.txt"; teardown

elif [[ "$PHASE" == "dsa_radixon" ]]; then
  echo "=== AC-7 DSA @0.8 radix-ON serving smoke $(date -u +%H:%M:%SZ) ==="
  build_server_args dsa08
  ARGS=(); for a in "${SERVER_ARGS[@]}"; do [[ "$a" == "--disable-radix-cache" ]] && continue; ARGS+=("$a"); done
  slog="$HERE/r8_dsa08_radixon_serve.log"
  python -m sglang.launch_server "${ARGS[@]}" > "$slog" 2>&1 &
  if ready_wait "$slog"; then
    CAP=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); SMK=$(smoke)
    RADIX=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;print(json.load(sys.stdin).get('disable_radix_cache','?'))")
    NOTE=OK
  else NOTE="BOOT_FAIL"; CAP=NA; SMK=-; RADIX=?; fi
  { echo "probe=dsa08_radixon status=$NOTE max_total_num_tokens=${CAP} disable_radix_cache=$RADIX (expect False) smoke=$SMK";
    grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -3; } > "$HERE/r8_dsa08_radixon_evidence.txt"
  cat "$HERE/r8_dsa08_radixon_evidence.txt"; teardown
else
  echo "unknown phase $PHASE"; exit 2
fi
echo "=== R8 phase=$PHASE done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
