#!/usr/bin/env bash
# Loop 11 R7 task4: int8 served config + table-aware pool sizing — M1 closer.
#   phase int8boot : DS signature_dtype=int8, mem 0.8, right-sized envelope
#                    (--max-running-requests 64 --cuda-graph-max-bs 64) -> capacity
#                    readout (bs cap = floor(max_total_num_tokens/4608) >= 64?),
#                    graph capture, single-request smoke. The table-aware cell-size
#                    reserves the int8 TokenLabelTable deliberately.
#   phase dsa_ac7  : DSA-native mem 0.8 -> 410560 unchanged (sizing change is DS-only). AC-7.
#   phase ladder   : DS int8 @0.8 rs + conc 16/32/64 directional ladder (iteration signal)
#                    via benchmark_baseline.sh; conc-64 achieved-concurrency = AC-1.1 serve smoke.
# One TP=8 server at a time; foreground per phase; frozen _env.sh reused read-only (int8 is a
# one-field flip of DS_CONFIG); never expandable_segments.
set -uo pipefail
PHASE="${1:?usage: stage_r7_task4.sh <int8boot|dsa_ac7|ladder>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_r7_task4_${PHASE}.log"; exec > >(tee "$LOG") 2>&1
unset SGLANG_DS_PROBE_TABLE_TOKENS SGLANG_DS_PROBE_SKIP_INDEXER || true
echo "=== R7 task4 phase=$PHASE start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
git diff --quiet -- python/ && echo "production tree = committed" || echo "NOTE: uncommitted python diff present"

# int8 served config = the frozen DS_CONFIG with signature_dtype flipped fp16 -> int8.
INT8_CONFIG="${DS_CONFIG/\"signature_dtype\": \"fp16\"/\"signature_dtype\": \"int8\"}"
RS_ARGS=(--max-running-requests 64 --cuda-graph-max-bs 64)

ready_wait() {
  local slog="$1"
  for i in $(seq 1 80); do
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory" "$slog" 2>/dev/null; then
      return 1
    fi
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
boot_fields() {  # $1=serve.log $2=evidence.txt $3=label
  local slog="$1" ev="$2" name="$3" cap smk note
  if ready_wait "$slog"; then
    cap=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); smk=$(smoke); note="OK"
  else
    note="BOOT_FAIL:$(grep -aiE 'SIGQUIT|exception|out of memory|capture cuda graph failed|Not enough host memory' "$slog" | head -1 | cut -c1-70)"
    cap=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); smk="-"
  fi
  {
    echo "probe=$name status=$note"
    echo "max_total_num_tokens=${cap:-NA} bs_cap=floor(cap/4608)=$(( ${cap:-0} / 4608 ))"
    echo "smoke=$smk"
    grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens|Memory pool end)" "$slog" | head -4
    grep -aiE "host memory for DSA indexer|token_label_table|Allocating .* signature" "$slog" | head -2 || true
  } > "$ev"
  echo ">>> $name status=$note cap=${cap:-NA} bs=$(( ${cap:-0} / 4608 )) smoke=$smk"
}

teardown
if [[ "$PHASE" == "int8boot" ]]; then
  echo "=== DS int8 @0.8 rs (table-aware sizing) $(date -u +%H:%M:%SZ) ==="
  ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.8 --enable-double-sparsity --double-sparsity-config "$INT8_CONFIG" "${RS_ARGS[@]}")
  slog="$HERE/r7_ds_int8_080rs_serve.log"
  python -m sglang.launch_server "${ARGS[@]}" > "$slog" 2>&1 &
  boot_fields "$slog" "$HERE/r7_ds_int8_080rs_evidence.txt" ds_int8_080rs
  teardown

elif [[ "$PHASE" == "dsa_ac7" ]]; then
  echo "=== DSA-native @0.8 (AC-7: sizing change is DS-only) $(date -u +%H:%M:%SZ) ==="
  build_server_args dsa08
  slog="$HERE/r7_dsa_080_ac7_serve.log"
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$slog" 2>&1 &
  boot_fields "$slog" "$HERE/r7_dsa_080_ac7_evidence.txt" dsa_080_ac7
  teardown

elif [[ "$PHASE" == "ladder" ]]; then
  echo "=== DS int8 @0.8 rs + conc 16/32/64 directional ladder $(date -u +%H:%M:%SZ) ==="
  ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.8 --enable-double-sparsity --double-sparsity-config "$INT8_CONFIG" "${RS_ARGS[@]}")
  slog="$HERE/r7_ds_int8_ladder_serve.log"
  python -m sglang.launch_server "${ARGS[@]}" > "$slog" 2>&1 &
  if ! ready_wait "$slog"; then echo "!! ladder boot FAIL"; tail -40 "$slog"; teardown; exit 1; fi
  CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
  echo ">>> int8 ladder server cap=$CAP bs=$(( CAP / 4608 ))"
  set +e
  PORT=30000 HOST=127.0.0.1 MODE=ds_int8 RESULTS_DIR="$HERE/serving_r7_int8" \
    CONCURRENCIES="16 32 64" TRIALS=1 \
    WARMUP_SECONDS="${LADDER_WARMUP:-60}" MEASUREMENT_WINDOW_S="${LADDER_WINDOW:-180}" NUM_PROMPTS=64 \
    bash development/benchmark_baseline.sh
  echo ">>> ladder rc=$?"
  set -e
  {
    echo "probe=ds_int8_ladder cap=$CAP bs_cap=$(( CAP / 4608 ))"
    for c in 16 32 64; do
      f=$(ls "$HERE/serving_r7_int8/"*"c${c}_t1.jsonl" 2>/dev/null | head -1)
      [[ -n "$f" ]] && python3 -c "import json;r=json.loads(open('$f').readline());print('conc=%d ach_conc=%.2f decTPS_p50=%.2f agg_tok/s=%.1f ttft_p99=%.2fs done=%d'%($c,r['concurrency'],r['median_decode_throughput_tps'],r['output_throughput'],r['p99_ttft_ms']/1000,r['completed']))" || echo "conc=$c MISSING"
    done
  } > "$HERE/r7_ds_int8_ladder_evidence.txt"
  cat "$HERE/r7_ds_int8_ladder_evidence.txt"
  teardown
else
  echo "unknown phase $PHASE"; exit 2
fi
echo "=== R7 task4 phase=$PHASE done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$HERE"/r7_*_evidence.txt 2>/dev/null || true
