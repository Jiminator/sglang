#!/usr/bin/env bash
# Stage 2 (DSA best, mem 0.8): boot -> directional serving sweep (native_nsa column)
# -> derive DSA max batch -> case3 torch decode profile @ DSA max batch. One boot.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/stage2_dsa08.log"; exec > >(tee "$LOG") 2>&1
echo "=== STAGE2 DSA@0.8  start $(date -u +%H:%M:%S)Z ==="

teardown
build_server_args dsa08
TRACE="$HERE/case3_dsa08/torch/trace"; mkdir -p "$TRACE"
SERVE_LOG="$HERE/case3_dsa08/serve.log"; mkdir -p "$(dirname "$SERVE_LOG")"

echo ">>> booting DSA server (mem 0.8, DS-off) ..."
env SGLANG_TORCH_PROFILER_DIR="$TRACE" \
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
if ! wait_ready; then echo "!! FAIL: DSA@0.8 not ready"; tail -50 "$SERVE_LOG"; teardown; exit 1; fi

CAP=$(max_batch_from_server); DSA_MAXBATCH=$(( CAP / 4608 ))
# Cap at 64 — the locked DSA "best" batch in the frozen baseline (avoid a runaway
# single-batch that no longer matches the served regime).
[[ "$DSA_MAXBATCH" -gt 64 ]] && DSA_MAXBATCH=64
echo ">>> DSA@0.8 cap=${CAP} => DSA_MAXBATCH=${DSA_MAXBATCH}"
echo "$DSA_MAXBATCH" > "$HERE/dsa_maxbatch.txt"

echo "=== DSA directional serving sweep (conc 16/32/64, 1 trial, 180s) $(date -u +%H:%M:%S)Z ==="
set +e
PORT=30000 HOST=127.0.0.1 MODE=native_nsa \
  RESULTS_DIR="$HERE/serving" \
  CONCURRENCIES="16 32 64" TRIALS=1 \
  WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
  bash development/benchmark_baseline.sh
SWEEP_RC=$?; set -e
echo ">>> DSA serving sweep rc=$SWEEP_RC"

echo "=== case3 DSA@0.8 torch decode profile @ bs ${DSA_MAXBATCH} $(date -u +%H:%M:%S)Z ==="
BENCH_LOG="$HERE/case3_dsa08/bench.log"; RESULT="$HERE/case3_dsa08/result.jsonl"
set +e
env SGLANG_TORCH_PROFILER_DIR="$TRACE" \
  python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
  --model-path "$GLM" --trust-remote-code \
  --batch-size "$DSA_MAXBATCH" --input-len 4096 --output-len 512 --temperature 0 --show-report \
  --profile --profile-by-stage --profile-activities CPU GPU --profile-steps 10 \
  --profile-output-dir "$TRACE" --result-filename "$RESULT" > "$BENCH_LOG" 2>&1
PROF_RC=$?; set -e
NTRACE=$(find "$TRACE" -name '*.trace.json.gz' 2>/dev/null | wc -l)
echo ">>> case3 torch rc=$PROF_RC trace_files=$NTRACE"
grep -iE "decode|throughput|latency|batch" "$BENCH_LOG" | tail -25
teardown
echo "=== STAGE2 DSA@0.8 done $(date -u +%H:%M:%S)Z sweep_rc=$SWEEP_RC prof_rc=$PROF_RC ntrace=$NTRACE ==="
