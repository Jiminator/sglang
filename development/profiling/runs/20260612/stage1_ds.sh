#!/usr/bin/env bash
# Stage 1 (DS, current HEAD): boot DS@0.7 -> derive max batch -> directional serving
# sweep -> case1 torch decode profile @ max batch. One boot; teardown at end.
# Critical-path validation gate: if DS won't boot on this HEAD or numbers are insane,
# we find out here before spending the rest of the budget.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
cd /sgl-workspace/sglang

LOG="$HERE/stage1_ds.log"
exec > >(tee "$LOG") 2>&1
echo "=== STAGE1 DS  start $(date -u +%H:%M:%S)Z  HEAD=$(git rev-parse --short HEAD) ==="

teardown
build_server_args ds07

TRACE="$HERE/case1_ds/torch/trace"; mkdir -p "$TRACE"
SERVE_LOG="$HERE/case1_ds/serve.log"; mkdir -p "$(dirname "$SERVE_LOG")"

echo ">>> booting DS server (mem 0.7, DS-on, GLM mask) ..."
env SGLANG_TORCH_PROFILER_DIR="$TRACE" \
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
if ! wait_ready; then
  echo "!! FAIL: DS server not ready"; echo "---- serve.log tail ----"; tail -50 "$SERVE_LOG"; teardown; exit 1
fi

CAP=$(max_batch_from_server)
DS_MAXBATCH=$(( CAP / 4608 ))
echo ">>> DS KV capacity max_total_num_tokens=${CAP}  => DS_MAXBATCH=floor(${CAP}/4608)=${DS_MAXBATCH}"
[[ "$DS_MAXBATCH" -ge 1 ]] || { echo "!! FAIL: could not derive DS max batch (cap=$CAP)"; tail -30 "$SERVE_LOG"; teardown; exit 1; }
echo "$DS_MAXBATCH" > "$HERE/ds_maxbatch.txt"

echo "=== DS directional serving sweep (conc 16/32/64, 1 trial, 180s window) $(date -u +%H:%M:%S)Z ==="
set +e
PORT=30000 HOST=127.0.0.1 MODE=double_sparsity \
  RESULTS_DIR="$HERE/serving" \
  CONCURRENCIES="16 32 64" TRIALS=1 \
  WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
  bash development/benchmark.sh
SWEEP_RC=$?
set -e
echo ">>> DS serving sweep rc=$SWEEP_RC"

echo "=== case1 DS torch decode profile @ bs ${DS_MAXBATCH} (10 steps) $(date -u +%H:%M:%S)Z ==="
BENCH_LOG="$HERE/case1_ds/bench.log"
RESULT="$HERE/case1_ds/result.jsonl"
set +e
env SGLANG_TORCH_PROFILER_DIR="$TRACE" \
  python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
  --model-path "$GLM" --trust-remote-code \
  --batch-size "$DS_MAXBATCH" --input-len 4096 --output-len 512 --temperature 0 --show-report \
  --profile --profile-by-stage --profile-activities CPU GPU --profile-steps 10 \
  --profile-output-dir "$TRACE" \
  --result-filename "$RESULT" > "$BENCH_LOG" 2>&1
PROF_RC=$?
set -e
NTRACE=$(find "$TRACE" -name '*.trace.json.gz' 2>/dev/null | wc -l)
echo ">>> case1 torch profile rc=$PROF_RC  trace_files=$NTRACE"
echo "---- bench.log show-report tail ----"; grep -iE "decode|throughput|latency|batch" "$BENCH_LOG" | tail -25

teardown
echo "=== STAGE1 DS done $(date -u +%H:%M:%S)Z  sweep_rc=$SWEEP_RC prof_rc=$PROF_RC ntrace=$NTRACE ==="
