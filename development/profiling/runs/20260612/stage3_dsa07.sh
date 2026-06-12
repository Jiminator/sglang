#!/usr/bin/env bash
# Stage 3 (DSA apples-to-apples, mem 0.7): boot -> case2 torch decode profile @ the
# SAME batch DS was capped to (ds_maxbatch.txt). Clean DS-overhead diff = case1-case2.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/stage3_dsa07.log"; exec > >(tee "$LOG") 2>&1
echo "=== STAGE3 DSA@0.7 start $(date -u +%H:%M:%S)Z ==="

DS_MAXBATCH=$(cat "$HERE/ds_maxbatch.txt" 2>/dev/null || echo 29)
echo ">>> case2 will profile at DS_MAXBATCH=${DS_MAXBATCH}"

teardown
build_server_args dsa07
TRACE="$HERE/case2_dsa07/torch/trace"; mkdir -p "$TRACE"
SERVE_LOG="$HERE/case2_dsa07/serve.log"; mkdir -p "$(dirname "$SERVE_LOG")"

env SGLANG_TORCH_PROFILER_DIR="$TRACE" \
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
if ! wait_ready; then echo "!! FAIL: DSA@0.7 not ready"; tail -50 "$SERVE_LOG"; teardown; exit 1; fi

echo "=== case2 DSA@0.7 torch decode profile @ bs ${DS_MAXBATCH} $(date -u +%H:%M:%S)Z ==="
BENCH_LOG="$HERE/case2_dsa07/bench.log"; RESULT="$HERE/case2_dsa07/result.jsonl"
set +e
env SGLANG_TORCH_PROFILER_DIR="$TRACE" \
  python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
  --model-path "$GLM" --trust-remote-code \
  --batch-size "$DS_MAXBATCH" --input-len 4096 --output-len 512 --temperature 0 --show-report \
  --profile --profile-by-stage --profile-activities CPU GPU --profile-steps 10 \
  --profile-output-dir "$TRACE" --result-filename "$RESULT" > "$BENCH_LOG" 2>&1
PROF_RC=$?; set -e
NTRACE=$(find "$TRACE" -name '*.trace.json.gz' 2>/dev/null | wc -l)
echo ">>> case2 torch rc=$PROF_RC trace_files=$NTRACE"
grep -iE "decode|throughput|latency|batch" "$BENCH_LOG" | tail -25
teardown
echo "=== STAGE3 DSA@0.7 done $(date -u +%H:%M:%S)Z prof_rc=$PROF_RC ntrace=$NTRACE ==="
