#!/usr/bin/env bash
# Stage 4 (DS nsys cross-check): boot DS@0.7 under nsys -> case1 nsys capture @ DS max
# batch, OSL 64 (kernel mix is stationary; short window keeps the .nsys-rep finalizable
# in seconds — confirms the f32 all-reduce / serialization story on current HEAD).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/stage4_ds_nsys.log"; exec > >(tee "$LOG") 2>&1
echo "=== STAGE4 DS nsys start $(date -u +%H:%M:%S)Z ==="

DS_MAXBATCH=$(cat "$HERE/ds_maxbatch.txt" 2>/dev/null || echo 29)
DELAY="${DELAY:-240}"
ND="$HERE/case1_ds/nsys"; mkdir -p "$ND"
SERVE_LOG="$ND/serve.log"; BENCH_LOG="$ND/bench.log"; RESULT="$ND/result.jsonl"

teardown
build_server_args ds07
echo ">>> booting DS server under nsys (delay ${DELAY}s, OSL 64, bs ${DS_MAXBATCH}) ..."
START=$(date +%s)
nsys profile --output "$ND/trace" --force-overwrite true \
  --trace cuda,nvtx,cublas --cuda-graph-trace node --trace-fork-before-exec true \
  --delay "$DELAY" --duration 900 \
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
if ! wait_ready; then echo "!! FAIL: DS nsys server not ready"; tail -50 "$SERVE_LOG"; teardown; exit 1; fi
NOW=$(date +%s); ELAPSED=$((NOW-START))
if [[ $ELAPSED -lt $((DELAY+3)) ]]; then SLEEP=$((DELAY+3-ELAPSED)); echo ">>> wait ${SLEEP}s for nsys collection live"; sleep "$SLEEP"; fi

set +e
python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
  --model-path "$GLM" --trust-remote-code \
  --batch-size "$DS_MAXBATCH" --input-len 4096 --output-len 64 --temperature 0 --show-report \
  --result-filename "$RESULT" > "$BENCH_LOG" 2>&1
BENCH_RC=$?; set -e
SID=$(nsys sessions list 2>/dev/null | grep -oE "profile-[0-9A-Za-z]+" | head -1)
echo ">>> nsys stop session=$SID"; [[ -n "$SID" ]] && nsys stop --session="$SID" 2>&1 | tail -3 || echo "WARN no session id"
sleep 5
teardown
REP=$(find "$ND" -name '*.nsys-rep' 2>/dev/null | head -1)
echo ">>> case1 nsys rc=$BENCH_RC rep=${REP:-NONE} $([[ -n "$REP" ]] && du -h "$REP" | cut -f1)"
echo "=== STAGE4 DS nsys done $(date -u +%H:%M:%S)Z ==="
