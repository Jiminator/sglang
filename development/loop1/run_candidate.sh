#!/usr/bin/env bash
# Run one GLM-5.1-FP8 serving candidate end to end on 8x H200:
#   teardown -> fresh server (supplied flags) -> wait ready
#   -> development/benchmark.sh (unchanged) -> record metrics -> teardown.
#
# Every knob under test lives in EXTRA_ARGS; the fixed base is intentionally
# minimal (model / host / port / tp / parsers) so candidates are reproducible
# from EXTRA_ARGS + any SGLANG_* env alone.
#
# Caller supplies via environment:
#   TAG         short candidate name (filenames are derived from it)  [required]
#   EXTRA_ARGS  server flags beyond the fixed base                    [default ""]
#   RATIONALE   one-line note for the sweep table                     [optional]
#   plus any SGLANG_* env vars for the run.
set -uo pipefail

ROOT=/sgl-workspace/sglang
DIR="$ROOT/development/loop1"
LOGS="$DIR/logs"
RESULTS="$DIR/results"
mkdir -p "$LOGS" "$RESULTS"

# GLM-5.1-FP8 is pre-staged on shared cluster storage (705 GB, HF-cache layout);
# serve straight from the snapshot dir so launches are fully offline.
MODEL="${MODEL:-/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/cluster-storage/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
TAG="${TAG:?set TAG}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RATIONALE="${RATIONALE:-}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"

SERVE_LOG="$LOGS/serve_${TAG}.log"
BENCH_LOG="$LOGS/bench_${TAG}.log"

note() { echo "[$(date +%H:%M:%S)] $*"; }

teardown() {
  pkill -9 -f "sglang serve"        2>/dev/null || true
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "$MODEL"              2>/dev/null || true
  # wait for GPU memory to drain so the next launch starts clean
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-999999}" -lt 2000 ] && break
    sleep 2
  done
}

trap teardown EXIT
teardown
note "TAG=$TAG  EXTRA_ARGS=[$EXTRA_ARGS]"

# shellcheck disable=SC2086
sglang serve \
  --model-path "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --tp "$TP" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  $EXTRA_ARGS \
  > "$SERVE_LOG" 2>&1 &
SPID=$!

ready=0
deadline=$(( SECONDS + READY_TIMEOUT ))
while (( SECONDS < deadline )); do
  if ! kill -0 "$SPID" 2>/dev/null; then note "SERVER DIED during startup"; break; fi
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/health" 2>/dev/null || true)
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 5
done

if [ "$ready" != "1" ]; then
  note "NOT READY (timeout or death) — last 40 server-log lines:"
  tail -40 "$SERVE_LOG" || true
  echo "CANDIDATE_RESULT TAG=$TAG STATUS=launch_failed"
  exit 2
fi
note "server READY — running benchmark"

TAG="$TAG" RESULTS_DIR="$RESULTS" PORT="$PORT" HOST="$HOST" \
  bash "$ROOT/development/benchmark.sh" > "$BENCH_LOG" 2>&1
brc=$?
note "benchmark exit=$brc"
if [ "$brc" != "0" ]; then
  note "benchmark FAILED — last 30 lines:"; tail -30 "$BENCH_LOG" || true
fi

python3 "$DIR/parse_result.py" \
  --tag "$TAG" --serve-log "$SERVE_LOG" \
  --extra-args "$EXTRA_ARGS" --rationale "$RATIONALE"

echo "CANDIDATE_RESULT TAG=$TAG STATUS=done"
