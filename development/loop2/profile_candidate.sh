#!/usr/bin/env bash
# Profile-only diagnostic run for one GLM-5.1-FP8 serving candidate on 8x H200.
#
# This is NON-SCORING: it never produces the official gate number. It launches a
# fresh server with the SAME flags as the gate run, then replays the SAME
# generated-shared-prefix workload as development/benchmark.sh (identical ISL/OSL/
# concurrency/seed) via a SEPARATE bench_serving command carrying torch-profiler
# flags. The fixed development/benchmark.sh is never modified or invoked here.
#
# A bounded window deep in the run is captured (start_step skips warmup/cold
# prefill, num_steps bounds a steady-state window) so the trace is dominated by the
# speculative decode loop. We intentionally DO NOT pass --profile-by-stage, because
# it classifies TARGET_VERIFY as prefill (forward_batch_info.py:109-118); a plain
# window keeps DECODE + TARGET_VERIFY + DRAFT_EXTEND together for correct grouping.
#
# Raw traces land in profiling/raw/<tag>/ and are meant to be analyzed into a
# markdown insights file, after which the raw trace is deleted (disk hygiene).
#
# Caller supplies via environment:
#   TAG         short candidate name                                  [required]
#   EXTRA_ARGS  server flags beyond the fixed base                    [default ""]
#   plus any SGLANG_* env vars for the run.
set -uo pipefail

ROOT=/sgl-workspace/sglang
DIR="$ROOT/development/loop2"
LOGS="$DIR/logs"
RAW_BASE="$DIR/profiling/raw"
mkdir -p "$LOGS" "$RAW_BASE"

MODEL="${MODEL:-/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/cluster-storage/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
TAG="${TAG:?set TAG}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"

# Workload identical to development/benchmark.sh (do NOT change these).
SEED="${SEED:-31234}"
CONCURRENCY="${CONCURRENCY:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-$(( 5 * CONCURRENCY ))}"
SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512

# Profiler window: skip warmup/cold prefill, capture a bounded steady-state window.
PROFILE_START_STEP="${PROFILE_START_STEP:-150}"
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-40}"
PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU GPU}"

RAW_OUT="$RAW_BASE/$TAG"
rm -rf "$RAW_OUT"; mkdir -p "$RAW_OUT"
export SGLANG_TORCH_PROFILER_DIR="$RAW_OUT"

SERVE_LOG="$LOGS/profile_serve_${TAG}.log"
BENCH_LOG="$LOGS/profile_bench_${TAG}.log"

note() { echo "[$(date +%H:%M:%S)] $*"; }

teardown() {
  pkill -9 -f "sglang serve"         2>/dev/null || true
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "$MODEL"               2>/dev/null || true
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-999999}" -lt 2000 ] && break
    sleep 2
  done
}

trap teardown EXIT
teardown
note "PROFILE TAG=$TAG  EXTRA_ARGS=[$EXTRA_ARGS]  profiler_dir=$RAW_OUT"

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
  echo "PROFILE_RESULT TAG=$TAG STATUS=launch_failed"
  exit 2
fi
note "server READY — running profile-only bench (window: start=$PROFILE_START_STEP num=$PROFILE_NUM_STEPS)"

# shellcheck disable=SC2086
python3 -m sglang.bench_serving \
  --backend sglang \
  --host "$HOST" --port "$PORT" \
  --seed "$SEED" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group "$NUM_PROMPTS" \
  --gsp-system-prompt-len "$SYS_LEN" \
  --gsp-question-len "$Q_LEN" \
  --gsp-output-len "$OUT_LEN" \
  --gsp-range-ratio 1.0 \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --output-details \
  --profile \
  --profile-activities $PROFILE_ACTIVITIES \
  --profile-start-step "$PROFILE_START_STEP" \
  --profile-num-steps "$PROFILE_NUM_STEPS" \
  --profile-output-dir "$RAW_OUT" \
  --profile-prefix "$TAG" \
  > "$BENCH_LOG" 2>&1
brc=$?
note "profile bench exit=$brc"

traces=$(find "$RAW_OUT" -type f \( -name '*.trace.json*' -o -name '*.json.gz' -o -name '*.pt.trace.json*' \) 2>/dev/null | wc -l)
note "trace files captured: $traces  (in $RAW_OUT)"
if [ "$brc" != "0" ] || [ "$traces" -lt 1 ]; then
  note "PROFILE bench failed or no trace — last 30 lines:"; tail -30 "$BENCH_LOG" || true
  echo "PROFILE_RESULT TAG=$TAG STATUS=profile_failed traces=$traces"
  exit 4
fi
echo "PROFILE_RESULT TAG=$TAG STATUS=done traces=$traces dir=$RAW_OUT"
