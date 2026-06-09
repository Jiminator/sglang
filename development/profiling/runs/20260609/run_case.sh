#!/usr/bin/env bash
# One profiling run end-to-end: boot server -> wait_ready -> drive ONE bench batch
# -> finalize -> teardown -> verify artifact. Exits 0 only if the expected trace
# artifact + a completed bench batch are present.
#
#   run_case.sh <casedir> <case> <mode> <bs>
#     casedir : output subdir, e.g. case1_ds
#     case    : case1|case2|case3 (selects server args)
#     mode    : nsys|torch
#     bs      : batch size
#
# Env overrides: DELAY (nsys collection start, default 210), PROFILE_STEPS (torch,
# default 10), WITH_STACK=0 to export SGLANG_PROFILE_WITH_STACK=False on the server
# (torch profiler with_stack-bug fallback).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"

CASEDIR="$1"; CASE="$2"; MODE="$3"; BS="$4"
DELAY="${DELAY:-210}"
PROFILE_STEPS="${PROFILE_STEPS:-10}"
OSL="${OSL:-512}"   # decode steps the bench generates; nsys runs use a shorter window (kernel mix is stationary)
RUNDIR="$OUT/$CASEDIR/$MODE"
mkdir -p "$RUNDIR"
SERVE_LOG="$RUNDIR/serve.log"
BENCH_LOG="$RUNDIR/bench.log"
RESULT="$RUNDIR/result.jsonl"

build_server_args "$CASE" || exit 2
echo "=== RUN $CASEDIR/$MODE  case=$CASE bs=$BS  $(date -u +%H:%M:%S)Z ==="
echo "    server args: ${SERVER_ARGS[*]}"

teardown   # ensure clean slate

fail() { echo "!! FAIL: $1"; echo "---- tail serve.log ----"; tail -40 "$SERVE_LOG" 2>/dev/null; echo "---- tail bench.log ----"; tail -40 "$BENCH_LOG" 2>/dev/null; teardown; exit 1; }

if [[ "$MODE" == "torch" ]]; then
  TR="$RUNDIR/trace"; mkdir -p "$TR"
  STACK_ENV=()
  [[ "${WITH_STACK:-1}" == "0" ]] && STACK_ENV=(SGLANG_PROFILE_WITH_STACK=False)
  env SGLANG_TORCH_PROFILER_DIR="$TR" "${STACK_ENV[@]}" \
    python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "server not ready (torch $CASEDIR)"
  env SGLANG_TORCH_PROFILER_DIR="$TR" \
    python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
    --model-path "$GLM" --trust-remote-code \
    --batch-size "$BS" --input-len 4096 --output-len "$OSL" --temperature 0 --show-report \
    --profile --profile-by-stage --profile-activities CPU GPU --profile-steps "$PROFILE_STEPS" \
    --profile-output-dir "$TR" \
    --result-filename "$RESULT" > "$BENCH_LOG" 2>&1
  BENCH_RC=$?
  teardown
  [[ $BENCH_RC -eq 0 ]] || fail "bench exited $BENCH_RC (torch $CASEDIR)"
  NTRACE=$(find "$TR" -name '*.trace.json.gz' 2>/dev/null | wc -l)
  [[ "$NTRACE" -gt 0 ]] || fail "no torch trace files under $TR"
  echo ">>> OK torch: $NTRACE trace files under $TR"

elif [[ "$MODE" == "nsys" ]]; then
  ND="$RUNDIR"
  START=$(date +%s)
  nsys profile --output "$ND/trace" --force-overwrite true \
    --trace cuda,nvtx,cublas --cuda-graph-trace node --trace-fork-before-exec true \
    --delay "$DELAY" --duration 900 \
    python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "server not ready (nsys $CASEDIR)"
  # ensure nsys collection (starts at DELAY s) is active before the bench batch
  NOW=$(date +%s); ELAPSED=$((NOW-START))
  if [[ $ELAPSED -lt $((DELAY+3)) ]]; then
    SLEEP=$((DELAY+3-ELAPSED)); echo ">>> waiting ${SLEEP}s so nsys collection is live before bench"; sleep "$SLEEP"
  fi
  python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
    --model-path "$GLM" --trust-remote-code \
    --batch-size "$BS" --input-len 4096 --output-len "$OSL" --temperature 0 --show-report \
    --result-filename "$RESULT" > "$BENCH_LOG" 2>&1
  BENCH_RC=$?
  # finalize the trace instantly via session stop (more robust than waiting for --duration)
  SID=$(nsys sessions list 2>/dev/null | grep -oE "profile-[0-9A-Za-z]+" | head -1)
  echo ">>> nsys stop session=$SID"
  [[ -n "$SID" ]] && nsys stop --session="$SID" 2>&1 | tail -3 || echo "WARN: no nsys session id found"
  sleep 5
  teardown
  [[ $BENCH_RC -eq 0 ]] || fail "bench exited $BENCH_RC (nsys $CASEDIR)"
  REP=$(find "$ND" -name '*.nsys-rep' 2>/dev/null | head -1)
  [[ -n "$REP" ]] || fail "no .nsys-rep under $ND"
  echo ">>> OK nsys: $REP ($(du -h "$REP" | cut -f1))"
else
  echo "unknown mode $MODE"; exit 2
fi

# verify the bench produced a completed batch (decode throughput present)
if grep -qiE "decode.*throughput|output_throughput|Decode\. +[0-9]" "$BENCH_LOG" 2>/dev/null || [[ -s "$RESULT" ]]; then
  echo ">>> bench batch completed; result=$RESULT"
else
  fail "bench log shows no completed batch / empty result"
fi
echo "=== DONE $CASEDIR/$MODE  $(date -u +%H:%M:%S)Z ==="
