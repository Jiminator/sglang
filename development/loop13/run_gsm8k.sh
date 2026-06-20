#!/usr/bin/env bash
# Run GSM8K (dense + sparse) against the live server using the DEV clone's
# run_eval (no PYTHONPATH).  Greedy, completion API (no thinking template, no
# shot/question leakage).
#   Usage:  run_gsm8k.sh <label>     e.g.  run_gsm8k.sh dsa   /   run_gsm8k.sh ds
# Run BACKGROUNDED (the sparse arm takes a few minutes).
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
# shellcheck source=_env.sh
source "$HERE/_env.sh" || exit 1

LABEL="${1:?usage: run_gsm8k.sh <label>}"
# THREADS controls concurrency: 64 = batched (default), 1 = serial control.
THREADS="${THREADS:-64}"
cd "$EVID"

run() {
  local tag="$1" shots="$2" n="$3"
  echo ">>> [$LABEL/$tag] gsm8k --api completion --num-shots $shots --num-examples $n --num-threads $THREADS"
  python3 -m sglang.test.run_eval --eval-name gsm8k --api completion \
    --host "$HOST" --port "$PORT" \
    --num-shots "$shots" --num-examples "$n" \
    --temperature 0 --max-tokens 512 --num-threads "$THREADS" \
    > "$EVID/${LABEL}_${tag}.out" 2>&1
  echo "    rc=$? -> $(grep -E '^Score:' "$EVID/${LABEL}_${tag}.out" | tr '\n' ' ')"
}

# REGIME selects which arms to run: dense | sparse | both (default both).
REGIME="${REGIME:-both}"
case "$REGIME" in
  dense) run dense  5 200 ;;                                   # ~763 tok: seq < top_k(2048)
  sparse) run sparse 24 150 ;;                                 # ~4.2k tok: seq > 2048
  both) run dense 5 200; run sparse 24 150 ;;
  *) echo "FATAL: REGIME must be dense|sparse|both"; exit 2 ;;
esac
echo ">>> done"
