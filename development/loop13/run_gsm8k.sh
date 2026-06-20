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
cd "$EVID"

run() {
  local tag="$1" shots="$2" n="$3"
  echo ">>> [$LABEL/$tag] gsm8k --api completion --num-shots $shots --num-examples $n"
  python3 -m sglang.test.run_eval --eval-name gsm8k --api completion \
    --host "$HOST" --port "$PORT" \
    --num-shots "$shots" --num-examples "$n" \
    --temperature 0 --max-tokens 512 --num-threads 64 \
    > "$EVID/${LABEL}_${tag}.out" 2>&1
  echo "    rc=$? -> $(grep -E '^Score:' "$EVID/${LABEL}_${tag}.out" | tr '\n' ' ')"
}

run dense  5 200   # ~763 tok prompts: seq < top_k(2048) -> DS dense path
run sparse 24 150  # ~4.2k tok prompts: seq > 2048 -> DS sparse selection active
echo ">>> done"
