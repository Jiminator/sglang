#!/usr/bin/env bash
# Run two GSM8K configs against the already-running server on $PORT.
# Uses the DEFAULT editable sglang install (= dev clone /sgl-workspace/sglang).
# MODE label only tags output files; the server identity (ds/dsa) is whatever is booted.
set -uo pipefail
HOST=127.0.0.1; PORT=30000
MODE="${1:?mode}"
OUT=/sgl-workspace/sglang/development/loop12/gsm8k_evidence
cd "$OUT"
run() {
  local tag="$1" shots="$2" n="$3"
  echo ">>> [$MODE/$tag] gsm8k --api completion --num-shots $shots --num-examples $n"
  python3 -m sglang.test.run_eval \
    --eval-name gsm8k --api completion \
    --host "$HOST" --port "$PORT" \
    --num-shots "$shots" --num-examples "$n" \
    --temperature 0 --max-tokens 512 --num-threads 64 \
    > "$OUT/${MODE}_${tag}.out" 2>&1
  echo "    rc=$? -> $(grep -E '^(Score|Total latency|Output throughput):' "$OUT/${MODE}_${tag}.out" | tr '\n' ' ')"
}
run short 5 200
run long 24 150
echo ">>> done"
