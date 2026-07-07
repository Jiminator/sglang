#!/usr/bin/env bash
# Concurrency sweep for the agentic multi-turn dataset (see README.md).
# Runs one (num_prompts, concurrency) point per step, advancing
# --agentic-offset so every step replays fresh conversations. The server
# must run with --enable-cache-report.
#
# Usage: ./sweep.sh [extra sglang.benchmark.serving args...]
# Env overrides: BACKEND, HOST, PORT, NUM_CONVERSATIONS, OUTPUT_FILE,
#                SWEEP_PAIRS ("num_prompts:concurrency" pairs)

set -euo pipefail

BACKEND=${BACKEND:-sglang-oai-chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
NUM_CONVERSATIONS=${NUM_CONVERSATIONS:-128}
OUTPUT_FILE=${OUTPUT_FILE:-agentic_sweep.jsonl}
SWEEP_PAIRS=${SWEEP_PAIRS:-"4:1 8:2 8:4 16:8 32:16"}

# One large-prefill warmup absorbs the first-run JIT compile/autotune that
# would otherwise land in step 1; step 1's --flush-cache wipes its KV.
echo "=== warmup: one large-prefill request to absorb first-run JIT ==="
python3 -m sglang.benchmark.serving \
  --backend "$BACKEND" \
  --host "$HOST" --port "$PORT" \
  --dataset-name random-ids \
  --random-input-len 8192 --random-output-len 1 --random-range-ratio 1 \
  --num-prompts 1 \
  --warmup-requests 0

offset=0
for pair in $SWEEP_PAIRS; do
  num_prompts=${pair%%:*}
  concurrency=${pair##*:}
  echo "=== agentic sweep step: num_prompts=${num_prompts}" \
    "concurrency=${concurrency} offset=${offset} ==="
  python3 -m sglang.benchmark.serving \
    --backend "$BACKEND" \
    --host "$HOST" --port "$PORT" \
    --dataset-name agentic \
    --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" \
    --agentic-num-conversations "$NUM_CONVERSATIONS" \
    --agentic-offset "$offset" \
    --warmup-requests 0 \
    --flush-cache \
    --cache-report \
    --output-file "$OUTPUT_FILE" \
    "$@"
  offset=$((offset + num_prompts))
done

echo "Sweep complete; results appended to ${OUTPUT_FILE}"
