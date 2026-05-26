#!/usr/bin/env bash
# bench_serving sweep — gsp, 4096 ISL / 512 OSL, ~55% prefix cache hit
set -euo pipefail

PORT="${PORT:-30000}"
MODE="${MODE:-dsv32}"

SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512
NUM_PROMPTS=$(( 5 * 64 ))
NUM_GROUPS=1

declare -A SEEDS=( [16]=213 [32]=431 [64]=31234 )
CONCURRENCIES="${CONCURRENCIES:-64}"

for CONCURRENCY in ${CONCURRENCIES}; do
  SEED="${SEEDS[$CONCURRENCY]}"
  OUTPUT_FILE="${MODE}_gsp_isl4096_osl512_c${CONCURRENCY}.jsonl"
  echo ">>> mode=${MODE} concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS} seed=${SEED} output=${OUTPUT_FILE}"

  python3 -m sglang.bench_serving \
    --backend sglang \
    --port "${PORT}" \
    --seed "${SEED}" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "${NUM_GROUPS}" \
    --gsp-prompts-per-group "${NUM_PROMPTS}" \
    --gsp-system-prompt-len "${SYS_LEN}" \
    --gsp-question-len "${Q_LEN}" \
    --gsp-output-len "${OUT_LEN}" \
    --gsp-range-ratio 1.0 \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --output-file "${OUTPUT_FILE}" \
    --output-details
done
