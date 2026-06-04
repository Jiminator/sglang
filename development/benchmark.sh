#!/usr/bin/env bash
# bench_serving — GLM-5.1-FP8 client SLO workload (see development/CLIENT_SLOS.md)
#
# Workload: 4096 ISL / 512 OSL, max-concurrency 64, ~55% prefix cache hit.
# The generated-shared-prefix dataset produces the cache hit: a single shared
# system prompt (SYS_LEN) is reused across every request, so the steady-state
# prefix-cache hit rate ≈ SYS_LEN / (SYS_LEN + Q_LEN) = 2253 / 4096 ≈ 0.55.
#
# A fixed seed keeps the generated prompts identical across runs for
# reproducibility. The server must already be running on PORT (default 30000).
# Results (JSONL + console log) are written to RESULTS_DIR.

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SEED="${SEED:-31234}"
CONCURRENCY="${CONCURRENCY:-64}"
# 5 epochs at the requested concurrency.
NUM_PROMPTS="${NUM_PROMPTS:-$(( 5 * CONCURRENCY ))}"

# ISL = SYS_LEN + Q_LEN = 2253 + 1843 = 4096; cache hit = 2253 / 4096 ≈ 0.55
SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
mkdir -p "${RESULTS_DIR}"

TAG="${TAG:-glm51}"
OUTPUT_FILE="${RESULTS_DIR}/${TAG}_isl4096_osl512_c${CONCURRENCY}.jsonl"
LOG_FILE="${OUTPUT_FILE%.jsonl}.log"

echo ">>> bench_serving: concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS} seed=${SEED}"
echo "    output = ${OUTPUT_FILE}"
echo "    log    = ${LOG_FILE}"

python3 -m sglang.bench_serving \
  --backend sglang \
  --host "${HOST}" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group "${NUM_PROMPTS}" \
  --gsp-system-prompt-len "${SYS_LEN}" \
  --gsp-question-len "${Q_LEN}" \
  --gsp-output-len "${OUT_LEN}" \
  --gsp-range-ratio 1.0 \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${CONCURRENCY}" \
  --output-file "${OUTPUT_FILE}" \
  --output-details \
  2>&1 | tee "${LOG_FILE}"
