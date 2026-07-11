#!/usr/bin/env bash
# nvidia/GLM-5.2-NVFP4 | day-0 snapshot | 8xB300 | TEP8: server + sweep client.
# Launch-day flags on purpose: no --bf16-gemm-backend, no fused-top-k or
# deferred-finalize env vars, and the day-0 --cuda-graph-max-bs spelling.
set -euo pipefail
cd "$(dirname "$0")/.."
source common.sh

OUT=results/b300/day0
sweep_already_done "$OUT/tep8" && exit 0
ensure_evalscope
ensure_day0_checkout
export PYTHONPATH="$DAY0_SGLANG/python"

start_server "$OUT/server_tep8.log" python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 8 \
    --ep-size 8 \
    --quantization modelopt_fp4 \
    --context-length 90000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs 16 \
    --mem-fraction-static 0.87 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --host localhost \
    --port "$PORT"

./run_client.sh nvidia/GLM-5.2-NVFP4 "$OUT" tep8
stop_server
