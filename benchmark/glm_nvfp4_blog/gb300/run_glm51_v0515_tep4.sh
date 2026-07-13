#!/usr/bin/env bash
# nvidia/GLM-5.1-NVFP4 | release/v0.5.15 | 4xGB300 | TEP4: server + sweep client.
set -euo pipefail
cd "$(dirname "$0")/.."
source common.sh

OUT=results/gb300/glm51_v0515
sweep_already_done "$OUT/tep4" && exit 0
ensure_evalscope
ensure_v0515_checkout
export PYTHONPATH="$V0515_SGLANG/python"

export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1
start_server "$OUT/server_tep4.log" python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.1-NVFP4 \
    --tensor-parallel-size 4 \
    --ep-size 4 \
    --quantization modelopt_fp4 \
    --context-length 90000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs-decode 16 \
    --mem-fraction-static 0.87 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --bf16-gemm-backend cutedsl \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --host localhost \
    --port "$PORT"

./run_client.sh nvidia/GLM-5.1-NVFP4 "$OUT" tep4
stop_server
