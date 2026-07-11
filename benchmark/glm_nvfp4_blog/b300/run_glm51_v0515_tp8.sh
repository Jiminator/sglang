#!/usr/bin/env bash
# nvidia/GLM-5.1-NVFP4 | release/v0.5.15 | 8xB300 | TP8: server + sweep client.
set -euo pipefail
cd "$(dirname "$0")/.."
source common.sh

OUT=results/b300/glm51_v0515
sweep_already_done "$OUT/tp8" && exit 0
ensure_evalscope

export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1
start_server "$OUT/server_tp8.log" python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.1-NVFP4 \
    --tensor-parallel-size 8 \
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

./run_client.sh nvidia/GLM-5.1-NVFP4 "$OUT" tp8
stop_server
