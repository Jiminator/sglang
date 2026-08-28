#!/usr/bin/env bash
# zai-org/GLM-5.3 (FP8) | container SGLang | 8xB300 | TEP8: server + sweep client.
set -euo pipefail
cd "$(dirname "$0")/.."
source common.sh

OUT=results/b300/glm53
sweep_already_done "$OUT/tep8" && exit 0
ensure_evalscope

export SGLANG_OPT_USE_TOPK_V2=1
start_server "$OUT/server_tep8.log" python3 -m sglang.launch_server \
    --model-path zai-org/GLM-5.3 \
    --tensor-parallel-size 8 \
    --ep-size 8 \
    --context-length 90000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs 16 \
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

./run_client.sh zai-org/GLM-5.3 "$OUT" tep8
stop_server
