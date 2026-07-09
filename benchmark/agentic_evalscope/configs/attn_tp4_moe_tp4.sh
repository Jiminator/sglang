#!/usr/bin/bash

set -euo pipefail

: "${HOST:=localhost}"
: "${PORT:=8002}"

# GLM-5.2-NVFP4 on 4 GPUs: attention TP4, MoE TP4. Spec 5/6, mem 0.87.
# hicache intentionally DISABLED — never on for this sweep.
#
# --fp4-gemm-backend and --moe-runner-backend are intentionally UNSET →
# both resolve to 'auto'. Confirmed NO-OP for GLM-5.2-NVFP4: its FP4 (E2m1) weights
# are all MoE experts (routed through the moe runner = auto flashinfer_trtllm on
# sm100); there are NO FP4 *dense* linears for --fp4-gemm-backend to act on (dense
# attn/MLP linears are BF16 → --bf16-gemm-backend cutedsl handles those, PR#30117).
#
# No PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — it breaks CUDA-graph
# capture on this image.
exec python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \
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
    --host "${HOST}" \
    --port "${PORT}"
