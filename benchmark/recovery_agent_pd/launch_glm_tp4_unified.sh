#!/usr/bin/env bash
# GLM-5.2-NVFP4 TP4 unified (no PD) on one 4-GPU GB300 node. STAGE selects
# how far past the v0.5.16-verified baseline the config steps:
#   STAGE=verified  the v0.5.16-verified GB300 NVFP4 recipe (EAGLE 5-1-6)
#   STAGE=doc-spec  verified recipe with the reference 3-1-4 speculation
#   STAGE=doc-full  reference attention/MoE backends on top of doc-spec
#
#   MODEL_PATH=/scratch/models/GLM-5.2-NVFP4 STAGE=verified \
#     bash launch_glm_tp4_unified.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/scratch/models/GLM-5.2-NVFP4}
STAGE=${STAGE:-verified}
PORT=${PORT:-30000}

args=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --tp 4
  --quantization modelopt_fp4
  --kv-cache-dtype fp8_e4m3
  --chunked-prefill-size 8192
  --max-prefill-tokens 8192
  --mem-fraction-static 0.85
  --max-running-requests 16
  --cuda-graph-max-bs 16
  --reasoning-parser glm45
  --host 0.0.0.0 --port "$PORT"
)

case "$STAGE" in
verified)
  args+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 5 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
    --bf16-gemm-backend cutedsl
  )
  ;;
doc-spec)
  args+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --bf16-gemm-backend cutedsl
  )
  ;;
doc-full)
  args+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --attention-backend dsa
    --moe-runner-backend flashinfer_trtllm_routed
  )
  ;;
*)
  echo "unknown STAGE=$STAGE (verified|doc-spec|doc-full)" >&2
  exit 1
  ;;
esac

exec python3 -m sglang.launch_server "${args[@]}"
