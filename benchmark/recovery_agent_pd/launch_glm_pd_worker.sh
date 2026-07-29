#!/usr/bin/env bash
# One GLM-5.2-NVFP4 TP4 PD worker for the 2 prefill + 2 decode deployment
# (one worker per 4-GPU GB300 node). ROLE selects prefill|decode; STAGE ramps
# features so failures attribute cleanly:
#   STAGE=1  dense baseline (--attention-backend triton, no speculation)
#   STAGE=2  +DSA attention (prefill adds --dsa-prefill-backend trtllm)
#   STAGE=3  +EAGLE 3-1-4 speculation
#   STAGE=4  +HiCache host tier on prefill workers (write_back)
#
#   ROLE=prefill STAGE=3 MODEL_PATH=/scratch/models/GLM-5.2-NVFP4 \
#     bash launch_glm_pd_worker.sh
set -euo pipefail

ROLE=${ROLE:?prefill or decode}
STAGE=${STAGE:-1}
MODEL_PATH=${MODEL_PATH:-/scratch/models/GLM-5.2-NVFP4}
PORT=${PORT:-30001}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
HICACHE_SIZE_GB=${HICACHE_SIZE_GB:-32} # per DP rank; reference used 160

export SGLANG_DISAGGREGATION_NIXL_BACKEND=${SGLANG_DISAGGREGATION_NIXL_BACKEND:-UCX}
export UCX_TLS=${UCX_TLS:-cuda_ipc,cuda_copy,rc}
# UCX_NET_DEVICES is deliberately not defaulted: discover the host's HCAs
# (ibv_devices) and export it yourself, or leave unset to let UCX choose.
#
# MNNVL (NVLink-fabric) KV transfer — the validated fast path on GB300
# NVL72 trays sharing one fabric clique. The KV pool MUST be fabric-
# allocated for cross-node cuda_ipc to engage: SGLANG_MOONCAKE_CUSTOM_MEM_POOL
# gates a generic cuMem fabric allocator (env-only, works for nixl too).
# Never substitute PYTORCH_CUDA_ALLOC_CONF=expandable_segments — torch VMM
# exports POSIX-FD handles, not fabric handles, and segfaults on first copy.
# Measured (1 session, 12k ctx): mean TTFT 2244ms (TCP) -> 437ms (fabric).
# Set SGLANG_MOONCAKE_CUSTOM_MEM_POOL="" to fall back to non-fabric transfer.
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=${SGLANG_MOONCAKE_CUSTOM_MEM_POOL-NVLINK}
export UCX_CUDA_IPC_ENABLE_MNNVL=${UCX_CUDA_IPC_ENABLE_MNNVL:-y}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-1}

args=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --tp 4
  --quantization modelopt_fp4
  --kv-cache-dtype fp8_e4m3
  --disaggregation-mode "$ROLE"
  --disaggregation-transfer-backend nixl # v0.5.16 defaults to mooncake
  --reasoning-parser glm45
  --enable-metrics --enable-cache-report
  --host 0.0.0.0 --port "$PORT"
)

if [ "$ROLE" = prefill ]; then
  args+=(
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
    --context-length 262144
    --mem-fraction-static 0.93
    --chunked-prefill-size 16384
    --max-prefill-tokens 32768
    --max-running-requests 32
    --disable-overlap-schedule
    --cuda-graph-config '{"prefill": {"backend": "disabled"}, "decode": {"backend": "disabled"}}'
  )
else
  args+=(
    --moe-a2a-backend none
    --fp4-gemm-backend flashinfer_cutlass
    --mem-fraction-static 0.87
    --max-running-requests 128
    --num-continuous-decode-steps 3
    --flashinfer-allreduce-fusion-backend auto
    --cuda-graph-config '{"decode": {"backend": "full", "max_bs": 128}}'
  )
fi

if [ "$STAGE" -ge 2 ]; then
  args+=(--attention-backend dsa)
  if [ "$ROLE" = prefill ]; then
    args+=(--dsa-prefill-backend trtllm)
  fi
else
  # Explicit dense attention: v0.5.16 auto-selects DSA for this model, so the
  # baseline stage must pin a non-DSA backend to be genuinely distinct.
  args+=(--attention-backend triton)
fi

if [ "$STAGE" -ge 3 ]; then
  args+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
  if [ "$ROLE" = decode ]; then
    args+=(--speculative-attention-mode decode)
  fi
fi

if [ "$STAGE" -ge 4 ] && [ "$ROLE" = prefill ]; then
  args+=(
    --enable-hierarchical-cache
    --hicache-size "$HICACHE_SIZE_GB"
    --hicache-io-backend direct
    --hicache-mem-layout page_first_direct
    --hicache-write-policy write_back
  )
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '#COMMAND %s\n' "pd-worker:$ROLE:stage$STAGE"
  printf '%s\n' "${args[@]}"
  exit 0
fi

exec python3 -m sglang.launch_server "${args[@]}"
