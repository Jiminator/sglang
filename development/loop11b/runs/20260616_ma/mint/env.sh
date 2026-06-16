#!/usr/bin/env bash
# loop11b M-A radix re-mint + capacity env. Reuses loop-11 R24 boot infra
# (COMMON_ARGS op-point, DSA_ARGS, ready_wait/smoke/teardown/gpu_idle_wait) but points
# the DS channel-mask config at the REGENERATED loop11b mask. ONE TP=8 server at a time.
# NEVER set PYTORCH_CUDA_ALLOC_CONF=expandable_segments (breaks custom-all-reduce IPC).
set -uo pipefail

source "/sgl-workspace/sglang/development/loop11/runs/20260616_r24/tablefree_radix/r24_env.sh"

REPO=/sgl-workspace/sglang
# Regenerated mask (loop11b task4). DEC-1 pins the tensor-content SHA, so the path is
# free; the serve default and these configs point at the durable cluster-storage copy.
MASK=/cluster-storage/models/glm51-fp8-channel-mask-s256.safetensors
MASK_CONTENT_SHA256=a4be98c4c4989ea828b6ac128968af72336994b04cc1b6086408dbb208aa800d

# DS table-free serving config (the op-point the fixture fingerprint binds to). The
# recall_oracle / selection_capture variants append the per-probe toggles.
_BASE='"top_k": 2048, "page_size": 64, "channel_mask_path": "'"$MASK"'", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0'
DS_CFG_TABLEFREE="{$_BASE}"
DS_CFG_RECALL="{\"recall_oracle\": true, $_BASE}"
DS_CFG_SELCAP="{\"selection_capture\": true, $_BASE}"
DS_CFG_RECALL_SELCAP="{\"recall_oracle\": true, \"selection_capture\": true, $_BASE}"

DEFAULT_SINK="$REPO/.sglang_ds_oracle/sink.jsonl"

echo "[ma_env] MASK=$MASK content_sha256=$MASK_CONTENT_SHA256"
echo "[ma_env] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo SET)"
