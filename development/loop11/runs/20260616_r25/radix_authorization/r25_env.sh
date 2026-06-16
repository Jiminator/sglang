#!/usr/bin/env bash
# Loop 11 R25 — table-free DS radix-on authorization under the NEW value-equivalence
# criterion (schema ds_radix_fixture_state_tablefree_v2). Reuses the EXACT R24 boot
# infra (r24_env.sh: COMMON_ARGS op-point, ready_wait/smoke/teardown/gpu_idle_wait).
# No new production code; all harness lives under this run dir.
#
# Op-point (all boots): GLM-5.1-FP8, TP=8, fp8_e4m3, page 64, mem 0.8, max-running 64.
# DS table-free config: scorer_norm off, head_agg max, top_k 2048, page_size 64,
# channel_mask /models/glm51-fp8-channel-mask-s256.safetensors, device_buffer_size 4096,
# anchor off. NEVER set expandable_segments. ONE TP=8 server at a time.
set -uo pipefail

R25_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R24_ENV="$R25_HERE/../../20260616_r24/tablefree_radix/r24_env.sh"
# shellcheck disable=SC1090
source "$R24_ENV"   # GLM, MASK, PORT, COMMON_ARGS, DSA_ARGS, ready_wait/smoke/teardown/gpu_idle_wait

REPO=/sgl-workspace/sglang
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db

# The exact table-free DS config for this op-point (the serving config the artifact
# fingerprint is bound to). recall_oracle / selection_capture are toggled per-probe
# by appending the JSON; this base has BOTH off (used for the no-override boot).
DS_CFG_TABLEFREE='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

# recall_oracle ON variant (EAGER probes)
DS_CFG_RECALL='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

# selection_capture ON variant (EAGER probes)
DS_CFG_SELCAP='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "selection_capture": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

# recall_oracle + selection_capture ON (edge probe needs both)
DS_CFG_RECALL_SELCAP='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": true, "selection_capture": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

DEFAULT_SINK="$REPO/.sglang_ds_oracle/sink.jsonl"

echo "[r25_env] REPO=$REPO PORT=$PORT GLM=$GLM"
echo "[r25_env] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"
