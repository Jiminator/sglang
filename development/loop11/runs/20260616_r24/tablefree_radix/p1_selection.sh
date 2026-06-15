#!/usr/bin/env bash
# P1 (CORRECTED) — cold/warm SELECTED-INDEX equivalence. Boots DS radix-ON via
# SGLANG_DS_RADIX_OVERRIDE=1 (drops --disable-radix-cache), EAGER
# (--disable-cuda-graph so the selection-capture host dump fires), config-borne
# selection_capture=true. IDENTICAL cold/warm prompt; all 8 TP ranks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r24_env.sh"
REPO=/sgl-workspace/sglang
cd "$REPO"
LOG="$HERE/p1_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== P1 selection cold/warm start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

export SGLANG_DS_RADIX_OVERRIDE=1
# selection_capture dumps to the worker CWD (repo root) -> .sglang_ds_selcap.
DS_CONFIG_P1='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": false, "selection_capture": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'
P1_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_P1" --disable-cuda-graph)

teardown
# Ensure both driver-side and worker-side resolve the same capture dir.
export SGLANG_DS_SELECTION_CAPTURE_DIR="$REPO/.sglang_ds_selcap"
rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$HERE/cold" "$HERE/warm" 2>/dev/null || true

slog="$HERE/p1_serve.log"
echo "=== boot DS radix-ON eager selection_capture $(date -u +%H:%M:%SZ) ==="
echo "DS_CONFIG_P1=$DS_CONFIG_P1"
SGLANG_DS_RADIX_OVERRIDE=1 python -m sglang.launch_server "${P1_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! P1 boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
  { echo "probe=p1_selection status=BOOT_FAIL rc_wait=$rc_wait";
    grep -anE "Error|ValueError|RuntimeError|OOM|Traceback|radix" "$slog" | tail -40; } > "$HERE/p1_evidence.txt"
  teardown; gpu_idle_wait; exit 1
fi
echo ">>> server ready (radix-ON, selection_capture). smoke=$(smoke)"
grep -aE "disable_radix_cache=False" "$slog" | head -1 || true

echo "=== run P1 selection driver $(date -u +%H:%M:%SZ) ==="
PORT=$PORT SGLANG_DS_SELECTION_CAPTURE_DIR="$SGLANG_DS_SELECTION_CAPTURE_DIR" \
python "$HERE/p1_selection_driver.py" --prefix-tokens 2400 --expect-ranks 8 \
  --out "$HERE/p1_selection_verdict.json"
DRV_RC=$?

teardown
{
  echo "probe=p1_selection (radix-ON, eager, selection_capture, IDENTICAL prompt) drv_rc=$DRV_RC"
  echo "--- verdict ---"
  python3 -c "import json;d=json.load(open('$HERE/p1_selection_verdict.json'));print('status',d.get('status'));print('warm_cached_tokens',d.get('warm_cached_tokens'));print('ranks',d.get('ranks_compared'));print('layers_per_rank',d.get('layers_per_rank'));print('all_cold_eq_warm',d.get('all_cold_eq_warm'));print('cross_rank_identical',d.get('cross_rank_identical'));import sys;[print('rank',r,'cold_eq_warm',v.get('cold_eq_warm'),'matches_rank0',v.get('matches_rank0'),'compare_len',v.get('compare_len')) for r,v in sorted(d.get('per_rank',{}).items(),key=lambda x:int(x[0]))]" 2>&1
} > "$HERE/p1_evidence.txt"
cat "$HERE/p1_evidence.txt"
echo "=== P1 done $(date -u +%H:%M:%SZ) ==="
gpu_idle_wait
exit $DRV_RC
