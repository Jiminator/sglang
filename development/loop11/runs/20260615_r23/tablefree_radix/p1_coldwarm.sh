#!/usr/bin/env bash
# P1 — cold/warm SELECTION equivalence on a REAL radix cache hit (table-free).
# Boots DS radix-ON (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache), EAGER
# (--disable-cuda-graph; the latent-capture decode hook is capture-safe and only
# the eager prompt-prefix slots are host-readable), with the config-borne
# latent_capture diagnostic on. Then runs the cold/warm comparator driver.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r23_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/p1_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== P1 cold/warm start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

export SGLANG_DS_RADIX_OVERRIDE=1
export SGLANG_DS_LATENT_CAPTURE_DIR="$HERE/.sglang_ds_latentcap"
rm -rf "$SGLANG_DS_LATENT_CAPTURE_DIR" "$HERE/cold" "$HERE/warm" 2>/dev/null || true

# DS radix-ON config WITH latent_capture. EAGER (--disable-cuda-graph).
DS_CONFIG_P1='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": false, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "latent_capture": true}'
P1_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_P1" --disable-cuda-graph)

teardown
echo "=== boot DS radix-ON eager latent_capture $(date -u +%H:%M:%SZ) ==="
echo "DS_CONFIG_P1=$DS_CONFIG_P1"
slog="$HERE/p1_serve.log"
SGLANG_DS_RADIX_OVERRIDE=1 SGLANG_DS_LATENT_CAPTURE_DIR="$SGLANG_DS_LATENT_CAPTURE_DIR" \
  python -m sglang.launch_server "${P1_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! P1 boot FAIL (rc_wait=$rc_wait) — tail:"
  tail -n 60 "$slog"
  { echo "probe=p1_cold_warm status=BOOT_FAIL rc_wait=$rc_wait";
    grep -anE "Error|ValueError|AssertionError|RuntimeError|OOM|out of memory|Traceback|radix" "$slog" | tail -40; } > "$HERE/p1_evidence.txt"
  teardown
  exit 1
fi
echo ">>> server ready; confirming radix-ON authorization in log"
grep -aE "radix|disable_radix_cache=False|RADIX_OVERRIDE" "$slog" | head -8 || true
echo "smoke=$(smoke)"

echo "=== run cold/warm comparator driver $(date -u +%H:%M:%SZ) ==="
SGLANG_DS_LATENT_CAPTURE_DIR="$SGLANG_DS_LATENT_CAPTURE_DIR" PORT="$PORT" \
  python "$HERE/p1_coldwarm_driver.py" --prefix-tokens 2400 --out "$HERE/p1_coldwarm_verdict.json"
DRV_RC=$?

{
  echo "probe=p1_cold_warm_selection_equivalence status=$([[ $DRV_RC -eq 0 ]] && echo PASS || echo FAIL)"
  echo "--- verdict ---"
  cat "$HERE/p1_coldwarm_verdict.json" 2>/dev/null
  echo ""
  echo "--- radix authorization (server log) ---"
  grep -aE "disable_radix_cache|radix" "$slog" | head -6
} > "$HERE/p1_evidence.txt"
cat "$HERE/p1_evidence.txt"

echo "=== P1 done; tearing down $(date -u +%H:%M:%SZ) ==="
teardown
gpu_idle_wait
exit $DRV_RC
