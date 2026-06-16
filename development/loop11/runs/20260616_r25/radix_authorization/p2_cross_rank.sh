#!/usr/bin/env bash
# R25 PROBE 2 — cross_rank_selection_identity. selection_capture=true, EAGER. For
# radix-ON and (separately) radix-OFF: assert all 8 TP ranks produce identical
# selected indices per (layer, step) within each path. Reports per-rank SHA
# agreement. Preserves the per-path capture dirs for the no_dense_fallback check.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r25_env.sh"
cd "$REPO"
LOG="$HERE/probes/p2_cross_rank_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== R25 P2 cross-rank identity start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
OUT="$HERE/probes/p2_cross_rank"; mkdir -p "$OUT"
export SGLANG_DS_SELECTION_CAPTURE_DIR="$REPO/.sglang_ds_selcap"

run_path() {  # $1=label(radix_on|radix_off)  $2..=extra launch args ; reads RADIX env per caller
  local label="$1"; shift
  local slog="$OUT/serve_${label}.log"
  teardown
  rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true
  echo "=== boot DS ($label) eager selection_capture $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_SELCAP" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! P2 boot FAIL ($label rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($label). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  PORT=$PORT SGLANG_DS_SELECTION_CAPTURE_DIR="$SGLANG_DS_SELECTION_CAPTURE_DIR" \
  python "$HERE/p2_cross_rank_driver.py" --path-label "$label" --prefix-tokens 2400 \
    --expect-ranks 8 --out "$OUT/${label}_verdict.json"
  local drv_rc=$?
  # preserve the capture dir for the no-dense-fallback scan
  rm -rf "$OUT/cap_${label}" 2>/dev/null || true
  cp -r "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$OUT/cap_${label}" 2>/dev/null || true
  teardown
  gpu_idle_wait
  return $drv_rc
}

# radix-ON (probe-gathering override)
export SGLANG_DS_RADIX_OVERRIDE=1
run_path radix_on
ON_RC=$?
unset SGLANG_DS_RADIX_OVERRIDE || true

# radix-OFF (shipped DS default)
run_path radix_off --disable-radix-cache
OFF_RC=$?

echo "=== R25 P2 done (on_rc=$ON_RC off_rc=$OFF_RC) $(date -u +%H:%M:%SZ) ==="
{
  echo "probe=p2_cross_rank_selection_identity (selection_capture, EAGER, 8 TP ranks)"
  echo "radix_on_driver_rc=$ON_RC radix_off_driver_rc=$OFF_RC"
  for L in radix_on radix_off; do
    if [[ -f "$OUT/${L}_verdict.json" ]]; then
      echo "--- $L ---"
      python3 -c "import json;d=json.load(open('$OUT/${L}_verdict.json'));print('status',d.get('status'));print('ranks_present',d.get('ranks_present'));print('all_ranks_identical_all_steps',d.get('all_ranks_identical_all_steps'));print('common_steps',d.get('common_steps'));print('reason',d.get('reason'))"
    fi
  done
  STATUS=$([[ $ON_RC -eq 0 && $OFF_RC -eq 0 ]] && echo PASS || echo FAIL)
  echo "status=$STATUS"
} > "$HERE/probes/p2_cross_rank_evidence.txt"
cat "$HERE/probes/p2_cross_rank_evidence.txt"
gpu_idle_wait
[[ $ON_RC -eq 0 && $OFF_RC -eq 0 ]] && exit 0 || exit 1
