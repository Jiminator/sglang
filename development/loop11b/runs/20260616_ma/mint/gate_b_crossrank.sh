#!/usr/bin/env bash
# loop11b GATE B — cross_rank_selection_identity + (capture for) no_dense_fallback.
# selection_capture=true, EAGER. For radix-ON and radix-OFF: assert all 8 TP ranks
# produce identical selected indices per (layer, step). Preserves per-path capture dirs
# for the no-dense-fallback scan. Reuses the loop-11 R25 driver p2_cross_rank_driver.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$REPO"
OUT="$HERE/probes/gate_b_crossrank"; mkdir -p "$OUT"
LOG="$OUT/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b GATE B cross-rank identity $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="
R25=/sgl-workspace/sglang/development/loop11/runs/20260616_r25/radix_authorization
DRIVER="$R25/p2_cross_rank_driver.py"
NDF="$R25/no_dense_fallback_check.py"
export SGLANG_DS_SELECTION_CAPTURE_DIR="$REPO/.sglang_ds_selcap"

run_path() {  # $1=label(radix_on|radix_off) $2..=extra launch args
  local label="$1"; shift
  local slog="$OUT/serve_${label}.log"
  teardown; rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true
  echo "=== boot DS ($label) eager selection_capture $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_SELCAP" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0; ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != 0 ]]; then echo "!! boot FAIL ($label rc=$rc_wait)"; tail -n 60 "$slog"; teardown; gpu_idle_wait; return 10; fi
  echo ">>> ready ($label). smoke=$(smoke)"
  PORT=$PORT SGLANG_DS_SELECTION_CAPTURE_DIR="$SGLANG_DS_SELECTION_CAPTURE_DIR" \
  python "$DRIVER" --path-label "$label" --prefix-tokens 2400 --expect-ranks 8 --out "$OUT/${label}_verdict.json"
  local drv_rc=$?
  rm -rf "$OUT/cap_${label}" 2>/dev/null || true
  cp -r "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$OUT/cap_${label}" 2>/dev/null || true
  teardown; gpu_idle_wait; return $drv_rc
}

export SGLANG_DS_RADIX_OVERRIDE=1; run_path radix_on; ON_RC=$?; unset SGLANG_DS_RADIX_OVERRIDE || true
run_path radix_off --disable-radix-cache; OFF_RC=$?

# no-dense-fallback scan over the preserved captures (GATE B p4)
echo "=== no-dense-fallback scan $(date -u +%H:%M:%SZ) ==="
NDF_RC=99
if [[ -f "$NDF" ]]; then
  python "$NDF" --capture-dirs "$OUT/cap_radix_on" "$OUT/cap_radix_off" \
    --out "$HERE/probes/gate_b_no_dense_fallback_verdict.json" 2>&1 | tee "$OUT/ndf.log"
  NDF_RC=${PIPESTATUS[0]}
else
  echo "!! no_dense_fallback_check.py not found at $NDF"
fi

{
  echo "probe=gate_b_cross_rank_selection_identity (selection_capture, EAGER, 8 TP ranks)"
  echo "radix_on_driver_rc=$ON_RC radix_off_driver_rc=$OFF_RC no_dense_fallback_rc=$NDF_RC"
  for L in radix_on radix_off; do
    if [[ -f "$OUT/${L}_verdict.json" ]]; then
      echo "--- $L ---"
      python3 -c "import json;d=json.load(open('$OUT/${L}_verdict.json'));print('status',d.get('status'));print('ranks_present',d.get('ranks_present'));print('all_ranks_identical_all_steps',d.get('all_ranks_identical_all_steps'));print('reason',d.get('reason'))"
    fi
  done
  echo "status=$([[ $ON_RC -eq 0 && $OFF_RC -eq 0 && $NDF_RC -eq 0 ]] && echo PASS || echo FAIL)"
} > "$HERE/probes/gate_b_evidence.txt"
cat "$HERE/probes/gate_b_evidence.txt"
gpu_idle_wait
[[ $ON_RC -eq 0 && $OFF_RC -eq 0 && $NDF_RC -eq 0 ]] && exit 0 || exit 1
