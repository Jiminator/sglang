#!/usr/bin/env bash
# R25 PROBE 3 — edge_probe (page 64), radix-ON, recall_oracle + selection_capture,
# EAGER. Exercises page-boundary-aligned full reuse (a), partial-page hit (b), and
# eviction-then-recompute (c). PASS = recall stays equivalent to the cold baseline
# (+/-0.5pp) and selection stays valid across all three; no stale slot served.
# Also preserves the selection-capture dir for the no_dense_fallback scan.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r25_env.sh"
cd "$REPO"
LOG="$HERE/probes/p3_edge_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== R25 P3 edge probe start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
OUT="$HERE/probes/p3_edge"; mkdir -p "$OUT"
export SGLANG_DS_SELECTION_CAPTURE_DIR="$REPO/.sglang_ds_selcap"
mkdir -p "$REPO/.sglang_ds_oracle"

teardown
rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true
: > "$DEFAULT_SINK"

export SGLANG_DS_RADIX_OVERRIDE=1
slog="$OUT/serve_radix_on.log"
echo "=== boot DS radix-ON eager recall_oracle+selection_capture $(date -u +%H:%M:%SZ) ==="
ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_RECALL_SELCAP" --disable-cuda-graph)
python -m sglang.launch_server "${ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! P3 boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
  { echo "probe=p3_edge status=BOOT_FAIL rc_wait=$rc_wait"; } > "$HERE/probes/p3_edge_evidence.txt"
  teardown; gpu_idle_wait; unset SGLANG_DS_RADIX_OVERRIDE || true; exit 1
fi
echo ">>> server ready. smoke=$(smoke)"
grep -aE "disable_radix_cache=False" "$slog" | head -1 || true

echo "=== run edge driver (L4096 idx3, page 64, top_k 2048) $(date -u +%H:%M:%SZ) ==="
PORT=$PORT DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
python "$HERE/p3_edge_driver.py" --length 4096 --idx 3 --page-size 64 --top-k 2048 \
  --max-delta-pp 0.5 --evict-prompts 6 --out "$HERE/probes/p3_edge_verdict.json"
EDGE_RC=$?

# preserve selection-capture dir for the no-dense-fallback scan
rm -rf "$OUT/cap_edge" 2>/dev/null || true
cp -r "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$OUT/cap_edge" 2>/dev/null || true
teardown
unset SGLANG_DS_RADIX_OVERRIDE || true
gpu_idle_wait

{
  echo "probe=p3_edge_probe (radix-ON, EAGER, recall_oracle+selection_capture) drv_rc=$EDGE_RC"
  echo "--- verdict ---"
  python3 -c "import json;d=json.load(open('$HERE/probes/p3_edge_verdict.json'));print('status',d.get('status'));import sys;[print(k, {kk:vv for kk,vv in v.items() if kk in ('cached_tokens','delta_pp_vs_cold','recall2048','cache_hit')}) for k,v in d.get('cases',{}).items()];print('problems',d.get('problems'))" 2>&1
} > "$HERE/probes/p3_edge_evidence.txt"
cat "$HERE/probes/p3_edge_evidence.txt"
echo "=== R25 P3 done $(date -u +%H:%M:%SZ) ==="
gpu_idle_wait
exit $EDGE_RC
