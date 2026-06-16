#!/usr/bin/env bash
# GATE B (cross-rank identity + no-dense-fallback) and GATE C (edge correctness:
# eviction==cold, no stale slot) — both gathered on radix-ON boots, fresh under v2.
# Reuses the existing R25 drivers (p2_cross_rank_driver.py selection_capture cross-rank,
# no_dense_fallback_check.py over the captures, p3_edge_driver.py edge correctness).
#
# GATE B: radix-ON, selection_capture=true, EAGER. All 8 TP ranks must produce
#         byte-identical selected indices per (layer, step). Then scan the captured
#         rows: every row with seq_len>=top_k selects exactly top_k (no dense degrade).
# GATE C: radix-ON, recall_oracle+selection_capture, EAGER. p3_edge_driver exercises
#         page-boundary full reuse, partial-page hit, and eviction->recompute. The
#         RESCOPED correctness criterion is no-stale-slot: eviction-recompute recall ==
#         cold baseline EXACTLY (delta 0.0pp). (The edge capture also feeds GATE B's
#         no-dense-fallback scan.)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RADIX_DIR="$HERE/../../radix_authorization"
source "$HERE/../../radix_authorization/r25_env.sh"
cd "$REPO"

mkdir -p "$HERE/logs" "$HERE/cap_radix_on" "$REPO/.sglang_ds_oracle"
LOG="$HERE/logs/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== GATE B+C start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
echo "[gateBC] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"
export SGLANG_DS_SELECTION_CAPTURE_DIR="$REPO/.sglang_ds_selcap"

# ---- GATE B: cross-rank selection identity (radix-ON, selection_capture) ----
teardown
rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true
export SGLANG_DS_RADIX_OVERRIDE=1
b_slog="$HERE/logs/serve_gateB.log"
echo "=== boot DS radix-ON eager selection_capture (GATE B) $(date -u +%H:%M:%SZ) ==="
B_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_SELCAP" --disable-cuda-graph)
python -m sglang.launch_server "${B_ARGS[@]}" > "$b_slog" 2>&1 &
rc_wait=0; ready_wait "$b_slog" || rc_wait=$?
B_RC=99
if [[ "$rc_wait" == "0" ]]; then
  echo ">>> server ready (GATE B). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$b_slog" | head -1 || true
  PORT=$PORT SGLANG_DS_SELECTION_CAPTURE_DIR="$SGLANG_DS_SELECTION_CAPTURE_DIR" \
  python "$RADIX_DIR/p2_cross_rank_driver.py" --path-label radix_on --prefix-tokens 2400 \
    --expect-ranks 8 --out "$HERE/gate_b_cross_rank_verdict.json"
  B_RC=$?
  rm -rf "$HERE/cap_radix_on" 2>/dev/null || true
  cp -r "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$HERE/cap_radix_on" 2>/dev/null || true
else
  echo "!! GATE B boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 60 "$b_slog"
fi
teardown; unset SGLANG_DS_RADIX_OVERRIDE || true; gpu_idle_wait

# ---- GATE C: edge correctness (radix-ON, recall_oracle+selection_capture) ----
teardown
rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true
: > "$DEFAULT_SINK"
export SGLANG_DS_RADIX_OVERRIDE=1
c_slog="$HERE/logs/serve_gateC.log"
echo "=== boot DS radix-ON eager recall_oracle+selection_capture (GATE C) $(date -u +%H:%M:%SZ) ==="
C_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_RECALL_SELCAP" --disable-cuda-graph)
python -m sglang.launch_server "${C_ARGS[@]}" > "$c_slog" 2>&1 &
rc_wait=0; ready_wait "$c_slog" || rc_wait=$?
C_RC=99
if [[ "$rc_wait" == "0" ]]; then
  echo ">>> server ready (GATE C). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$c_slog" | head -1 || true
  PORT=$PORT DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$RADIX_DIR/p3_edge_driver.py" --length 4096 --idx 3 --page-size 64 --top-k 2048 \
    --max-delta-pp 0.5 --evict-prompts 6 --out "$HERE/gate_c_edge_verdict.json"
  C_RC=$?
  rm -rf "$HERE/cap_edge" 2>/dev/null || true
  cp -r "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$HERE/cap_edge" 2>/dev/null || true
else
  echo "!! GATE C boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 60 "$c_slog"
fi
teardown; unset SGLANG_DS_RADIX_OVERRIDE || true; gpu_idle_wait

# ---- GATE B (cont): no-dense-fallback scan over the captured selection ----
echo "=== GATE B no-dense-fallback scan over captures $(date -u +%H:%M:%SZ) ==="
python "$RADIX_DIR/no_dense_fallback_check.py" \
  --capdirs "$HERE/cap_radix_on" "$HERE/cap_edge" --top-k 2048 \
  --out "$HERE/gate_b_no_dense_fallback_verdict.json"
P4_RC=$?

echo "=== GATE B+C done (B_rc=$B_RC C_rc=$C_RC p4_rc=$P4_RC) $(date -u +%H:%M:%SZ) ==="
{
  echo "=== GATE B+C evidence ==="
  echo "GATE B cross-rank: drv_rc=$B_RC"
  [[ -f "$HERE/gate_b_cross_rank_verdict.json" ]] && python3 -c "import json;d=json.load(open('$HERE/gate_b_cross_rank_verdict.json'));print('  status',d.get('status'),'ranks',d.get('ranks_present'),'identical',d.get('all_ranks_identical_all_steps'),'steps',d.get('common_steps'),'layers',d.get('layers'),'bs',d.get('bs'))"
  echo "GATE B no_dense_fallback: scan_rc=$P4_RC"
  [[ -f "$HERE/gate_b_no_dense_fallback_verdict.json" ]] && python3 -c "import json;d=json.load(open('$HERE/gate_b_no_dense_fallback_verdict.json'));print('  status',d.get('status'),'files',d.get('files_scanned'),'rows',d.get('rows_checked'),'full_budget',d.get('rows_full_sparse_budget(seq>=top_k,vl==top_k)'),'violations',d.get('num_violations'))"
  echo "GATE C edge correctness: drv_rc=$C_RC"
  [[ -f "$HERE/gate_c_edge_verdict.json" ]] && python3 -c "import json;d=json.load(open('$HERE/gate_c_edge_verdict.json'));c=d.get('cases',{});ev=c.get('c_eviction_recompute',{});cb=c.get('cold_baseline',{});print('  cold recall',cb.get('recall2048',{}).get('pct'),'evict recall',ev.get('recall2048',{}).get('pct'),'evict delta_pp_vs_cold',ev.get('delta_pp_vs_cold'),'prefix_evicted',ev.get('prefix_evicted(cached_fell_vs_boundary)'),'overall_status',d.get('status'))"
} > "$HERE/gate_bc_evidence.txt"
cat "$HERE/gate_bc_evidence.txt"
echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=== GATE B+C complete $(date -u +%H:%M:%SZ) ==="
gpu_idle_wait
