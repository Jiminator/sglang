#!/usr/bin/env bash
# Loop-9 M0 baseline freeze — two serialized server runs on the Case-1 op-point.
#
#   Phase A (graph mode): boot the Case-1 DS server with the config-borne
#     selection_capture diagnostic, drive the fixed deterministic workload
#     twice (run-to-run identity), verify cross-rank bit-identity + output
#     contract, and freeze the digest.
#   Phase B (eager mode): boot the Case-1 DS server EAGER with the config-borne
#     recall oracle, run the fail-closed NIAH oracle sweep, and freeze the
#     recall@2048 baseline summary.
#
# Reuses the frozen Case-1 recipe (development/profiling/runs/20260609/_env.sh):
# model, mask, DS config, common server args, wait_ready, teardown. One TP=8
# server at a time. Never sets PYTORCH_CUDA_ALLOC_CONF (the env file unsets it).
#
# Usage: bash development/loop9/run_m0_baselines.sh [outdir]
#        SKIP_SELCAP=1 / SKIP_RECALL=1 skip a phase.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source "$REPO/development/profiling/runs/20260609/_env.sh"

M0OUT="${1:-$REPO/development/loop9/runs/20260610_m0}"
mkdir -p "$M0OUT"

fail() { echo "!! FAIL: $1"; tail -40 "$2" 2>/dev/null; teardown; exit 1; }

# ---------------- Phase A: selection-capture baseline (graph mode) ----------------
if [[ "${SKIP_SELCAP:-0}" != "1" ]]; then
  DS_CONFIG_SELCAP="${DS_CONFIG%\}}, \"selection_capture\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_SELCAP"
    --cuda-graph-max-bs 4)
  SERVE_LOG="$M0OUT/selcap_serve.log"
  echo "=== M0 Phase A: selection-capture baseline (graph mode) $(date -u +%H:%M:%S)Z ==="
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "selcap server not ready" "$SERVE_LOG"

  python "$REPO/development/loop9/selection_capture_tool.py" run \
    --out "$M0OUT/selcap_baseline" --decode-steps 8 --repeat 2 \
    > "$M0OUT/selcap_run.log" 2>&1 || fail "selcap run failed" "$M0OUT/selcap_run.log"
  teardown

  python "$REPO/development/loop9/selection_capture_tool.py" verify \
    --run-dir "$M0OUT/selcap_baseline" --ranks 8 --expected-steps 32 \
    --digest "$M0OUT/selcap_baseline_digest.json" \
    > "$M0OUT/selcap_verify.log" 2>&1 || fail "selcap verify failed" "$M0OUT/selcap_verify.log"
  echo ">>> Phase A OK: digest at $M0OUT/selcap_baseline_digest.json"
fi

# ---------------- Phase B: NIAH recall-oracle baseline (eager mode) ----------------
if [[ "${SKIP_RECALL:-0}" != "1" ]]; then
  DS_CONFIG_ORACLE="${DS_CONFIG%\}}, \"recall_oracle\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_ORACLE"
    --disable-cuda-graph)
  SERVE_LOG="$M0OUT/recall_serve.log"
  echo "=== M0 Phase B: NIAH recall-oracle baseline (eager mode) $(date -u +%H:%M:%S)Z ==="
  [[ -f "$GLM/tokenizer.json" ]] || { echo "!! no tokenizer.json under $GLM"; exit 1; }
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "recall server not ready" "$SERVE_LOG"

  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$REPO/development/loop7/niah_oracle_sweep.py" \
    --lengths 1024 4096 16384 --num 20 --decode-steps 4 \
    --out "$M0OUT/oracle_trials_index.jsonl" \
    > "$M0OUT/recall_sweep.log" 2>&1 || fail "oracle sweep failed" "$M0OUT/recall_sweep.log"
  teardown

  python "$REPO/development/loop9/oracle_recall_summary.py" \
    --out "$M0OUT/recall_baseline.json" \
    > "$M0OUT/recall_summary.log" 2>&1 || fail "recall summary failed" "$M0OUT/recall_summary.log"
  echo ">>> Phase B OK: baseline at $M0OUT/recall_baseline.json"
fi

echo "=== M0 baselines complete $(date -u +%H:%M:%S)Z ==="
