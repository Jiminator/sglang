#!/usr/bin/env bash
# Loop-9 M1 landing gates — three serialized runs on the Case-1 op-point with
# the bf16 score reduce active (served default; DS config recipe unchanged):
#
#   Phase A: selection-capture run (graph mode) -> cross-rank verify + exact
#            index diff vs the frozen production oracle (attribution).
#   Phase B: NIAH recall-oracle run (eager) -> recall@2048 gate vs the frozen
#            baseline (fail-closed at ±0.5pp).
#   Phase C: Case-1 torch re-profile (recipe verbatim) -> per-bucket ledger
#            column + reduce-backend kernel evidence from the trace.
#
# Usage: bash development/loop9/run_m1_gates.sh [outdir]
#        SKIP_SELCAP=1 / SKIP_RECALL=1 / SKIP_PROFILE=1 skip a phase.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source "$REPO/development/profiling/runs/20260609/_env.sh"

M1OUT="${1:-$REPO/development/loop9/runs/20260610_m1}"
BASE="$REPO/development/loop9/runs/20260610_m0"
mkdir -p "$M1OUT"

fail() { echo "!! FAIL: $1"; tail -40 "$2" 2>/dev/null; teardown; exit 1; }

if [[ "${SKIP_SELCAP:-0}" != "1" ]]; then
  DS_CONFIG_SELCAP="${DS_CONFIG%\}}, \"selection_capture\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_SELCAP"
    --cuda-graph-max-bs 4)
  SERVE_LOG="$M1OUT/selcap_serve.log"
  echo "=== M1 gate A: selection capture (graph mode) $(date -u +%H:%M:%S)Z ==="
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "selcap server not ready" "$SERVE_LOG"
  python "$REPO/development/loop9/selection_capture_tool.py" run \
    --out "$M1OUT/selcap" --decode-steps 8 --repeat 2 \
    > "$M1OUT/selcap_run.log" 2>&1 || fail "selcap run failed" "$M1OUT/selcap_run.log"
  teardown
  python "$REPO/development/loop9/selection_capture_tool.py" verify \
    --run-dir "$M1OUT/selcap" --ranks 8 --expected-steps 32 \
    --digest "$M1OUT/selcap_digest.json" \
    > "$M1OUT/selcap_verify.log" 2>&1 || fail "selcap verify failed" "$M1OUT/selcap_verify.log"
  python "$REPO/development/loop9/selection_capture_tool.py" diff \
    --a "$BASE/selcap_baseline/pass0" --b "$M1OUT/selcap/pass0" \
    --out "$M1OUT/selcap_diff_vs_frozen.json" \
    > "$M1OUT/selcap_diff.log" 2>&1 || fail "selcap diff failed" "$M1OUT/selcap_diff.log"
  echo ">>> gate A done (cross-rank verify PASS; diff is diagnostic)"
fi

if [[ "${SKIP_RECALL:-0}" != "1" ]]; then
  DS_CONFIG_ORACLE="${DS_CONFIG%\}}, \"recall_oracle\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_ORACLE"
    --disable-cuda-graph)
  SERVE_LOG="$M1OUT/recall_serve.log"
  echo "=== M1 gate B: recall oracle (eager) $(date -u +%H:%M:%S)Z ==="
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "recall server not ready" "$SERVE_LOG"
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$REPO/development/loop7/niah_oracle_sweep.py" \
    --lengths 1024 4096 16384 --num 20 --decode-steps 4 \
    --out "$M1OUT/oracle_trials_index.jsonl" \
    > "$M1OUT/recall_sweep.log" 2>&1 || fail "oracle sweep failed" "$M1OUT/recall_sweep.log"
  teardown
  python "$REPO/development/loop9/oracle_recall_summary.py" \
    --out "$M1OUT/recall_gate.json" \
    --baseline "$BASE/recall_baseline.json" --max-delta-pp 0.5 \
    > "$M1OUT/recall_gate.log" 2>&1 || fail "RECALL GATE FAILED" "$M1OUT/recall_gate.log"
  echo ">>> gate B done (recall within ±0.5pp)"
fi

if [[ "${SKIP_PROFILE:-0}" != "1" ]]; then
  echo "=== M1 gate C: Case-1 re-profile $(date -u +%H:%M:%S)Z ==="
  bash "$REPO/development/profiling/runs/20260609/run_case.sh" \
    "../../../loop9/runs/20260610_m1/case1_ds" case1 torch 29 \
    || fail "re-profile failed" "/dev/null"
  D="$M1OUT/case1_ds/torch/trace"
  F="$REPO/development/profiling/runs/20260609"
  python "$F/summarize_torch.py" "$D" "$M1OUT/summary_torch.txt"
  python "$F/compare_decode.py" "$D" m1 "$F/case2_dsa07/torch/trace" case2 \
    "$M1OUT/cmp_m1_vs_case2.txt"
  python "$F/compare_decode.py" "$D" m1 \
    "$REPO/development/loop9/runs/m0_dryrun/case1_ds/torch/trace" m0dryrun \
    "$M1OUT/cmp_m1_vs_m0dryrun.txt"
  echo ">>> gate C done"
fi

echo "=== M1 gates complete $(date -u +%H:%M:%S)Z ==="
