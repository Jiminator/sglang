#!/usr/bin/env bash
# Loop-10 landing gates — cloned from development/loop9/run_r1_gates.sh with the
# selection-capture diff PROMOTED TO A HARD GATE and a second, op-point-shaped
# selection-capture phase (bs-29 concurrent decode under CUDA-graph replay).
#
#   Phase A1: bs-1 selection-capture run (graph mode, loop-9-identical
#             workload) -> cross-rank verify -> HARD digest comparison vs the
#             frozen bs-1 baseline (zero index diff or FAIL).
#   Phase A2: op-point selection-capture run (29 concurrent ~4k-token
#             requests, full production graph ladder) -> verify with hard
#             bucket-identity requirements (raw_bs=29, graph-replay path,
#             padded_bs=32) -> HARD digest comparison vs the frozen op-point
#             baseline.
#   Phase B:  NIAH recall-oracle run (eager) -> recall@2048 gate vs the frozen
#             loop-9 baseline (fail-closed at +/-0.5pp).
#   Phase C:  Case-1 torch re-profile (frozen recipe verbatim) -> per-bucket
#             ledger column.
#
# Usage: bash development/loop10/run_gates.sh <outdir> [baseline_dir]
#   outdir        run directory for this gate run (created)
#   baseline_dir  directory holding the frozen digests to compare against
#                 (default: development/loop10/runs/20260611_m0_freeze)
#
# Env:
#   FREEZE=1            freeze mode: this run WRITES the frozen digests. The
#                       bs-1 digest is still hard-compared against the loop-9
#                       R1 digest fingerprint (cross-boot reproducibility
#                       proof); the op-point digest is recorded as the first
#                       frozen baseline.
#   ALLOW_IDENTITY_CHANGE="field ..."  declared bucket-identity evolution for
#                       an exact change that legitimately moves identity
#                       fields (e.g. selector_width after the compact patch);
#                       passed through to diff-digest. Default: none.
#   SKIP_SELCAP=1 / SKIP_OPPOINT=1 / SKIP_RECALL=1 / SKIP_PROFILE=1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source "$REPO/development/profiling/runs/20260609/_env.sh"

OUTDIR="${1:?usage: run_gates.sh <outdir> [baseline_dir]}"
BASELINE="${2:-$REPO/development/loop10/runs/20260611_m0_freeze}"
LOOP9_R1="$REPO/development/loop9/runs/20260611_r1"
RECALL_BASE="$REPO/development/loop9/runs/20260610_m0"
TOOL="$REPO/development/loop10/selection_capture_tool.py"
mkdir -p "$OUTDIR"

fail() { echo "!! FAIL: $1"; tail -40 "$2" 2>/dev/null; teardown; exit 1; }

ALLOW_ARGS=()
if [[ -n "${ALLOW_IDENTITY_CHANGE:-}" ]]; then
  # shellcheck disable=SC2206
  ALLOW_ARGS=(--allow-identity-change ${ALLOW_IDENTITY_CHANGE})
fi

if [[ "${SKIP_SELCAP:-0}" != "1" ]]; then
  DS_CONFIG_SELCAP="${DS_CONFIG%\}}, \"selection_capture\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_SELCAP"
    --cuda-graph-max-bs 4)
  SERVE_LOG="$OUTDIR/selcap_bs1_serve.log"
  echo "=== gate A1: bs-1 selection capture (graph mode) $(date -u +%H:%M:%S)Z ==="
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "bs-1 selcap server not ready" "$SERVE_LOG"
  python "$TOOL" run \
    --out "$OUTDIR/selcap_bs1" --decode-steps 8 --repeat 2 \
    > "$OUTDIR/selcap_bs1_run.log" 2>&1 || fail "bs-1 selcap run failed" "$OUTDIR/selcap_bs1_run.log"
  teardown
  python "$TOOL" verify \
    --run-dir "$OUTDIR/selcap_bs1" --ranks 8 --expected-steps 32 \
    --digest "$OUTDIR/bs1_digest.json" \
    > "$OUTDIR/selcap_bs1_verify.log" 2>&1 || fail "bs-1 selcap verify failed" "$OUTDIR/selcap_bs1_verify.log"
  if [[ "${FREEZE:-0}" == "1" ]]; then
    # Cross-boot reproducibility proof: the fresh pre-change digest must match
    # the loop-9 R1 fingerprint bit-exactly (no pipeline change has landed).
    python "$TOOL" diff-digest \
      --a "$LOOP9_R1/selcap_digest.json" --b "$OUTDIR/bs1_digest.json" \
      --out "$OUTDIR/bs1_diff_vs_loop9r1.json" \
      > "$OUTDIR/selcap_bs1_diff.log" 2>&1 || fail "bs-1 HARD GATE: fresh digest != loop-9 R1 fingerprint" "$OUTDIR/selcap_bs1_diff.log"
  else
    python "$TOOL" diff-digest \
      --a "$BASELINE/bs1_digest.json" --b "$OUTDIR/bs1_digest.json" \
      --out "$OUTDIR/bs1_diff_vs_baseline.json" "${ALLOW_ARGS[@]}" \
      > "$OUTDIR/selcap_bs1_diff.log" 2>&1 || fail "bs-1 HARD GATE: selection indices differ from frozen baseline" "$OUTDIR/selcap_bs1_diff.log"
    if [[ -d "$BASELINE/selcap_bs1/pass0" ]]; then
      python "$TOOL" diff \
        --a "$BASELINE/selcap_bs1/pass0" --b "$OUTDIR/selcap_bs1/pass0" \
        --out "$OUTDIR/bs1_rowdiff_vs_baseline.json" --fail-on-diff \
        > "$OUTDIR/selcap_bs1_rowdiff.log" 2>&1 || fail "bs-1 HARD GATE: per-row diff nonzero" "$OUTDIR/selcap_bs1_rowdiff.log"
    fi
  fi
  echo ">>> gate A1 done (cross-rank verify PASS; digest comparison PASS)"
fi

if [[ "${SKIP_OPPOINT:-0}" != "1" ]]; then
  DS_CONFIG_SELCAP="${DS_CONFIG%\}}, \"selection_capture\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_SELCAP")
  SERVE_LOG="$OUTDIR/selcap_op_serve.log"
  echo "=== gate A2: op-point selection capture (bs-29 graph replay) $(date -u +%H:%M:%S)Z ==="
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "op-point selcap server not ready" "$SERVE_LOG"
  python "$TOOL" run-op \
    --out "$OUTDIR/selcap_op" --concurrency 29 --decode-steps 12 --repeat 2 \
    > "$OUTDIR/selcap_op_run.log" 2>&1 || fail "op-point selcap run failed" "$OUTDIR/selcap_op_run.log"
  teardown
  python "$TOOL" verify \
    --run-dir "$OUTDIR/selcap_op" --ranks 8 --expected-steps 12 \
    --require-raw-bs 29 --require-replay --require-padded-bs 32 \
    --digest "$OUTDIR/op_digest.json" \
    > "$OUTDIR/selcap_op_verify.log" 2>&1 || fail "op-point selcap verify failed (bucket-identity requirements)" "$OUTDIR/selcap_op_verify.log"
  if [[ "${FREEZE:-0}" != "1" ]]; then
    python "$TOOL" diff-digest \
      --a "$BASELINE/op_digest.json" --b "$OUTDIR/op_digest.json" \
      --out "$OUTDIR/op_diff_vs_baseline.json" "${ALLOW_ARGS[@]}" \
      > "$OUTDIR/selcap_op_diff.log" 2>&1 || fail "op-point HARD GATE: selection indices differ from frozen baseline" "$OUTDIR/selcap_op_diff.log"
    if [[ -d "$BASELINE/selcap_op/pass0" ]]; then
      python "$TOOL" diff \
        --a "$BASELINE/selcap_op/pass0" --b "$OUTDIR/selcap_op/pass0" \
        --out "$OUTDIR/op_rowdiff_vs_baseline.json" --fail-on-diff \
        > "$OUTDIR/selcap_op_rowdiff.log" 2>&1 || fail "op-point HARD GATE: per-row diff nonzero" "$OUTDIR/selcap_op_rowdiff.log"
    fi
  fi
  echo ">>> gate A2 done (op-point verify PASS; raw_bs=29 on graph replay proven)"
fi

if [[ "${SKIP_RECALL:-0}" != "1" ]]; then
  DS_CONFIG_ORACLE="${DS_CONFIG%\}}, \"recall_oracle\": true}"
  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
    --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_ORACLE"
    --disable-cuda-graph)
  SERVE_LOG="$OUTDIR/recall_serve.log"
  echo "=== gate B: recall oracle (eager) $(date -u +%H:%M:%S)Z ==="
  teardown
  python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
  wait_ready || fail "recall server not ready" "$SERVE_LOG"
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$REPO/development/loop7/niah_oracle_sweep.py" \
    --lengths 1024 4096 16384 --num 20 --decode-steps 4 \
    --out "$OUTDIR/oracle_trials_index.jsonl" \
    > "$OUTDIR/recall_sweep.log" 2>&1 || fail "oracle sweep failed" "$OUTDIR/recall_sweep.log"
  teardown
  python "$REPO/development/loop9/oracle_recall_summary.py" \
    --out "$OUTDIR/recall_gate.json" \
    --baseline "$RECALL_BASE/recall_baseline.json" --max-delta-pp 0.5 \
    > "$OUTDIR/recall_gate.log" 2>&1 || fail "RECALL GATE FAILED" "$OUTDIR/recall_gate.log"
  echo ">>> gate B done (recall within +/-0.5pp)"
fi

if [[ "${SKIP_PROFILE:-0}" != "1" ]]; then
  echo "=== gate C: Case-1 re-profile $(date -u +%H:%M:%S)Z ==="
  RELOUT="../../../loop10/runs/$(basename "$OUTDIR")/case1_ds"
  bash "$REPO/development/profiling/runs/20260609/run_case.sh" \
    "$RELOUT" case1 torch 29 \
    || fail "re-profile failed" "/dev/null"
  D="$OUTDIR/case1_ds/torch/trace"
  F="$REPO/development/profiling/runs/20260609"
  python "$F/summarize_torch.py" "$D" "$OUTDIR/summary_torch.txt"
  python "$F/compare_decode.py" "$D" loop10 "$F/case2_dsa07/torch/trace" case2 \
    "$OUTDIR/cmp_vs_case2.txt"
  python "$F/compare_decode.py" "$D" loop10 \
    "$LOOP9_R1/case1_ds/torch/trace" loop9r1 \
    "$OUTDIR/cmp_vs_loop9r1.txt"
  echo ">>> gate C done"
fi

echo "=== loop-10 gates complete $(date -u +%H:%M:%S)Z ==="
