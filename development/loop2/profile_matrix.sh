#!/usr/bin/env bash
# Profile every LAUNCHABLE DSA matrix cell (no pruning, DEC-3). For each cell:
#   profile-only run -> category rollup (TP-0) + analyzer triage -> per-cell md
#   -> DELETE raw trace (disk hygiene, DEC-4).
# combo_baseline (sparse/fa3) is already profiled; decode=flashmla_auto cells are
# launch-rejected (skipped here, recorded in coverage_log.md).
set -uo pipefail
ROOT=/sgl-workspace/sglang
cd "$ROOT"
BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"
ANALYZER=.claude/skills/llm-torch-profiler-analysis/scripts/analyze_sglang_torch_profile.py
ROLLUP=development/loop2/profiling/_work/rollup.py
PROFDIR=development/loop2/profiling
LEDGER="development/loop2/profile_matrix_results.txt"
echo "# profile-matrix started $(date +%H:%M:%S)" > "$LEDGER"

# launchable, non-incumbent cells (prefill__decode); decode=flashmla_auto excluded (rejected)
CELLS="flashmla_sparse__flashmla_sparse flashmla_sparse__flashmla_kv \
flashmla_kv__flashmla_sparse flashmla_kv__flashmla_kv flashmla_kv__fa3 \
flashmla_auto__flashmla_sparse flashmla_auto__flashmla_kv flashmla_auto__fa3 \
fa3__flashmla_sparse fa3__flashmla_kv fa3__fa3"

for cell in $CELLS; do
  P="${cell%%__*}"; D="${cell##*__}"
  TAG="dsa_${cell}"
  echo ">>> $(date +%H:%M:%S) PROFILE prefill=$P decode=$D (tag=$TAG)" | tee -a "$LEDGER"
  SGLANG_ENABLE_SPEC_V2=1 TAG="$TAG" \
    EXTRA_ARGS="$BASE --dsa-prefill-backend $P --dsa-decode-backend $D" \
    READY_TIMEOUT=900 \
    bash development/loop2/profile_candidate.sh > "development/loop2/logs/profmx_${TAG}.out" 2>&1
  prc=$?
  rawdir=$(find "$PROFDIR/raw/$TAG" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
  tp0=$(find "$PROFDIR/raw/$TAG" -name '*TP-0.trace.json.gz' 2>/dev/null | head -1)
  if [ "$prc" = "0" ] && [ -n "$tp0" ]; then
    {
      echo "# Decode-phase profile — $TAG (prefill=$P decode=$D)"
      echo
      echo "## Category rollup (summed kernel GPU time, TP-0)"
      echo '```'
      python3 "$ROLLUP" "$tp0" 2>/dev/null
      echo '```'
    } > "$PROFDIR/${TAG}.md"
    python3 "$ANALYZER" --framework sglang --input "$tp0" \
      --output-dir "$PROFDIR/_work" --kernel-table-limit 20 --overlap-table-limit 8 \
      > "$PROFDIR/_work/${TAG}_triage.md" 2>/dev/null
    echo "CELL $cell prc=$prc rollup_written delete_raw" | tee -a "$LEDGER"
  else
    echo "CELL $cell prc=$prc PROFILE_FAILED (see logs/profmx_${TAG}.out)" | tee -a "$LEDGER"
  fi
  rm -rf "$PROFDIR/raw/$TAG"
done
echo "PROFILE_MATRIX_DONE $(date +%H:%M:%S)" | tee -a "$LEDGER"
