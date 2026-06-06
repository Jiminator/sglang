#!/usr/bin/env bash
# Drive the DSA prefill x decode cross-product GATE sweep on the combo base.
# Every cell is launch-attempted and recorded under a deterministic taxonomy
# (owner decision: no pruning, no silent skips). Launchable cells get a full
# fresh-server gate run via the unchanged development/benchmark.sh; cells that
# fail (e.g. decode=flashmla_auto -> first-decode assert) are recorded with the
# decisive serve-log line. Profiling of launchable cells is a separate pass.
set -uo pipefail
ROOT=/sgl-workspace/sglang
cd "$ROOT"

BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"
LEDGER="development/loop2/dsa_matrix_results.txt"
echo "# DSA matrix gate-sweep started $(date +%H:%M:%S)" > "$LEDGER"

BACKENDS="flashmla_sparse flashmla_kv flashmla_auto fa3"

for P in $BACKENDS; do
  for D in $BACKENDS; do
    if [ "$P" = "flashmla_sparse" ] && [ "$D" = "fa3" ]; then
      echo "CELL prefill=$P decode=$D STATUS=already_done(combo_baseline=24.08TPS)" | tee -a "$LEDGER"
      continue
    fi
    TAG="dsa_${P}__${D}"
    CELLOUT="development/loop2/logs/cell_${TAG}.out"
    echo ">>> $(date +%H:%M:%S) CELL prefill=$P decode=$D (tag=$TAG)" | tee -a "$LEDGER"
    SGLANG_ENABLE_SPEC_V2=1 TAG="$TAG" \
      EXTRA_ARGS="$BASE --dsa-prefill-backend $P --dsa-decode-backend $D" \
      RATIONALE="DSA matrix prefill=$P decode=$D (bf16)" \
      READY_TIMEOUT=900 \
      bash development/loop2/run_candidate.sh > "$CELLOUT" 2>&1
    rc=$?
    status=$(grep -oE "STATUS=[a-z_]+" "$CELLOUT" | tail -1)
    decis=$(grep -hnE "Unsupported .*dsa_decode_impl|assert False|only page size 64 is supported|raise ValueError|NotImplementedError|CUDA error|out of memory" \
              "development/loop2/logs/serve_${TAG}.log" 2>/dev/null | tail -2 | tr '\n' '~')
    echo "CELL prefill=$P decode=$D rc=$rc $status decisive=[$decis]" | tee -a "$LEDGER"
  done
done
echo "DSA_MATRIX_GATE_SWEEP_DONE $(date +%H:%M:%S)" | tee -a "$LEDGER"
