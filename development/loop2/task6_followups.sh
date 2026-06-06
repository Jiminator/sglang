#!/usr/bin/env bash
# Profile-directed follow-up GATE sweep (flags-only, in-scope), layered on the
# combo incumbent base. Directed by combo_baseline profile: comms ~16.5-19% and
# indexer/topk ~8.5% are the material non-MoE slices. Each candidate is a fresh
# server gate run via unchanged development/benchmark.sh. Promising movers are
# profiled in a follow-up step.
set -uo pipefail
ROOT=/sgl-workspace/sglang
cd "$ROOT"
BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"
LEDGER="development/loop2/task6_results.txt"
echo "# task6 follow-up gate-sweep started $(date +%H:%M:%S)" > "$LEDGER"

run_cell () {
  local tag="$1"; shift
  local extra="$1"; shift
  local why="$1"; shift
  echo ">>> $(date +%H:%M:%S) $tag : $extra" | tee -a "$LEDGER"
  SGLANG_ENABLE_SPEC_V2=1 TAG="$tag" \
    EXTRA_ARGS="$BASE $extra" RATIONALE="$why" READY_TIMEOUT=900 \
    bash development/loop2/run_candidate.sh > "development/loop2/logs/t6_${tag}.out" 2>&1
  local rc=$?
  local status=$(grep -oE "STATUS=[a-z_]+" "development/loop2/logs/t6_${tag}.out" | tail -1)
  local tps=$(grep -oE "client_TPS    = [0-9.]+" "development/loop2/logs/t6_${tag}.out" | tail -1)
  local decis=$(grep -hnE "raise ValueError|assert|Error|NotImplemented|incompatible|not supported" \
                  "development/loop2/logs/serve_${tag}.log" 2>/dev/null | tail -2 | tr '\n' '~')
  echo "RESULT $tag rc=$rc $status $tps decisive=[$decis]" | tee -a "$LEDGER"
}

run_cell "t6_fused_moe_sum_ar"  "--enable-fused-moe-sum-all-reduce"  "task6: fuse MoE-sum + all-reduce (targets comms 16.5%)"
run_cell "t6_topk_flashinfer"   "--dsa-topk-backend flashinfer"      "task6: DSA topk backend flashinfer (targets indexer/topk 8.5%)"
run_cell "t6_topk_torch"        "--dsa-topk-backend torch"           "task6: DSA topk backend torch (targets indexer/topk 8.5%)"
run_cell "t6_contdecode2"       "--num-continuous-decode-steps 2"    "task6: continuous decode steps (scheduling; profile shows <1% idle)"
echo "TASK6_GATE_SWEEP_DONE $(date +%H:%M:%S)" | tee -a "$LEDGER"
