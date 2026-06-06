#!/usr/bin/env bash
set -uo pipefail
ROOT=/sgl-workspace/sglang; cd "$ROOT"
BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"
LED="development/loop2/task6_results.txt"
echo "t6_fused_moe_sum_ar client_TPS=23.33 (neutral/slightly-worse vs 24.08; comms fusion no help)" >> "$LED"
run_cell () {
  local tag="$1"; local extra="$2"; local why="$3"
  echo ">>> $(date +%H:%M:%S) $tag : $extra" >> "$LED"
  SGLANG_ENABLE_SPEC_V2=1 TAG="$tag" EXTRA_ARGS="$BASE $extra" RATIONALE="$why" READY_TIMEOUT=900 \
    bash development/loop2/run_candidate.sh > "development/loop2/logs/t6_${tag}.out" 2>&1
  local rc=$?
  local tps=$(grep -oE "client_TPS    = [0-9.]+" "development/loop2/logs/t6_${tag}.out" | tail -1)
  local status=$(grep -oE "STATUS=[a-z_]+" "development/loop2/logs/t6_${tag}.out" | tail -1)
  local decis=$(grep -haE "raise ValueError|assert|NotImplemented|not supported|incompatible" "development/loop2/logs/serve_${tag}.log" 2>/dev/null | tail -2 | tr '\n' '~')
  echo "RESULT $tag rc=$rc $status $tps decisive=[$decis]" >> "$LED"
}
run_cell "t6_topk_flashinfer" "--dsa-topk-backend flashinfer" "task6: DSA topk backend flashinfer (indexer/topk 8.5%)"
run_cell "t6_topk_torch"      "--dsa-topk-backend torch"      "task6: DSA topk backend torch (indexer/topk 8.5%)"
run_cell "t6_contdecode2"     "--num-continuous-decode-steps 2" "task6: continuous decode steps (scheduling)"
echo "TASK6_REST_DONE $(date +%H:%M:%S)" >> "$LED"
