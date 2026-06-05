#!/usr/bin/env bash
# Lower-risk knob ladder: run each candidate on the safe incumbent base
# (EAGLE steps3/topk1/draft4, mem0.85, max-running-requests 64, chunked-prefill 4096,
#  schedule-policy lpm), fresh server per candidate, recording a sweep-table row or a
# launch-failure reason. Sequential because only one TP8 server fits on the node.
set -uo pipefail
ROOT=/sgl-workspace/sglang
cd "$ROOT"
export SGLANG_ENABLE_SPEC_V2=1
export READY_TIMEOUT=2400

BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"

run() {  # $1=TAG  $2=EXTRA_ARGS  $3=RATIONALE
  TAG="$1" EXTRA_ARGS="$2" RATIONALE="$3" bash "$ROOT/development/loop1/run_candidate.sh" \
    > "$ROOT/development/loop1/logs/cand_$1.out" 2>&1
  echo "==== $1 done (rc=$?) ===="
  grep -aE "CANDIDATE|launch_failed|median_itl|mean_tpot|p99_ttft|completed" "$ROOT/development/loop1/logs/cand_$1.out" | tail -6
}

run combo_mrr80 \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 80 --chunked-prefill-size 4096 --schedule-policy lpm" \
  "lower-risk: combo + max-running-requests 80 (admission headroom)"

run combo_mrr96 \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 96 --chunked-prefill-size 4096 --schedule-policy lpm" \
  "lower-risk: combo + max-running-requests 96"

run combo_mem90_cg64 \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.90 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm --cuda-graph-max-bs 64" \
  "lower-risk: combo + mem-fraction 0.90 + cuda-graph-max-bs 64 (KV headroom + graph-capture decode batch)"

run eagle_xlight \
  "--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm" \
  "lower-risk: combo + lightest EAGLE steps1/draft2 (least verify overhead)"

run dsa_decode_sparse \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm --kv-cache-dtype bfloat16 --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_sparse" \
  "lower-risk: combo + bf16 DSA decode=flashmla_sparse (vs default fa3)"

run dsa_pf_auto \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm --dsa-prefill-backend flashmla_auto" \
  "lower-risk: combo + DSA prefill=flashmla_auto (bf16)"

echo "ALL_LOWRISK_DONE"
