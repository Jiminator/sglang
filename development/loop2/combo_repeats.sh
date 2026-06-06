#!/usr/bin/env bash
set -uo pipefail
cd /sgl-workspace/sglang
BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"
for r in rep2 rep3; do
  SGLANG_ENABLE_SPEC_V2=1 TAG="combo_$r" EXTRA_ARGS="$BASE" \
    RATIONALE="AC-2.1 finalist repeat $r of incumbent combo" READY_TIMEOUT=900 \
    bash development/loop2/run_candidate.sh > "development/loop2/logs/combo_$r.out" 2>&1
  echo "combo_$r: $(grep -oE 'client_TPS    = [0-9.]+' development/loop2/logs/combo_$r.out | tail -1)" >> development/loop2/combo_repeats.txt
done
echo "COMBO_REPEATS_DONE $(date +%H:%M:%S)" >> development/loop2/combo_repeats.txt
