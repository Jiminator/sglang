#!/usr/bin/env bash
# Winner-level page-size probes (page-size flexibility evidence): combo and combo+IndexCache, each + --page-size 32.
set -uo pipefail
ROOT=/sgl-workspace/sglang; cd "$ROOT"
export SGLANG_ENABLE_SPEC_V2=1 READY_TIMEOUT=2400
PAT='FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS'
run(){ TAG="$1" EXTRA_ARGS="$2" RATIONALE="$3" bash "$ROOT/development/loop1/run_candidate.sh" \
  > "$ROOT/development/loop1/logs/cand_$1.out" 2>&1; echo "==== $1 rc=$? ===="; }
run combo_page32 \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm --page-size 32" \
  "page-size check: safe combo + --page-size 32 (check effective page size)"
run indexcache_page32 \
  "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm --page-size 32 --json-model-override-args {\"index_topk_pattern\":\"${PAT}\"}" \
  "page-size check: indexcache + --page-size 32 (check effective page size)"
echo "PAGE_PROBES_DONE"
