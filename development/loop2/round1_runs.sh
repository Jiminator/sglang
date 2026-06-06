#!/usr/bin/env bash
# Round 1: close Codex-flagged gaps.
#  (1) combo+IndexCache (accuracy-risk best-achievable): gate + profile + 2 repeats
#  (2) fa3/fa3 safe finalist: 2 gate repeats
#  (3) profile the 3 launchable task6 follow-ups
# Each profile run -> rollup + analyzer triage -> per-candidate md -> delete raw (DEC-4).
set -uo pipefail
ROOT=/sgl-workspace/sglang; cd "$ROOT"
BASE="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm"
IDXPAT='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
ANALYZER=.claude/skills/llm-torch-profiler-analysis/scripts/analyze_sglang_torch_profile.py
ROLLUP=development/loop2/profiling/_work/rollup.py
PROFDIR=development/loop2/profiling
LED=development/loop2/round1_results.txt
echo "# round1 started $(date +%H:%M:%S)" > "$LED"

gate () {  # tag, extra, rationale
  local tag="$1"; local extra="$2"; local why="$3"
  echo ">>> $(date +%H:%M:%S) GATE $tag" >> "$LED"
  SGLANG_ENABLE_SPEC_V2=1 TAG="$tag" EXTRA_ARGS="$BASE $extra" RATIONALE="$why" READY_TIMEOUT=900 \
    bash development/loop2/run_candidate.sh > "development/loop2/logs/r1_${tag}.out" 2>&1
  local rc=$?
  echo "GATE $tag rc=$rc $(grep -oE 'client_TPS    = [0-9.]+|STATUS=[a-z_]+' development/loop2/logs/r1_${tag}.out | tr '\n' ' ')" >> "$LED"
}

profile_and_extract () {  # tag, extra
  local tag="$1"; local extra="$2"
  echo ">>> $(date +%H:%M:%S) PROFILE $tag" >> "$LED"
  SGLANG_ENABLE_SPEC_V2=1 TAG="$tag" EXTRA_ARGS="$BASE $extra" READY_TIMEOUT=900 \
    bash development/loop2/profile_candidate.sh > "development/loop2/logs/r1prof_${tag}.out" 2>&1
  local prc=$?
  local tp0=$(find "$PROFDIR/raw/$tag" -name '*TP-0.trace.json.gz' 2>/dev/null | head -1)
  if [ "$prc" = "0" ] && [ -n "$tp0" ]; then
    python3 "$ROLLUP" "$tp0" > "development/loop2/logs/rollup_${tag}.txt" 2>/dev/null
    python3 "$ANALYZER" --framework sglang --input "$tp0" --output-dir "$PROFDIR/_work" \
      --kernel-table-limit 20 --overlap-table-limit 8 > "$PROFDIR/_work/${tag}_triage.md" 2>/dev/null
    echo "PROFILE $tag prc=$prc rollup+triage_ok" >> "$LED"
  else
    echo "PROFILE $tag prc=$prc FAILED" >> "$LED"
  fi
  rm -rf "$PROFDIR/raw/$tag"
}

# (1) IndexCache best-achievable
gate    "indexcache_loop2" "--json-model-override-args $IDXPAT" "AC-9 best-achievable: combo+IndexCache (ACCURACY-RISK)"
profile_and_extract "indexcache_loop2" "--json-model-override-args $IDXPAT"
gate    "indexcache_rep2"  "--json-model-override-args $IDXPAT" "AC-2.1 IndexCache repeat 2"
gate    "indexcache_rep3"  "--json-model-override-args $IDXPAT" "AC-2.1 IndexCache repeat 3"

# (2) fa3/fa3 safe finalist repeats
gate    "fa3fa3_rep2" "--dsa-prefill-backend fa3 --dsa-decode-backend fa3" "AC-2.1 fa3/fa3 repeat 2"
gate    "fa3fa3_rep3" "--dsa-prefill-backend fa3 --dsa-decode-backend fa3" "AC-2.1 fa3/fa3 repeat 3"

# (3) profile launchable task6 follow-ups (same flags as their gate runs)
profile_and_extract "t6_fused_moe_sum_ar" "--enable-fused-moe-sum-all-reduce"
profile_and_extract "t6_topk_flashinfer"  "--dsa-topk-backend flashinfer"
profile_and_extract "t6_contdecode2"      "--num-continuous-decode-steps 2"

echo "ROUND1_RUNS_DONE $(date +%H:%M:%S)" >> "$LED"
