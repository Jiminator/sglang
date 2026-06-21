# Round 1 Contract

## Mainline Objective
Fix the AC-8 perf wrapper so it runs the EXACT loop-11b conc-64 generated-shared-prefix workload
(1 prefix group, all prompts sharing the system prompt) instead of the stock GSP default (64 groups
× 16 = 1024 requests), then re-prove AC-8 on the live DS server and replace the perf evidence.

## Target ACs
- **AC-8** (perf parity) — the only AC Codex left open. AC-1..AC-7, AC-9, AC-10 are Completed and
  Verified per Codex's review; do not re-litigate them.

## Blocking Side Issues (in scope — they block AC-8)
- [P1] `benchmarks/bench_double_sparsity.py` omits `--gsp-num-groups`/`--gsp-prompts-per-group`, so
  the stock default (64 groups × 16 prompts) generated 1024 requests across 64 prefix groups — NOT
  the loop-11b workload (NUM_GROUPS=1, `--gsp-prompts-per-group ${NUM_PROMPTS}`). The 29.34 TPS / 23.29 s
  datapoint is on the wrong workload shape. Fix per Codex's directive: pin `--gsp-num-groups 1` +
  `--gsp-prompts-per-group <num_prompts>`, record grouping + `actual_completed` in the verdict, and
  FAIL the wrapper if `actual_completed != num_prompts`. Rerun AC-8, replace evidence, re-push.

## Queued Side Issues (out of scope as the objective, but addressed opportunistically)
- [P3] Shipped code/comments still contain plan-tracking markers (`AC-`, `Milestone`, `Phase`,
  `Step`) — contradicts plan:417 Code Style. Non-blocking for AC-8, but it is a plan requirement and
  this is the close-out round, so I will sweep + reword these durable comments during the GPU rerun
  wait. It must not displace the AC-8 fix.

## Success Criteria
1. Wrapper pins `--gsp-num-groups 1` + `--gsp-prompts-per-group <num_prompts>` over stock bench_serving.
2. Wrapper records `gsp_num_groups`, `gsp_prompts_per_group`, `expected_prompts`, `actual_completed`
   in `verdict.json`, and fails if `actual_completed != num_prompts`.
3. AC-8 rerun on the live DS server: `actual_completed == 256`, 1 prefix group, within band
   (p50 decode TPS ≥ 24.2 AND P99 TTFT ≤ 30.1 s). Evidence replaced in `development/loop12/perf_evidence/`.
4. `benchmarks/DOUBLE_SPARSITY.md` shows the exact rerun command + actual completed count.
5. No plan-terminology markers remain in shipped code/comments (P3).
6. Final sweeps green (AC-1/AC-2/AC-3/AC-4/AC-10 unchanged); corrected branch pushed to the fork.

## Hard Operational Constraints (unchanged)
One TP=8 server at a time; tear down + wait for GPU idle; NEVER `expandable_segments` for serving;
no blanket nvidia-smi kills / parent-matching pkill; push only to `Jiminator/sglang`.
