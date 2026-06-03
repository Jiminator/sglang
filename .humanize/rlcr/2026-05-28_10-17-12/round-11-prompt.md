Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop5/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop5/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-11-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 10/11 addressed (9 met, AC-11 executed with a recorded directional MISS, AC-12 not run) | Forgotten items: 0 | Unjustified deferrals: 0 | Active original-plan tasks remaining: 2

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review knowledge, `round-10-prompt.md`, `round-10-contract.md`, `round-10-summary.md`, `goal-tracker.md`, and Round 7-9 summaries/reviews. Reviewed commit `a24bc469c`, the AC-11 comparator changes, comparator tests, and AC-11 artifacts under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python python development/benchmark_compare.py --ac11 \
  --ac11-baseline-results runs/20260528_dsv32_mvp/ac11_results/native_nsa_gsp_isl4096_osl512_c{16,32,64}_t{1,2,3}.jsonl \
  --ac11-ds-results runs/20260528_dsv32_mvp/ac11_results/double_sparsity_gsp_isl4096_osl512_c{16,32,64}_t{1,2,3}.jsonl
# exit 3, directional MISS, effective-vs-nominal table present

PYTHONPATH=python pytest \
  test/registered/unit/development/test_ac11_comparator.py \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py \
  test/registered/unit/development/test_option_b_scripts.py -q
# 359 passed, 24 warnings, 28 subtests
```

I also parsed all AC-11 JSONLs and sidecars: 18 raw JSONLs exist locally, 18 sidecars exist, every JSONL duration is >= 600s, all runs use 120s warmup / 600s measurement window / NUM_PROMPTS=64 / ISL=4096 / OSL=512, and the locked fields match (`tp_size=8`, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `chunked_prefill_size=8192`, `disable_radix_cache=false`, graph flags, DSA/DS backends). DS mem fraction is consistently 0.6 within DS trials; DSA mem fraction is consistently 0.85 within DSA trials.

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Previously verified hardware capture + producer regression. |
| AC-4 | MET | Previously verified native-FP8 sharded calibration, mask validation, and SHA. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, and invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on a >top_k prompt. |
| AC-1b | MET | Round 9 AC-1b probe passed at the radix-on operating point. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status, distinct from disabled piecewise graph. |
| AC-8 / AC-9 | MET | Round-4 smoke DS/DSA benchmark pair + comparator verified. |
| AC-10 | MET | Round-8 no-env-override radix flip verified via config-bound fixture artifact and final radix-on DS boot. |
| AC-11 | EXECUTED, DIRECTIONAL MISS | Round 10 produced the 3-trial radix-on sweep and comparator report. Per DEC-7, the miss is recorded as AC-11 failure + follow-up, not a build-break. |
| AC-12 | NOT MET | No NIAH 4K/16K/64K + MMLU 5-shot run or artifacts found. |
| AC-Q | MET | Round-8 corrected AC-Q gate verified with `all_pass=true`. |

## Mainline Gaps

1. **AC-12 remains unrun and is still a hard Loop4-compatible MVP gate.**

   `test/manual/test_double_sparsity_v32.py` is the required harness. I found no `ac12_*`, NIAH, or MMLU artifacts under `runs/20260528_dsv32_mvp/` or `development/results/`, so task14 is still pending. This is not an acceptable final deferral.

   Directive implementation plan for task14:

   1. Start a DSA baseline server and a DS radix-on server on the two available H200 nodes. Use `development/serve_native_nsa.sh` for DSA and `development/serve_double_sparsity.sh` for DS with `RADIX_FIXTURE_ARTIFACT=runs/20260528_dsv32_mvp/ds_radix_fixture_state.json`, the validated channel mask, TP=8, fp8 KV, page 64, chunked prefill 8192, and the same Option-B graph flags.
   2. Capture `/get_server_info` from both servers into `runs/20260528_dsv32_mvp/ac12_dsa_server_info.json` and `runs/20260528_dsv32_mvp/ac12_ds_server_info.json`; preserve both boot logs.
   3. Pre-populate or point `AC12_MMLU_DATA_DIR` at a valid MMLU `dev/` + `test/` CSV tree. Do not allow the MMLU path to skip after servers are configured.
   4. Run `PYTHONPATH=python DS_BASE_URL=<ds> DSA_BASE_URL=<dsa> AC12_NIAH_NUM_PROMPTS=20 AC12_MMLU_NUM_EXAMPLES=200 python -m pytest test/manual/test_double_sparsity_v32.py -v`.
   5. Copy every `development/results/ac12_*.json` artifact into `runs/20260528_dsv32_mvp/ac12_results/`, record the exact env/command, and summarize pass/fail in `runs/20260528_dsv32_mvp/ac12_analysis.md`.
   6. If NIAH 64K fails because of the known top_k-bounded recall risk, publish that as a hard AC-12 failure with evidence. Do not reclassify AC-12 as directional.

2. **task15 evidence bundle remains incomplete until AC-12 exists.**

   There is no final evidence index/bundle file. The bundle must not be assembled as complete until task14 has real pass/fail artifacts.

   Directive implementation plan for task15 after AC-12:

   1. Create `runs/20260528_dsv32_mvp/evidence_bundle.md`.
   2. Include an AC-by-AC table with artifact paths for AC-0, AC-4, AC-1, AC-1.1, AC-1b, AC-6, AC-8/9, AC-10, AC-11, AC-12, and AC-Q.
   3. Include raw JSONL locations for smoke and AC-11 even though JSONLs are gitignored; include committed sidecars and comparator JSON/Markdown.
   4. Include mask provenance (`calibrate.log`, validation output, SHA), server args/server_info, CUDA-graph status, chunked-prefill status, radix fixture artifacts, and the AC-10 label-capture provenance note.
   5. State AC-11 accurately as “executed; directional TTFT/TPS target missed; #F admission caveat and follow-up filed,” not as a green performance pass.
   6. If AC-12 fails, the bundle must say the Loop4-compatible MVP is not complete.

## Blocking Side Issues

No unresolved blocking side issue remains for the Round-10 AC-11 execution artifact. #F is accounted in the comparator/report for this artifact and has a follow-up in `ac11_analysis.md`.

## Queued Side Issues

1. **Comparator validation hole: `mem_fraction_static` is ignored within a side, not only across DSA vs DS.**

   Round 10 correctly avoids refusing DSA 0.85 vs DS 0.6, but `_normalize_ac11_server_args()` excludes `mem_fraction_static` for both `_validate_cross_side_agreement()` and `_validate_per_side_agreement()` (`development/benchmark_compare.py:648`, `development/benchmark_compare.py:829`). A synthetic reproducer with DSA trial mem fractions `0.85/0.80/0.75` and DS `0.6/0.6/0.6` exits 0, so the comparator would median across per-side mismatched launch knobs. The current AC-11 artifact is not invalidated because its sidecars are constant within each side, but the comparator should be tightened before any AC-11 rerun.

   Required fix when the comparator is next touched: keep `mem_fraction_static` ignored for cross-side agreement, but compare it within each side. Add a regression that per-side mem-fraction drift refuses with exit 2 while DSA 0.85 vs DS 0.6 still proceeds.

2. **AC-11 directional miss follow-up remains performance work, not next-round mainline.**

   The TokenLabelTable/KV-budget and conc-64 admission profile work in `ac11_analysis.md` is valid, but it should not take over the next round ahead of AC-12 and task15 unless the user explicitly reprioritizes performance tuning.

3. **Existing queued bundle cleanup remains.**

   The stale `calibrate.py` operator recipe docstring and AC-10 label-capture provenance note are still valid task15 cleanup items.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added a Round-10 review correction entry.
- Moved task13 / AC-11 from Active to Completed and Verified as executed with a recorded directional MISS.
- Kept task14 / AC-12 and task15 evidence bundle Active.
- Left `Explicitly Deferred` empty.

Original plan work remains pending.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-11-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-11-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
