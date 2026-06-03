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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-9-contract.md

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
ACs: 8/11 addressed (8/11 met) | Forgotten items: 0 | Unjustified deferrals: 0 accepted; Active original-plan tasks still pending: 4

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline, taste-review knowledge, `round-8-prompt.md`, `round-8-contract.md`, `round-8-summary.md`, `goal-tracker.md`, and recent Round 5-7 summaries/reviews. Reviewed the Round-8 commits `d47dcbadb`, `fa4473694`, `67422e698`, and `0cb6b597b`, the changed AC-Q matcher, AC-10 validator/server-args/launcher code, manual fixture change, unit regressions, and Round-8 hardware artifacts under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python pytest \
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py \
  test/registered/unit/manual/test_m3b_label_capture_verdict.py \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 291 passed, 24 warnings in 11.76s

bash -n development/serve_double_sparsity.sh \
  development/serve_native_nsa.sh \
  development/benchmark.sh \
  development/benchmark_baseline.sh
# pass

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
# 2 failed, 21 passed
```

I also recomputed `runs/20260528_dsv32_mvp/dsv32_quality_smoke_concise.json` from its saved outputs with the corrected matcher and matched the committed gates exactly: `all_pass=true`, prefix 0.95, ROUGE-L 0.944, NIAH 5/5, first8 divergence 0.

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Previously verified hardware capture + unit suite. |
| AC-4 | MET | Previously verified calibrated FP8 mask + loader validation. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on >top_k prompt. |
| AC-1b | NOT MET | Chunked-prefill probe has not run; must precede AC-11. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status. |
| AC-8 / AC-9 | MET | Round-4 smoke benchmark pair + comparator verified. |
| AC-10 | MET | Config-bound artifact path authorizes radix-on before validation; both fixtures pass; final DS boot has `disable_radix_cache=false` with no env override. |
| AC-11 | NOT MET | No 3-trial radix-on 120s/600s sweep; #F must be resolved or explicitly accounted for first. |
| AC-12 | NOT MET | Full NIAH 4K/16K/64K + MMLU 5-shot gate has not run. |
| AC-Q | MET | #J false-pass fixed; concise artifact recomputed cleanly under the corrected first-8 gate. |

## Mainline Gaps

1. **The original plan remains incomplete.**

   Round 8 closes AC-Q and AC-10, but the Loop4-compatible tier is still not complete. Task12 AC-1b, task13 AC-11, task14 AC-12, and task15 evidence bundle remain active. These are not acceptable final deferrals; they are required original-plan work.

   Directive implementation plan for the remaining mainline:

   1. Fix #K below first so registered launcher-contract CI matches the evolved radix plan.
   2. Run task12 AC-1b before any AC-11 benchmark. Record chunked-prefill status in the run directory. If it fails, disable chunked prefill on both DS and DSA and record matching sidecars.
   3. Resolve or explicitly account for #F, then run task13 AC-11: 3 DS trials and 3 DSA trials at conc 16/32/64, 120s warmup, 600s measurement window, radix-on parity, comparator output with TPS/TTFT pass-or-fail summary.
   4. Run task14 AC-12: NIAH 4K/16K/64K plus MMLU 5-shot through `test_double_sparsity_v32.py`; record pass/fail at thresholds.
   5. Complete task15: assemble the evidence bundle with raw JSONL retention/location, sidecars, server args, CUDA-graph status, AC-Q, AC-10, AC-11, AC-12, mask provenance, comparator reports, and explicit notes for any known artifact limitations.

## Blocking Side Issues

1. **#K: registered Option-B launcher tests are stale and now fail.**

   `test/registered/unit/development/test_option_b_scripts.py` still encodes the old pre-Round-4/pre-AC-10 launcher contract. It forbids any `--disable-radix-cache` path in `serve_native_nsa.sh` at lines 79-88, but Round 4 intentionally added `DISABLE_RADIX_CACHE=1` for TIER-1 smoke parity while keeping the default radix-on. It also requires the obsolete fixed-line AC-10 marker to precede the first textual `--disable-radix-cache` at lines 99-123, but Round 8 changed DS to `RADIX_ARGS` with `RADIX_FIXTURE_ARTIFACT` (`development/serve_double_sparsity.sh:56-68`).

   Required fix: update the registered test to assert the evolved contract instead of the old one:

   1. DSA launcher default is radix-on, with an explicit smoke-only `DISABLE_RADIX_CACHE=1` path.
   2. DS launcher default is radix-off, and artifact-driven radix-on passes `--double-sparsity-radix-fixture-artifact` without requiring the old fixed marker.
   3. Remove or rewrite the obsolete marker test.
   4. Rerun `PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q` plus the DS/sequential suites.

## Queued Side Issues

1. **Plan-specific terms were newly added to production comments/help.**

   The plan explicitly says implementation code/comments should not introduce plan markers such as `AC-`, `DEC-`, `Tier`, or `Phase`. Round 8 added several in production-facing code and launcher comments, for example `development/serve_double_sparsity.sh:50-64`, `python/sglang/srt/layers/attention/double_sparsity/validator.py:297`, and `python/sglang/srt/server_args.py:6117,7213`. This does not invalidate AC-10, but it should be cleaned while updating the launcher-contract tests.

2. **AC-10 label-capture artifact provenance is weaker than the final bundle should be.**

   `ac10_label_capture.json` passes the cached-prefix proof (`cached_tokens_warm_pass=128`; first divergence is after cached position 128), but it records `server_args: null` because the fixture looks only for nested `server_args` while `/get_server_info` returns server args at top level. It also records `local_commit_sha=fa4473694` because the artifact was generated before the artifact/fix commit. This is not an AC-10 blocker because final radix-on server info and logs cover the serving config, but task15 should include an explicit provenance note or regenerate/copy the fixture artifact with complete server-info metadata.

3. **#F remains queued for AC-11.**

   DS effective concurrency at `mem_fraction_static=0.6` can make TTFT comparison queue-dominated unless resolved or reported before task13.

4. The stale `calibrate.py` operator recipe docstring remains queued cleanup.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added the Round-8 review correction entry.
- Moved task9 / AC-Q and task11 / AC-10 to Completed and Verified.
- Kept task12-task15 active.
- Added #K to Blocking Side Issues.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-9-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-9-summary.md

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
