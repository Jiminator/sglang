# Code Review - Round 11

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-11-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 11 Summary

Mainline: **fix the AC-4 selector-behavior mismatch** my R10 change introduced — the one CPU blocker
Codex asked to fix before the GPU close-out. CPU-only round; no server launched.

## Work Completed
- **Bug (mine, R10):** the "DS effective" table column rendered the resolved config DEFAULTS
  (`selector_width_buckets=[5120]`, `score_reduce_dtype=bf16`) for the REFERENCE arms — but the reference
  selector path (`selector_impl=reference_*`) **bypasses** those knobs: it dequantizes to fp32 and runs
  the exact absorbed channel-dot + full-width `torch.topk`, with no fp8-in-register, no bf16 reduce, no
  radix kernel, no selector-width bucketing (`deepseek_v2.py:2137-2143`, `config.py:132-136`). So the
  table falsely showed `ref_faithful`/`ref_cosine`/`ref_cosine_noinc` as `W[5120] · bf16`.
- **Fix:** split config-object provenance from actual selector behavior.
  - Added `ds_selector_behavior_for(arm)` — path-aware on `selector_impl`. **production** → resolved
    width/reduce + `blocked/radix` top-k + `fp8 absorbed in-register` + `raw-dot (scorer_norm=off)`.
    **reference_*** → `full (no bucketing)` / `none (per-rank-local fp32; no cross-TP reduce)` /
    `exact torch.topk` / `exact fp32 dequant` / raw-dot|cosine, with a note that the production knobs are
    bypassed.
  - Each DS arm JSON now records `ds_selector_behavior` alongside `effective_ds_config` (kept as
    config-object provenance, now noted that a set knob isn't necessarily used).
  - `evidence_table.md` renders `ds_selector_behavior` (path·width·reduce·topk·scorer·head-agg): production
    arms show `[5120] · bf16` (or `fp32` for ds_reduce_fp32) · `blocked/radix`; reference arms show
    `full · none · exact torch.topk`. Column header renamed to "DS selector behavior".
  - **Fail-closed assertion:** a `reference_*` arm's `ds_selector_behavior` must NOT show production
    `5120`/`bf16` as used — **verified it fires** when re-broken. `findings.md`/`ROOT_CAUSE.md` already
    describe the reference arms as exact-fp32/full-width/no-reduce (no contradiction; confirmed by scan).

## Files Changed (committed `482ff8083`)
- `build_ledger.py` (ds_selector_behavior_for + rec field + table renders behavior + header + reference-arm
  bypass assertion), `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `build_ledger.py` → provenance consistent (blob `4f83d15605ca`); reference-arm behavior guard **asserts
  (exit 1)** when a reference arm is made to show `bf16`; effective-config + ds_reduce_fp32 + DS-config
  assertions still hold.
- Table check: production_ds `prod · [5120] · bf16 · blocked/radix · raw-dot · max`; ds_reduce_fp32
  `prod · [5120] · fp32 · …`; ref_* `ref · full (no bucketing) · none … · exact torch.topk · raw-dot|cosine`.
  No reference arm shows production width/reduce.
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix` — all exit 0.
- CPU-only (GPUs idle). No `.pt`/`.humanize` committed. No selection/adapter fix.

## Remaining Items (for AC-8 COMPLETE — all GPU/instrumentation)
- **AC-2.1** forced-all physical-slot assertions (`forced_all_assertions.json`) + **AC-4** length-cap
  garbage counters — guarded `logical_to_physical`→`transform_index_page_table_decode` adapter
  instrumentation + a GPU run (shared physical-slot boundary).
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — resident-latent capture +
  offline materialize/compare at top-2048.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; GPU run, labeled corroboration.
- **AC-4** remaining serial cells (DSA-radix serial, production DS sparse serial); selected-vs-total gaps.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-effective-config-not-launch-overrides
- Notes: Added the R11 corollary — a resolved config OBJECT is not the runtime BEHAVIOR: a knob being set
  doesn't mean the dispatch path uses it. For an AC-4 comparison/display surface, derive behavior from the
  dispatch key (`selector_impl`), not the config object (production → resolved knobs used; reference →
  full/none/exact-topk, knobs bypassed); display the behavior view, keep the config object as provenance
  only, and guard that a bypassed knob can never be shown as "used".

## Goal Tracker Update Request

### Requested Changes:
- Close **R10-review blocking: "DS effective" reports dormant defaults as reference-arm behavior** —
  fixed via `ds_selector_behavior` + table + the reference-arm bypass guard (verified fires).
- Mark **AC-4 (task9)** advanced: config-object + behavior surfaces both correct; remaining AC-4 gaps are
  GPU/instrumentation (garbage counters, serial cells, selected-vs-total).
- Plan Evolution Round-11 row added.

### Justification:
This was Codex's single new CPU blocker — a provenance bug I introduced in R10 that would mislead the
AC-4/AC-8 production-vs-reference comparison (reference arms shown as if they used the production width
ladder + bf16 reduce). The behavior surface now derives from the actual dispatch path and is guarded, so
the table reflects what each selector truly does. All remaining close-out items (AC-2.1/2.4/3.1/4-garbage/
serial/8) require GPU capture or adapter instrumentation and are the next sequence toward AC-8 COMPLETE.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
96202e4c4 [loop11b] R3: corrected verdict evidence (results_r3) + supersede results_r2
2ce2adf4e [loop11b] R3: ledgers to one current state (AC-8) — results_r3, mask=regenerated, close-out ACTIVE-until-push
e0935e5a9 [loop11b] R3: AC-8 close-out COMPLETE — pushed to owner fork Jiminator/sglang
da12616a5 [loop11b] R3 review fix [P3]: build_corpus.py creates the output dir before writing
101926d76 [loop11b] R3 review fixes [P2 x2]: report verdict vs exit consistency + fail-closed partial DS evidence
9ab62e6ad [loop11b] R3 review fixes: DS abort test rename + comparator report verdict/labels + green test suite
3058bdc35 [loop12] add gen-plan output + draft; record pensieve doctor state
aaefdaf1e [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
8f88e1aef [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
4706b2138 [loop12] document double-sparsity v2 performance numbers
480fd70ba [loop12] add double-sparsity v2 run-and-evaluate runbook
2babc5afa [loop12] add gsm8k evidence + refresh perf verdict numbers
180f6dd6d [loop13] add diagnosis-loop plan, draft, and guarded harness
fc6ac20a7 [loop13] diagnostic DS selectors: fp32 raw-dot reference + forced-all dense control
29ec137bf [loop13] harness: ref/ds_capture/ds_forced_all serve modes + AC-1 baseline evidence
16caf4f5b [loop13] reference selector: gather-then-dequant (full-pool dequant was intractable)
5a1da871d [loop13] verdict: DS dense degradation = current-decode-slot exclusion (H3)
fea920c06 [loop13] reference selector: served cosine + faithful/leak-free ceiling
62ad64346 [loop13] Round 1 verdict FLIP: ceiling GOOD; sparse = raw-dot scorer_norm=off lock
ac479aeb3 [loop13] Round 2: per-arm evidence ledger, baseline consistency, captures, cleanup
29ed825fa [loop13] Round 3: ledger SHA provenance, capture row-identity, exact-join analyzer
393966c02 [loop13] Round 4: AC-2.3 RESOLVED on real captured rows; fail-closed analyzer; ledger provenance
c7b66f04b [loop13] Round 5 (drift recovery): pruning-valid AC-2.3 + AC-6 scorer×current-slot 2×2
8b55dfba3 [loop13] Round 6: AC-6 corroboration + complete per-leg bisection matrix
8281361e7 [loop13] Round 7: measure the score-reduce leg + dense current-slot corroboration
4d874b89e [loop13] gitignore transient DS capture scratch dirs
752752f6d [loop13] Round 8: fix ds_reduce_fp32 metadata; settle AC-2.2 + AC-4 sample IDs
5d48cbd0d [loop13] Round 9: reconcile evidence package — full DS configs + AC-2.2 consistency
75158e505 [loop13] Round 10: finish evidence-package consistency (head-agg + effective DS config)
482ff8083 [loop13] Round 11: AC-4 selector-behavior surface (reference arms != production knobs)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-10-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-10-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-9-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-9-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-8-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-8-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: Required Finding Classification

You MUST classify your findings into these lanes:
- **Mainline Gaps**: plan-derived work or AC progress that is missing, incomplete, or regressing
- **Blocking Side Issues**: bugs or implementation issues that block the current mainline objective from succeeding safely
- **Queued Side Issues**: valid non-blocking follow-up issues that should be documented but must NOT take over the next round

Also include a one-line verdict:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
```

This verdict line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

If Claude mostly worked on queued side issues and failed to advance the mainline, say so explicitly.

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Blocking Side Issues" or "Queued Side Issues" as appropriate
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If you reject a requested tracker change**: Include in your review why it was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New blocking issues: Add to "Blocking Side Issues"
- New queued issues: Add to "Queued Side Issues"
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-11-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
