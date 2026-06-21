# Code Review - Round 7

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-7-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 7 Summary

Mainline: **finish the AC-6 bisection matrix honestly.** Round 6 was ADVANCED; Codex flagged two correct
gaps — a runnable leg wrongly marked "blocked", and a sparse-only corroboration used for the dense cost.
Both fixed and measured on the actual workload.

## Work Completed
- **Score-reduce leg now MEASURED** (was wrongly "blocked"). `serve.sh ds_reduce_fp32` = the production
  `ds` config + `score_reduce_dtype="fp32"` (same graph mode; the **only** variable vs `production_ds`;
  config accepts `{fp32,bf16}` — a runnable route, not a fix). GSM8K: **dense 0.620 / sparse 0.000 —
  identical to production_ds** ⇒ the reduce dtype is **not** a culprit. Selection-level corroboration
  (`ac6_score_reduce_corrob.py` → `ac6_score_reduce_fp32_corrob.json`): reduce the SAME captured per-rank
  `pre_reduce_scores` (validated `sum(pre)==post` **702/702**) in bf16 vs fp32 → median selected-set
  **Jaccard 0.998** (127/702 identical); only bottom-of-top-k near-ties reshuffle.
- **Dense current-slot corroboration added** (Codex gap #2). `ac6_corrob_ref_cosine_noinc.py` restructured
  into regime sections with **distinct** invariants, on real captures:
  - **sparse** (seq_len>top_k) **4992/4992** — full top-k, include SWAPS the current slot in (symdiff==2,
    Jaccard (k−1)/(k+1)).
  - **dense** (seq_len≤top_k) **3744/3744** — room for all, exclude DROPS the current slot
    (`valid_length==seq_len-1`), include ADDS it (`valid_length==seq_len`, symdiff==1, no eviction).
  Real dense captures (seq_len ~790) came from a separate eager run.
- **fp8-absorbed leg re-verified as the only blocker** with a precise, source-checked citation: no
  production config flag toggles fp8-vs-fp32 absorbed scoring (the graph selector scores the fp8 resident
  latent in-register, `deepseek_v2.py:2602`→`absorbed_latent_kernel.py`); exact-fp32 absorbed exists only
  on the `reference_*` path, which bundles current-slot/TF32/radix/width/reduce (no single-variable
  isolation). Bounded second-order (≤~1.3 pp) now reduce is measured and radix/width retired.
- **Matrix + ledger + generated text reconciled:** `ac6_bisection_matrix.json` legs
  measured[2,3,7]/retired[4,5]/not-a-difference[1]/blocked[6=fp8 only]; `build_ledger.py` adds the
  `ds_reduce_fp32` arm (measured_source + ac6_leg + corroboration), and the AC-6 guard now protects BOTH
  AC-6 arms (verified: asserts when either corroboration artifact is missing); the generated
  `evidence_table.md` verdict text no longer says fp8/bf16-reduce/head_agg are out of scope
  (**0 occurrences**). `cheap_controls.json._status` pointer fixed to `superseded_round3_join_summary`.

## Files Changed (committed `8281361e7`)
- NEW: `development/loop13/ac6_score_reduce_corrob.py`, `evidence/ac6_score_reduce_fp32_corrob.json`.
- MODIFIED: `serve.sh` (ds_reduce_fp32 mode), `build_ledger.py` (arm + verdict text), `ac6_bisection_matrix.py`,
  `ac6_corrob_ref_cosine_noinc.py` (dual-regime), `ROOT_CAUSE.md`, `evidence/findings.md`,
  `evidence/cheap_controls.json`, `evidence/ac6_bisection_matrix.json`, `evidence/ac6_ref_cosine_noinc_corrob.json`,
  `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`, `.gitignore`.

## Validation
- `test_reference_selectors.py` → **all 5 pass**.
- `verify_ac2_3.py` (sparse) → 4992/4992, exit 0.
- `ac6_corrob_ref_cosine_noinc.py` → sparse 4992/4992 + dense 3744/3744, exit 0.
- `ac6_score_reduce_corrob.py` → 702 groups, sum==post 702/702, median Jaccard 0.998, exit 0.
- `ac6_bisection_matrix.py` → measured[2,3,7]/retired[4,5]/not-a-difference[1]/blocked[6], exit 0.
- `build_ledger.py` → provenance consistent (blob `1280fa0339`); AC-6 guard **asserts** when either
  corroboration artifact is removed.
- `evidence_table.md` → **0** "out of scope"/"Untested numeric" occurrences; `ds_reduce_fp32` row 0.620/0.000.
- Discipline: one TP=8 server at a time (eager capture run + graph measurement run, each torn down to
  0 MiB). No `.pt`/`.humanize` committed. No selection/adapter fix landed.

## Remaining Items (for AC-8 close-out)
- fp8-absorbed per-leg blocker awaits review sign-off (the only un-measured AC-6 leg; genuinely blocked).
- AC-2.4 recall-oracle@2048 (NIAH-only instrument; GSM8K has no oracle).
- AC-2.1 `forced_all_assertions.json`; AC-3.1 captured-row materialized-K; AC-2.2 head-agg semantics
  (R7 note: `sum(pre_reduce)==post` 702/702 confirms the reduce, but AC-2.2's SUM-vs-global-max question
  is separate); AC-4 sample IDs/order + garbage counters; AC-8 final writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-corroboration-cardinality-dependent
- Notes: Added a lesson that selected-index corroboration invariants are cardinality-dependent — a
  fixed-size top-k SWAP (sparse, seq_len>top_k, symdiff==2) is a different mechanism than a pure ADD
  (dense, seq_len≤top_k, symdiff==1, valid_length seq_len-1→seq_len); corroborate each regime with its own
  invariant, do not reuse one for the other. Also UPDATED BL-20260621-per-leg-blocker-not-blanket with the
  R7 refinement: before declaring a leg `blocked`, verify no existing CONFIG ROUTE can test it on the same
  servable path (the R6 reduce leg was wrongly blocked because `score_reduce_dtype="fp32"` is a runnable
  toggle on the raw-dot path); run the route first, block only when none exists (e.g. fp8-absorbed).

## Goal Tracker Update Request

### Requested Changes:
- Close **R6 mainline gap: reduce leg wrongly blocked** — `ds_reduce_fp32` measured (0.620/0.000 =
  production; corrob median Jaccard 0.998); matrix leg 7 = measured.
- Close **R6 mainline gap: sparse-only current-slot corroboration** — dense regime added (3744/3744,
  symdiff==1, valid_length invariant), distinct from the sparse swap.
- Close **R6 blocking: contradictory generated evidence** — `evidence_table.md` regenerated (0 "out of
  scope"); `cheap_controls.json._status` pointer fixed.
- Mark **task11 (AC-6)**: all legs measured/retired/not-a-difference except fp8-absorbed, which carries an
  accepted per-leg blocker (no production config for fp32 absorbed scoring) awaiting review sign-off.
- Plan Evolution Round-7 row added.

### Justification:
Both R6-review mainline gaps are now measured/corroborated on the actual workload with the correct
per-regime invariants. The reduce result (fp32 reduce = production 0.620/0.000, selection Jaccard 0.998)
is decisive that the reduce dtype is innocent. The sole remaining AC-6 leg (fp8-absorbed) has no
production config route — exact-fp32 absorbed scoring exists only on the multi-variable reference path,
so isolating it would require a new production kernel (a fix forbidden by the Ultimate Goal) — making it a
genuine documented per-leg blocker, not a deferral.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
c16c0d202 [loop11b] R2: wire GLM/dsa-backend DS per-request summary (AC-5) — host-side, graph-robust
b5c4d72be [loop11b] R2: verdict re-established + AC-5 PASS + raw evidence committed (lossless)
8062039d8 [loop11b] R2: AC-8 ledgers regenerated to final state + push status; de-AC the new backend comment
df18a93d0 [loop11b] R3: fix total_tokens metric semantics (AC-5) — explicit field, not rate-inverse
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-6-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-6-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-5-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-5-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-4-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-4-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-7-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
