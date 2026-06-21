# Code Review - Round 6

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-6-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 6 Summary

Mainline: **close out AC-6.** Round 5 was ADVANCED; Codex blocked close-out on four items — all
addressed this round, entirely on CPU (no server launched; GPUs idle throughout).

## Work Completed
- **AC-6 corroboration for `ref_cosine_noinc`** (`ac6_corrob_ref_cosine_noinc.py` →
  `evidence/ac6_ref_cosine_noinc_corrob.json`). Replays the REAL `_select_topk_with_optional_current`
  — the ONLY code differing between `ref_cosine` and `ref_cosine_noinc` — on the 4992 captured sparse
  pruning rows, include_current True vs False. **4992/4992**: the include flag swaps **exactly** the
  current decode slot into the selected set (Jaccard **0.999024 = (2048−1)/(2048+1)**, symmetric
  difference **== 2** on every row), and the current slot is **−inf-masked** in every capture (the
  production `_slot_written` exclusion). Ties the selection-level difference to the measured
  0.940→0.625 dense / 0.940→0.313 sparse cost. Fail-closed on zero pruning rows / mechanism violation.
- **Complete per-leg AC-6 bisection matrix** (`ac6_bisection_matrix.py` →
  `evidence/ac6_bisection_matrix.json`) — no blanket "out of scope"; every leg classified:
  - **scorer** (raw-dot↔cosine) — measured (2×2 + materialized-raw selection equality)
  - **current-slot** (incl↔excl) — measured (`ref_cosine_noinc` + the corroboration above)
  - **radix top-k**, **selector width** — retired (AC-2.3, 4992/4992)
  - **head_agg** — not-a-differing-variable (`max` on both paths; cross-TP sum-of-max is AC-2.2)
  - **fp8-absorbed**, **bf16-reduce** — **blocked** with a specific code citation: these live only in the
    production absorbed-latent Triton kernel (`absorbed_latent_kernel.py`, called at
    `deepseek_v2.py:2588/2602`), which implements **only** `scorer_norm="off"`; `config.py:110`
    `_ALLOWED_SCORER_NORM=("off",)` + validation at `:170` hard-reject cosine — testing fp8/reduce
    under cosine needs a new production-path cosine kernel = a selection-path fix (forbidden). Bounded
    second-order (raw-dot exact-fp32 0.013 vs fp8+bf16 0.000 ⇒ ≤~1.3 pp).
- **Blocking evidence-integrity fixes:**
  - `ref_cosine_noinc` measured provenance corrected: `measured_git_sha` now `393966c02` (run HEAD,
    dirty) + a `measured_source` recording the `serve.sh` blob `e1c83e22` that defined the mode
    (committed `c7b66f04b`) — replayable. Was `fea920c06` (where the mode did not exist).
  - `cheap_controls.json`: the stale Round-3 join result (`81/546`, `…=false`) moved out of the
    authoritative `summary` into `superseded_round3_join_summary`; `summary` now carries the
    pruning-valid `4992/4992` AC-2.3 verdict — one machine-readable verdict, no contradiction.
  - `build_ledger.py`: **fails loud** if an AC-6 arm records GSM8K scores but has no corroboration
    artifact on disk (verified: asserts when the artifact is removed). Arm JSON now carries
    `ac6_leg` + `corroboration_artifact` + `measured_source`.
- ROOT_CAUSE.md / findings.md updated with the per-leg matrix + citations (replacing the blanket
  out-of-scope text).

## Files Changed (committed `8b55dfba3`)
- NEW: `development/loop13/ac6_corrob_ref_cosine_noinc.py`,
  `development/loop13/ac6_bisection_matrix.py`,
  `evidence/ac6_ref_cosine_noinc_corrob.json`, `evidence/ac6_bisection_matrix.json`.
- MODIFIED: `build_ledger.py` (measured_source/ac6_leg/corroboration guard), `ROOT_CAUSE.md`,
  `evidence/findings.md`, `evidence/cheap_controls.json`, `evidence/evidence_table.md`,
  `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `test_reference_selectors.py` → **all 5 pass**.
- `verify_ac2_3.py` (sparse) → 4992/4992, exit 0.
- `ac6_corrob_ref_cosine_noinc.py` → 4992/4992 single-swap, exit 0.
- `ac6_bisection_matrix.py` → 7 legs classified, exit 0.
- `build_ledger.py` → provenance consistent (blob `3757eb5363`); AC-6 guard **asserts (exit 1)** when
  the corroboration artifact is removed, passes when present.
- GPUs idle (0 MiB), no server launched this round. No `.pt`/`.humanize` committed. No fix landed.

## Remaining Items (for AC-8 close-out)
- The two per-leg blockers (fp8-absorbed, bf16-reduce) need review sign-off that the code citation is
  accepted as a valid non-fix-route block.
- AC-2.4 recall-oracle@2048 (the `recall_oracle` instrument is NIAH-only per DEC; GSM8K has no oracle —
  selected-index/current-slot corroboration is the accepted GOOD-branch alternative).
- AC-2.1 `forced_all_assertions.json`; AC-4 per-example sample IDs/order + length-cap garbage counters;
  AC-3.1 captured-row materialized-K; AC-2.2 head-agg `pre_reduce` semantics; AC-8 final writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-corroborate-via-selection-replay, BL-20260621-per-leg-blocker-not-blanket
- Notes: (1) Corroborate a served single-variable bisection arm by replaying the EXACT selection
  function on already-captured score rows instead of a second GPU capture+join — with a fixed-size
  top-k, force-including one position is a SWAP (symdiff==2, Jaccard exactly (k−1)/(k+1)), assert the
  exact value. (2) A planned diagnostic deferred under "no fix" needs a PER-LEG blocker citing the exact
  code path (verified against source this round: config.py:110/170 + the raw-dot-only kernel), not a
  blanket "out of scope"; distinguish `blocked` from `not-a-differing-variable`.

## Goal Tracker Update Request

### Requested Changes:
- Close **Blocking: AC-6 arm lacks corroboration** — `ac6_ref_cosine_noinc_corrob.json` (4992/4992
  single-swap) + `ac6_bisection_matrix.json` + the build_ledger AC-6 guard.
- Close **Blocking: ref_cosine_noinc measured provenance inaccurate** — now records run HEAD
  `393966c02` + serve.sh blob `e1c83e22` (was `fea920c06`).
- Close **Blocking: cheap_controls.json stale AC-2.3 summary** — moved to
  `superseded_round3_join_summary`; `summary` carries 4992/4992.
- Mark **task11 (AC-6)**: corroborated + per-leg matrix complete; only review sign-off on the two
  fp8/bf16 per-leg blockers remains (they are genuinely blocked — config.py:110/170 + raw-dot-only
  kernel — not deferred).
- Plan Evolution Round-6 row added.

### Justification:
Every Round-5-review item now has a concrete generated artifact with code citations, validated and
fail-closed. The two remaining numeric legs are blocked by a real, source-verified two-level lock
(config validation + a raw-dot-only kernel); isolating them under cosine would require a new
production-path cosine kernel, which is a selection-path fix forbidden by the Ultimate Goal's "no fix"
constraint — so they are documented per-leg blockers awaiting sign-off, not silent deferrals.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
f1b90c797 [loop11b] R1: AC-8 close-out — results.md + queue.md regenerated to the R1 publishable state
44310f230 [loop11b] R1: complete evidence package — DSA server_info + crash-probe txt + crash-log hashes
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-5-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-5-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-4-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-4-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-3-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-3-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-6-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
