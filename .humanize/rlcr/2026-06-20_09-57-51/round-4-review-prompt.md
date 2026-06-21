# FULL GOAL ALIGNMENT CHECK - Round 4

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 4 Summary

## Work Completed
- **AC-2.3 RESOLVED** (the stalled Round-3 mainline), via a **cleaner direct method** instead of
  the shared-decode-step-id join Codex prescribed. The Round-3 stall was a score-vs-selection
  *step-misalignment* artifact (81/546), not a radix discrepancy. The direct proof sidesteps the
  pairing entirely: take each captured **post-reduce score row** (the authoritative top-k input the
  production radix consumed) and run BOTH top-k methods on the SAME row —
  - `select_topk_sequence_order` (exact torch reference == `torch.topk` semantics)
  - `blocked_topk_sequence_order` (the deterministic blocked/radix ALGORITHM the production Triton
    kernel `select_topk_sequence_order_triton` implements)
  — then compare selected-index sets. **Result: 624/624 identical**, and selector-width `[5120]`-vs-full
  **624/624 identical** on the same rows. The radix and selector-width AC-2.3 suspects are **retired on
  real GLM-5.1 score distributions** (not just on `topk_kernel.py` documentation). `verify_ac2_3.py` is
  fail-closed (nonzero exit on zero rows or any real mismatch).
- **Both real Round-3 review bugs fixed:**
  - `analyze_captures.py` is now **fail-CLOSED**: exits nonzero (rc=2) on zero score-capture groups,
    zero equivalence rows, or any unmatched join row. Verified it exits 2 on an empty capture dir.
  - `build_ledger.py` records **unambiguous generator-source provenance**: the generator file's
    git **blob hash** (commit-independent) + head-at-generation SHA + worktree dirty/clean marker, so
    the ledger source is pinned despite build_ledger emitting evidence one commit before its own commit
    exists. `run_meta.json` `git_sha_current` synced to HEAD; full SHAs.

## Files Changed
- `development/loop13/verify_ac2_3.py` (NEW) — the direct AC-2.3 proof; writes
  `evidence/ac2_3_radix_width_equivalence.json`.
- `development/loop13/evidence/ac2_3_radix_width_equivalence.json` (NEW) — 624/624 radix==torch.topk +
  624/624 width==full, on real captured rows.
- `development/loop13/analyze_captures.py` — fail-closed (nonzero exit on zero/unmatched rows).
- `development/loop13/build_ledger.py` — generator blob-hash + worktree provenance; regenerated
  `evidence/meta/arms/*.json`, `evidence/evidence_table.md`, `evidence/meta/run_meta.json`.
- `development/loop13/evidence/cheap_controls.json`, `evidence/findings.md` — AC-2.3 marked RESOLVED.
- `python/.../double_sparsity/selection_capture.py` — `req_pool_indices` retained (row identity).

## Validation
- `python3 development/loop13/test_reference_selectors.py` → **ALL 5 reference-selector tests pass**.
- `python3 development/loop13/verify_ac2_3.py` → **624/624** radix==torch.topk AND **624/624** width==full;
  wrote `evidence/ac2_3_radix_width_equivalence.json`; exit 0 (would exit 2 on any mismatch).
- `analyze_captures.py` on an empty dir → **exits 2** (fail-closed verified).
- Committed as `393966c02`; tree clean. **No server launched this round** (CPU-only analysis on already-captured
  artifacts); GPUs idle, one-TP=8-server-at-a-time invariant not exercised, not violated.

## Remaining Items (next mainline)
- **AC-6 production-path one-variable bisection** — the largest substantive gap. Reference-ceiling cliff
  (faithful raw-dot 0.013 vs faithful cosine 0.940 ≈ DSA 0.973) already names the candidate; the production
  path still needs a guarded diagnostic production-style cosine arm + per-variable arms (head_agg,
  fp8-vs-fp32 reduce, reduce dtype, radix, width) to attribute the 0.000 on the *served* path, not only the
  reference ceiling.
- **AC-2.1** forced-all physical-slot assertions; **AC-4** per-step length-cap garbage counters + per-example
  sample IDs/order (adapter/harness instrumentation, listed `fields_not_instrumented`, not faked).
- **AC-3.1** captured-row materialized-K proof; **AC-2.2** head-agg `pre_reduce` semantics confirmation.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-ds-capture-step-alignment
- Notes: The lesson originally flagged that score_capture and selection_capture use independent step
  counters so rows can't be paired at a decode step. Round 4 found the **cleaner resolution**: when the
  question is "does the production radix top-k match exact top-k," you don't need to pair captured selection
  rows to score rows at all — run BOTH the radix algorithm and exact `torch.topk` on the SAME captured
  *score* row and compare selected sets. The score row IS the authoritative top-k input, so this is
  conclusive and alignment-free. Added the corollary: prefer re-running both algorithms on one captured input
  over cross-instrument row joins when the captured input is the algorithm's direct argument.

## Goal Tracker Update Request
- Mark **AC-2.3 RESOLVED** (radix==torch.topk + width==full, 624/624 on real captured rows;
  `evidence/ac2_3_radix_width_equivalence.json`). The radix and selector-width suspects are retired.
- Promote **AC-6 production-path one-variable bisection** to the **next round's mainline** (largest remaining
  substantive gap; reference ceiling already names the cosine candidate).
- Keep **AC-2.2** (head-agg pre_reduce semantics), **AC-2.1/AC-4** (forced-all physical-slot assertions,
  garbage counters, sample IDs/order), and **AC-3.1** (captured-row materialized-K proof) active as
  instrumentation follow-ups.
- Note the two Round-3 review blockers (fail-open analyzer; stale ledger generated-SHA) are **CLOSED**.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
811c40420 [loop11b] R1: AC-5 no-op proof (dense_fallback=0 + structural sparsity) + GLM meta_info gap doc
9d2c4253d [loop11b] R1: headline M-B verdict + AC-4 dedicated per-step tax (both PASS)
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-3-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-3-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-2-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-2-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-1-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-1-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop13/plan.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

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

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop13/plan.md, we have completed **5 iterations** (Round 0 to Round 4).

The project's `.humanize/rlcr/2026-06-20_09-57-51/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-06-20_09-57-51/round-3-review-result.md` (previous round)
- `@.humanize/rlcr/2026-06-20_09-57-51/round-2-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-06-20_09-57-51/round-3-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-4-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-4-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
