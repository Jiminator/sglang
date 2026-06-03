# Code Review - Round 23

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-23-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 23 Summary — deterministic top-k tie-break + AC-5 verifier workload-volume

## Mainline objective (round contract)
Fix the two Codex R22 prerequisites for the owner-chosen graph-safe blocked-topk kernel: (1) the blocked
top-k "exact oracle" was NOT identical to the monolithic selection on finite ties; (2) the AC-5 verifier was
fail-open on workload VOLUME (a 1-request artifact passed). Codex requires the tie contract fixed FIRST so
the kernel is written against a correct oracle.

## What landed (commit `2715b7382`)
1. **Deterministic top-k tie-break (shared contract).** Codex counterexample: all-ones scores, K=3,
   block_width=4 → monolithic `[4,5,6]` vs blocked `[4,6,7]`. Added a shared `_topk_by_score_then_pos(vals,
   pos, k)` helper that selects the top-K by **(score DESCENDING, then logical position ASCENDING)** — a
   stable position-ascending sort then a stable score-descending argsort, so equal scores resolve toward the
   lower position. Both `select_topk_sequence_order` (monolithic) and `blocked_topk_sequence_order` (the
   oracle/eager fallback) now use it → **bit-identical, including on ties** (the all-ones case is `[0,1,2]`
   in both). 4 finite-tie regressions added (all-equal, ties crossing block boundaries, ties at the K
   boundary, ties mixed with `-inf`). **289 DS unit tests pass.** Moved the test-file `__main__` guard to the
   end (Codex queued #3 — direct class invocation now works).
2. **AC-5 verifier workload-VOLUME hardened.** Codex showed a 1-request artifact with self-consistent
   headlines passed. Now a **code-owned `EXPECTED_WORKLOAD`** constant is the authority (the JSON copy is
   documentation, asserted == code); `--verify` asserts per conc `completed == 192`, `duration_s >= window
   (300 s)`, and sidecar `trial_id` present, on top of the workload-identity + recompute-from-raw checks.
   **5 volume tamper tests each exit 1** (reduced completed=1 with consistent headlines, short duration,
   coordinated expected_workload+arrays mutation, JSON-doc tamper, trial_id removed); clean PASS.

## Result
The top-k exactness contract is now correct and shared — the graph-safe Triton blocked top-k (next round)
can be written against a correct oracle (bit-identical to the monolithic path incl. ties). The AC-5
full-context verifier is now fail-closed on metrics (recompute-from-raw + means), workload identity, AND
volume/duration/trial — Codex's R20/R21/R22 verifier gaps are all closed.

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `_topk_by_score_then_pos`
  helper; `select_topk_sequence_order` + `blocked_topk_sequence_order` use the shared deterministic tie-break;
  docstring updated.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — 4 finite-tie regressions; `__main__`
  guard moved to the end.
- `runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py` + `ac5_fullctx_arrays.json` — code-owned
  `EXPECTED_WORKLOAD` + completed/duration/trial assertions.
- `.humanize/bitlesson.md` — new lesson `BL-20260531-topk-deterministic-tiebreak`; goal-tracker (R23 row);
  round-23 contract/summary (gitignored loop state).

## Validation
- `pytest test_double_sparsity_unit.py` → **289 passed** (9 subtests); direct `TestBlockedTopKExactness`
  invocation works. The all-ones K=3/bw=4 counterexample now matches in both selectors; ties-crossing-blocks
  and ties+(-inf) match.
- `ac5_fullctx_metrics_tool.py --verify` → PASS; 5 workload-volume tamper tests each exit 1 (incl. the exact
  Codex R22 reduced-completed gap). `git diff --check` clean; commit `2715b7382` pushed to `jimmy`. GPUs free
  (CPU/data round; no server booted).

## Remaining Items (the owner-chosen path)
- **Graph-safe Triton blocked top-k** in `retrieve_topk_graph_safe` (DSGraphState partial-score/partial-index
  scratch; per-block top-K; SKIP blocks entirely past each request's `seq_len`; merge under the now-correct
  deterministic tie-break; zero-alloc under CUDA-graph; ABI lock intact) + CUDA-graph replay/zero-alloc tests.
- **Full-context closed-batch conc-16 ≥30 TPS** proof, then the **full AC-5 client rerun** (np64-approved)
  with the hardened verifier.
- **Gated AC-10** — after AC-5 verified. Cross-node smoke (future-gated), DSA conc-64 TPS ~29.4 (queued).

## Goal Tracker Update Request
### Requested Changes:
- Mark Codex's R22 prerequisite blockers RESOLVED: the blocked-topk finite-tie hole (shared deterministic
  tie-break, bit-identical incl. ties, 4 regressions) and the verifier workload-volume fail-open (code-owned
  EXPECTED_WORKLOAD + completed/duration/trial asserts, 5 tamper tests). The AC-5 verifier is now fail-closed
  on metrics + identity + volume.
- AC-5/task6 stays Active for the owner-chosen graph-safe Triton kernel + the post-kernel rerun.
### Justification:
Codex required the tie contract fixed FIRST (before the kernel) and the verifier volume-anchored; both are
done and tamper-demonstrated. The remaining AC-5 work is the owner-chosen graph-safe kernel (writable now
against a correct oracle) + the np64 rerun.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260531-topk-deterministic-tiebreak
Notes: New lesson — when ≥2 implementations of a top-k/selection must agree (monolithic vs blocked vs a future
Triton kernel; an "exact oracle"), they DIVERGE on finite score ties unless they share an explicit
deterministic tie-break, because `torch.topk`/argsort break ties arbitrarily and differently across a
whole-array topk vs a per-block-then-merge topk (Codex's all-ones K=3/bw=4 counterexample). Fix: one shared
ordering (score DESC, then position ASC) via a stable pos-asc sort then a stable score-desc argsort; test ties
explicitly (distinct-score tests miss the divergence). Applied: BL-20260527-torch-topk-aliasing-corrupts-input
(fresh argsort outputs), BL-20260530-durable-tracked-acceptance-evidence (the verifier-volume hardening
extends "prove the artifact IS the claimed run" to completed/duration/trial with code-owned expected constants).
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
1aa24cfc1 [Sparsity] Loop-6: refined plan v1 + QA ledger + DEC-5 roadmap deferral
88c6498e5 [Sparsity] Loop-6 R0: strategic recall-R&D gate + footprint feasibility budget
84d3410b9 [Sparsity] Loop-6 R1: int8-symmetric compact TokenLabelTable (flag-gated, fp16 default, CUDA-graph-safe)
e85cd2564 [Sparsity] Loop-6 R2: scale-aware proof/sanity consumers + AC-3.1/AC-6 evidence
5d8e47fb3 [Sparsity] Loop-6 R3: serve_double_sparsity.sh exposes SIGNATURE_DTYPE (compact-table selection)
8a05b1688 [Sparsity] Loop-6 R3: real-mask NIAH non-regression PASS (int8 DS vs fp16 Loop-5 baseline, TP=8)
75e68053f [Sparsity] Loop-6 R4: AC-4 mem-fraction sweep PASS (int8 lifts no-OOM ceiling 0.6->0.7, TP=8)
91e9c20a3 [Sparsity] Loop-6 R5: AC-4 evidence addendum (full HBM budget + durable no-OOM proof)
8883848e9 [Sparsity] Loop-6 R6: AC-5 client-SLO directional result (int8 @ 0.7 radix-on, TP=8) + attribution
51dd009b8 [Sparsity] Loop-6 R7: durable AC-5 evidence + corrected per-conc attribution
bd09d1ca7 [Sparsity] Loop-6 R8: exact-recomputable AC-5 evidence + reconciled attribution
57f86b66f [Sparsity] Loop-6 R9: exact ITL source + fail-closed AC-5 verifier
d6e884aa9 [Sparsity] Loop-6 R10: AC-9 within_budget from real usage.prompt_tokens
daad92923 [Sparsity] Loop-6 R10: AC-9 within-budget gate re-run on hardware (real tokens)
2fd2c6937 [Sparsity] Loop-6 R10: AC-6 opt-in / DSA-default product proof on hardware
0e1ce974d [Sparsity] Loop-6 R11: AC-6 redo — proper-methodology DSA SLO + radix-on toggle
d0cc9fdc9 [Sparsity] Loop-6 R12: benchmark scripts pass --host to bench_serving
f9bc51b13 [Sparsity] Loop-6 R12: recomputable DSA SLO evidence + honest AC-6 verdict
5e6d3afb5 [Sparsity] Loop-6 R13: AC-7 — 3-trial DS+DSA re-sweep at the lifted point (characterized)
147b6d05f [Sparsity] Loop-6 R14: AC-7 exact-recomputable metrics + fail-closed verifier
99e51ad00 [Sparsity] Loop-6 R14: AC-7 profiling discharged at AC-7 methodology
40ccc4b63 [Sparsity] Loop-6 R15: AC-7 evidence review-clean (verifier precision + provenance + reconciliation)
9915630ca [Sparsity] Loop-6 R16: AC-8 64K servability PASS at lifted DS int8/mem-0.7
ece26eb52 [Sparsity] Loop-6 R17: AC-5 decode-throughput remediation -- DS selection over-scan fix
fcc2d1cdb [Sparsity] Loop-6 R18: AC-5 conc-16 strict-decode PASS via bounded-context op-point
7f896b454 [Sparsity] Loop-6 R19: bench_serving fail-closed on empty-latency streaming runs
96bc789cc [Sparsity] Loop-6 R20: AC-5 full-context measured client workload + attribution
991666b58 [Sparsity] Loop-6 R21: rebuild AC-5 full-context evidence to R9 fail-closed standard
704be382f [Sparsity] Loop-6 R22: harden AC-5 verifier to fail closed on workload identity
8ab6c7db0 [Sparsity] Loop-6 R22: exact blocked top-k algorithm + adversarial regression suite
2715b7382 [Sparsity] Loop-6 R23: deterministic top-k tie-break + AC-5 verifier workload-volume
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-22-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-22-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-21-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-21-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-20-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-20-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-23-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
