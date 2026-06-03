# Code Review - Round 18

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-18-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 18 Summary — Loop 7

## Mainline objective (round-18-contract.md)
**Complete AC-4: the graph-captured TP=8 lifted-width selector-equality artifact
(task16's last item) + a fully-consistent production-ready `m9` disposition
(task17), re-reviewed.**

## Outcome: ACHIEVED — AC-4 MET (production-ready); task16 + task17 closed.

## Work Completed
### task16 — graph-captured TP=8 lifted-width selector equality (the R17-review gap)
I first attempted the literal ask: an 8-rank NCCL + raw `torch.cuda.graph` capture of
`retrieve_topk_graph_safe`. **It deadlocked** (540 s timeout, SIGTERM + orphan cleanup)
— capturing an NCCL collective in a naive per-rank `torch.cuda.graph` needs the
production `cuda_graph_runner`'s coordination (shared graph pool + comm registration),
which a standalone unit harness cannot provide. So I proved the property by **composed
evidence**:
- **(a)** single-rank `retrieve_topk_graph_safe` at **4096/8192** captured in a real
  `torch.cuda.CUDAGraph`: **zero-alloc replay** + **bit-identical to the eager logical
  reference** (`TestLiftedWidthSelectionGraphCaptured`).
- **(b)** the eager 8-rank all-reduce equality at 4096/8192 (`TestTP8LiftedWidthDeterminism`)
  — the SUM all-reduce is rank-symmetric + deterministic.
- **(c)** the **live R17 TP=8 server** ran the selection under production CUDA graph and
  served correct **95%** recall (divergent ranks → corrupt all-reduced selection →
  degenerate output, which did not occur).

### task17 — `m9` full production-ready consistency (the R17-review gap)
Rewrote every contradictory section: validator no longer requires `--disable-cuda-graph`,
launcher no longer forces eager, `dequantize_k_cache_paged_out` + fixed-shape scratch is
the production decode path, graph-mode 4K NIAH 95% is the binding recall; removed every
"deferred / eager-required" claim; cleaned the stale "eager-only" comments in
`serve_double_sparsity.sh`, `selection_kernel.py`, and `dsa_backend.py`. **Re-reviewed via
`/humanize:ask-codex` (twice)**: R18 returned **"No runtime/design gap found blocking
AC-4"**, the **`(a)+(b)+(c)` composed evidence is an acceptable production-readiness close**
(a raw per-rank NCCL `torch.cuda.graph` harness is NOT required), and the speculative guard
is sound. Integrated its wording fixes (the two bullets that overclaimed "8-rank NCCL graph
capture" → the exact `(a)+(b)+(c)`).

### Bundled (R17-review queued hazard, now resolved)
A fail-closed validator guard rejecting `enable_lifted_budget_decode` + `--speculative-algorithm`
(the lifted CUDA-graph scratch is sized by `max_bs`, but speculative target-verify expands
the decode rows) + `test_validator_lifted_rejects_speculative`.

## Files Changed
- `test_ds_scorer_tp_determinism.py` (single-rank lifted-width selection graph-capture),
  `validator.py` (lifted+speculative guard), `test_scorer_variants.py` (guard test),
  `m9_tier2a_disposition.md` (consistency rewrite + re-review + wording fixes),
  `serve_double_sparsity.sh` / `selection_kernel.py` / `dsa_backend.py` (stale-comment cleanup).
- Commit `f9f6ec056` (local — loop hook keeps commits local until completion).

## Validation
- `TestLiftedWidthSelectionGraphCaptured` (4096/8192) + `TestLiftedBudgetABI` (13, incl.
  the speculative guard) pass; the focused lifted/TP subset → **60 passed**.
- Full DS unit suite → **350 passed + 9 subtests**, no regression. Default-off byte-identical.

## AC status after R18
- **AC-4 → MET (production-ready)**; **task16 + task17 closed**. With AC-1/3/5 (prior),
  **5/6 ACs MET**.
- AC-2 PARTIAL (task20), AC-6 PARTIAL (task19).

## Remaining Items (active mainline)
- **task19 (AC-6, next mainline)** — consolidated perf guardrails at conc-1/16 (TTFT,
  decode TPS/req, GPU mem, graph-replay, admission) + Tier-1 non-regression + the
  DS-vs-DSA recall/perf report.
- **task20 (AC-2)** — final strategic-gate supersession decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-nccl-collective-graph-capture-needs-runtime-not-unit-test
- Notes: a standalone N-rank NCCL-collective-under-raw-`torch.cuda.graph` unit test
  DEADLOCKS — NCCL collective graph capture needs the production `cuda_graph_runner`'s
  coordination (graph pool + comm registration), not a naive per-rank capture. Prove the
  property by composed evidence instead: (a) single-rank capture (`process_group=None`),
  (b) eager N-rank cross-rank equality (gloo), (c) a live server boot WITH CUDA graph + TP=N
  serving correct output (the real collective-under-production-capture). Hit the hard way
  (540 s timeout) and confirmed by the ask-codex re-review.

## Goal Tracker
Updated directly (Plan Version 25): R18 row; task16 + task17 → Completed and Verified;
**AC-4 MET (production-ready)**; the lifted+speculative queued issue → RESOLVED; Active =
task19, task20. No Goal Tracker Update Request needed.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
9ca1f5133 [Sparsity] Loop-7 recall R&D: plan + QA (active plan gitignored)
9914a3004 [Sparsity] Loop-7 M0: selection-recall oracle diagnostic math
8074cb1cf [Sparsity] Loop-7 M0: oracle sink + AC-1.1 force + flag-gated hook
c6ffcdea6 [Sparsity] Loop-7 M0: DS served-recall baseline at mem 0.7 (N=20)
78f6b5d17 [Sparsity] Loop-7 M0: oracle budget-vs-scorer evidence (A-vs-B decider)
a1e2c72dc [Sparsity] Loop-7 M0: A-vs-B decision (Codex-adjudicated)
599d7cc99 [Sparsity] Loop-7 M1: flag-gated cosine scorer (Tier-2.B candidate)
e2674f4f4 [Sparsity] Loop-7 M1: cosine scorer MEASURED — 16K recall 5%->40%
c5a829def [Sparsity] Loop-7: oracle trial-file read fresh; gitignore transient artifacts
273622705 [Sparsity] Loop-7 R1: length-conditional hybrid scorer (best of both regimes)
72c704edf [Sparsity] Loop-7 R2: scorer variants correct + production-safe
fc8871372 [Sparsity] Loop-7 R3: fix anchor over-budget + TP=8 logical-path matrix
bf2ce9b2b [Sparsity] Loop-7 R4: oracle fail-closed + config-borne + 64K binding re-run
9f76ad659 [Sparsity] Loop-7 R5: binding DS-vs-DSA same-node served-recall matrix (AC-2)
cb02b6673 [Sparsity] Loop-7 R6: port Tier-2.B scorer to the graph-safe Triton path (AC-3)
9a37590ec [Sparsity] Loop-7 R7: binding AC-3 non-regression matrix (graph-mode N=50 + MMLU)
f05cb730e [Sparsity] Loop-7 R8: close AC-1 (oracle-off zero-hot-path + stride reference)
e7cf1f146 [Sparsity] Loop-7 R9: port anchor-budget variant to the graph-safe path (AC-3)
c41e5193a [Sparsity] Loop-7 R10: lifted-budget ABI + design record (AC-4 task13)
a62ce91de [Sparsity] Loop-7 R11: fail-closed lifted-budget decode opt-in at the validator
d187f59f4 [Sparsity] Loop-7 R12: lifted-budget decode index core + flash_mla_sparse_fwd kernel proof
2ba4dafc1 [Sparsity] Loop-7 R13: wire the served eager lifted-budget decode branch + enable the seam
0ad20774a [Sparsity] Loop-7 R14: binding served 4K recall recovery for the lifted-budget path
b70f48d36 [Sparsity] Loop-7 R15: Tier-2.A landing disposition (deferred-with-evidence) — closes AC-4
714cf62b2 [Sparsity] Loop-7 R16: graph-safe lifted-budget decode primitives + zero-alloc replay proof
6453562e9 [Sparsity] Loop-7 R17: wire graph-safe lifted decode into production CUDA-graph + relax validator
41e0af078 [Sparsity] Loop-7 R17: production-ready Tier-2.A disposition + graph-mode recall evidence (AC-4 close)
f9f6ec056 [Sparsity] Loop-7 R18: close AC-4 — graph-captured lifted-width selector proof + consistent production-ready disposition + lifted+spec guard
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-17-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-17-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-16-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-16-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-15-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-15-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-18-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
