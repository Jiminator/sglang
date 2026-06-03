# Code Review - Round 17

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-17-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 17 Summary — Loop 7

## Mainline objective (round-17-contract.md)
**task16 (part 2) — wire the R16 graph-safe primitives into production CUDA graph,
relax the validator, prove zero-alloc backend replay, and confirm a live CUDA-graph
boot.** (The R15 review STALLED the loop, overriding the deferred close and requiring
task16 to be implemented.)

## Outcome: ACHIEVED — task16 COMPLETE; AC-4 closes via the production-ready branch.

## Work Completed (`coding` + `analyze`, Claude + ask-codex)
1. **`DSGraphState` + `allocate_graph_state`** (`cuda_graph.py`): lifted scratch
   (`lifted_page_table`, `lifted_compact_indices`, `lifted_valid_counts`,
   `lifted_compact_kv`, `lifted_q_padded`), allocated **only when
   `enable_lifted_budget_decode`**; threaded from both metadata sites in `dsa_backend.py`.
2. **`_forward_lifted_budget`** (`dsa_backend.py`): graph path — slice the scratch to
   the captured bs/width, run `build_lifted_compact_kv_fixed` (fixed-shape builder +
   alloc-free `out=` dequant into scratch), attend via FlashMLA with a q-padding
   scratch. The eager `build_lifted_compact_kv` stays as the non-graph fallback
   (resolved via `getattr(self, "forward_metadata", None)` so partial test stubs work).
3. **`_forward_flashmla_sparse`**: optional q-padding scratch param (write real heads,
   the pad tail stays 0 from allocation, trimmed output); default callers byte-identical.
4. **Validator**: removed the lifted `--disable-cuda-graph` rejection (path is now
   graph-safe); the default `flashmla_kv` `dsa_index_topk` assert is untouched.
   `serve_double_sparsity.sh`: `LIFTED_BUDGET=1` no longer forces eager.
5. **task17 (production-ready disposition)**: rewrote `m9_tier2a_disposition.md` from
   deferred → **production-ready**; recorded `m10_lifted_graph_finding.md`. Re-reviewed
   via `/humanize:ask-codex` (**"No invalidating design gap found"**); integrated its 3
   points (reframed the graph-captured TP=8 item as integratively-evidenced; added the
   fp8-op-point scope caveat; cleaned the stale deferred prose).

## Validation
- **Offline (GPU)**: the wired backend `_forward_lifted_budget` replays **zero-alloc**
  under a real `torch.cuda.CUDAGraph` at **4096 and 8192**
  (`TestLiftedBudgetBackendGraphSafe`), matching the eager reference.
- **LIVE (8×H200)**: server booted **WITHOUT `--disable-cuda-graph`**; the full forward
  (incl. lifted decode) **captured** ("fired up"); decode batches log **`cuda graph:
  True`** (#token 4416); **graph-mode NIAH 4K N=20 = 95% (19/20)** — matches the eager
  95% and confirms the **+20pp recovery over DS-default-2048 (~75%) holds in production
  graph mode**; served 20/20, 0 admission failures; **3.4× faster** than eager (13.8s vs
  46.8s); ~14.5 tok/s; ~70 MB lifted scratch at `--cuda-graph-max-bs 8`.
  (`ds_meta=None` under graph is the **expected** eager-only-meta behavior, confirming
  the decode ran captured.) `m10_lifted_graph_finding.md`, `niah_ds_lifted4096_graph.json`.
- **Non-regression**: default-off path byte-identical; full DS unit suite → **347 passed
  + 9 subtests**.

## Files Changed
- `cuda_graph.py`, `dsa_backend.py`, `validator.py`, `serve_double_sparsity.sh`,
  `test_lifted_budget_decode.py`, `test_scorer_variants.py` — commit `6453562e9`.
- `m9_tier2a_disposition.md` (production-ready), `m10_lifted_graph_finding.md` (new),
  `niah_ds_lifted4096_graph.json` (new) — commit `41e0af078`. (Both local — loop hook.)

## AC status after R17
- **AC-4 → MET (production-ready)**; **task16 + task17 done**. With AC-1/3/5 (prior),
  **5/6 ACs MET**.
- **AC-2 PARTIAL** (task20 final decision record), **AC-6 PARTIAL** (task19 perf consolidation).

## Remaining Items (active mainline)
- **task19 (AC-6, next mainline)** — consolidated perf guardrails at conc-1/16 (TTFT,
  decode TPS/req, GPU mem, graph-replay, admission) + Tier-1 non-regression + the
  DS-vs-DSA recall/perf report (the lifted graph-mode 14.5 tok/s / mem note feeds it).
- **task20 (AC-2)** — final strategic-gate supersession decision record.
- Open follow-on (documented, non-blocking): a standalone graph-captured 8-rank TP=8
  selector-equality artifact (the live TP=8 graph serving + eager equality + graph-safe
  selection evidence it); the bf16-store lifted branch is out of the fp8 AC-4 scope.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260602-flash-mla-sparse-fwd-compact-decode-contract
- Notes: added the **production-wiring corollary** — allocate the fixed scratch once in
  the DS graph-state dataclass (only when opt-in), resolve it from the backend's own
  `self.forward_metadata` with a `getattr` eager fallback; q head-padding is alloc-free
  because the pad tail stays 0 from allocation and is never written (heads independent +
  output trimmed); bound the `[bs*width]` compact_kv footprint with `--cuda-graph-max-bs`;
  and VALIDATE with a wired-backend zero-alloc replay AND a live `cuda graph: True` boot
  (the host-syncing per-request meta is `None` under graph — its absence confirms capture).

## Goal Tracker
Updated directly (Plan Version 23): R17 row; task16 + task17 → Completed and Verified;
**AC-4 MET (production-ready)**; Active = task19, task20. No Goal Tracker Update Request needed.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-16-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-16-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-15-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-15-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-14-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-14-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-17-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
