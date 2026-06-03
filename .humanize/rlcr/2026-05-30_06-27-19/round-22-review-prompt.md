# Code Review - Round 22

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-22-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 22 Summary — AC-5 verifier hardened + owner decisions + blocked-topk foundation

## Mainline objective (round contract)
Harden the AC-5 full-context verifier to fail closed on the workload + operating-point IDENTITY (Codex R21
demonstrated a fail-open gap), and obtain the two owner decisions Codex says require explicit approval for
the loop to converge (np64-vs-np320 methodology; conc-16 full-context TPS kernel-vs-rescope).

## What landed
1. **Verifier hardened — fail-closed on workload identity** (commit `704be382f`). Codex R21 mutated the
   sidecar `mode`/`num_prompts`/`isl_total_tokens`/`osl_tokens` and `server_args.max_total_num_tokens` to
   garbage and `--verify` still passed. Now `ac5_fullctx_metrics_tool.py` embeds an `expected_workload` and
   asserts, on EVERY sidecar: `mode=double_sparsity`, sidecar `concurrency`==artifact key, ISL 4096 / OSL 512,
   `num_prompts`/warmup/window == the recorded np64-steady-state methodology, `chunked_prefill_size=8192`,
   `max_total_num_tokens=396096` — plus the existing flag invariants and the recompute-from-raw metric checks.
   **7 workload-metadata tamper tests each exit 1** (mode=baseline, num_prompts=320, isl=1, osl=1, max_total=1,
   conc-key mismatch, warmup=0); clean PASS.
2. **Owner decisions** (AskUserQuestion, R12/R18 precedent):
   - **(a) AC-5 methodology = np64 steady-state APPROVED.** The literal `NUM_PROMPTS=320` is rejected as
     cold-flood-misleading per the verified BitLesson (window cold-ramps; fixed-count floods the queue →
     P99 TTFT ≈ full 320-drain ~300s). The committed full-context AC-5 evidence's methodology is now
     **owner-approved** (recorded as plan evolution).
   - **(b) conc-16 full-context TPS path = implement the full-context blocked-topk kernel** (owner chose the
     exact research-grade kernel over the bounded-context rescope / directional-accept).
3. **Blocked-topk foundation** (commit `8ab6c7db0`): exact torch `blocked_topk_sequence_order` in
   `selection_kernel.py` returning the IDENTICAL ascending positions + valid_lengths as the monolithic
   `select_topk_sequence_order` (per-block top-min(K,bw) → merge → global top-K; exact because a global-top-K
   token has within-block rank ≤ its global rank ≤ K). This is the exactness oracle + eager fallback for the
   graph-safe Triton skip-kernel (whose value is skipping blocks entirely past each request's `seq_len`).
   **4 registered adversarial regressions (6 subtests)**: all-winners-in-one-block, masked/short sequences,
   block-boundary seq, padding (n not a multiple of bw), K≥block_width (single block), K>n. 285 DS unit
   tests pass. ABI lock untouched.

## Result
The AC-5 full-context evidence verifier is now fully fail-closed (metrics recompute-from-raw + workload
identity + operating point; 13 tamper tests across R21/R22 each exit 1). The AC-5 measurement methodology is
owner-approved (np64). The owner-chosen conc-16 TPS path (the full-context blocked-topk kernel) has its exact
algorithm + adversarial regression suite landed — the foundation the graph-safe Triton kernel must match.

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `blocked_topk_sequence_order`
  (exact, the oracle/eager-fallback).
- `runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py` + `ac5_fullctx_arrays.json` — verifier
  hardened (`expected_workload` + per-sidecar workload-identity assertions).
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — `TestBlockedTopKExactness` (4 tests).
- `.humanize/bitlesson.md` — `BL-20260530-durable-tracked-acceptance-evidence` extended (verifier must prove
  workload identity, not just a subset of flags); goal-tracker (R22 row + owner decisions); round-22
  contract/summary (gitignored loop state).

## Validation
- `ac5_fullctx_metrics_tool.py --verify` → PASS; 7 workload-metadata tamper tests each exit 1 (Codex's exact
  R21 gaps closed).
- `pytest test_double_sparsity_unit.py` → **285 passed** (281 + 4 new; 6 subtests). `git diff --check` clean;
  commits `704be382f` + `8ab6c7db0` pushed to `jimmy`. GPUs free (data/CPU round; no server booted).

## Remaining Items (the owner-chosen path)
- **Graph-safe Triton blocked top-k** in `retrieve_topk_graph_safe`: a zero-alloc kernel using DSGraphState
  partial-score/partial-index scratch that computes per-block top-K and SKIPS blocks entirely past each
  request's `seq_len` (sentinel-filled on device), then merges to the same result as the monolithic path
  (now oracle-tested). This is the actual perf win for full-context conc-16.
- **Full-context closed-batch conc-16 ≥30 TPS** re-measure after the kernel, then the **full AC-5 client
  workload rerun** (np64-approved) with the hardened verifier.
- **Gated AC-10** — after AC-5 verified. Cross-node smoke (future-gated), DSA conc-64 TPS ~29.4 (queued).

## Goal Tracker Update Request
### Requested Changes:
- Record the R22 **owner decisions** as accepted plan evolution: (a) AC-5 methodology = np64 steady-state
  (warmup120/window300) — supersedes the literal NUM_PROMPTS=320; (b) the conc-16 full-context TPS path =
  the exact full-context blocked-topk kernel (bounded-context rescope declined).
- Mark the **AC-5 full-context evidence verifier** as acceptance-grade/fail-closed — resolving Codex's R20/R21
  verifier blocking issues. AC-5 stays Active for the owner-chosen kernel + the post-kernel rerun.
### Justification:
The owner explicitly approved np64 and chose the kernel path, resolving the two decisions Codex flagged as
requiring owner approval. The verifier is now fail-closed on both metrics and workload identity (the specific
R21 gaps closed with tamper tests). The remaining AC-5 work is the owner-chosen graph-safe kernel + rerun,
whose exact algorithm + regression oracle landed this round.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended with the R22 workload-identity instance — a verifier that validates only a SUBSET of the
sidecar (selected server flags) is still fail-OPEN on workload identity (Codex mutated mode/num_prompts/ISL/
OSL/max_total_num_tokens and it passed); the verifier must prove the artifact IS the claimed AC run (workload
+ full operating point) via an `expected_workload` asserted on every sidecar, not just that the metric arrays
are self-consistent (7 workload tamper tests each exit 1). Applied existing lessons:
BL-20260527-torch-topk-aliasing-corrupts-input (the blocked top-k uses fresh topk outputs, no aliasing),
BL-20260530-cold-flood-not-steady-state-slo (the np64 methodology the owner approved). The blocked-topk
exactness property (global top-K ⊆ union of per-block top-K) is captured in the `blocked_topk_sequence_order`
docstring + its regression suite; a standalone lesson will be added when the graph-safe Triton kernel lands.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-21-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-21-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-20-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-20-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-19-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-19-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-22-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
