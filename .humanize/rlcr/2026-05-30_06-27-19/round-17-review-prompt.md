# Code Review - Round 17

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-17-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 17 Summary — AC-5 decode-throughput remediation (DS selection over-scan fix)

## Mainline objective (round contract)
Codex R16-review Required-Plan steps 2-3: AC-5 strict remediation as a **decode-throughput-first**
problem — profile the DS conc-16 decode hot path at the lifted DS int8 / mem-0.7 / radix-on point,
then make the smallest decode-path code change to move conc-16 per-req TPS toward ≥ 30, preserving
the ABI lock (`indices.shape[-1] == dsa_index_topk == 2048`). AC-10 and more AC-7/AC-8 evidence were
out of scope.

## What landed (commit `ece26eb52`)
1. **Profiled + localized the bottleneck (Codex step 3).** A CLOSED-batch pure-decode measurement (N
   parallel `/generate`, `ignore_eos`, no new arrivals → clean decode batch, `#queue-req:0`) gave
   **17.4 TPS/req at batch 16** (step 57.6 ms) — ≈ the AC-5 cold-flood 17.6, so conc-16 is **genuinely
   decode-bound, NOT a WARMUP=0 artifact**. A selection-width microbench showed the graph-safe DS
   selection scores + top-k over `max_seq_len = req_to_token.shape[1] = context_len = 163840` every
   layer (×61) every step — a **~35× over-scan** for a ~4096-token request (~32 ms of the 57.6 ms step
   is selection; ~23.5 ms is pure over-scan; the score kernel did the per-head loads/dots for the whole
   context and only masked the result).
2. **The fix — a numerically-identical, CUDA-graph-safe score-kernel early-exit.** `_logical_score_kernel`
   now skips token-blocks entirely past each request's `seq_len` (store -inf + return before the per-head
   loop). **No flag** (bit-identical output), **ABI lock untouched**, **AC-8 context preserved** (each
   program still scans its own seq, no context cap). Verified: selection **identical at width 4608 vs
   163840** (layers 0/7/30/60); selection @163840 **32.08 → 12.50 ms/step**; **281 DS unit tests pass**.
3. **End-to-end re-measure (patched, same operating point).** Closed-batch pure decode:
   conc-1 39.7→40.9, conc-8 24.6→**32.6 (now ≥30)**, conc-16 **17.4→27.1 TPS/req (+56%)**; step
   57.6→36.9 ms (−20.7 ms == the profiled over-scan savings). Coherence smoke unchanged.

## Result
The AC-5 decode bottleneck is **localized and the dominant component fixed**: conc-16 pure-decode
**17.4 → 27.1 TPS/req (+56%)**, conc-8 now passes ≥ 30 — from a bit-identical, graph-safe, no-flag,
AC-8-preserving kernel change. conc-16 strict (≥30) is **not yet reached** (step 36.9 vs the 33.3 ms
target, ~3.6 ms over): the residual is the first `torch.topk` over-scan (runs over the full captured
163840-wide score row). Shrinking it is capture-width-bound (needs a seq-aware blocked/partial top-k or
bucketed width, **not** a context cap) — the next round's lever. **AC-5 strict SLO stays a live mainline
blocker (DEC-3).**

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — score-kernel early-exit
  (the only production code change; ~16 lines, numerically identical).
- `runs/20260530_dsv32_loop6/ac5_decode_profile/` (NEW): `ac5_decode_remediation.md` (the profile +
  fix + before/after), `closed_batch_decode.py` (pure-decode profiler), `ds_closed_batch_decode.txt` /
  `ds_closed_batch_decode_patched.txt` (before/after curves), `selection_width_microbench.py`+`.json`
  (over-scan attribution), `verify_early_exit.py` (bit-identical equivalence + timing),
  `get_server_info_ds{,_patched}.json` (operating-point sidecars), `closed_batch_b{1,8,16}.json`.
- `.humanize/bitlesson.md` — new lesson `BL-20260531-ds-selection-fullwidth-overscan`; goal-tracker
  (R17 Plan Evolution row + AC-5/task6 note); round-17 contract/summary (gitignored loop state).

## Validation
- `verify_early_exit.py`: selection bit-identical at width 4608 vs 163840 (layers 0/7/30/60); selection
  @163840 32.08 → 12.50 ms/step.
- `pytest test/.../test_double_sparsity_unit.py`: **281 passed**.
- Closed-batch end-to-end (patched, same sidecar: mem 0.7 / int8 / radix-on / max_total 396096):
  conc-16 27.1 TPS/req, conc-8 32.6, conc-1 40.9; coherence "The capital of France is" → " Paris. The
  capital of the United States" (no degeneration). `git diff --check` clean; commit `ece26eb52` pushed
  to `jimmy`. GPUs freed at round end (all 8 at 0 MiB, no live `launch_server`).

## Remaining Items
- **Open mainline blocker (AC-5 strict):** conc-16 per-req TPS 27.1 < 30 — residual ~3.6 ms `torch.topk`
  over-scan (capture-width-bound). Next round: a seq-aware blocked/partial top-k (the
  `DSGraphState.scratch_partial_*` buffers exist) or bucketed selection width — without a context cap;
  then conc-32/64 TTFT tuning (the over-scan fix should help proportionally more there); then the full
  AC-5 client re-run (NUM_PROMPTS=320, conc 16/32/64) with exact arrays + a fail-closed verifier.
- **Gated AC-10** — only after AC-5 strict is verified. **Cross-node wrapper smoke** — future-gated.
  **DSA-default conc-64 TPS ~29.4** — queued pre-existing limit. No ABI-lock / FlashMLA-assert changes;
  DS-fair AC-12 gate unchanged.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260531-ds-selection-fullwidth-overscan
Notes: New lesson — the DS graph-safe decode selection over-scanned the full KV-index buffer width
(`ds_graph_state.max_seq_len = req_to_token.shape[1] = context_len = 163840`) every layer every step,
scoring the entire context for a ~4096-token request (~32 of the 57.6 ms decode step). Captures the
profile-first method (CLOSED-batch pure decode to isolate decode from prefill-interleave + a
scan-width microbench to attribute), the numerically-identical CUDA-graph-safe early-exit fix (skip
token-blocks past seq_len; no flag, no context cap, ABI lock intact), and the residual topk over-scan
caveat (capture-width-bound). Applied existing lessons: BL-20260530-cold-flood-not-steady-state-slo
(the closed-batch rejected the cold-flood hypothesis), BL-20260530-admission-restore-tps-tradeoff
(per-req TPS = 1/decode_step_time), BL-20260527-torch-topk-aliasing-corrupts-input (kept the topk
out=/scratch contract intact), BL-20260530-durable-tracked-acceptance-evidence (tracked profiler +
microbench + bit-identical verifier), BL-20260530-remote-server-launch (background boot + ps-grep +
pkill||true; no foreground sleep).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-16-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-16-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-15-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-15-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-14-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-14-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-17-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
