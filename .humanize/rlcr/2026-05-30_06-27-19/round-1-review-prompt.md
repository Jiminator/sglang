# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 1 Summary — AC-3 int8-symmetric compact TokenLabelTable

## Mainline objective (round contract)
Implement **AC-3** — the int8-symmetric compact `TokenLabelTable` path (the lever the feasibility budget selected), flag-gated with **fp16 default**, **CUDA-graph-safe**, threaded through every site that touches signatures. Plus the CPU-testable AC-3/AC-6 unit tests. Blocking prerequisite (Codex R0 review): revise the AC-2 artifact so int8 is the unambiguous binding path.

## Blocking fix landed first (Codex R0 review)
Revised `runs/20260530_dsv32_loop6/footprint_feasibility.md`: the Binding Lever Decision now reads unambiguously — **build int8 same-`label_dim` for the footprint reduction**, not conditional on any prior fp16 experiment. The fp16 lower-`mem_fraction` window is demoted to *optional comparison instrumentation* logged only during the AC-4 compact-table sweep; it must not gate/replace/precede the int8 build. Anchor B label corrected `≈0.77-0.8`→`≈0.70` (matches the recorded "0.7 OOMs during generation"). Tracker reconciled (blocking issue marked resolved; Plan Evolution updated).

## What was implemented (commit `84d3410b9`)
Design: store int8 signatures `[L,T,H,D]` + a static fp16 `scales [L,T,H]` (one symmetric scale per vector). Because `score = scale[t,h]·Σ_d(q_proj[h,d]·int8_sig[t,h,d])`, dequant is a single per-head multiply by `scale[t,h]` **after** the integer dot, **before** the cross-head max — so fp16 stays zero-overhead via a `HAS_SCALE` compile-time branch.

- **config.py** — explicit allowed field `signature_dtype` (`"fp16"` default | `"int8"`); unknown-field rejection preserved.
- **token_label_table.py** — compact mode allocates int8 signatures + static fp16 `scales`; `bytes_per_rank`/`estimate_hbm_bytes` count both (measured **0.5625×**); `is_compact`; fp16 path byte-identical (no scales).
- **token_label_write.py** — symmetric per-`(slot,head)` int8 quantize-on-write (`scale = max(|label|)/127`) + scale; fp16 path unchanged; zero-vector safe (no div-by-zero); no host sync.
- **selection_kernel.py** — dequant-at-scoring in the torch refs (`compute_token_scores`, `_compute_logical_token_scores`), **both** Triton kernels (`_compute_token_scores_kernel`, `_logical_score_kernel` — added `scale_ptr` + `HAS_SCALE: tl.constexpr`), and the allocation-free `retrieve_topk_graph_safe` scratch pipeline; `token_scales` threaded through.
- **selector.py / cuda_graph.py / deepseek_v2.py / dsa_backend.py** — pass `token_scales`/`scales`; bind maps `signature_dtype`→torch dtype; DSA-default (fp16) allocates no scales. The FlashMLA `indices.shape[-1]==dsa_index_topk` assert is untouched (AC-3.3 ABI lock).

## Files changed
8 production files (config, token_label_table, token_label_write, selection_kernel, selector, cuda_graph, dsa_backend, models/deepseek_v2), 1 test file (+230 lines), and the AC-2 artifact revision. Loop state (goal-tracker, round-1 contract/summary) is in `.humanize/rlcr/` (gitignored).

## Validation — 272 DS unit tests pass (260 existing + 12 new), GPU enabled, no regression
New `TestCompactInt8Signatures`:
- config: fp16 default, int8 opt-in, invalid dtype rejected, unknown-field still rejected.
- table: fp16 → no scales; int8 → static fp16 `[L,T,H]` scales; **byte ratio exactly 0.5625×**.
- quantize-on-write round-trip (dequant within one quant step) + zero-vector safety.
- **selection-equivalence: int8-vs-fp16 top-k overlap@2048 ≥ 0.99** (torch path — the binding gate).
- **GPU (H200, ran this round):** int8 Triton kernels match the torch reference (max score err **9.5e-7**; graph-safe logical overlap **1.0**); CUDA-graph **capture+replay** of the int8 path is **allocation-free** (`assert_no_alloc_in_region`) and **replay == eager**.

The RLCR dev box exposes 2 H200s, so the Triton-kernel int8 scale logic and CUDA-graph-safety — normally AC-4 concerns — were validated on real hardware this round, de-risking the serving round.

## Remaining items (gated on this round)
- **AC-3 hardware evidence (paired with the AC-4/AC-5 hardware round, needs a live served model):** real-mask NIAH non-regression on the Loop-5 mask vs the fp16 Loop-5 DS baseline, and the compact-vs-fp16 decode-scoring microbench against the 33.9→30 TPS/req margin. These AC-3.1 positive tests require serving, so per the plan ("a code-only round is acceptable if the next round validates on hardware") they belong to the next round — **not forgotten, explicitly deferred to AC-4** (tracked as `task4: partial`).
- **AC-4 (next mainline):** boot DS with `signature_dtype=int8` on TP=8, sweep `mem_fraction_static` 0.6→0.8 with full NVML/torch-residual HBM accounting + sustained long `/generate` no-OOM. Then AC-5 client-SLO benchmark with admission-vs-prefill attribution, AC-6 opt-in/DSA-default hardware proof, AC-7/AC-8/AC-9, then gated AC-10.

## Note for review
AC-3 code + unit/GPU evidence is complete this round; the two AC-3.1 **hardware** positive tests are intentionally paired with the AC-4 serving round. The optional `code-simplifier` agent is not installed; I did a manual simplification pass instead (consolidated three near-duplicate test alloc-helpers into one).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-3 was a planned feature implemented and validated in a single round (CPU torch-path tests + real-H200 Triton/CUDA-graph validation, 272 tests green) — not a defect solved across multiple rounds, so no new lesson is warranted. Applied existing lessons as design constraints: BL-20260527-torch-topk-aliasing-corrupts-input (left the graph-safe topk pipeline untouched; only inserted the scale multiply before topk), BL-20260528-ds-radix-capture-cuda-graph-safe / the CUDA-graph host-sync rule (scale is pre-allocated, static-shaped, device-only — capture/replay verified allocation-free), and BL-20260528-ds-private-server-args-attrs-crash-ipc (scales live inside the already `_`-prefixed TokenLabelTable, so the IPC filter still covers them).
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
1aa24cfc1 [Sparsity] Loop-6: refined plan v1 + QA ledger + DEC-5 roadmap deferral
88c6498e5 [Sparsity] Loop-6 R0: strategic recall-R&D gate + footprint feasibility budget
84d3410b9 [Sparsity] Loop-6 R1: int8-symmetric compact TokenLabelTable (flag-gated, fp16 default, CUDA-graph-safe)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-0-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-0-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
