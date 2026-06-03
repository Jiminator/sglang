# Code Review - Round 12

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-12-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 12 Summary — Loop 7

## Mainline objective (round-12-contract.md)
**task14 (foundation) — implement & validate the lifted-budget decode *index core*:**
the request-local physical→compact remap (padding-safety + within-row dedup +
prefix-sharing isolation), plus the direct `flash_mla_sparse_fwd` wider-than-2048
kernel proof.

## Outcome: ACHIEVED — index core landed + CPU-tested; kernel half proven on GPU.

## Context (why this is the right first chunk of task14)
The fp8 DS decode uses `flashmla_kv` (asserts `indices.shape[-1] == dsa_index_topk
== 2048`). The lifted path must instead use `flash_mla_sparse_fwd` (no cap) — which
is the **bf16** backend, so the fp8 KV is dequantized via `dequantize_k_cache_paged`
into a **compact** `[total_valid,1,576]` buffer, and the kernel attends it by
**request-local compact ordinals**, not physical slots. That compact remap is the
trap-laden correctness core (request-local spans, prefix sharing, `-1` masking,
within-row dedup) the m7 design + every Codex review flag — and it is pure tensor
logic, fully CPU-testable now. This round lands it + proves the kernel half.

## Work Completed (`coding`, Claude)
1. **`double_sparsity/lifted_budget.py::build_compact_decode_index`** — pure-tensor,
   deterministic. Given per-request selected physical slots (selector order, fixed
   padded width) + `valid_lengths`, it emits:
   - `page_table_1_flattened` — valid physical slots only, batch-major/selection-rank
     order, **never `-1`** (it is the literal input `dequantize_k_cache_paged` loads);
   - `compact_indices` — request-local ordinals `request_base + rank` for valid lanes,
     `-1` for pad lanes (the kernel masks `<0`/`>=s_kv`);
   - within-row **dedup keeps the highest selection rank** (stable value-sort,
     first-of-run) and counts drops; **prefix-sharing isolated** to per-request spans;
     selector order preserved.
2. **CPU unit tests** (`test_lifted_budget_decode.py::TestCompactDecodeIndex`, 8):
   request-local mapping, prefix sharing (shared slot → distinct per-request spans),
   no `-1` in flattened table, within-row dedup keep-first (+ keep-highest-rank when
   the first repeats later), zero-valid-row base accounting, `valid_lengths` prefix,
   order preservation.
3. **GPU kernel smokes** (`TestLiftedBudgetKernelSmoke`, 2, H200/sm90):
   - `flash_mla_sparse_fwd` attends a request selecting **3000 > 2048** rows inside a
     4096-wide padded budget and matches a reference attention — **no-cap proof**, plus
     `-1` pad masking + request-local spans;
   - full **fp8 → `dequantize_k_cache_paged` → `flash_mla_sparse_fwd`** pipe with
     prefix-sharing matches a reference attending the dequantized selected slots, and
     the compact rows are **bit-identical** to the full-dequant gather.
4. **Discovered + recorded a kernel ABI constraint** (a `width=8` smoke hit
   `Assertion params.topk % (2*B_TOPK) == 0`): the padded index width
   (`lifted_budget_top_k`) must be a **multiple of 128**; the kernel masks indices
   `<0`/`>=s_kv` (so `-1` pad lanes suffice). Captured in `m7_lifted_budget_design.md`
   + a new BitLesson; the next-round wiring must enforce `lifted_budget_top_k % 128 == 0`.
5. `m7_lifted_budget_design.md` updated (landed core + kernel proof + the confirmed
   contract; resolved the "`flash_mla_sparse_fwd` accuracy >512 unproven" open risk).

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/lifted_budget.py` (new module).
- `test/registered/unit/layers/attention/test_lifted_budget_decode.py` (new: 8 CPU + 2 GPU).
- `development/loop7/m7_lifted_budget_design.md` (landed core + kernel contract + risks).
- Commit `d187f59f4` (local — loop hook keeps commits local until completion).

## Validation
- `TestCompactDecodeIndex` → **8 passed** (CPU).
- `TestLiftedBudgetKernelSmoke` → **2 passed** (GPU, H200/sm90).
- Full DS unit suite (`test_lifted_budget_decode`, `test_scorer_variants`,
  `test_double_sparsity_unit`, `test_ds_scorer_tp_determinism`) → **332 passed +
  9 subtests** (was 322+9; +8 CPU remap +2 GPU smoke), no regressions.
- No existing runtime path changed; `ds_lifted_budget_decode_available()` stays
  `False` (no half-wired path can boot); default DSA/DS-hybrid/oracle untouched.

## Remaining Items (active mainline, NOT queued-out)
- **task14 (wiring, next mainline)** — widen the selector budget
  `max_top_k`→`lifted_budget_top_k` for the opt-in eager path; route the opt-in
  decode to `flashmla_sparse` via `dequantize_k_cache_paged` feeding this round's
  remap; flip `ds_lifted_budget_decode_available()` to `True` **gated eager-only**
  (validator still requires `--disable-cuda-graph`); enforce `lifted_budget_top_k %
  128 == 0`; preserve the R23 tie-break.
- **task15 (remaining)** — served correctness + TP=8 selected-index equality at 4096/8192.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph), gated behind the recall win.
- **task17** — Tier-2.A landing disposition record.
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
**task14 advanced** (index core + kernel proof). **AC-4 NOT MET** — the decode-branch
wiring, served recall evidence, TP=8 equality, and task16/17 remain.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-flash-mla-sparse-fwd-compact-decode-contract
- Notes: records the fp8 lifted-budget decode contract — dequant→compact buffer +
  request-local ordinal indices (not physical), `-1`/`>=s_kv` masking, within-row
  dedup, prefix-sharing isolation, and the `lifted_budget_top_k % 128 == 0`
  (`topk % (2*B_TOPK)`) kernel-block requirement — so the next-round wiring and any
  future `flash_mla_sparse_fwd` consumer honor it. Reusable, non-obvious, directly
  load-bearing for the upcoming decode-branch wiring.

## Goal Tracker
Updated directly (Plan Version 15): R12 Plan Evolution row added; task14 → "in
progress (index core + kernel proof done R12)" with the remaining wiring listed;
task15 → "partial (kernel smokes done R12)". No Goal Tracker Update Request needed.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-11-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-11-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-10-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-10-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-9-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-9-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-12-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
