# Code Review - Round 10

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-10-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 10 Summary — AC-6 (opt-in / DSA-default product) + AC-9 (real-token within-budget) on hardware

## Mainline objective (round contract)
Per Codex's R9 directive (AC-5 evidence verified/resolved; move to hardware): land
the DS-opt-in / DSA-default product property (AC-6) and the real-token within-budget
gate (AC-9) on hardware, via one cross-node bring-up (DS int8 @ 0.7 on node 0,
DSA-default on node 1). The AC-5 directional verdict + open strict-SLO blocker stay
tracked but are not this round's objective.

## AC-9 — within-budget gate from real `usage.prompt_tokens` (commits d6e884aa9 code, daad92923 evidence)
- Harness edit: `_generate` returns `(text, prompt_tokens)` from `usage.prompt_tokens`
  (chat) / `meta_info.prompt_tokens` (generate); threaded through `_GenAttempt` →
  `_summarize_prompt_tokens` (max-over-served `input_tokens` + `usage_missing`
  fail-closed signal) → `_run_niah` → `_niah_record`. `within_budget` now computed
  from real `input_tokens` (→ fail-closed `None` if usage missing); records
  `input_tokens`, `dsa_input_tokens`, and the old `within_budget_wordcount_proxy`.
  `test_niah_within_budget` asserts the premise from real tokens. Renamed the
  misleading `length_tokens` → `length_words`. **DS-fair gate definition UNCHANGED**
  (INDEX_TOPK=2048, 5 pp tolerance, 1024/1536-word lengths). Dry-run verified the
  parsing + fail-closed logic before hardware.
- Live re-run (DS node0 + DSA node1) **PASSED** (1 passed, 2 subtests, 26.5 s):
  1024 words → `input_tokens=1128`; 1536 words → `1678`; both `within_budget=True`,
  `usage_missing=False`; DS recall 100% vs DSA 100% (Δ0.0 pp). The real-token
  `within_budget` **matches** the word-count proxy at both lengths ⇒ **the proxy was
  safe** (recorded per-length). Artifacts: `ac9_real_token_within_budget.md`,
  `ac9_within_budget/ac12_niah_{1024,1536}_*.json`.

## AC-6 — DS opt-in; DSA stays the default (DEC-2 "Both")
- **The opt-in flag toggles the compact DS path** (`ac6_product_proof/get_server_info_keys.json` + boot logs):
  - DS opt-in (node 0): `enable_double_sparsity=True`, `double_sparsity_config={…,"signature_dtype":"int8"}`,
    `token_label_table: 6.48 GB/rank … dtype=torch.int8 scales=float16` on all 8 TP ranks.
  - DSA-default (node 1): `enable_double_sparsity=False`, `double_sparsity_config=None`,
    **0** `token_label_table` lines, full **910784**-token KV pool — **no DS table allocated**.
  - Identical Option B operating point (fp8 KV, page 64, flashmla_kv prefill+decode, overlap/piecewise disabled).
- **DSA-default admits full nominal concurrency** and serves cleanly: achieved 16.00 / 32.00 / [64]
  at conc 16/32/64, completed 64/64 each, **errors 0** (`ac6_product_proof/dsa_default_slo.txt`).
- **"Meets the SLO unchanged":** the authoritative DSA steady-state SLO is the established
  Loop-5 baseline (P99 TTFT 0.73 / 1.37 / 2.04 s, ≥30 TPS) at this identical operating point;
  this fresh boot reproduces that operating point exactly. The fresh `WARMUP=0` confirmation run
  is **cold-ramp-dominated** (DSA P99 TTFT 22.5 s / TPS 16.9 at conc 16) — the **same flood
  artifact AC-5 documented for DS** (min TTFT 1.6 s; tight median≈p99); under identical
  `WARMUP=0` methodology DSA is **not** faster than DS (AC-5 conc-16 12.8 s), confirming the
  inflation is the cold ramp, not DS-specific. A clean all-trials steady-state DSA sweep is AC-7.

## Result
AC-9 met (code + live rerun, proxy shown safe). AC-6 product property met (opt-in toggle +
DSA no-table + full admission + clean serving). The **strict-SLO miss remains the open
mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc) — unchanged.

## Files Changed
- `test/manual/test_double_sparsity_v32.py` (AC-9 harness; commit d6e884aa9).
- `runs/20260530_dsv32_loop6/`: `ac9_real_token_within_budget.md`, `ac9_within_budget/` (daad92923);
  `ac6_optin_dsa_default_product.md`, `ac6_product_proof/` (get_server_info ×2 + keys, boot excerpts, dsa_default_slo.txt).
- `.humanize/bitlesson.md` (+1 lesson `cold-flood-not-steady-state-slo`), goal-tracker, round-10 contract/summary (gitignored loop state).

## Validation
- Cross-node bring-up: DS int8@0.7 node0 (`token_label_table 6.48 GB/rank int8`, no OOM) + DSA-default node1 (no table, pool 910784).
- AC-9 gate PASSED on hardware; `within_budget` from real tokens, proxy safe.
- AC-6 toggle/no-table proven from `/get_server_info` + boot logs; DSA full admission (achieved==nominal), errors 0.
- AC-9 code dry-run (mock responses) confirmed usage capture + fail-closed before hardware; `git diff --check` clean; commits pushed to `jimmy`.
- Servers killed and GPUs freed after capture.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc).
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, 120/600 s — also gives the clean steady-state DSA SLO),
  **AC-8** (~70K-token servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-cold-flood-not-steady-state-slo
Notes: Added a lesson from the AC-6 SLO confirmation: a `WARMUP=0` / `request_rate=inf` + `max_concurrency` flood run inflates TTFT/TPS for the NATIVE baseline (DSA) too (DSA conc-16 P99 TTFT 22.5 s / TPS 16.9 vs the established steady-state 0.73 s) because the cold ramp floods `max_concurrency` simultaneous 4096-prefills → prefill/decode contention. Such a run validates ADMISSION (achieved==nominal) + clean SERVING (errors 0) but NOT steady-state latency; the SLO number must come from a proper-warmup baseline at the identical operating point. Tell: tight median≈p99 well above a small `min`. Cross-checking the native baseline under identical methodology also retro-validates the AC-5 directional caveat (its WARMUP=0 run over-states DS TTFT). Reinforces BL-20260530-admission-restore-tps-tradeoff. Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (tracked .json/.txt proofs + get_server_info under runs/) and the `pkill -f 'sglang::router'` router-kill gotcha for the cross-node bring-up.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-9-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-9-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-8-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-8-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-7-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-7-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-10-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
