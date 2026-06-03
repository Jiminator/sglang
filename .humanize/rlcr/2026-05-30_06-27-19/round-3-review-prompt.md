# Code Review - Round 3

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-3-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 3 Summary — launcher signature_dtype + real-mask NIAH non-regression (AC-3.1)

## Mainline objective (round contract)
Run the real-mask NIAH non-regression on TP=8 hardware — boot DS with the compact int8 table on the Loop-5 mask and prove int8 does not regress DS needle recall vs the fp16 Loop-5 DS baseline. This was the last outstanding AC-3.1 evidence item (it gates AC-4). Blocking prerequisite: the launcher couldn't select the compact table.

## Correction to a prior-round premise
The RLCR box is **node 0 with 8× H200** (plus a reachable node 1) — V3.2 TP=8 serving is feasible here. My R1/R2 "2-GPU dev box" claim came from a capped `nvidia-smi -L | head -2`; the resulting "Explicitly Deferred" classification for the NIAH was wrong, and Codex was right to reject it. (Captured as a BitLesson — see below.)

## Blocking fix landed first (Codex R2 review)
`serve_double_sparsity.sh` built `DS_CONFIG` without `signature_dtype`, so the documented `bash serve_double_sparsity.sh` silently booted the **fp16** table (config default) — any compact-table hardware run would have validated full precision. Fixed (commit `5d8e47fb3`): added `SIGNATURE_DTYPE` env (default fp16), included `"signature_dtype": "${SIGNATURE_DTYPE}"` in `DS_CONFIG`, echoed it in the launch log, plus a behavioral test (stub `python3` captures `--double-sparsity-config`): default → fp16, `SIGNATURE_DTYPE=int8` → int8 and parses as a valid config.

## Real-mask NIAH non-regression — PASS (commit `8a05b1688`)
Setup (real TP=8 hardware):
- **DS-int8** on node 0:30000 — `SIGNATURE_DTYPE=int8`, mem 0.6, Loop-5 mask (`7b3207cae888`). Boot proof: `token_label_table: 0.87 GB/rank ... dtype=torch.int8 scales=float16` (the **0.5625× = 1.55→0.87 GB** reduction confirmed on hardware) + `double_sparsity_config='{...,"signature_dtype": "int8"}'`; decode coherent (" Paris.").
- **DSA (live reference)** on node 1:30001 (cross-node), mem 0.85.
- `test_double_sparsity_v32.py -k niah`, `AC12_NIAH_NUM_PROMPTS=20`, DS=node0 / DSA=node1 → **2 passed, 2 skipped, 5 subtests passed** (308 s).

| length | int8 DS (now) | fp16 Loop-5 DS | live DSA | int8 ≥ fp16? |
|---:|---:|---:|---:|:--:|
| 1024 (within budget) | 100% | 100% | 100% | ✅ |
| 1536 (within budget) | 100% | 100% | 100% | ✅ |
| 4K | 85% | 75% | 100% | ✅ (+10pp) |
| 16K | 5% | 5% | 100% | ✅ (=) |
| 64K | 0% (unservable) | 0% (unservable) | 100% | ✅ (=) |

**Verdict: PASS** — int8 matches or exceeds fp16 DS recall at every length (no regression; the +10pp at 4K is within the ±5pp/needle granularity at 20 prompts), no new unservable error (64K is an admission limit identical for both at mem-0.6), and the live DSA reference is 100% everywhere (paired cross-node setup sound). Combined with the decode-scoring microbench (TPS-neutral), the compact path is recall-neutral **and** TPS-neutral. fp16 baseline: `runs/20260528_dsv32_mvp/ac12_results/`.

## Files changed
- `development/serve_double_sparsity.sh` (+SIGNATURE_DTYPE), DS test file (+2 launcher tests) — commit `5d8e47fb3`.
- `runs/20260530_dsv32_loop6/real_mask_niah_nonregression.md` + `real_mask_niah_int8/` (5 `ac12_niah_*.json` + pytest log + int8 boot proof) — commit `8a05b1688`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-3 contract/summary (gitignored loop state).

## Validation
- Launcher tests: 2 passed (default fp16, int8 selectable through the real script).
- DS-int8 booted clean on TP=8 (after clearing a 4-day-old stale `sglang::router` that had grabbed port 30000); served coherently.
- Real-mask NIAH paired run: PASS, int8 ≥ fp16 at every length; artifacts copied under `runs/`.
- Servers killed; both nodes' GPUs freed.

## Remaining items
**AC-3 is now fully evidenced** (synthetic top-k overlap@2048≥0.99 + scale-sidecar proofs + decode-scoring microbench + real-mask NIAH + DSA-default). Nothing in AC-3 is deferred.
- **AC-4 (next mainline):** mem-fraction sweep `0.6→0.8` with `SIGNATURE_DTYPE=int8`, full NVML/torch-residual HBM accounting, `/get_server_info`, sustained long `/generate` with no OOM / no monotonic growth. Then AC-5 client-SLO (with admission-vs-prefill attribution), AC-6 hardware opt-in/DSA-default, AC-7/AC-8/AC-9, then gated AC-10.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-verify-hardware-before-deferring
Notes: Added a lesson capturing the R1→R3 mistake: a capped `nvidia-smi -L | head -2` was read as the GPU inventory, leading to a wrong "2-GPU, can't serve V3.2" deferral of the real-mask NIAH that review had to reject. The lesson: probe the FULL hardware inventory (`nvidia-smi --query-gpu=...`, CLUSTER.md, weights/rank×tp ≤ HBM/rank) before declaring a hardware step infeasible or deferring it; never let a display cap become a capacity claim. (Also surfaced but already covered by existing router-kill lessons: a 4-day-old stale `sglang::router` answered `/health` 200 with `no_available_workers`, masking my server's port-bind failure — diagnosed and cleared by killing the stale router and rebooting.)
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-2-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-2-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-1-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-1-review-result.md
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-3-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
