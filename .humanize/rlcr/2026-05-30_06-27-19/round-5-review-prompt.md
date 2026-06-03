# Code Review - Round 5

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-5-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 5 Summary — AC-4 evidence addendum (durable, complete)

## Mainline objective (round contract)
Close the two AC-4 acceptance-completeness gaps Codex's R4 review found, so task5/AC-4 is durably verifiable (it gates AC-5). **No verdict change** — AC-4 still PASSES at the lifted operating point 0.7; this round makes the *evidence* complete and tracked (an evidence/packaging round; no production code).

## What was done (commit `91e9c20a3`)

### 1. Full per-rank HBM budget (`ac4_hbm_budget_addendum.md`)
Assembled the complete torch-tracked budget for f=0.6/0.7/0.8 from the server's own memory stage-deltas (`Init torch distributed` / `Load weight end mem usage=80.63 GB` / `KV Cache` / pool-end / `Capture cuda graph` / headroom):

| component (GB/rank) | 0.6 | 0.7 | 0.8 |
|---|---:|---:|---:|
| weights | 80.63 | 80.63 | 80.63 |
| KV pool | 2.38 | 17.73 | 33.09 |
| table+scales (int8) | 0.87 | 6.48 | 12.10 |
| written+scratch+flashmla-meta+bind | 2.63 | 2.66 | (OOM in capture) |
| cuda-graph pool | 11.61 | 11.59 | partial→OOM |
| headroom | 38.43 | 17.65 | — |
| torch_used (=139.80−avail) / NVML used | 101.4 / 101.4 | 122.2 / 122.2 | OOM @ `134.41 GiB alloc` |

The budget **closes** (Σnamed + residual ≈ 139.80 GiB to ~0.5 GiB driver reserve) — *not only named tensors*. `torch.memory_reserved/allocated` per rank aren't HTTP-exposed, so the addendum provides: torch-tracked stage deltas, `torch_used`, NVML used (== per-process — confirmed `nvidia-smi --query-compute-apps` = per-GPU since the server is the sole process), a labeled residual bucket, and the 0.8 OOM's literal `134.41 GiB allocated by PyTorch`. `written` per rank = `bool[L, max_tokens]` = 0.023 GB (negligible).

### 2. Durable, tracked no-OOM proof at 0.7
The R4 NVML series was a **gitignored `.csv`** (never committed) and the 97/97 was only summarized. Re-ran the sustained stress (one re-boot) and tracked everything as `.txt`:
- `stress_0.7_client.txt` — `SUMMARY: 97/97 ok, 0 failed, 92.7s` (32-conc 4096-ISL × 3 rounds + 30K long-context).
- `stress_0.7_server_excerpt.txt` — server scheduler log: prefill 8192-chunks → **Decode batch `#running-req: 32`**, token-usage 0.39, gen ~380 tok/s; **generation-time OOM line count = 0**.
- `nvml_timeseries_0.7.txt` — used **1,005,832 → plateau 1,041,136 MiB** (last == max), min-free steady — **no monotonic growth**.
- `get_server_info_0.7.json` — `mem_fraction_static=0.7`, `max_total_num_tokens=396096`.

### 3. Hygiene
Stripped trailing whitespace from `mf_*.txt` + the sweep `.md` (`git diff --check` now **clean**); removed the gitignored `nvml_*.csv`.

## Files changed
- New/updated under `runs/20260530_dsv32_loop6/`: `ac4_hbm_budget_addendum.md`, `memfraction_sweep_int8/{get_server_info_0.7.json, nvml_timeseries_0.7.txt, stress_0.7_client.txt, stress_0.7_server_excerpt.txt}`, whitespace-fixed `mf_*.txt` + `memfraction_sweep_int8.md` — commit `91e9c20a3`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-5 contract/summary (gitignored loop state).

## Validation
- `git diff --check` clean on the new artifacts.
- Re-boot @0.7 confirmed int8 table (`dtype=torch.int8`), `/health` 200, sustained stress 97/97 with 0 server-side OOM lines, NVML plateau. GPUs freed after.

## Remaining items
**AC-4 evidence complete and tracked.** Nothing in AC-4 deferred.
- **AC-5 (next mainline):** the headline result — full client-SLO benchmark at the lifted 0.7 int8 operating point: `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL / ~55% cache, radix-on proven from `.meta.json` sidecars, `client_slo_report.md` asserting strict `< 22.0` TTFT + `≥ 30 TPS/req`, the pre-declared trial-aggregation rule, and the **required measured admission-wait vs prefill-compute attribution**. Then AC-6 hardware opt-in/DSA-default, AC-7/AC-8/AC-9, gated AC-10.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Added a lesson capturing the R4→R5 packaging failure: a hardware AC's durable artifact summarized the result and referenced a `.csv` that the repo gitignores (so it never committed), and the HBM "budget" listed big named tensors without closing. The lesson: embed acceptance evidence as tracked `.txt`/`.md` (never a gitignored format — check `git ls-files`/`status --ignored` + `git diff --check`), and CLOSE the HBM budget from the server's own torch memory stage-deltas (`torch_used = total − avail`, NVML == per-process for a sole-process GPU, + a labeled residual bucket summing to total). Applied existing lessons as context: BL-20260530-int8-memfraction-ceiling-is-cudagraph-capture (the 0.8 cuda-graph boot-OOM framing) and BL-20260530-ds-... router-kill gotcha (explicit-PID kill for the re-boot).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-4-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-4-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-3-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-3-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-2-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-2-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-5-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
