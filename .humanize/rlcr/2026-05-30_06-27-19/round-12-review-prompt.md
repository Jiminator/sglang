# Code Review - Round 12

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-12-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 12 Summary — AC-6 honestly resolved (non-regression, user decision) + benchmark `--host` fix

## Mainline objective (round contract)
Honestly resolve AC-6: fix the benchmark `--host` bug, commit recomputable DSA-default SLO
evidence, correct the AC-6 doc/tracker (no overclaim), and surface the one genuine open
question — DSA-default's conc-64 per-req TPS ~29.4 is a ~2% pre-existing miss of `≥30 TPS`
(not DS-introduced) — as a user decision, since no re-run changes it and it determines
AC-6's verdict. Codex R11 verified AC-9 (MET) and raised these as the AC-6 blockers.

## What landed
1. **`--host` benchmark-script fix (commit d0cc9fdc9).** `benchmark.sh` + `benchmark_baseline.sh`
   now pass `--host "${HOST}"` to `bench_serving` (was `--port`-only → silent localhost
   targeting; this had mislabeled R10's "DSA" bench, which actually hit node 0). Both the
   load path and the `/get_server_info` sidecar now use the same `${HOST}`. (R11 runtime
   evidence `bench_serving --host node1` → "Server ready 0.0s" proves the mechanism; the
   full script-level cross-node smoke is the AC-7 bring-up gate.) Recorded as BitLesson
   `bench-host-targeting`.
2. **Recomputable DSA-default SLO evidence (commit f9bc51b13).** `ac6_product_proof/dsa_slo_metrics_tool.py`
   + `dsa_slo_arrays.json` — exact per-request `ttfts`/`tpots`/`input_lens`/`output_lens`,
   errors-all-empty, source JSONL SHA256; `--verify` recomputes P99 TTFT + per-req TPS from
   the committed JSON alone and is **fail-closed** (exit 1 on mismatch). Verify PASS: conc
   16/32/64 P99 TTFT 0.89/1.49/2.18 s (all <22), per-req TPS 46.1/37.0/29.4.
3. **Honest AC-6 verdict + USER DECISION (R12).** The user ruled AC-6 is a **non-regression /
   opt-in product test**: DSA-default is byte-identical to the pre-DS Loop-5 baseline and
   reproduces it (0.89/1.49/2.18 s, 46.1/37.0/29.4 ≈ 0.97/1.39/2.02 s, 46.7/37.6/29.5), so
   enabling the DS opt-in code leaves DSA-default **unchanged**; the DS opt-in flag toggles
   the compact int8 path at the radix-on locked point. **AC-6 = MET.** The conc-64 per-req
   TPS ~29.4 (<30) is a **pre-existing DSA + H200 decode-batch-64 limit** (29.5 in Loop-5),
   **not introduced by DS** — recorded as a separate Queued client-SLO-vs-DSA tension that
   (per the user decision) does not block this non-regression AC.

## Result
AC-6 MET (per the R12 user decision: non-regression/opt-in test). The **AC-5 DS strict-SLO
miss remains the open mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30). Remaining:
AC-7 (3-trial DS+DSA re-sweep via the `--host`-fixed scripts), AC-8 (~70K probe), gated AC-10.

## Files Changed
- `development/benchmark.sh`, `development/benchmark_baseline.sh` (`--host "${HOST}"`).
- `runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md` (AC-6 verdict = non-regression MET; conc-64 TPS tension recorded; recomputable-evidence reference).
- `runs/20260530_dsv32_loop6/ac6_product_proof/dsa_slo_metrics_tool.py`, `dsa_slo_arrays.json` (new, recomputable + fail-closed verifier).
- `.humanize/bitlesson.md` (+1 lesson `bench-host-targeting`), goal-tracker (R12 row, plan-evolution = AC-6 non-regression grading per user decision; AC-6/`--host` blockers → RESOLVED; conc-64 TPS → Queued), round-12 contract/summary (gitignored loop state).

## Validation
- `dsa_slo_metrics_tool.py --verify`: recomputed == stored + sanity PASS; prints the honest SLO verdict (TTFT <22 all conc; TPS ≥30 conc 16/32; conc-64 ~29.4 marginal miss).
- `--host` fix: both scripts pass `--host "${HOST}"`; both sites use the same `${HOST}` (static-verified); R11 runtime banner confirmed `--host` targets the named node.
- `git diff --check` clean; commits d0cc9fdc9 + f9bc51b13 pushed to `jimmy`; no servers left running (GPUs were freed at R11 end; none booted this round).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc).
- **Queued (not blocking):** DSA-default conc-64 per-req TPS ~29.4 < 30 — pre-existing DSA/H200 limit, a client-SLO-vs-DSA tension independent of DS.
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, radix-on both, `num_prompts=64` per R11's steady-state finding, via the `--host`-fixed scripts + a cross-node host-targeting smoke), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-bench-host-targeting
Notes: Added BL-20260530-bench-host-targeting: a cross-node benchmark wrapper that passes only `--port` (not `--host`) to `sglang.bench_serving` silently load-tests localhost while its `/get_server_info` sidecar points at the remote host — this mislabeled loop6 R10's "DSA" run (it hit node0 DS). Fix: thread `--host "${HOST}"` to bench_serving so the load path and the sidecar target the same host, or run the wrapper on the server node; smoke-test that bench_serving's readiness banner names the intended host before trusting a cross-node artifact. Validated R10 (bug)/R11 (direct --host workaround)/R12 (script fix, commit d0cc9fdc9). The R12 AC-6 grading (non-regression/opt-in test; conc-64 TPS is a pre-existing DSA limit, not DS-introduced) is a USER DECISION recorded in the goal-tracker Plan Evolution Log, not a BitLesson. Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (AC-5-grade recomputable arrays + fail-closed verifier for the DSA SLO) and BL-20260530-cold-flood-not-steady-state-slo (num_prompts=64 steady-state vs the 320 cold-ramp).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-11-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-11-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-10-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-10-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-9-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-9-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-12-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
