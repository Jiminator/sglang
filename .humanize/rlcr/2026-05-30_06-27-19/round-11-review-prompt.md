# Code Review - Round 11

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-11-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 11 Summary — AC-6 redo: proper-methodology DSA SLO + radix-on toggle

## Mainline objective (round contract)
Finish AC-6 properly: a DSA-default client-SLO confirmation under the proper
steady-state methodology that actually passes, plus a DS-opt-in toggle proof at the
locked **radix-on** operating point, and an AC-6 doc that claims only what the artifacts
prove. Codex's R10 review verified AC-9 (MET) but rejected AC-6: the WARMUP=0 smoke
showed DSA *failing* the SLO (cold-ramp), and the DS boot was radix-OFF while DSA was
radix-ON. AC-5's directional verdict + the open strict-SLO blocker stay tracked, not this round's objective.

## What landed (commit 0e1ce974d, pushed to `jimmy`)
Cross-node bring-up: DS int8 @ 0.7 **radix-on** (node 0) + DSA-default (node 1).

**Gap 2 fixed — DS opt-in toggle at the locked radix-on point.** Re-booted DS with the
int8 radix fixture: `/get_server_info` proves `enable_double_sparsity=True`,
`signature_dtype=int8`, **`disable_radix_cache=False`**, `double_sparsity_radix_fixture_artifact`
set; boot log shows the int8 `token_label_table` 6.48 GB/rank on **all 8 ranks** + the
radix fixture recorded PASSED (sha f3b67943). DSA-default: `enable_double_sparsity=False`,
`config=None`, 0 table lines, full 910784 pool, **`disable_radix_cache=False`** too. Both
radix-on ⇒ differ by DS enablement (and the mem-fraction it forces: 0.7+table vs 0.85+full-pool).

**Gap 1 fixed — DSA-default SLO under proper methodology.** The fresh DSA-default boot is
**byte-identical to the tracked Loop-5 DSA SLO baseline** (`dsa_default_matches_loop5_baseline.txt`:
all 11 operating-point fields match), so the established baseline applies after the DS
changes (DSA-default runs no DS path). Baseline + **fresh R11 `num_prompts=64` reproduction**
(`dsa_default_slo_np64.txt`, cross-node, warmup 120 / window 600):

| conc | P99 TTFT (fresh / L5) | per-req TPS (fresh / L5) | SLO `<22` & `≥30` |
|---:|---:|---:|:--|
| 16 | 0.89 / 0.97 s | 46.1 / 46.7 | ✅ / ✅ |
| 32 | 1.49 / 1.39 s | 37.0 / 37.6 | ✅ / ✅ |
| 64 | 2.18 / 2.02 s | 29.4 / 29.5 | ✅ / ⚠ ~29.4 (marginal, pre-existing) |

DSA-default meets **P99 TTFT < 22 s at every conc** (0.89/1.49/2.18 s); TPS ≥ 30 at conc
16/32; **conc-64 TPS ~29.4 is marginally below 30 in the DSA baseline itself** — a
pre-existing DSA characteristic at the threshold (decode batch of 64), reproduced fresh,
**not** introduced by the DS opt-in code. completed 832/1344/2048, errors 0, achieved == nominal.

**Methodology finding (why R10/this round's NUM_PROMPTS=320 run failed):** a `NUM_PROMPTS=320`
run has an epoch (~558 s at conc-16, request_rate=inf) **longer than the 120 s warmup**, so
the measurement captures the synchronized first-epoch cold-ramp (P99 TTFT 17.2/34.2 s), not
steady state. `num_prompts=64` (epoch ≈ 35 s ≪ warmup) reproduces the baseline. (Also: R10's
"DSA" bench actually hit node0 because `benchmark_baseline.sh` never passes `--host`; R11
targets node1 DSA directly via `bench_serving --host`.) The 320-prompt run is kept only as
the cold-ramp datapoint in `dsa_default_slo.txt`.

## Result
AC-6 met: DS ships opt-in (int8 table toggles at the locked radix-on point), DSA stays the
production default (no DS table, full admission, SLO < 22 s at every conc, unchanged by the
DS code). The **strict-SLO miss remains the open mainline blocker** (AC-5 conc-32/64) — unchanged.

## Files Changed
- `runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md` (rewritten honestly: radix-on parity, proper-methodology SLO, methodology note).
- `runs/20260530_dsv32_loop6/ac6_product_proof/`: `ds_opt_in_get_server_info.json` (radix-on), `ds_table_boot_excerpt.txt` (8 ranks + fixture PASSED), `dsa_default_get_server_info.json`, `dsa_notable_boot_excerpt.txt`, `get_server_info_keys.json` (both radix-on), `dsa_default_matches_loop5_baseline.txt` (NEW), `dsa_default_slo_np64.txt` (NEW fresh SLO), `dsa_default_slo.txt` (320 cold-ramp datapoint).
- `.humanize/bitlesson.md` (extended `cold-flood-not-steady-state-slo` with the epoch>warmup trap), goal-tracker, round-11 contract/summary (gitignored loop state).

## Validation
- DS radix-on: `get_server_info` `disable_radix_cache=false` + fixture path + int8 table 8 ranks (fixture PASSED).
- DSA-default operating point byte-identical to Loop-5 baseline (11/11 fields); fresh `num_prompts=64` reproduces baseline (0.89/1.49/2.18 s); errors 0; achieved == nominal.
- bench correctly targeted node1 DSA (`bench_serving --host 10.220.51.5`, smoke "Server ready in 0.0s").
- `git diff --check` clean; commit 0e1ce974d pushed; servers killed, both nodes' GPUs freed (0 MiB).

## Remaining Items
- **Open mainline blocker:** AC-5 strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc) for DS.
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, 120/600, radix-on both — note: use `num_prompts=64` per this round's methodology finding, and target the right server), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3); DS-fair thresholds unchanged (AC-9 done).

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-cold-flood-not-steady-state-slo
Notes: Extended the cold-flood lesson with the R11 finding: there are TWO cold-ramp traps, not one. Trap 1 (R10) is `WARMUP=0`. Trap 2 (R11) is `WARMUP>0` but the EPOCH longer than the warmup — with `request_rate=inf`, bench_serving processes `num_prompts` per epoch; `num_prompts=320` at conc-16 gives an epoch ~558 s ≫ 120 s warmup, so the measurement still captures the synchronized first-epoch burst (DSA P99 TTFT 17.2/34.2 s), whereas `num_prompts=64` (epoch ~35 s ≪ warmup) reproduces the steady-state baseline (0.89/1.49/2.18 s, matching Loop-5 0.97/1.39/2.02). Rule added: choose `num_prompts` so the epoch ≪ warmup, or reuse the established small-`num_prompts` baseline. Validation/Source updated to R10+R11. (Also surfaced, recorded in the round summary not the lesson: `benchmark_baseline.sh` never passes `--host`, so a cross-node "baseline" run silently hits localhost — invoke `bench_serving --host` directly or run on the server node.) Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (baseline numbers captured in a tracked `.txt` since the Loop-5 `.jsonl` are gitignored; fresh np64 is self-contained tracked evidence) and the `pkill -f 'sglang::router'` router-kill gotcha.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-10-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-10-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-9-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-9-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-8-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-8-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-11-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
