# Code Review - Round 13

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-13-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 13 Summary — AC-7: 3-trial DS+DSA re-sweep at the lifted point (characterized, DEC-9)

## Mainline objective (round contract)
AC-7: the 3-trial DS+DSA directional re-sweep at the lifted operating point (DS int8 @ 0.7
radix-on, DSA-default @ 0.85 radix-on), conc 16/32/64, num_prompts=64, 120/600, TRIALS=3, then
refresh the ac11 reports showing DS achieved-concurrency now tracks nominal. Gated by the
cross-node wrapper host smoke (Codex R12).

## What landed (commit 5e6d3afb5, pushed)
**3-trial DS+DSA re-sweep** (num_prompts=64, warmup 120 / window 600, conc 16/32/64, radix-on
both). **Both sides ran on node 0 localhost, sequentially** (DS, then DSA) — see Methodology note.

**Headline — admission RESTORED (the footprint→admission spine's payoff), effective concurrency, median of 3:**
| conc | DS @ lifted | DS/nominal | Loop-5 DS (mem 0.6) | DSA |
|---:|---:|:--:|---:|---:|
| 16 | **16.0** | **100%** | 14.5 (91%) | 16.0 |
| 32 | **32.0** | **100%** | 24.6 (77%) | 32.0 |
| 64 | **47.0** | **73%** | 35.7 (56%) | 63.9 |

DS now admits full nominal concurrency at conc 16/32 and improves conc-64 to 73% (from Loop-5's
56%); errors 0 across all 18 runs.

**DS-vs-DSA parity gates FAIL** (comparator `ac11_resweep.md`): DS TPS 0.31–0.38× DSA; DS P99
TTFT 18–49× DSA. This is the **expected** DSA-trained-indexer advantage + the admission-restore
TPS tradeoff (AC-5) — a **DEC-7 directional follow-up, not a footprint regression**; AC-7 is soft
(DEC-9, may be characterized). The comparator's profiling obligation is discharged by the AC-5
measured attribution (queue-vs-prefill + decode-batch root cause) at the identical workload.
DSA-default reproduces its baseline (0.72/1.28/2.04 s, 46.9/37.5/29.5 TPS; conc-64 TPS ~29.5 =
the queued pre-existing limit).

## Methodology note (justified plan deviation)
The intended cross-node bring-up (DS node 0 + DSA node 1) was abandoned: **node-1 remote server
boot proved intractable this round** — setsid/nohup/tmux-arg launches all failed (fast ssh-close
teardown + accumulated zombie procs; no DSA weights ever loaded; ~2h lost). I pivoted to run
**both sweeps on node 0 localhost, sequentially**, which is **comparator-clean** (same
node/session/commit; only per-side mem differs, as in Loop-5) and avoids any cross-node
host-mismatch. Because neither sweep is cross-node, the **cross-node wrapper smoke is N/A this
round**; the R12 `--host` fix is verified in-wrapper (the DS sweep `bench_serving` banner
`Waiting up to 60s for http://127.0.0.1:30000` + the matching DS `.meta.json`), and R11 separately
proved `bench_serving --host node1` targets node 1. Captured as BitLesson `remote-server-launch`.

## Result
AC-7 characterized/soft-met (DEC-9): admission restored (the spine validated across 3 trials);
DS-vs-DSA parity miss recorded as a DEC-7 directional follow-up, attributed via AC-5, not a
footprint regression. The **AC-5 DS strict-SLO miss remains the open mainline blocker**.

## Files Changed
- `runs/20260530_dsv32_loop6/ac7_resweep/`: `ac11_resweep.md` (comparator), `ac11_analysis.md` (verdict/characterization), `ac7_resweep_metrics.json` (recomputable per-trial DS+DSA + medians + source JSONL SHA256), 18 `.meta.json` sidecars.
- `.humanize/bitlesson.md` (+1 lesson `remote-server-launch`), goal-tracker (R13 row; task8/AC-7 → done-characterized; cross-node-host blocker → RESOLVED/N-A; the host bug + node1 deviation documented), round-13 contract/summary (gitignored loop state).

## Validation
- 9 DS + 9 DSA runs, errors 0; DS achieved 16/32/47 effective (100/100/73%), DSA 16/32/64.
- Comparator exit 3 (gates fail = expected directional); recomputable metrics JSON validates (per-trial + SHA256); `.meta.json` confirm radix-on + per-side mem (DS 0.7 / DSA 0.85) + commit SHA.
- `--host` fix verified in-wrapper (localhost banner + matching sidecar). `git diff --check` clean; commit 5e6d3afb5 pushed to `jimmy`; both nodes' GPUs freed (0 MiB).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Per Codex's plan, the AC-5 remediation (smallest scheduling/decode/operating-point change to restore both, with the AC-7 data in hand) is the next focus after AC-8.
- **AC-8** (~70K-token servability probe at the lifted mem fraction — HTTP 200 with capacity, or a characterized ceiling), gated **AC-10** (after AC-3–AC-9 verified). Queued: DSA conc-64 TPS ~29.5 (pre-existing). No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-remote-server-launch
Notes: Added BL-20260530-remote-server-launch capturing ~2h of node-1 boot failures: ssh-launched detached servers (setsid/nohup) are torn down when the ssh channel closes fast; `tmux new-session "<cmd>"` bypasses the shell so env/redirects don't apply (use `tmux send-keys` into the session's bash); remote login cwd is /root so use ABSOLUTE script paths; `pgrep -f` false-matches the launcher's own command line (use `ps | grep "[s]glang.launch_server"`); `pkill` no-match exit-1 trips `set -e` (use `|| true`, not a trailing `; true`); foreground `sleep` is blocked in harness Bash. Reliable fallback: run both servers on the local node sequentially via the Bash run_in_background tool (comparator-clean), which is what AC-7 used. Applied existing lessons: BL-20260530-cold-flood-not-steady-state-slo (num_prompts=64 steady-state methodology for the sweep), BL-20260530-durable-tracked-acceptance-evidence (recomputable per-trial metrics + source SHA256 since raw .jsonl are gitignored), BL-20260530-bench-host-targeting (the --host fix, verified in-wrapper), and the push-between-rounds preference.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-12-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-12-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-11-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-11-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-10-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-10-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-13-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
