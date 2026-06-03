# Code Review - Round 18

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-18-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 18 Summary — AC-5 conc-16 strict-decode PASS (bounded-context op-point) + owner done-criterion

## Owner decision (R12-style)
Surfaced the structural finding via AskUserQuestion: `≥30 TPS/req at every conc 16/32/64` is unattainable
for DS — per-request decode TPS falls as the decode batch grows, and even **DSA (the faster path DS cannot
exceed) is 29.4 at conc-64** (DSA 46.1/37.0/29.4). The owner chose **"conc-16 strict + characterize 32/64"**
(confirmed as my recommendation): strict gate = conc-16 (≥30 TPS/req AND P99 TTFT <22 s); conc-32/64 are
characterized as the structural decode-batch ceiling (DS ≤ DSA), not a DS/footprint defect. Recorded as a
Plan Evolution row in `goal-tracker.md`.

## Mainline objective (round contract)
Make DS strict-pass the client SLO at conc-16 and characterize conc-32/64, per Codex's R17 plan (residual
top-k over-scan first, then the client numbers), preserving the ABI lock and 64K servability (AC-8).

## What landed (commit `fcc2d1cdb`; no production code change — operating point + evidence)
1. **Technical finding on the residual top-k over-scan.** R17's score-kernel early-exit left the first
   `torch.topk(scores[:bs,:max_seq_len], 2048)` scanning the full `max_seq_len = req_to_token.shape[1] =
   context_len = 163840`. Under **CUDA-graph capture the topk score-buffer width is fixed at capture and
   `torch.topk` (a monolithic reduction) cannot skip** rows past `seq_len` — so a no-context-cap graph-safe
   topk speedup needs a research-grade K=2048 skipping kernel (a torch two-level/reshape topk still processes
   the full width; the stubbed `DSGraphState.scratch_partial_*` two-stage path was never implemented).
2. **The cheap, exact lever — bounded-context client-SLO operating point.** The topk scan width == the model
   context length, and the client workload is 4096 ISL + 512 OSL = 4608 tokens, so `--context-length 8192`
   shrinks `req_to_token` width 163840→8192 (the topk then scans 8192), KV pool unchanged
   (`max_total_num_tokens=396224`, mem 0.7), **+9 GB headroom**. 64K servability (AC-8) is the **separate
   full-context operating point**, already validated in R16 — two honest operating points.
3. **conc-16 strict-decode MET.** Closed-batch pure decode (own client, `ignore_eos`, real 512-step decode,
   server-log-confirmed; no prefill interleave) at DS int8 / mem-0.7 / radix-on / **ctx 8192** + R17 score-fix:
   conc-16 **27.1 → 30.3 TPS/req (PASS ≥30)**, conc-8 36.0, conc-1 43.6. Fail-closed verifier
   `ctx8192_decode_metrics_tool.py --verify` recomputes per-req TPS = median(gen)/batch from committed
   samples (conc-16 ≥30 asserted; tampered conc-16 sample 29.38 → exit 1; clean → exit 0 PASS).
4. **conc-32/64 characterized.** 27.2 / 22.6 TPS/req at ctx 8192 (up from full-ctx ~20/~16, still < 30) — the
   decode-batch→TPS structural ceiling; DS < DSA (37 / 29.4); conc-64 ≥30 unattainable even for DSA.

## Result
The conc-16 decode-TPS axis (the previously-failing axis) now strict-passes (30.3 ≥ 30, verifier-checked)
at the bounded-context client-SLO operating point, with the ABI lock intact and 64K servability preserved
as the separate full-context deployment. conc-32/64 are characterized as the structural decode-batch
ceiling per the owner decision + DEC-3.

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_conc16_strict/` (NEW): `ac5_conc16_strict.md` (the report + the two-operating-points
  framing + the technical finding), `ctx8192_decode_curve.json` + `ctx8192_decode_metrics_tool.py` (exact
  closed-batch samples + fail-closed verifier), `ctx8192_decode_curve.txt` / `closed_batch_ctx8192.txt`
  (decode-curve excerpts), `get_server_info_ctx8192.json` (operating-point sidecar).
- `.humanize/bitlesson.md` — extended `BL-20260531-ds-selection-fullwidth-overscan` with the R18 addendum
  (capture-width-bound topk; bounded-context lever; bench window-mode caveat); goal-tracker (R18 owner-decision
  Plan Evolution row + task6 note); round-18 contract/summary (gitignored loop state).
- (No production code change this round; the R17 score-kernel fix `selection_kernel.py` is already committed.)

## Validation
- `ctx8192_decode_metrics_tool.py --verify`: PASS (conc-16 30.33 ≥30; conc-32/64 27.17/22.6 < 30; monotone
  sanity); tamper (conc-16 → 29.38) exits 1.
- Operating point proven: `get_server_info_ctx8192.json` (int8 / mem 0.7 / radix-on / context_len 8192 /
  pool 396224 / TP=8). Coherence on the ctx8192 server: "The capital of France is" → " Paris. The capital of
  the United States is Washington, D" (no degeneration).
- GPUs freed at round end (all 8 at 0 MiB). `git diff --check` clean; commit `fcc2d1cdb` pushed to `jimmy`.

## Remaining Items
- **Open residual (conc-16 TTFT):** conc-16 P99 TTFT <22 s is supported by the R6 Codex-verified full-context
  12.8 s (ctx8192 decodes faster → TTFT only lower), but a **fresh ctx8192 TTFT-under-flood number** was not
  captured: `development/benchmark.sh` bench_serving WINDOW mode returned empty per-request latency arrays +
  impossible aggregate throughput in this build (at WARMUP=0 and 120) while the server generated correctly on
  direct `/generate` and under the closed-batch client. Resolving the window-mode harness (or using a working
  flood client) to publish a fresh conc-16 P99 TTFT is the item to fully close conc-16 strict.
- **conc-32/64** characterized (structural; DS ≤ DSA, conc-64 unattainable even for DSA) per the owner decision.
- **Gated AC-10** — only after AC-5 is met under the owner criterion + AC-3..AC-9 verified.
- Cross-node wrapper smoke (future-gated) and DSA-default conc-64 TPS ~29.4 (queued) unchanged. No ABI-lock /
  FlashMLA-assert change; DS-fair AC-12 gate unchanged.

## Goal Tracker Update Request
### Requested Changes:
- Record the **R18 owner decision** as accepted Plan Evolution: AC-5 done-criterion = **conc-16 strict-pass
  (≥30 TPS/req AND P99 TTFT <22 s) + conc-32/64 characterized** (already added to the Plan Evolution Log).
- Mark AC-5 **conc-16 decode-TPS axis MET** (30.3 ≥30, fail-closed verifier) at the bounded-context client-SLO
  operating point; keep task6/AC-5 Active only for the fresh-ctx8192 conc-16 P99 TTFT residual (harness-blocked,
  strongly supported by R6 12.8 s).
### Justification:
The all-conc strict pass is structurally impossible (conc-64 ≥30 unattainable even for DSA); per DEC-3 + the
Lower Bound the owner set the realizable MVP done-criterion. The conc-16 decode-TPS strict-pass is the novel,
previously-failing result and is now verifier-checked; the only remaining gap is a harness-blocked fresh TTFT
number whose target was already met + verified at the full-context point in R6.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260531-ds-selection-fullwidth-overscan
Notes: Added the R18 addendum — the residual topk over-scan is **capture-width-bound** (CUDA-graph fixes the
topk buffer width; `torch.topk` can't skip → a no-context-cap graph-safe speedup needs a research-grade K=2048
skipping kernel), so the cheap EXACT lever is the **bounded-context operating point** for a latency-sensitive
bounded workload (client SLO 4608 tokens → `--context-length 8192`), keeping long-context as a separate
full-context op-point; closed-batch conc-16 27.1→30.3 (clears ≥30), conc-32/64 27.2/22.6 (structural ceiling,
DS ≤ DSA). Plus the bench_serving WINDOW-mode empty-latency-array caveat (use the closed-batch / server-log
gen-throughput for pure decode TPS). Applied existing lessons: BL-20260530-admission-restore-tps-tradeoff
(per-req TPS = 1/decode_step_time; batch→TPS ceiling), BL-20260530-cold-flood-not-steady-state-slo (cold-flood
TTFT is a conservative upper bound), BL-20260530-durable-tracked-acceptance-evidence (exact samples + fail-closed
verifier), BL-20260530-remote-server-launch (background boot + pkill||true; foreground sleep blocked).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-17-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-17-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-16-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-16-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-15-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-15-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-18-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
