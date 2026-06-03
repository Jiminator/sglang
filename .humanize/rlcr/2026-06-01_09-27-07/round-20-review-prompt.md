# Code Review - Round 20

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-20-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 20 Summary — Loop 7

## Mainline objective (round-20-contract.md)
**task19 (AC-6 completion) — record the missing fresh conc-1/conc-16 TTFT guardrails
(plus a clean streaming decode-TPS) for DS-default, DS-hybrid, and DSA/native-NSA at the
Loop-7 op-point under CUDA graph, and update `m11_perf_consolidation.md` so AC-6 is claimed
only after TTFT is present.**

## Outcome: ACHIEVED — AC-6 MET; task19 done. 5/6 ACs MET.

## Why this round (R19-review gap)
The R19 review (ADVANCED) rejected full AC-6 closure: the plan
(`refined_plan_v1.md:80`) requires **`TTFT, decode TPS/req, GPU memory, graph-replay
success, admission` at conc-1/16**, and `perf_closed_batch.py` was a non-streaming
`/generate` probe that records only request wall time + completion tokens — it
structurally cannot measure time-to-first-token. The reviewer's authorized fix: "the
existing `bench_serving` path or **an equivalent streaming probe that records first-token
timestamps**."

## Work Completed (`coding`, live measurement; no production-code change)
1. **Extended `perf_closed_batch.py` with a `--stream` SSE mode.** It mirrors the
   canonical SGLang streaming parser (`data: {"text": <cumulative>, "meta_info":
   {"completion_tokens": N}}`), records per-request **TTFT** = first-streamed-token arrival
   − submit and a **clean post-first-token decode TPS** = `(completion_tokens − 1) /
   (t_last − t_first)`, and — per BL-20260531-bench-empty-stream-failclosed — **fails
   closed** on an HTTP-200 empty stream (raises rather than recording a no-token response
   as a completion). The non-streaming R19 path is preserved for reproducibility.
2. **Re-measured all three variants** (DS-default, DS-hybrid Tier-2.B, DSA/native-NSA) at
   conc-1/16 under CUDA graph at the Loop-7 op-point (int8 / mem 0.7 / fp8-KV / TP=8 / page
   64 / radix-off), two prompt regimes: a SHORT prompt (the R19 decode cross-check) and a
   ~770-token prompt (a prefill-bound TTFT guardrail, dense-prefill regime).

### TTFT (ms; conc-16 reported p50 / p99 across 16 concurrent reqs)
| variant | c1 short | c16 short (p50/p99) | c1 ~770-tok | c16 ~770-tok (p50/p99) |
|---|---|---|---|---|
| DSA (native-NSA) | 150.8 | 307.1 / 309.2 | 150.9 | 1161.5 / 1322.1 |
| DS-default | 183.3 | 371.7 / 374.0 | 180.4 | 1210.9 / 1400.2 |
| DS-hybrid (Tier-2.B) | 178.4 | 363.3 / 365.1 | 177.7 | 1218.1 / 1405.2 |

### Streaming decode-TPS cross-check (clean, post-first-token)
DSA **87.3 / 58.7**, DS-default **40.8 / 28.5**, DS-hybrid **41.1 / 28.5** (c1 / c16) —
reproduces the R19 closed-batch ordering and the DS ≈ 0.48–0.49× DSA structural ratio
(slightly higher than R19's e2e number because it excludes prefill+first-token — the
theoretically correct pure-decode rate). All decode batches `cuda graph: True`; served
16/16 every run.

## Key findings (non-regression)
- **DS-hybrid TTFT ≈ DS-default TTFT at every point** (178 vs 183 ms c1; 363 vs 372 ms
  c16-short p50; 1218 vs 1211 ms c16-p770 p50 — within run-to-run noise). The Tier-2.B
  hybrid scorer adds **no material TTFT cost** — the same decode-free result the R19
  decode-TPS table showed, now confirmed on first-token latency too.
- **DS TTFT is modestly above DSA** (~+30 ms c1, ~+60 ms c16-short) — the small per-step
  cost of the DS selection + logical→physical adapter, the same structural overhead as
  the decode-TPS gap; NOT a Loop-7 regression. In the prefill-bound c16-p770 case TTFT is
  prefill-dominated and DS ≈ DSA + ~5%.
- **Every measured TTFT is far below the Loop-6 directional P99 22 s ceiling** — heaviest
  point (DS conc-16, ~770-tok prefill) is P99 ≈ **1.4 s**. The Loop-6 directional P99
  13.13 s is retained only as the historical full-context reference (that path is
  unchanged because all Loop-7 work is opt-in/default-off).

## Files Changed
- `development/loop7/perf_closed_batch.py` (added `--stream` SSE mode + fail-closed guard;
  dev probe, not imported by tests).
- `development/loop7/m11_perf_consolidation.md` (TTFT table + streaming cross-check +
  TTFT findings + corrected conclusion-3 + AC-6-MET restated to include TTFT + provenance).
- `development/loop7/ttft_{ds_default,ds_hybrid,dsa}_c{1,16}{,_p770}.json` (12 artifacts).
- Commit `30173f08b` (local — loop hook). **No production-code change.**

## Validation
- TTFT + streaming decode-TPS per variant (above); GPU mem from `nvidia-smi` (DS 125 GB /
  DSA 133 GB, matching R19); graph from `cuda graph: True` decode batches (27 each);
  admission served 16/16 all variants; the probe fail-closes on empty streams so a
  completed run guarantees real per-request latency.
- Full DS unit suite → **350 passed + 9 subtests** (identical to R19 — confirms no
  production-code regression).
- GPUs freed (0 MiB) + all three servers stopped at round end.

## AC status after R20
- **AC-6 → MET** (the full guardrail set — TTFT + decode TPS + mem + graph + admission at
  conc-1/16 — is now recorded; landed deliverable non-regressing on both TPS and TTFT).
  With AC-1/3/4/5 (prior), **5/6 ACs MET**.
- **AC-2 PARTIAL** — only task20 (the final strategic-gate supersession decision record)
  remains, now **unblocked** (its corrected AC-6 source artifact exists). After task20,
  all 6 ACs are met and Loop 7 can close.

## Remaining Items (active mainline)
- **task20 (AC-2, next mainline + loop close)** — the final gate-supersession decision
  record: cite M0 regime attribution, AC-1 closure, AC-3 hybrid scorer, AC-4 production-ready
  lifted, AC-5 servability, AC-6 perf guardrails (this `m11`, now with TTFT); explicitly
  state what measured evidence superseded the Loop-6 Tier-2.A-primary ordering; cite/preserve
  the R8 oracle-sink provenance before relying on it.
- Evidence-hygiene queued (fold into task20): R8 stride/oracle provenance citation;
  plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: applied the selected lessons — BL-20260531-bench-empty-stream-failclosed (the
  `--stream` probe adopts the same fail-closed empty-stream guard and the standard SGLang
  SSE format), BL-20260528-dsv32-ds-serving-boot-chain + BL-20260529-dsv32-bench-smoke-sizing
  (TP=8 boot + mem-fraction op-point hygiene for the three measurement servers), and
  BL-20260527-shell-json-into-python-source (the probe consumes server JSON as data, never
  source). No NEW reusable pitfall surfaced: the streaming probe re-uses the established
  SSE parser + fail-closed pattern, so the findings (Tier-2.B is TTFT-free; DS ≈ 0.48–0.49×
  DSA) are project evidence recorded in `m11_perf_consolidation.md`, not a cross-round
  engineering lesson.

## Goal Tracker
Updated directly (Plan Version 29): R20 plan-evolution row; task19 → Completed (TTFT
recorded, AC-6 MET) and added to Completed-and-Verified with Verified = pending (R20
Review); Active = task20 only (marked unblocked). No Goal Tracker Update Request needed.
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
2ba4dafc1 [Sparsity] Loop-7 R13: wire the served eager lifted-budget decode branch + enable the seam
0ad20774a [Sparsity] Loop-7 R14: binding served 4K recall recovery for the lifted-budget path
b70f48d36 [Sparsity] Loop-7 R15: Tier-2.A landing disposition (deferred-with-evidence) — closes AC-4
714cf62b2 [Sparsity] Loop-7 R16: graph-safe lifted-budget decode primitives + zero-alloc replay proof
6453562e9 [Sparsity] Loop-7 R17: wire graph-safe lifted decode into production CUDA-graph + relax validator
41e0af078 [Sparsity] Loop-7 R17: production-ready Tier-2.A disposition + graph-mode recall evidence (AC-4 close)
f9f6ec056 [Sparsity] Loop-7 R18: close AC-4 — graph-captured lifted-width selector proof + consistent production-ready disposition + lifted+spec guard
68969deb0 [Sparsity] Loop-7 R19: AC-6 perf consolidation — conc-1/16 guardrails + DS-vs-DSA non-regression report
30173f08b [Sparsity] Loop-7 R20: AC-6 TTFT guardrails — streaming probe + fresh conc-1/16 TTFT (DS-default/hybrid/DSA)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-19-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-19-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-18-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-18-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-17-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-17-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-20-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
