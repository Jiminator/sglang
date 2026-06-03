# Code Review - Round 21

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-21-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 21 Summary — Loop 7

## Mainline objective (round-21-contract.md)
**task20 (AC-2) — write the final strategic-gate supersession decision record that closes
Loop 7**, with a **blocking prerequisite**: repair the AC-6/task19 TTFT evidence provenance
(the R20-review gap) so task20 can cite a correct `m11`.

## Outcome: ACHIEVED — AC-2 + AC-6 MET; ALL 6/6 ACs MET. Loop 7 ready to close.

## Why this round (R20-review gaps)
The R20 review (ADVANCED) accepted the TTFT values but found two mainline gaps before close:
(1) the `ttft_*.json` artifacts lacked a `run_provenance` object and `m11` cited offset
commit SHAs (`f9f6ec056`/`68969deb0` as if they were the R19/R20 commits); (2) task20 (the
final decision record) was still unwritten. The reviewer directed: reconstruct provenance
(do **not** rerun) → then write task20.

## Work Completed (`coding`; no production-code change)

### Blocking prerequisite — AC-6/task19 provenance repair (reconstructed, no rerun)
- Verified the commit history: **`f9f6ec056`=R18, `68969deb0`=R19, `30173f08b`=R20**; R19
  and R20 commits touched **only `development/loop7/`**, so the DS/DSA production serving
  code is unchanged across R18→R19→R20. The R20 TTFT servers were launched from the R19
  tree `68969deb0` with the `--stream` probe uncommitted (committed as `30173f08b`).
- **Added `build_run_provenance(...)` to `perf_closed_batch.py`** (single schema source) and
  wired `--stream` to emit a `run_provenance` block (auto-detect git commit/dirty + GPU,
  plus pass-through `--launch-cmd`/`--op-point`/`--mem-per-gpu`/`--graph-evidence`) — future
  runs now self-document.
- **`ttft_add_provenance.py`** backfilled all **12 `ttft_*.json`** with the reconstructed
  `run_provenance` (server-code commit, tool commit, tree-dirty-during-run, GPU NVIDIA H200
  ×8/TP=8, exact launch cmd, effective config, mem fraction, gpu_mem/GPU, `mem_source`,
  graph flag + a representative `cuda graph: True` decode log line, radix/overlap flags,
  served count, op-point, artifact path), marked `reconstructed=true, reconstructed_in_round=21`
  with **metric values unchanged** (verified: p99 still 374.0 etc.).
- **Corrected `m11`'s commit story** to be exact + internally consistent, and **reconciled
  the 4K recall cell** (graph-N=50 default **80%**; eager N=20 **75%** — the prior `75%`
  conflated the two).

### Mainline — task20 final decision record (`m12_final_decision.md`)
The gate-supersession / loop-close artifact (plan M4, `refined_plan_v1.md:165-167`):
- **Supersedes** `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`'s **Tier-2.A-primary**
  ordering with the **M0 regime attribution** (4K budget-limited / 16K budget-partial ~46%
  cap / 64K scorer-limited) → **Tier-2.B is the primary long-context lever; Tier-2.A is a
  bounded opt-in 4K lever**; states exactly what changed (the prior rationale was **sound
  when written**; the oracle data is what changed).
- Cites the full **per-AC evidence chain** (AC-1 oracle closure, AC-2 N=50 recall matrix +
  CIs, AC-3 hybrid-scorer non-regression, AC-4 production-ready lifted disposition, AC-5 64K
  servability, AC-6 perf+TTFT guardrails), including the **R8 stride/oracle provenance
  explicitly** (`oracle_stride_reference.json` `emitted_stride_value_counts {"1":14640}` +
  the `selection_kernel.py::_maybe_record_recall_oracle` stride=1 call site, raw sink noted
  gitignored), and confirms the **DEC-4 close-gate** (production-ready `m9` disposition
  exists → no dangling pursued-hardening).
- Records the **Ultimate-Goal outcome**: gap rigorously characterized (M0) + materially
  partially closed (16K 6%→38% via Tier-2.B decode-free; 4K 75%→95% via opt-in Tier-2.A);
  64K residual is a characterized scorer-limited negative result → the DEC-5 learned-selector
  follow-on; DSA default + Loop-6 Tier-1 op-point non-regressed.
- **Reviewed via `/humanize:ask-codex`**: "supersession logic supported; Loop-6 change +
  prior-soundness explicit; R8 stride provenance explicit; DEC-4 satisfied." Integrated its
  **2 high-signal factual fixes**: 4K is `recall@8192=100%` (not `@4096`); the AC-2
  graph-N=50 4K default is **80%** (not 75%). Also reconciled the same cell in `m11`.

## Files Changed
- `development/loop7/m12_final_decision.md` (NEW — the loop-close decision artifact).
- `development/loop7/ttft_add_provenance.py` (NEW — reconstruction backfill).
- `development/loop7/perf_closed_batch.py` (`build_run_provenance` + `--stream` self-docs).
- `development/loop7/m11_perf_consolidation.md` (exact commit story + 4K cell reconcile).
- `development/loop7/ttft_*.json` ×12 (added `run_provenance`; metrics unchanged).
- Commit `17782726f` (local — loop hook). **No production-code change.**

## Validation
- Provenance backfilled into all 12 artifacts; metric fields byte-unchanged (spot-checked).
- `m11`/`m12` commit story matches `git log`; `git diff --check` clean; both scripts
  `py_compile` clean.
- ask-codex factual review passed after integrating its 2 corrections.
- Full DS unit suite → **350 passed + 9 subtests** (no production-code regression).
- No GPU rerun (reviewer-directed reconstruction); no servers launched this round.

## AC status after R21
- **AC-2 → MET** (`m12_final_decision.md` is the final decision artifact); **AC-6 → MET**
  (provenance complete). With AC-1/3/4/5 (prior), **ALL 6/6 ACs MET**.
- **Loop 7 is ready to close** — all plan tasks (task1–task20) are complete; the M4
  close-gate (the production-ready Tier-2.A disposition exists, DEC-4) is satisfied.

## Remaining Items
- None blocking. Queued (explicitly out of scope, before any final merge — not loop-blocking):
  remove plan/workflow markers (`AC-*`, `task*`, `Tier-2`, `DEC-`) from production
  code/comments/tests (documentation-only, no runtime effect); the DEC-5 learned/distilled
  selector for the 16K/64K scorer-limited residual is the approved-but-deferred follow-on
  (its own loop).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: applied the selected lessons — BL-20260530-durable-tracked-acceptance-evidence
  (provenance embedded as tracked JSON, not referencing the gitignored `/tmp` boot logs;
  graph/mem evidence baked into each artifact), BL-20260527-shell-json-into-python-source
  (the backfill reads/writes JSON as data via `json.load`/`json.dump`),
  BL-20260529-gate-record-artifact-before-raise / -ds-radix-flip-config-bound-artifact
  (config-bound, self-describing artifact records). No NEW reusable cross-round pitfall
  surfaced: this round was evidence-provenance hygiene + a documentation-synthesis decision
  record reusing established patterns; the conclusions are project evidence in
  `m12_final_decision.md`, not a generalizable engineering lesson.

## Goal Tracker
Updated directly (Plan Version 31): R21 plan-evolution row; task19 + task20 → Completed and
Verified (Verified = pending R21 Review); Active Tasks = none; the R8 stride-provenance
queued item marked RESOLVED-R21 (cited in `m12`); **all 6/6 ACs MET**. No Goal Tracker
Update Request needed.
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
17782726f [Sparsity] Loop-7 R21: final decision record (task20/AC-2 loop close) + TTFT evidence provenance repair
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-20-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-20-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-19-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-19-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-18-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-18-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-21-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
