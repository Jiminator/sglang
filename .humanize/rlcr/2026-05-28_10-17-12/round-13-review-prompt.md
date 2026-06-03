# Code Review - Round 13

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-13-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 13 Summary

## Context
Codex's Round-12 review converged the loop: **0 active plan tasks, 0 mainline gaps, 0 blocking
issues**; the sole unmet criterion is **AC-12** (a hard fail), which is an inherent DS limit with no
DS-preserving fix. The mechanical COMPLETE condition is therefore unsatisfiable by autonomous code
work, so I **escalated the disposition to the user** (AskUserQuestion). The user directed a close-out:
do the queued cleanups first, then test whether a significantly larger DS selection budget passes
NIAH ("if not, then there is a serious issue"), then document the remaining work for the next loop.

## Work completed (user-directed)

### 1. Queued cleanups
- **Comparator per-side `mem_fraction_static` hole** (Codex queued #1): `_validate_per_side_agreement`
  now requires `mem_fraction_static` **constant within each side** while keeping it ignored
  **across** sides (the sanctioned DSA 0.85 vs DS 0.6 TokenLabelTable asymmetry). New regression
  `test_per_side_mem_fraction_drift_refused` (per-side drift → exit 2; constant-per-side cross-side
  asymmetry → proceeds).
- **Stale `calibrate.py` recipe docstring** (Codex queued #3): corrected to the validated command in
  `calibration_provenance.md` (`--tp 8` + local `--dataset` corpus + `-v`) and made accurate —
  `--dtype` is a recorded forward-stability hint (load uses `torch_dtype="auto"`), `--tp` is recorded
  metadata (`device_map="auto"` shards across visible GPUs; no distributed group spawned).

### 2. NIAH selection-budget investigation — answers the user's question
**The DS selection budget cannot be significantly increased**, and the AC-12 NIAH failure is a
**selection-quality** limit, not a decode bug:

- **Budget is kernel-locked to the model's DSA `index_topk=2048`.** Booting DS with `top_k=8192`
  fails twice (`ac12_topk_sweep/boot_evidence_topk_locked.txt`): the validator refuses
  (`top_k != index_topk`), and with `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` the **`flashmla_kv` decode
  kernel itself** asserts `indices.shape[-1] == self.dsa_index_topk` during CUDA-graph capture. DS
  reuses V3.2's native DSA decode kernel, which consumes exactly 2048 indices — so the budget can't
  be widened on this path. (Architectural constraint, not a bug.)
- **No serious decode bug.** A DS-only NIAH curve at `top_k=2048`
  (`ac12_topk_sweep/ds_recall_vs_length_topk2048.json`): recall **100%** at 1024 words (dense,
  budget ≥ seq), **100%** at 1536 (dense), **75%** at 4K (~50% selected), **5%** at 16K (~12.5%,
  from AC-12). DS recalls perfectly when its selection is complete (= a dense model), and recall
  tracks the selected fraction. Combined with MMLU == DSA (short context), DS decode is sound.
- **The DS-vs-DSA gap at the SAME 2048 budget is selection quality.** DSA recalls 100% at 16K with
  the same kernel + budget because its *trained* DSA indexer places the needle in its 2048; DS's
  *offline channel-mask* heuristic does not. On a model with a native trained sparse indexer, DS is
  capped at the native budget and selects worse, so it cannot match DSA's long-context recall, and a
  larger `top_k` is not an available lever. (`ac12_topk_sweep/analysis.md`.)

### 3. Next-loop issue list
`runs/20260528_dsv32_mvp/next_loop_issues.md`: the AC-12 disposition decision (accept smoke
milestone vs re-scope vs R&D), DS selector R&D (query-aware/learned indexer; a kernel accepting
`top_k > index_topk`), DS KV-budget/TokenLabelTable for 64K admission + AC-11 TTFT, the strategic
"is DS worthwhile on a model with native DSA?" question, and the cosmetic serve-header terms.

## Files changed
- `development/benchmark_compare.py` — per-side `mem_fraction_static` check.
- `test/registered/unit/development/test_ac11_comparator.py` — `test_per_side_mem_fraction_drift_refused`.
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — recipe docstring.
- `runs/20260528_dsv32_mvp/ac12_topk_sweep/` (`analysis.md`, `ds_recall_vs_length_topk2048.json`,
  `boot_evidence_topk_locked.txt`), `runs/20260528_dsv32_mvp/next_loop_issues.md`.
- Commits `ced03f374` (cleanups), `27434cee7` (investigation + docs). Both pushed.

## Validation
- **409 CPU tests pass** (the five-file suite; +1 comparator per-side mem-fraction regression).
- Hardware: confirmed the `top_k`-budget kernel lock (two boot failures with evidence) and ran the
  DS-only NIAH recall curve at `top_k=2048`. DS server shut down; node-0 GPUs freed (0 MiB);
  the pre-existing router was not touched.

## Remaining Items
- **AC-12 remains a recorded HARD FAIL** — now precisely characterized (selection-quality +
  kernel-budget-cap; no DS-preserving fix). The loop's COMPLETE condition stays unsatisfiable; the
  disposition (accept smoke milestone / re-scope / R&D) is the user's carried-over decision,
  documented in `next_loop_issues.md`.
- Queued (still non-blocking): DS selector/KV-budget R&D, AC-11 TTFT follow-up, cosmetic
  serve-header terms — all listed for the next loop.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260529-ds-longcontext-needle-recall-vs-topk
- Notes: Added a Round-13 addendum: the DS selection budget is **kernel-locked** to the model's DSA
  `index_topk=2048` (validator refusal + `flashmla_kv` `indices.shape[-1] == dsa_index_topk`
  assertion), so `top_k` cannot be raised on this backend; a DS-only recall curve proves decode is
  sound (dense ≤2048 → 100%, tracking the selected fraction); and the DS-vs-DSA gap at the same 2048
  budget is **selection quality** (offline channel-mask vs the trained DSA indexer), not budget size
  or a decode bug. This sharpens how to diagnose an AC-12 NIAH failure and bounds what can fix it.

## Goal Tracker Update Request

### Requested Changes:
- Confirm the **comparator per-side `mem_fraction_static`** and **`calibrate.py` recipe docstring**
  queued items RESOLVED (commit `ced03f374`, +1 regression, 409 CPU tests pass).
- Record the **AC-12 selection-budget investigation** outcome: budget kernel-locked to
  `index_topk=2048`; DS decode sound (dense=100%); AC-12 gap is selection quality vs the native DSA
  indexer — no DS-preserving fix and no raisable `top_k` on this backend (commit `27434cee7`).
- Keep **AC-12 as NOT met** (hard fail) and the loop4-compatible MVP incomplete; the disposition is
  the user's carried-over decision (see `next_loop_issues.md`). No immutable AC/threshold changed.

### Justification:
Per the user's Round-13 direction, this round cleared the two queued cleanups and answered the
selection-budget question empirically: DS cannot widen `top_k` (shared DSA decode kernel asserts the
2048 budget), and the AC-12 NIAH gap is a selection-quality limit (DS's offline channel-mask vs
V3.2's trained DSA indexer at the same budget), with DS decode proven sound (dense recall 100%). No
"serious issue" (decode/serving bug) exists — the limitation is architectural and inherent to running
Double Sparsity on a model that already has a superior native sparse indexer. AC-12 was not faked
green and no immutable AC was changed; the disposition is documented for the next loop.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
eb914678e [Sparsity] Loop-5: refined plan v1 + QA ledger
8979848ab [Sparsity] Loop-5: untrack active RLCR plan file
4f4c620df [Sparsity] Thread forward_batch into _write_token_labels (radix capture producer fix)
7cbbce088 [Sparsity] Calibration: native-FP8 sharded load + one-block dry-run mode
c99ed3644 [Sparsity] Calibration: load DeepSeek-V3.2 via deepseek_v3 remap + fail-closed dry-run
610f364c9 [Sparsity] Loop-5: V3.2 channel-mask calibration evidence (AC-4 complete)
df8d7c6c6 [Sparsity] Untrack .humanize/bitlesson.md (loop state, per .gitignore)
34b243b07 [Sparsity] Fix the DS serving path so DeepSeek-V3.2 boots on hardware
44a12d5d1 [Sparsity] Loop-5: round-2 DS boot evidence (AC-1 knobs + /generate probe)
610b65c15 [Sparsity] Loop-5: localize DS decode degeneration (DS-specific, selection over-count)
05a25f197 [Sparsity] Loop-5: refine decode diagnosis (eager scorer masks seq_len; instrument inputs in round 3)
2af5f4e65 [Sparsity] Fix DS decode selecting wrong domain: resolve req_to_token via ForwardContext
d9ad3066f [Sparsity] Loop-5: decode-degeneration is two bugs (req_to_token fixed; decode label-write open)
6429cf539 [Sparsity] Loop-5: complete bug #2 root cause (decode passes pre-projected k_nope, not latent)
8375b76a5 [Sparsity] Fix DS decode degeneration: label decode tokens (attn_mqa kv_b_proj + robust head_width)
b231942fa [Sparsity] Loop-5: DS genuine-sparse path OOB when seq_len>top_k (#18 finding)
da1ff651e [Sparsity] Loop-5: #18 deeper root cause — DS prefill selection bad req_pool_indices (long-prompt OOB)
802b51b84 [Sparsity] Loop-5: confirm #18 mechanism — DS selection uses decode batch shape, breaks on prefill per-token batch
ffe6c2b97 [Sparsity] Loop-5: critical review of loop4 DS scaffolding + pre-cutover loop5 fixes
eba4c640e [Sparsity] DS dense-prefill / sparse-decode: fix long-prompt OOB + unblock AC-1.1
590b0dc05 [Sparsity] Loop-5: extend code review to loops 1-3 foundational DS modules
3f9478128 [Sparsity] Loop-5: mark #18 resolved in review doc (dense-prefill fix)
8e9138af6 [Sparsity] Make radix fixture capture CUDA-graph-safe (no host copies during capture)
6f95a9711 [Sparsity] AC-0: radix-capture publish resolves req_to_token via backend/ForwardContext; dtype-safe SHA
bc534da7c [Sparsity] Fix /get_server_info crash (DS stashes tensors on server_args) + AC-0/AC-1 evidence
76eef9c80 [Sparsity] AC-1 negative test: invalid channel-mask path -> fail-closed validator rejection
6acdfb94f [Sparsity] Launcher parity: default MODEL_PATH to cluster weights; add DSA radix-off smoke knob
f2bc1eb6a [Sparsity] Make the TIER-1 smoke benchmark actually runnable on V3.2 FP8
2220a793f [Sparsity] TIER-1 smoke benchmark pair + comparator (AC-8/AC-9), radix-off both sides
99ac93691 [Sparsity] AC-Q quality smoke: single-node sequential capture/compare (#G)
d8fce372a [Sparsity] AC-Q evidence: single-node sequential quality smoke (3/4 gates; ROUGE-L miss analyzed)
bac3aaff6 [Sparsity] Quality smoke: generate via /v1/chat/completions (raw /generate is degenerate)
70bb52a15 [Sparsity] Diagnose AC-Q decode failure (#H): greedy degeneration, not a DS bug; harden ref validation (#I)
7861ca1d4 [Sparsity] AC-Q #H: reviewable DS-selection metadata proves no selection bug (greedy fragility)
85974608e [Sparsity] AC-Q: concise-answer measurement (user-approved) so the smoke tests answers, not greedy CoT
b0e43294c [Sparsity] AC-Q PASSES (all 4 gates) under user-approved concise measurement + first-8 prefix-overlap fix
d47dcbadb [Sparsity] Fix #J: first-8 overlap false-pass — alnum-subtoken normalization (not string prefix)
fa4473694 [Sparsity] AC-10 (DEC-5): no-env-override radix flip via a config-bound fixture state file
67422e698 [Sparsity] AC-10 MET on 8x H200: both radix fixtures pass; DS boots radix-on via artifact (no env)
0cb6b597b [Sparsity] gitignore development/results/ (benchmark + fixture runtime scratch outputs)
e7951a59d [Sparsity] Fix #K: update Option-B launcher-contract tests to the evolved radix contract; drop plan markers from new code
461119b46 [Sparsity] AC-1b chunked-prefill probe PASSES at the radix-on operating point
a24bc469c [Sparsity] AC-11 directional sweep (radix-on, 3-trial) + #F effective-concurrency accounting
7478c27a0 [Sparsity] Add HOST knob to Option-B launchers for cross-node AC-12
1a1293f01 [Sparsity] AC-12 full quality gate executed: MMLU pass, NIAH hard-fail (task14+task15)
d2f48bbd4 [Sparsity] Make AC-12 NIAH gate artifact-safe on server rejection (#L)
cc50bae38 [Sparsity] AC-12 64K durable artifact (#L) + analysis/bundle update
ced03f374 [Sparsity] Round-13 queued cleanups: comparator per-side mem-fraction check + calibrate recipe docstring
27434cee7 [Sparsity] Round-13 NIAH selection-budget investigation + next-loop issue list
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-12-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-12-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-11-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-11-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-10-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-10-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-13-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
