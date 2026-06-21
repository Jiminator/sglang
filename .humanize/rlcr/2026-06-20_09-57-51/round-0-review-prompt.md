# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 0 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Verdict (the deliverable)
**Two-part root cause, with live GSM8K evidence on GLM-5.1-FP8 (8×H200 TP=8, temp 0, completion API). No fix landed (diagnosis loop).**

1. **DENSE 0.620 → H3 (downstream-of-selection slot-validity bug): the current decode slot is excluded from its own attention.** `_select_topk_indices` invalidates the current slot in the `_slot_written` bitmap before scoring (`_slot_written[layer_id, out_cache_loc] = False`) and only restores it after the KV write, so the current token scores −∞ and is dropped from its selected set (DS keeps 715/716 in dense). Each decode token cannot attend to itself.
   - **Airtight isolation:** force-including ONLY the current slot (anchor-recency b=1) recovers dense **0.620 → 0.970** (≈ DSA 0.975). b=64 → 0.960, forced-all → 0.950. One token is the entire dense gap.
   - **Scorer/perf-opts exonerated:** the fp32-EXACT reference selector (no fp8-in-register, no bf16 reduce, no radix approx, no width bucketing) scores the SAME 0.620 dense → rules out H1. Recovery to ≈DSA with the current slot rules out H0/H2 for dense.

2. **SPARSE 0.000 → a distinct secondary failure (H0/H2 family), confounded by H3.** With real pruning (selected 2048 of ~5600), the collapse persists with the current slot included (anchor b=1/b=64 sparse 0.000/0.007) and with the fp32-exact scorer (reference sparse 0.000). The channel-importance top-2048 doesn't capture the long-context tokens DSA's learned indexer does — but this is confounded by the H3 current-slot bug on every decode step and can only be cleanly characterized after the H3 fix.

**Recommendation:** fix H3 first (force-include the current decode slot / restore `_slot_written` before the selected set is consumed — small, localized), then re-measure the sparse selection ceiling. No selection/adapter fix landed this loop.

## Evidence (per-arm GSM8K)
| Arm | Dense | Sparse |
|---|---|---|
| DSA (native) | 0.975 | 0.953 |
| DSA `--disable-radix-cache` | 0.960 | 0.940 |
| production DS | 0.620 | 0.000 |
| fp32-exact reference (raw-dot) | 0.620 | 0.000 |
| forced-all dense (incl current) | 0.950 | n/a |
| anchor-recency b=64 | 0.960 | 0.007 |
| anchor-recency b=1 (current only) | 0.970 | 0.000 |

Full writeup: `development/loop13/ROOT_CAUSE.md`. Table: `evidence/evidence_table.md`. Notes: `evidence/findings.md`. Codex review: `evidence/codex_review_h3.md`.

## Files Changed
Diagnostic code (config-gated; production behavior byte-identical when fields unset):
- `python/sglang/srt/layers/attention/double_sparsity/config.py` — `selector_impl` + `forced_all_dense_control` fields, validation, parse wiring.
- `python/sglang/srt/layers/attention/double_sparsity/absorbed_latent.py` — `dequantize_resident_latent`, `absorbed_latent_score_logical_fp8` (gather-then-dequant), `reference_rawdot_select`, `apply_forced_all_dense`.
- `python/sglang/srt/models/deepseek_v2.py` — `_reference_selector_topk` (fp32 reference at the `_select_topk_indices` seam, gated to decode-with-state 3-D cuda queries, mask indexed per-layer); forced-all override; one-time reference-active log.

Harness + evidence (`development/loop13/`):
- `serve.sh` modes `dsa_noradix`/`ds_capture`/`ref`/`ds_forced_all`/`ds_anchor`; `run_gsm8k.sh` `THREADS`+`REGIME` knobs; `analyze_captures.py`.
- `ROOT_CAUSE.md`, `evidence/evidence_table.md`, `evidence/findings.md`, `evidence/codex_review_h3.md`, `evidence/meta/{run_meta.json,ds_instruments.md}` (server logs gitignored).

## Validation
- CPU unit tests (passed): `dequantize_resident_latent` round-trip; `apply_forced_all_dense` (dense→[0..s-1], sparse untouched); `reference_rawdot_select` top-k == `torch.topk`; `absorbed_latent_score_logical_fp8` == full-pool dequant EXACTLY on finite scores; config validation (defaults / new fields / bad-enum rejection).
- Live: all arms boot from the dev clone (`_env.sh` guard passed), DS genuinely active by regime (sparse selected<total dense_fallback==0; dense selected==seq_len under forced-all). Regression reproduced (gate sound).

## Remaining Items (explicitly deferred — see goal-tracker Plan Evolution Log + Explicitly Deferred)
- AC-2.2/2.3/2.4 (TP head-agg, radix/width index-equivalence micro-tests, recall-oracle): SUPERSEDED — the fp32-exact reference exonerates the whole scorer; recall-oracle is NIAH-only. `analyze_captures.py` + `ds_capture` mode built and ready.
- AC-3.2 (served cosine): deprioritized — a different scorer can't recover while H3 corrupts the downstream feed.
- AC-7.1/7.2/7.3 (no-mask ablation, full knob sweep, per-head oracle): deferred — the BAD sparse ceiling is confounded by H3; the verdict does NOT assert a clean H0. Partial: anchor-recency b=1/b=64 sweep run for the H3/sparse separation.
- AC-6 (GOOD-branch bisection): N/A — gate is BAD; cause is H3, not a perf-opt regression.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260620-ds-current-slot-exclusion
Notes: Captured (a) the root cause — DS drops the current decode slot via the `_slot_written` invalidation so each decode token can't attend to itself (dense 0.62, sparse 0.00) — and (b) the reusable diagnostic method: a fp32-EXACT reference selector exonerates the entire scorer in one shot; forced-all / anchor-recency-budget-sweep isolates downstream-of-selection and pins it to a single token (current slot, b=1→0.970); gather-then-dequant keeps the eager reference tractable; the sparse collapse stays confounded until the H3 fix.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
4ceba0ead [loop11b] R1: queue checkpoint — bench evidence + cleanup done, mb_v2 clean re-run running
86ddf6faf [loop11b] R1: fix stale a4be98c4 capacity claim (Codex gap 5) — note ld32 504640 reconfirm
1a29be00d [loop11b] R1: fix DS error-abort crash — check_finished was renamed upstream (#25725)
99ac584ac [loop11b] R1: document DS crash finding + selector reuse-edge; mb_v2 emits selector-error count
8fbe848ed [loop11b] R1: M-B verdict re-established clean — comparators ACCEPT both op-points
811c40420 [loop11b] R1: AC-5 no-op proof (dense_fallback=0 + structural sparsity) + GLM meta_info gap doc
9d2c4253d [loop11b] R1: headline M-B verdict + AC-4 dedicated per-step tax (both PASS)
f1b90c797 [loop11b] R1: AC-8 close-out — results.md + queue.md regenerated to the R1 publishable state
44310f230 [loop11b] R1: complete evidence package — DSA server_info + crash-probe txt + crash-log hashes
c16c0d202 [loop11b] R2: wire GLM/dsa-backend DS per-request summary (AC-5) — host-side, graph-robust
b5c4d72be [loop11b] R2: verdict re-established + AC-5 PASS + raw evidence committed (lossless)
8062039d8 [loop11b] R2: AC-8 ledgers regenerated to final state + push status; de-AC the new backend comment
df18a93d0 [loop11b] R3: fix total_tokens metric semantics (AC-5) — explicit field, not rate-inverse
96202e4c4 [loop11b] R3: corrected verdict evidence (results_r3) + supersede results_r2
2ce2adf4e [loop11b] R3: ledgers to one current state (AC-8) — results_r3, mask=regenerated, close-out ACTIVE-until-push
e0935e5a9 [loop11b] R3: AC-8 close-out COMPLETE — pushed to owner fork Jiminator/sglang
da12616a5 [loop11b] R3 review fix [P3]: build_corpus.py creates the output dir before writing
101926d76 [loop11b] R3 review fixes [P2 x2]: report verdict vs exit consistency + fail-closed partial DS evidence
9ab62e6ad [loop11b] R3 review fixes: DS abort test rename + comparator report verdict/labels + green test suite
3058bdc35 [loop12] add gen-plan output + draft; record pensieve doctor state
aaefdaf1e [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
8f88e1aef [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
4706b2138 [loop12] document double-sparsity v2 performance numbers
480fd70ba [loop12] add double-sparsity v2 run-and-evaluate runbook
2babc5afa [loop12] add gsm8k evidence + refresh perf verdict numbers
180f6dd6d [loop13] add diagnosis-loop plan, draft, and guarded harness
fc6ac20a7 [loop13] diagnostic DS selectors: fp32 raw-dot reference + forced-all dense control
29ec137bf [loop13] harness: ref/ds_capture/ds_forced_all serve modes + AC-1 baseline evidence
16caf4f5b [loop13] reference selector: gather-then-dequant (full-pool dequant was intractable)
5a1da871d [loop13] verdict: DS dense degradation = current-decode-slot exclusion (H3)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
