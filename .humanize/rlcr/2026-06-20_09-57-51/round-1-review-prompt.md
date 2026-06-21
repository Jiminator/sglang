# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 1 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Headline: the verdict FLIPPED — the ceiling is GOOD
Round-0 left a confounded "sparse = H0/H2" wording. Codex's Round-0 review correctly demanded a
served cosine arm + a faithful, leak-free ceiling. Building those overturned the conclusion:

| arm (faithful: current slot incl, TF32 off, exact fp32) | dense | sparse |
|---|---|---|
| raw-dot | 0.950 | 0.013 |
| **cosine** | 0.940 | **0.940** |
| DSA | 0.975 | 0.953 |

**The cosine scorer recovers sparse 0.013 → 0.940 ≈ DSA.** AC-5 gate recomputed from
best(raw,cosine): dense 0.950 (2.5 pp), sparse 0.940 (1.3 pp) → **GOOD**. The channel-importance
algorithm DOES transfer to GLM-5.1 MLA. The production collapse is **two regressions**:
1. **Dense (0.620) = H3**: the current decode slot is excluded from its own attention (`_slot_written`). Including it → 0.950/0.970. Cost ~33 pp.
2. **Sparse (0.000) = the `scorer_norm="off"` raw-dot lock**: the Loop-11 table-free rewrite (`01e3ff238`, deletes `TokenLabelTable`) dropped the Loop-7 cosine scorer. Single-variable: raw-dot sparse 0.013 vs cosine 0.940. Cost ~92.7 pp.

NOT H0 (cosine transfers) and NOT H2 (same mask reaches ≈DSA under cosine) → the BAD-branch
no-mask ablation (AC-7) is moot; the GOOD branch (AC-6 bisection) is the taken branch and is answered.

## Mainline objective (round-1-contract.md): DONE
Make the reference a valid AC-5 gate input (cosine + faithful + leak-free) and recompute the gate.

## Work Completed (diagnostic code; production unchanged when flags unset)
- `reference_cosine` selector: cosine on a materialized per-head signature
  (`|K_label_h|=||absorbed_w_sel[h]@c_kv||`, `|Q_label_h|=||w_c⊙q_{S_h}||`, normalize after gather).
  A `normalize=False` mode gives the materialized-raw single-variable control. (AC-3.2)
- `reference_include_current` config flag: force-include the current decode slot → H3-clean ceiling
  (dense reports `selected==seq_len`). (AC-3.3)
- TF32 disabled (`allow_tf32=False` + cuDNN) in the reference path → leak-free fp32. (AC-3.4)
- `serve.sh` modes `ref_faithful` and `ref_cosine`; per-arm metadata JSON under `evidence/meta/arms/`.

ACs addressed: **AC-3.2** (cosine served, dense 0.940/sparse 0.940, DS active by regime),
**AC-3.3** (faithful dense `selected==714==seq_len`; sparse `selected 2048<5610`, `dense_fallback 0`),
**AC-3.4** (TF32 disabled), **AC-5** (gate GOOD, valid best-of), **AC-6** (two culprits single-variable,
costs + responsible change; cosine-vs-rawdot sparse delta 141 vs 2 of 150 is unambiguous).

## Files Changed
- Code: `config.py`, `absorbed_latent.py`, `deepseek_v2.py` (cosine, include_current, TF32-off, normalize control).
- Harness/tests: `serve.sh` (ref_faithful/ref_cosine), `development/loop13/test_reference_selectors.py`.
- Evidence/writeup: `ROOT_CAUSE.md` (rewritten), `evidence/{gate_ac5.md, evidence_table.md, findings.md, codex_review_gate.md, meta/arms/*.json}`.
Two atomic commits (`fea920c06`, `62ad64346`); tree clean; one TP=8 server at a time; GPUs idle.

## Validation
- CPU: `python3 development/loop13/test_reference_selectors.py` → 5/5 pass, including the decisive
  `test_materialized_raw_equals_absorbed_raw` (cosine path `normalize=False` is selection-equal to the
  absorbed raw-dot, max |Δ| 4.8e-6, bit-identical top-k) — proving NORMALIZATION is the sole
  cosine-vs-rawdot variable (Codex Round-1 MUST_DO #1, addressed offline → conclusive).
- Live: ref_faithful + ref_cosine boot from the dev clone (guard passed), DS active by regime,
  0 selector errors; cosine sparse 0.940 verified with selected 2048<5610, dense_fallback 0,
  impl=reference_cosine confirmed.
- Codex gate verification: `evidence/codex_review_gate.md` — GOOD gate arithmetically valid, cosine
  norm formula correct, no-mask ablation no longer required.

## Remaining Items (queued / next round — not verdict-changing)
- AC-2 capture artifacts (`ds_capture`, `cheap_controls.json`, forced-all physical-slot-assertion JSON) + the full per-arm AC-4 ledger with length-cap garbage-rate columns.
- Production-style cosine deployability control (Codex MUST_DO #2) — a FIX-loop concern.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260620-ds-rawdot-scorer-lock
- Notes: Captured that the DS sparse collapse is the `scorer_norm="off"` raw-dot lock (the table-free
  rewrite dropped the Loop-7 cosine scorer), recoverable by a cosine scorer whose per-head signature
  norms are computable from the bind-time absorbed projection (no TokenLabelTable rebuild needed); and
  the diagnostic method (toggleable raw-dot/cosine reference + materialized-raw selection-equality
  proof) that turns "does the algorithm transfer?" into a one-variable GSM8K experiment and makes the
  no-mask ablation moot once best-of(raw,cosine) is GOOD.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
fea920c06 [loop13] reference selector: served cosine + faithful/leak-free ceiling
62ad64346 [loop13] Round 1 verdict FLIP: ceiling GOOD; sparse = raw-dot scorer_norm=off lock
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-0-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-0-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
