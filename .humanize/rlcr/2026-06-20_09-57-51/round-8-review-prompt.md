# Code Review - Round 8

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-8-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 8 Summary

Mainline: **advance original-plan close-out** — settle the two offline-computable evidence items
(AC-2.2, AC-4 sample IDs) on a corrected, self-consistent ledger. Round 7 was ADVANCED; Codex flagged a
real metadata bug (#1 P1) + the close-out backlog. Entirely CPU this round (no server launched).

## Work Completed
- **Blocking fix — `ds_reduce_fp32` ledger metadata** (Codex #1 P1). When the arm switched to graph mode
  in R7, `build_ledger.py` still hard-coded `--disable-cuda-graph`, so the arm JSON recorded
  `cuda_graph: "off"` — contradicting the actual graph-enabled run (`serve_ds_reduce_fp32.log`:
  `disable_cuda_graph=False`, decode `cuda graph: True`) and making the single-variable arm look
  multi-variable. Fixed `extra` to match `serve.sh` (`--disable-radix-cache --enable-double-sparsity`);
  regenerated arm JSON/table/run_meta (`cuda_graph: "on (piecewise off)"`); recorded
  `ds_config={score_reduce_dtype: fp32, ...}`; added a **fail-closed consistency check** (server_args must
  not contain `--disable-cuda-graph`, cuda_graph graph-enabled, ds_config has `score_reduce_dtype=fp32`)
  — verified it asserts when the extra is re-broken.
- **AC-2.2 SETTLED — TP head-aggregation micro-test** (`ac2_2_head_agg.py` → `head_agg_tp_semantics.json`),
  offline from the validated per-rank `pre_reduce_scores` (702 8-rank groups; `sum(pre)==post` **702/702**,
  resolving the long-standing PRELIMINARY blocker). Served cross-TP **SUM** (= `reduce_token_scores`) vs
  **global-MAX** over heads: median Jaccard **0.679** (78/702 identical) → the served `head_agg="max"` +
  SUM is **not** a global max over heads (the plan's negative test). SUM vs **global-MEAN**: Jaccard 1.0
  (scale-only). **Exoneration:** `build_absorbed_projection` uses `num_local_heads` and the reference path
  does NO cross-TP reduce (verified — no `reduce_token_scores`/all-reduce in `_reference_selector_topk`),
  so production (SUM) and the reference (per-rank-local) use *different* head aggregation — yet cosine
  recovers under both and raw-dot collapses under both (production-SUM 0.000 ≈ reference-local 0.013). So
  cross-TP head aggregation is **not** the accuracy driver (consistent with AC-6).
- **AC-4 sample IDs/order** (`ac4_sample_ids.py` → `gsm8k_sample_ids.json`). The stock
  `simple_eval_gsm8k` loader is deterministic (no seed/shuffle) → re-derived the exact ordered eval
  slices (dense `lines[5:205]`, sparse `lines[24:174]`) with per-example `(line, question sha16)` +
  `test.jsonl` sha256; all arms share the identical set. Wired into the ledger
  (`gsm8k.sample_ids_artifact`); removed sample IDs from `fields_not_instrumented` (the prior "seed-42
  slice" note was wrong — selection is deterministic). Only garbage counters remain not-instrumented.

## Files Changed (committed `752752f6d`)
- NEW: `development/loop13/ac2_2_head_agg.py`, `ac4_sample_ids.py`,
  `evidence/head_agg_tp_semantics.json`, `evidence/gsm8k_sample_ids.json`.
- MODIFIED: `build_ledger.py` (ds_reduce_fp32 extra + ds_config + consistency check + sample_ids wiring +
  footer text), `evidence/findings.md` (AC-2.2 section), `evidence/cheap_controls.json` (AC-2.2 status),
  `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `build_ledger.py` → provenance consistent (blob `0d914406af8b`); ds_reduce_fp32 arm `cuda_graph: "on
  (piecewise off)"`, no `--disable-cuda-graph`, `ds_config` records `score_reduce_dtype=fp32`; the
  consistency check **asserts (exit 1)** when the extra is re-broken.
- `ac2_2_head_agg.py` → 702 groups, `sum(pre)==post` 702/702, SUM-vs-MAX median Jaccard 0.679, exit 0.
- `ac4_sample_ids.py` → test.jsonl sha, dense [5:205]/200, sparse [24:174]/150, exit 0.
- Full suite re-run: `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac6_bisection_matrix` — all exit 0.
- No `.pt`/`.humanize` committed. No selection/adapter fix landed. CPU-only (GPUs idle throughout).

## Remaining Items (for AC-8 COMPLETE)
- **AC-2.1** forced-all physical-slot assertions (`forced_all_assertions.json`) — needs guarded
  instrumentation of the `logical_to_physical`→`transform_index_page_table_decode` adapter (dump physical
  slots + `req_to_token`, assert no dup/`-1`/unwritten/out-of-range, adapter errors 0). GPU + capture.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; run the NIAH dense/sparse oracle as corroboration. GPU.
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — needs the resident latent
  captured alongside scores, then offline materialize + compare top-2048. GPU capture + offline.
- **AC-4** length-cap garbage counters — same adapter instrumentation as AC-2.1.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-ledger-metadata-tracks-serve-mode
- Notes: Added a lesson that a ledger/evidence generator which HAND-CODES per-arm server args / cuda_graph
  state separately from the serve script will drift when the serve mode changes (here eager→graph), silently
  contradicting the actual run and making a single-variable arm look multi-variable. Fix: match the
  generator to serve.sh, record the actual config knobs, and add a fail-closed check tying the recorded
  args/graph-state to the serve mode — verified to actually fire when re-broken.

## Goal Tracker Update Request

### Requested Changes:
- Close **R7-review blocking: ds_reduce_fp32 wrong CUDA-graph metadata** — fixed + guarded (cuda_graph
  graph-enabled, ds_config records fp32, fail-closed check verified).
- Mark **AC-2.2 (task3) DONE** — `head_agg_tp_semantics.json`: served SUM ≠ global-max (Jaccard 0.679),
  exonerated as the bottleneck; the PRELIMINARY `sum==post` blocker resolved (702/702).
- Mark **AC-4 sample IDs/order done** (task9 partial→advanced) — `gsm8k_sample_ids.json` wired into the
  ledger; only garbage counters remain not-instrumented.
- Plan Evolution Round-8 row added.

### Justification:
The metadata bug was a genuine AC-1/AC-4/AC-6 integrity defect (now fixed and guarded against recurrence).
AC-2.2 was settleable offline once the captures were validated (`sum(pre)==post` 702/702), and the
exoneration uses verified code facts (`num_local_heads`, no reference cross-TP reduce) + the measured GSM8K
numbers. AC-4 sample IDs are deterministic and thus exactly re-derivable. The remaining close-out items
(AC-2.1/2.4/3.1/4-garbage/8) each require GPU capture or adapter instrumentation and are the explicit
next sequence toward AC-8 COMPLETE.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
ac479aeb3 [loop13] Round 2: per-arm evidence ledger, baseline consistency, captures, cleanup
29ed825fa [loop13] Round 3: ledger SHA provenance, capture row-identity, exact-join analyzer
393966c02 [loop13] Round 4: AC-2.3 RESOLVED on real captured rows; fail-closed analyzer; ledger provenance
c7b66f04b [loop13] Round 5 (drift recovery): pruning-valid AC-2.3 + AC-6 scorer×current-slot 2×2
8b55dfba3 [loop13] Round 6: AC-6 corroboration + complete per-leg bisection matrix
8281361e7 [loop13] Round 7: measure the score-reduce leg + dense current-slot corroboration
4d874b89e [loop13] gitignore transient DS capture scratch dirs
752752f6d [loop13] Round 8: fix ds_reduce_fp32 metadata; settle AC-2.2 + AC-4 sample IDs
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-7-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-7-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-6-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-6-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-5-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-5-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-8-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
