# Code Review - Round 18

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-18-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 18 Summary

Mainline: **AC-2.4 NIAH recall-oracle@2048 corroboration** for the production DS scorer (Codex required-plan
item #1). Diagnostic/guarded instrumentation only; no selection/adapter fix.

## Feasibility / approach
The recall-oracle is fully wired: the config-borne `recall_oracle` flag latches the cross-process sink on
the TP workers (`oracle_artifact_sink.py`), and the selector hook records the needle's score rank +
`recall_at_k` per (request,trial,layer,step) (`selection_kernel.py:1210` → `selection_recall_oracle.py`).
The loop7 driver is unusable (its `test/manual` helper was deleted + it targets DeepSeek-V3.2), so I wrote
a self-contained GLM driver. The design is FAIL-CLOSED and self-verifying: a wrong needle span makes the
server emit a `span_out_of_range` hard-failure marker, so an incorrect offline tokenization fails LOUD, not
silent.

## Work Completed
1. **`serve.sh ds_recall_oracle`** — production DS config + `recall_oracle:true`, eager (the host-side
   oracle record is illegal under graph capture). Also added `ds_reduce_fp32` + `ds_recall_oracle` to the
   mode-error string (R17-review queued nit).
2. **`niah_recall_oracle.py`** (new, self-contained) — GLM tokenizer; filler + unique magic-number needle
   near the middle + a recall question; needle span via raw-prompt offset mapping (`add_special_tokens=
   False`). An **alignment probe** measures the server-vs-offline token delta (a BOS prefix would shift all
   logical positions) on a representative prompt of each regime and asserts it is one consistent small
   offset, then shifts every span by it — **measured delta = 0** (GLM adds no BOS, so the offline span
   matches the server KV domain exactly). Per trial: `set_active_trial` → `/generate` (`ignore_eos`, a few
   decode steps) → `clear_active_trial`. After the sweep it reads the sink, **fails closed** on any missing
   trial record or `span_out_of_range`/`exception` marker, and reduces to per-regime recall@2048.
3. **GPU** — one TP=8 server (`ds_recall_oracle`, eager), launched with cwd=`evidence/` so the TP worker's
   oracle-dir default (`cwd/.sglang_ds_oracle`) matches the driver's `--oracle-dir` (env does NOT reach TP
   workers). 8 dense + 8 sparse trials. Torn down to 0 MiB.
4. **`build_ledger.py`** — `validate_recall_oracle_artifact()` fail-closes (both regimes present, non-zero
   records, `corroboration_only` label) before recording the per-regime summary in `run_meta.json`.
   `findings.md` + the evidence-table footer record the result.

## Result (`evidence/ac2_4_recall_oracle.json`, CORROBORATION ONLY — not exoneration)
Fail-closed checks passed: 8/8 trials produced records in BOTH regimes, 0 `span_out_of_range`, 0
`exception`, token delta 0.

| regime | prompt tok | recall@2048 | needle_worst_rank (min/median/max) | selected_contains_needle |
|---|---|---|---|---|
| dense  | ~1136 (< top_k) | **1.0**    | 54 / 777 / 1139   | 1.0 |
| sparse | ~4310 (> top_k) | **0.4103** | 72 / **2524** / 4313 | 0.4103 |

- **Dense** selects every token (recall trivially 1.0; needle always kept) → the dense regression is NOT a
  scorer-ranking failure; it is the H3 current-slot exclusion (AC-2.1/AC-4), independently corroborated.
- **Sparse**: the production raw-dot scorer (`scorer_norm=off`) ranks the needle inside the 2048 budget only
  **~41%** of the time (median worst-rank 2524 > 2048), so the needle is pruned out > half the time. This
  corroborates that the sparse collapse is **scorer-driven** (the raw-dot lock) — consistent with the sparse
  0.000 GSM8K and the cosine recovery. `selected_contains_needle_rate == recall@2048` in both regimes (the
  AC-1 oracle invariant holds — internal consistency check).

## Files Changed (committed `4a16c082a`)
- `development/loop13/serve.sh` (+`ds_recall_oracle` mode, mode-error string), `development/loop13/niah_recall_oracle.py`
  (new), `development/loop13/build_ledger.py` (`validate_recall_oracle_artifact` + run_meta wiring),
  `development/loop13/evidence/ac2_4_recall_oracle.json` (new), `development/loop13/evidence/findings.md`
  (AC-2.4 section), `evidence/evidence_table.md` + `evidence/meta/*` (regenerated), `.gitignore`
  (+`.sglang_ds_oracle/`).

## Validation
- CPU suite, explicit args: `ac4_garbage_counters` (production + ref_faithful + ref_cosine),
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent; verified `validate_recall_oracle_artifact()` ABORTS ledger
  generation on an empty-regime artifact, then restored.
- The driver is itself fail-closed (exit 2 on missing record / hard failure / inconsistent token delta).
- One TP=8 server, eager, torn down to 0 MiB. No sink/`.pt`/`.humanize` raw artifacts committed. No
  selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured decode-row materialized fp32 `K_label` selected-index equality (the committed
  `ac3_1_materialized_k.json` is a synthetic CPU proof; the plan wants it on captured rows).
- **AC-4** serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial — the
  reference eager serial runs are very slow) + selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-niah-recall-oracle-fail-closed-span-self-verify
- Notes: To reuse a model-specific NIAH recall-oracle on a NEW model (DeepSeek-V3.2 driver → GLM-5.1-FP8),
  the server-side oracle (rank/recall computation) is model-agnostic; only the driver (tokenizer + NIAH
  prompt + needle-span) is model-specific. Two correctness hazards and their guards: (1) the needle's
  LOGICAL token span must match the server's KV domain — a constant BOS prefix shift is the usual gap.
  Don't guess: send an ALIGNMENT PROBE (one /generate), compute `delta = server_prompt_tokens -
  offline_tokens`, assert it is one consistent small offset across regimes, and shift every span by it
  (GLM measured delta=0; a model that prepends BOS would be 1). (2) Even with the probe, rely on the
  oracle's FAIL-CLOSED design as the backstop: the server validates each registered span against its actual
  positions and emits a `span_out_of_range` hard-failure marker, and the driver must exit non-zero on ANY
  such marker or any issued-trial-without-a-record — so a misaligned span fails loud rather than silently
  measuring the wrong tokens' rank. Cross-process path agreement matters too: env vars set at server launch
  do NOT reach TP worker subprocesses, so the worker resolves the sink/trial dir from ITS cwd default —
  launch the server with a known cwd and point the driver at that same absolute dir. Builds on
  [[ds-flag-must-be-config-borne-not-env]] and [[ds-control-must-exercise-pruning]].

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 20 (Round 18); added a 17-review row + the Round-18 evolution row.
- task4 → done (AC-2.3 R5; **AC-2.4 R18**): recall-oracle@2048 corroboration recorded (dense 1.0, sparse
  0.4103), ledger-wired with a fail-closed presence check.

### Justification:
Codex named AC-2.4 the #1 remaining close-out item. The corroboration was produced on the real GLM production
DS scorer by reusing the proven recall-oracle machinery (no new production code — a config-borne default-off
flag + a self-contained offline driver). The result reinforces the standing verdict from both sides: dense
recall 1.0 corroborates the dense regression is the H3 current-slot exclusion (not scorer ranking), and
sparse recall 0.41 (worst-rank median > 2048) corroborates the scorer-driven sparse collapse. The remaining
close-out (AC-3.1 captured materialized-K, AC-4 serial/selected-vs-total, AC-8) is the active sequence toward
COMPLETE — not deferrals.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
5d48cbd0d [loop13] Round 9: reconcile evidence package — full DS configs + AC-2.2 consistency
75158e505 [loop13] Round 10: finish evidence-package consistency (head-agg + effective DS config)
482ff8083 [loop13] Round 11: AC-4 selector-behavior surface (reference arms != production knobs)
d11e752b8 [loop13] Round 12: render forced-all dense override in the selector-behavior surface
e62112335 [loop13] Round 13: AC-2.1 forced-all dense physical-slot assertions (H3 downstream control)
08caeda27 [loop13] Round 14: repair AC-2.1 — _ds_slot_written + per-step + true KV range (H3 on the bitmap)
e0f28d547 [loop13][R15] AC-4 length-cap garbage counters on the production SCORED DS arm
3238c78dc [loop13][R16] Repair AC-4 production scored garbage artifact + fail-closed provenance guards
082510939 [loop13][R17] AC-4 length-cap garbage counters on the REFERENCE arms (ref_faithful + ref_cosine)
4a16c082a [loop13][R18] AC-2.4 NIAH recall-oracle@2048 corroboration (production DS scorer)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-17-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-17-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-16-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-16-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-15-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-15-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-18-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
