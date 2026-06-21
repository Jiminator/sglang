# Code Review - Round 15

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-15-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 15 Summary

Mainline: **AC-4 length-cap garbage counters on the production SCORED DS arm** — the complement to the
R14 forced-all control. Diagnostic/guarded instrumentation only (no selection/adapter fix; production
byte-identical when off).

## Work Completed
Reused the **verified R14 `forced_all_assert` capture** but WITHOUT `forced_all_dense_control`, so it dumps
the **REAL production scored top-k** (post-adapter physical slots + `_ds_slot_written` bits + KV capacity,
per `(rank, req, layer, step)`) instead of the forced sweep.

1. **`serve.sh ds_garbage`** — production `ds` config (`scorer_norm=off`, `head_agg=max`, top_k 2048) +
   `"forced_all_assert": true`, NO forced-all override, eager (`--disable-cuda-graph`). One TP=8 server;
   captured a small dense (5-shot) AND sparse (24-shot) run into `.sglang_ds_garbage`, torn down to 0 MiB.
2. **`ac4_garbage_counters.py`** — garbage-only reducer for the SCORED selection: NO sweep / `req_to_token`
   equality (the selection is scored, not `[0..seq_len-1]`); auto-splits **dense (seq_len≤top_k)** vs
   **sparse (seq_len>top_k)**; per regime counts duplicate / live-lane `-1` / out-of-range (vs the true KV
   capacity) / adapter-error, and unwritten split into **current-slot (H3 marker, if selected)** vs
   **non-current (real garbage)**. Fail-closed on zero rows / missing fields / any real garbage (verified
   exit 2 on an empty dir).
3. **`ac2_1` hardened** — `h3_finding` prose is now conditional on a new `h3_marker_on_all_rows` field, so a
   future rerun that does NOT have `current_unwritten == dense_rows` cannot inherit the "every row" claim
   (Codex R14 reuse note; cheap, done this round).
4. **Ledger + findings wired** — `build_ledger.py` attaches `garbage_counters_artifact =
   evidence/ac4_garbage_counters.json` to the production_ds arm; footer + `NOT_INSTRUMENTED` updated (only
   the reference arms remain); `findings.md` records the AC-4 scored-arm result.

## Result (CLEAN — H3 from the selection side)
On **41808 dense + 37440 sparse** captured scored rows, the production selection has **zero real garbage**
in BOTH regimes: 0 duplicate, 0 live-lane `-1`, 0 out-of-range, 0 adapter-error, **0 non-current
unwritten**. And **`current_slot_unwritten = 0`** — the production scored selection does **not** include
the current decode slot (it is masked/excluded by the `_slot_written` invalidation).

This is the **complement** to AC-2.1: the forced-all control *forces* the current slot in and finds it
marked unwritten on 61776/61776 rows; the production scored path simply *excludes* it (never selects it).
Both views pin the dense regression to the **current-slot invalidation (H3)** — the adapter + selected-index
path itself is clean (no duplicate/stale/out-of-range slots) in the real production selection, dense and
sparse alike.

## Files Changed (committed `e0f28d547`)
- `development/loop13/serve.sh` (+`ds_garbage` mode), `development/loop13/ac4_garbage_counters.py` (new),
  `development/loop13/evidence/ac4_garbage_counters.json` (new), `development/loop13/ac2_1_forced_all_assertions.py`
  (conditional H3 prose), `development/loop13/build_ledger.py` (artifact wiring + footer/NOT_INSTRUMENTED),
  `development/loop13/evidence/findings.md`, `evidence/forced_all_assertions.json` (regenerated: +`h3_marker_on_all_rows`),
  `evidence/evidence_table.md`, `evidence/meta/*` (regenerated), `.gitignore` (+`.sglang_ds_garbage/`).

## Validation
- CPU reducers/tests that consume committed evidence — `ac4_garbage_counters` (CLEAN), `ac2_1_forced_all_assertions`
  (61776/61776 PASS, `h3_marker_on_all_rows=true`), `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `test_reference_selectors` (5/5) — **all exit 0**.
- `verify_ac2_3.py` is a GPU-capture re-derivation (reads the ephemeral `.sglang_ds_scorecap` sparse dir,
  which is gitignored and not on disk this turn); its committed artifact `ac2_3_radix_width_equivalence.json`
  (4992/4992 pruning rows) is the canonical evidence and was left intact — NOT re-run from a stale dir.
- `ac4_garbage_counters.py` fail-closes (exit 2) on an empty dir; real (non-current) garbage is a number (0),
  not prose.
- `build_ledger.py` → provenance consistent; production_ds arm carries `garbage_counters_artifact`.
- One TP=8 server at a time, torn down to 0 MiB. No `.pt`/`.humanize` committed. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality (extend latent-VALUE capture).
- **AC-2.4** recall-oracle@2048 (NIAH-only).
- **AC-4** garbage counters on the REFERENCE arms (`ref_faithful`/`ref_cosine`); remaining serial cells
  (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial); selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-forced-include-vs-scored-exclude-complementary-h3
- Notes: One instrumentation hook gives TWO complementary views of the same downstream bug depending on
  whether you force-include the suspect slot or read the real selection. Forced-all (force the current slot
  IN) shows it `_ds_slot_written`-False on every row; the production scored path simply EXCLUDES it
  (`current_slot_unwritten=0`). Same H3 cause (the `_slot_written[layer,out_cache_loc]=False`
  invalidation), seen from both sides — and the scored-exclude view also proves the real production
  selection has zero adapter/stale garbage in BOTH regimes (dense AND sparse), which the forced-all dense
  control alone could not show. Reuse a verified capture before building a new one; just drop the override.

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-4 production_ds scored garbage counters DONE** (task9 advanced) — `ac4_garbage_counters.json`:
  41808 dense + 37440 sparse scored rows, real garbage 0 both regimes, `current_slot_unwritten=0` (H3 from
  the selection side); wired as `garbage_counters_artifact` on the production_ds ledger arm.
- Narrow the standing AC-4 garbage-counter blocker to the **REFERENCE arms only** (production_ds + the
  forced-all control are now both instrumented).
- Note **AC-2.1 reducer hardened** (conditional `h3_finding` + `h3_marker_on_all_rows`).
- Plan Evolution Round-15 row added; Plan Version → 16.

### Justification:
The R14-review left AC-4 scored-arm garbage counters as an explicit open item. This round measured them on
the real production workload with the already-verified R14 instrumentation (no new production code — the
only change is a config-borne default-off capture and offline reducers). The result strengthens the H3
verdict from both sides: forced-include shows the current slot unwritten on every row, scored-exclude shows
it is never selected, and the scored path has zero adapter/stale garbage across dense AND sparse. Remaining
close-out items (AC-3.1, AC-2.4, AC-4 reference-arm garbage / serial / selected-vs-total, AC-8) are the next
sequence toward COMPLETE.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
5d48cbd0d [loop13] Round 9: reconcile evidence package — full DS configs + AC-2.2 consistency
75158e505 [loop13] Round 10: finish evidence-package consistency (head-agg + effective DS config)
482ff8083 [loop13] Round 11: AC-4 selector-behavior surface (reference arms != production knobs)
d11e752b8 [loop13] Round 12: render forced-all dense override in the selector-behavior surface
e62112335 [loop13] Round 13: AC-2.1 forced-all dense physical-slot assertions (H3 downstream control)
08caeda27 [loop13] Round 14: repair AC-2.1 — _ds_slot_written + per-step + true KV range (H3 on the bitmap)
e0f28d547 [loop13][R15] AC-4 length-cap garbage counters on the production SCORED DS arm
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-14-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-14-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-13-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-13-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-12-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-12-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-15-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
