# Code Review - Round 20

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-20-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 20 Summary

Mainline: **AC-3.1 CAPTURED decode-row materialized fp32 `K_label` selected-index equality** — Codex named
the committed `ac3_1_materialized_k.json` a SYNTHETIC CPU proof; the plan wants the equality on CAPTURED
rows. Diagnostic/guarded instrumentation only; no selection/adapter fix.

## Approach (reuse the proven math on real data)
The identity is already implemented + proven by the passing unit test
`test_materialized_raw_equals_absorbed_raw`: the served `reference_rawdot_select` →
`absorbed_latent_score_logical_fp8` (absorbed raw-dot) and `absorbed_latent_cosine_logical_fp8(normalize=
False)` (the raw dot on the MATERIALIZED per-head `K_label` signature) take the IDENTICAL arg set and select
the same top-k. So the captured-row proof = dump those inputs at a served decode row and re-run BOTH
functions offline on the SAME captured inputs — zero new math, just real data.

## Work Completed
1. **Config flag** — `materialized_k_capture: bool = False` wired in all 4 config places.
2. **Capture module** (`materialized_k_capture.py`, new) — guarded, default-off, eager-only. Dumps a
   SELF-CONTAINED minimal **bs=1 reconstruction** per (rank,layer,regime,step), regime-aware-capped: the
   per-request query + the **GATHERED live** fp8 latent/scales/`_ds_slot_written` (only the request's live
   slots, NOT the whole KV pool — so the reducer needs no pool) + the per-layer `w_sel`/channel mask.
3. **Hook** in `_reference_selector_topk` (deepseek_v2.py), inside the `not is_current_stream_capturing()`
   guard, before `reference_rawdot_select`, copy-only — production byte-identical when off (the 5 reference
   unit tests still pass).
4. **`serve.sh ref_faithful_matk`** — ref_faithful config + `materialized_k_capture`, eager. One TP=8
   server, small dense+sparse capture (192 rows = 96 dense + 96 sparse), torn down to 0 MiB.
5. **CPU reducer** (`ac3_1_materialized_k_equality.py`, new) — rebuilds each captured row
   (`req_to_token=[[0..seq_len-1]]` over the captured live latent so the functions gather exactly the
   captured slots) and runs `absorbed_latent_score_logical_fp8` (raw) vs
   `absorbed_latent_cosine_logical_fp8(normalize=False)` (materialized), `select_topk_sequence_order(@2048)`,
   asserts per-row selected-index SET equality. Fail-closed: requires BOTH regimes, writes the canonical
   artifact ONLY via atomic `.tmp`→`os.replace` when every row matches.
6. **Ledger** — `build_ledger.validate_materialized_k_artifact()` independently asserts both-regimes /
   all-rows-equal / source basename / index_topk before recording `run_meta.materialized_k_captured_row_
   equality`. The synthetic proof is marked SUPERSEDED. `findings.md` records the captured-row result.

## Result (`evidence/ac3_1_materialized_k_selected_index_equality.json`)
On **96 dense + 96 sparse REAL captured decode rows**, the absorbed raw-dot reference and the materialized
fp32 `K_label` score select the **IDENTICAL top-2048 indices** — 96/96 in both regimes, max abs score diff
**2e-9 dense / 7e-9 sparse** (fp32 round-off). So the served raw-dot ceiling **is** the materialized fp32
`K_label` ceiling on real data (the captured-row form of the exact-algebra identity).

## Verification (the guards fire)
- Reducer: an empty / single-regime capture dir → exit 2, canonical artifact UNTOUCHED (both-regimes +
  atomic write).
- Ledger: an injected `all_selected_index_equal=false` / partial (`eq<rows`) / missing-regime / wrong-source
  artifact each makes `build_ledger.py` ABORT; restored → provenance consistent.

## Files Changed (committed `e67f1b5f3`)
- `python/.../double_sparsity/config.py` (flag), `python/.../double_sparsity/materialized_k_capture.py`
  (new), `python/.../models/deepseek_v2.py` (guarded hook), `development/loop13/serve.sh`
  (ref_faithful_matk), `development/loop13/ac3_1_materialized_k_equality.py` (new),
  `development/loop13/build_ledger.py` (validate + wiring), `evidence/ac3_1_materialized_k_selected_index_equality.json`
  (new), `evidence/ac3_1_materialized_k.json` (superseded note), `evidence/findings.md`,
  `evidence/evidence_table.md` + `evidence/meta/*` (regenerated), `.gitignore`.

## Validation
- CPU suite, explicit args: `ac3_1_materialized_k_equality` (EQUAL), `ac4_garbage_counters`,
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5 — production byte-identical when
  the capture flag is off) — **all exit 0**.
- `py_compile` clean; `build_ledger.py` → provenance consistent. No `.pt`/`.humanize` raw artifacts
  committed. One TP=8 server, eager, torn down to 0 MiB. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-4** serial cells (production DS sparse serial + dsa_noradix serial graph-mode; ref_faithful/ref_cosine
  serial eager-slow) + selected-vs-total verification (values are populated; make them capture-backed).
- **AC-8** final root-cause writeup (after AC-4).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-captured-row-proof-via-minimal-reconstruction-of-the-served-fn
- Notes: To turn a SYNTHETIC algebra/unit-test proof into the CAPTURED-row evidence a reviewer demands,
  don't reimplement the math offline — capture the EXACT inputs of the served function and re-call the SAME
  function on real data. Two keys: (1) capture a SELF-CONTAINED MINIMAL reconstruction, not the whole hot
  state — the served scorer gathers `latent[req_to_token[req,:seq]]` from the full KV pool, so capturing the
  pool is intractable; instead capture the GATHERED live slots ([seq_len,…]) + set `req_to_token=[[0..seq_len
  -1]]` offline so the same function gathers exactly the captured slots (verified: scores matched to fp32
  round-off ~1e-9, top-k bit-identical). Read the function's gather/mask indexing FIRST so the minimal
  reconstruction is faithful (here `written[physical_slots]` and the per-block scale layout). (2) the
  capture is a guarded config-borne default-off flag (all 4 config places) hooked inside the existing
  `not is_current_stream_capturing()` guard, copy-only — prove byte-identical-when-off by re-running the
  existing unit tests. The reducer + ledger follow the now-standard fail-closed contract (both regimes,
  atomic write only when every row passes, independent ledger gate; verified on negatives). Builds on
  [[forced-all-downstream-isolation-control]] and [[niah-recall-oracle-fail-closed-span-self-verify]].

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 23 (Round 20); added a 19-review row + the Round-20 evolution row.
- task7 → done (R20 captured-row): `ac3_1_materialized_k_selected_index_equality.json`, 96/96 dense + 96/96
  sparse identical top-2048, fail-closed reducer + ledger gate; synthetic proof superseded.

### Justification:
Codex named AC-3.1 a remaining close-out item and the synthetic proof insufficient. The captured-row
artifact proves the absorbed raw-dot reference ceiling IS the materialized fp32 `K_label` ceiling on REAL
served decode rows in BOTH regimes, by reusing the proven served functions on captured inputs (zero math
risk) under the standard fail-closed producer + independent ledger gate. Remaining close-out (AC-4
serial/selected-vs-total, then AC-8) is the active sequence toward COMPLETE.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
8a179067d [loop13][R19] Harden the AC-2.4 recall-oracle fail-closed contract (producer + consumer + harness)
e67f1b5f3 [loop13][R20] AC-3.1 CAPTURED-row materialized fp32 K_label selected-index equality
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-19-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-19-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-18-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-18-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-17-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-17-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-20-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
