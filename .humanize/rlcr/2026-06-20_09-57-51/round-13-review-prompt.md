# Code Review - Round 13

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-13-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 13 Summary

Mainline: **deliver AC-2.1 — the forced-all dense physical-slot assertions**, the plan's load-bearing
downstream-isolation control (the first GPU/instrumentation close-out item). Guarded diagnostic
instrumentation (no fix; production byte-identical when off) + one eager GPU run + a fail-closed reducer.

## Work Completed
- **Guarded instrumentation (default-off):**
  - `config.py`: new bool flag `forced_all_assert`, wired in all four places (`_ALLOWED_FIELDS`, the
    dataclass, validation, and the `parse_double_sparsity_config` explicit constructor).
  - `forced_all_assert_capture.py` (new): `maybe_dump_forced_all_assert()` dumps, per (rank, req, layer),
    the post-`logical_to_physical` **physical** slots, the forced **logical** positions, the request's
    `req_to_token[req, 0:seq_len]` slice, and the adapter `error_count`. Host-side copy only.
  - `deepseek_v2.py`: a guarded call right after `logical_to_physical`, **inside** the existing
    `not torch.cuda.is_current_stream_capturing()` guard (mirrors `_publish_ds_request_summary`). Fires
    only when the flag is on; the selected set is unchanged either way → off-path byte-identical (verified:
    reference tests pass, flag default `False`, `py_compile` clean, unknown-field guard intact).
- **Run + reducer:** `serve.sh ds_forced_all_assert` (= `ds_forced_all` + `forced_all_assert`, eager) —
  one TP=8 server, small dense drive, torn down to 0 MiB. `ac2_1_forced_all_assertions.py` →
  `evidence/forced_all_assertions.json` (fail-closed: nonzero exit on zero dense rows / missing field /
  any failing assertion; verified exit 2 on an empty dir).
- **Result (PASS) on 4368/4368 real dense rows** (median seq_len 793): forced logical sweep
  `[0..seq_len-1]` 4368/4368; physical == `req_to_token[req, 0:seq_len]` (element-wise) 4368/4368;
  **0** duplicate, **0** live-lane `-1`, **0** out-of-range, **0** adapter `error_count`. ⇒ When the dense
  selected set is forced to all tokens, the `logical_to_physical`→`transform_index_page_table_decode`
  adapter maps it to **exactly the request's own KV slots** (the same DSA feeds) with zero garbage. So the
  forced-all dense selection is a **provable no-op**, which confirms the dense regression is **downstream
  of selection** (the `_slot_written` current-slot exclusion, H3) — on live physical slots, not theory.
  The same counters are the **AC-4 garbage-rate** for the forced-all control (all zero); "unwritten" is
  subsumed by the physical==req_to_token equality. Wired into the ledger
  (`ds_forced_all.forced_all_assertions_artifact`) + the `findings.md` AC-2.1 section.

## Files Changed (committed `e62112335`)
- NEW (production, guarded diagnostic): `python/.../double_sparsity/forced_all_assert_capture.py`;
  `config.py` (flag); `deepseek_v2.py` (hook).
- NEW (loop13): `ac2_1_forced_all_assertions.py`; `evidence/forced_all_assertions.json`.
- MODIFIED: `serve.sh` (ds_forced_all_assert mode), `build_ledger.py` (artifact wiring + NOT_INSTRUMENTED
  update), `evidence/findings.md`, `evidence/evidence_table.md`, `evidence/meta/run_meta.json`,
  `evidence/meta/arms/*.json`, `.gitignore` (forcedall capture dir).

## Validation
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix`,
  `ac2_1_forced_all_assertions` — **all exit 0**.
- Off-path: `forced_all_assert` defaults `False`; config parses/validates; `py_compile` clean; reference
  selector tests unchanged → production byte-identical when off.
- `build_ledger.py` → provenance consistent (blob `80e818a7ff84`); reducer fail-closed on empty dir (exit 2).
- One TP=8 server at a time; torn down to 0 MiB. No `.pt`/`.humanize` committed. No selection/adapter **fix**
  (guarded instrumentation only).

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — extend `latent_capture` to
  store bounded latent/scales/query, then an offline analyzer at top-2048 on captured decode rows.
- **AC-2.4** recall-oracle@2048 — NIAH-only (`recall_oracle` flag + `.sglang_ds_oracle/trial.json`); GPU.
- **AC-4** garbage counters on the SCORED arms (enable the same instrumentation on production_ds/ref_*),
  remaining serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial),
  and selected-vs-total gaps.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-forced-all-downstream-isolation-control
- Notes: Added a lesson on the downstream-isolation control technique (force a stage's output to the
  trivial value and ASSERT the downstream mapping is exact ⇒ residual degradation is downstream) and the
  safe way to instrument a hot decode seam (config-borne default-off flag wired in all four config places;
  capture module mirroring score_capture; hook inside the existing not-CUDA-capturing guard; host-side copy
  only ⇒ byte-identical when off; eager run + fail-closed reducer).

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-2.1 (task2) DONE** — `forced_all_assertions.json`: 4368/4368 dense rows, physical==req_to_token,
  0 garbage, PASS ⇒ dense selection is a provable no-op (H3 confirmed downstream).
- Note **AC-4 garbage counters** instrumented for the forced-all control (all zero); enabling on the scored
  arms is the remaining AC-4 garbage-counter work.
- Plan Evolution Round-13 row added.

### Justification:
AC-2.1 is the plan's lower-bound downstream-isolation control; its PASS on live physical slots removes the
selected-index/adapter family from the dense cause and confirms the slot-validity (H3) verdict with
measured evidence, not theory. The instrumentation is a guarded, default-off, byte-identical-when-off
diagnostic (no fix), reusing the existing capture pattern. The remaining close-out items (AC-3.1, AC-2.4,
AC-4 scored-arm garbage/serial/selected-vs-total, AC-8) are the next sequence toward COMPLETE.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
5d48cbd0d [loop13] Round 9: reconcile evidence package — full DS configs + AC-2.2 consistency
75158e505 [loop13] Round 10: finish evidence-package consistency (head-agg + effective DS config)
482ff8083 [loop13] Round 11: AC-4 selector-behavior surface (reference arms != production knobs)
d11e752b8 [loop13] Round 12: render forced-all dense override in the selector-behavior surface
e62112335 [loop13] Round 13: AC-2.1 forced-all dense physical-slot assertions (H3 downstream control)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-12-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-12-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-11-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-11-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-10-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-10-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-13-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
