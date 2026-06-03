# Code Review - Round 12

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-12-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 12 Summary

## Mainline objective (met)
Closed blocking side issue **#L**: made the AC-12 NIAH gate **artifact-safe** so a server rejection
still produces a durable per-gate JSON, regenerated the missing `ac12_niah_65536_*.json` on
hardware, and corrected `ac12_analysis.md` + `evidence_bundle.md`. task14/task15 evidence is now
complete. The AC-12 verdict is unchanged — **HARD FAIL, loop4-compatible MVP not complete** (not
reclassified as directional; DEC-7 is AC-11-only).

## Work completed

### #L — AC-12 NIAH gate is artifact-safe (mainline)
`_run_niah()` called `_generate()` directly, so when DS rejected the ~70K-token 64K prompt with
HTTP 400 (prompt exceeds the mem-0.6 KV pool of 53,056 tokens) the `urllib.error.HTTPError` escaped
before `_niah_assert()` reached `_record_artifact()` — leaving no `ac12_niah_65536_*.json`. Fix:
- **`_GenAttempt`** (`text, ok, http_status, error, body`) + **`_generate_attempt()`** wrap
  `_generate` so a 4xx/5xx (`HTTPError`) or transport failure (`URLError`) is captured, not raised.
- **`_run_niah()`** collects DSA-then-DS attempts, summarizes served counts + the first error per
  side, and computes recall over `num_prompts` (an unservable prompt is a miss). `_NIAHRunResult`
  gains `dsa_served`/`ds_served`/`dsa_error`/`ds_error`.
- **`_niah_assert()`** ALWAYS records the per-length artifact (served counts, per-side error,
  `verdict`) **before** asserting, and the failure message distinguishes a recall miss from an
  admission failure. 4K/16K/MMLU behavior is unchanged when no server error occurs.
- New registered regression **`test_niah_64k_ds_rejection_records_failure_artifact`**: patches
  `_generate` so DSA returns the needle and DS raises `HTTPError(400)`; asserts the gate fails
  cleanly (1 failure, 0 errors) and records exactly one `niah_65536` artifact with `verdict=FAIL`,
  `ds_served=0`, and the DS error body.

### Hardware 64K rerun (mainline)
Reran the AC-12 64K gate on the same locked two-node operating point (DS radix-on via fixture,
mem 0.6, node 0; DSA radix-on, mem 0.85, node 1, bound `0.0.0.0`, reached cross-node). Durable
artifact `ac12_results/ac12_niah_65536_20260529T093912Z.json`:

```
dsa_served=20  dsa_hits=20  dsa_recall_pct=100.0
ds_served=0    ds_recall_pct=0.0   delta_pct=100.0   verdict=FAIL
ds_error="HTTP 400 ... Input length (69970 tokens) exceeds the maximum allowed length (53050 tokens)"
```

The test fails cleanly (assertion, not an uncaught exception). `ac12_analysis.md` +
`evidence_bundle.md` now reference all four per-gate JSONs and phrase the DSA 20/20 result from the
artifact (no longer overstating coverage).

### Plan-term reword (queued #4, folded in)
Reworded the Round-11-added plan-specific terms (`AC-12`/`AC-Q`/`BL-...`/"Option B") in the
`test_double_sparsity_v32.py` `_generate`/NIAH comments and both `serve_*.sh` HOST-knob comments
into behavior-based wording. Only the pre-existing file-header "Locked Option B operating point
(plan §13/DEC-1)" lines remain (predate Round 11).

## Files changed
- `test/manual/test_double_sparsity_v32.py` — `_GenAttempt`/`_generate_attempt`,
  error-aware `_run_niah`, record-before-assert `_niah_assert`; comment reword.
- `test/registered/unit/manual/test_ac12_helpers.py` — new DS-rejection regression.
- `development/serve_native_nsa.sh`, `development/serve_double_sparsity.sh` — HOST-comment reword.
- `runs/20260528_dsv32_mvp/` — `ac12_results/ac12_niah_65536_*.json` (new), updated
  `ac12_results/ac12_pytest_summary.txt`, `ac12_analysis.md`, `evidence_bundle.md`.
- Commits `d2f48bbd4` (#L code + regression + reword), `cc50bae38` (64K artifact + doc updates).
  Both pushed.

## Validation
- **408 CPU tests pass** (`test_ac11_comparator` + `test_double_sparsity_unit` +
  `test_dsv32_quality_smoke_sequential` + `test_option_b_scripts` + `test_ac12_helpers`); +1 from
  the new #L regression.
- Hardware: 64K gate reran to a clean failure (`1 failed`), producing the durable
  `ac12_niah_65536_*.json`. Both servers shut down afterward; both nodes' GPUs freed; the
  pre-existing port-30000 router was already down at round start and was not touched.

## Remaining Items
- **No mainline work remains** — all plan tasks (task1–task15) are executed and AC-12 evidence is
  now complete. AC-12 is a recorded hard failure; per the Ultimate Goal the deliverable is a
  TIER-1 smoke milestone with TIER-2 loop4 quality not met (recorded, not a build-break).
- **Queued (out of scope, documented):** (a) comparator per-side `mem_fraction_static` validation
  hole — tighten when the comparator is next touched; (b) AC-11 directional performance follow-up
  (TokenLabelTable / KV-budget — the same lever bounds AC-12 64K admission); (c) stale
  `calibrate.py` `--tp 1` recipe docstring (doc-accuracy only); (d) pre-existing "Option B" header
  lines in `serve_*.sh` (predate Round 11; reword if those headers are next edited).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-gate-record-artifact-before-raise
- Notes: Added the reusable lesson behind #L — a paired hardware quality/eval gate that records its
  per-case artifact only AFTER a raising measurement call loses all durable evidence when the
  server legitimately rejects a case (here: DS HTTP 400 because the 64K prompt exceeds the mem-0.6
  KV pool). Fix: an error-aware attempt wrapper that captures HTTP/URL errors instead of raising,
  count an unservable case as a recorded miss, and ALWAYS `_record_artifact(...)` before the
  assertion — a server rejection is a recordable hard FAIL, never a silent skip or an uncaught
  error — plus a CPU regression that drives the rejection path. Validated by the new regression
  (408 CPU pass) and the durable hardware 64K artifact.

## Goal Tracker Update Request

### Requested Changes:
- Move **task14 / AC-12** to Completed and Verified — evidence is now complete (all four per-gate
  JSONs present including the durable 64K admission artifact); AC-12 remains a recorded HARD FAIL.
- Move **task15 / evidence bundle** to Completed and Verified — `evidence_bundle.md` now references
  all four AC-12 JSONs and no longer overstates 64K coverage.
- Confirm **#L RESOLVED** (artifact-safe NIAH path + regression + durable hardware 64K artifact +
  analysis/bundle correction).
- Note the **plan-term reintroduction** queued item as addressed for the Round-11 additions (only
  pre-existing header lines remain).

### Justification:
Round 12 fixed exactly the blocking gap Codex raised: the AC-12 64K subgate now produces a durable
per-gate JSON (`ac12_niah_65536_*.json`) recording the DSA 20/20 reference and the DS HTTP-400
admission failure with `verdict=FAIL`, and a registered regression proves the failure path records
evidence and fails cleanly. The AC-12 verdict is unchanged (HARD FAIL; loop4-MVP not complete) — no
immutable AC or threshold was modified, and the quality failure was not reclassified as directional.
All plan tasks are now executed with complete evidence; the remaining items are explicitly-queued,
non-blocking cleanups.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-11-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-11-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-10-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-10-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-9-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-9-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-12-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
