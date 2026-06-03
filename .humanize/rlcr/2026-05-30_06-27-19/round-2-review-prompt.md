# Code Review - Round 2

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-2-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 2 Summary — scale-aware compact consumers + AC-3.1/AC-6 evidence

## Mainline objective (round contract)
Make the int8 compact path correct through **every** signature consumer (the compact label is `signatures * scales`, so any consumer treating raw `signatures` as authoritative proves the wrong thing), and close the dev-completable AC-3.1/AC-6 evidence gaps. This finishes AC-3 to the maximum extent possible without the TP=8 served model.

## Blocking correctness fix (Codex R1 review — reproduced false-pass)
Codex reproduced a `startup_sanity_probe()` false-pass by zeroing all compact scales. Root cause: the int8 sidecar (`scales`) was wired only into the hot scoring path; the proof/sanity/fingerprint consumers still hashed/scored raw `signatures`. Fixed across all of them (commit `e85cd2564`):

- **`validator.radix_fixture_config_fingerprint`** — added `signature_dtype`. The compare loop (`for k in current: recorded.get(k) != current[k]`) makes an fp16-recorded artifact (or any older artifact missing the key) **fail closed** against an int8 boot.
- **`radix_fixture_capture`** — `record_table_snapshot` / `build_request_capture` accept optional `scales` and record per-(layer,token) **scale SHAs**; `compare_cached_prefix` returns a `"scale"` divergence when equal int8 bytes carry different scales, or on a compact/fp16 mode mismatch. fp16 records are byte-identical (scale keys appear only in compact mode → backward compatible).
- **`dsa_backend._ds_radix_publish_extend_snapshot`** — passes `table.scales`.
- **`channel_mask.startup_sanity_probe`** — the compact plant uses **equal int8 magnitude with the signal living entirely in the scale**, scores with `token_scales`, and snapshots/restores scales. The probe now genuinely exercises the dequant path: a probe that ignored scales would see a flat int8 field and fail to find the needle.

## AC-3.1 / AC-6 evidence completed (dev-box)
- **DSA-default / no-table CPU regression (AC-6/AC-3.1):** `finalize_double_sparsity_bind` is a no-op with `use_double_sparsity=False` → `_bind_double_sparsity_runtime_data` is never invoked → no `TokenLabelTable` allocated.
- **compact-vs-fp16 decode-scoring microbench (AC-3.1), on H200:** the int8 overhead in the decode-time scorer is **+0.029 ms/token worst-case (conc 16)** against the **3.83 ms/token** budget (Loop-5 33.9→30 TPS/req margin) — ~**130× under budget**; at conc 32/64 int8 is *faster* (half the signature bytes → less memory bandwidth). The "TTFT-fixed-at-the-cost-of-TPS" failure mode does **not** occur. Artifact: `runs/20260530_dsv32_loop6/decode_scoring_microbench.md` + a GPU-guarded registered budget test that locks the property.

## Files changed
4 production files (validator, radix_fixture_capture, dsa_backend, channel_mask), the DS test file (+7 tests), and the new microbench artifact. Loop state in `.humanize/rlcr/` (gitignored).

## Validation — 279 DS unit tests pass (272 + 7 new), GPU enabled, no regression
New regressions: `test_compact_sanity_probe_finds_needle_and_restores_scales`, `test_compact_scorer_requires_scales` (the false-pass killer — needle selected only WITH scales), `test_radix_capture_diverges_on_scale_only_change`, `test_radix_capture_scale_mode_mismatch_diverges`, `test_fp16_radix_artifact_cannot_authorize_int8_boot` (fail-closed), `test_dsa_default_finalize_bind_is_noop_no_table`, `test_decode_scoring_overhead_within_tps_budget` (H200). Existing radix-capture + sanity-probe tests still green (backward compatible).

## Remaining items
- **Real-mask NIAH non-regression (AC-3.1) — explicitly deferred (hardware dependency, NOT avoidance):** requires the **TP=8 served DeepSeek-V3.2 FP8 model** (~672 GB weights) via `test_double_sparsity_v32.py`. The RLCR dev box has only **2 H200s** (~282 GB) — V3.2 cannot be served here. It is logged in the tracker's *Explicitly Deferred* section as the **first gate of the AC-4 cluster round** (run with `signature_dtype=int8` before any AC-4 mem-fraction-sweep claim, since task4 gates task5). Every other AC-3.1 item is complete.
- **AC-4 (next):** TP=8 DS boot with `signature_dtype=int8`, mem-fraction sweep 0.6→0.8, full NVML/torch-residual HBM accounting, no-OOM long `/generate` — preceded by the real-mask NIAH non-regression. Then AC-5/AC-6-hardware/AC-7/AC-8/AC-9, then gated AC-10.

## Note for review
The compact path is now correct end-to-end (scorer + proof + fingerprint + sanity probe) with the false-pass killed by a regression. The one outstanding AC-3.1 item (real-mask NIAH) is a genuine serving dependency on the 8-GPU cluster, not avoidance — it's tracked as the gate of the AC-4 round.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-compact-sidecar-consumer-coverage
Notes: Added a lesson capturing the review-found, R1→R2 defect class: when a compaction sidecar (int8 `scales`) changes the semantic value of an existing tensor (`signatures` → `signatures*scales`), every consumer that treats the base tensor as authoritative — proofs, config fingerprints, self-test/sanity probes, fixtures, IPC — must be threaded with the sidecar or it proves/authorizes the wrong thing (Codex reproduced a `startup_sanity_probe` false-pass with zeroed scales). The lesson records the enumerate-every-consumer fix pattern (scorer → bit-stability proof per-element sidecar SHA + explicit divergence → fingerprint storage-mode field fail-closed → sidecar-discriminative self-test) and the backward-compatibility rule (sidecar keys only in compact mode).
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
1aa24cfc1 [Sparsity] Loop-6: refined plan v1 + QA ledger + DEC-5 roadmap deferral
88c6498e5 [Sparsity] Loop-6 R0: strategic recall-R&D gate + footprint feasibility budget
84d3410b9 [Sparsity] Loop-6 R1: int8-symmetric compact TokenLabelTable (flag-gated, fp16 default, CUDA-graph-safe)
e85cd2564 [Sparsity] Loop-6 R2: scale-aware proof/sanity consumers + AC-3.1/AC-6 evidence
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-1-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-1-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-0-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-2-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
