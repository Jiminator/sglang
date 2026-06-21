# Code Review - Round 2

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop12/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-2-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Loop 12 — Round 2 Summary

## Mainline objective
Complete the close-out cleanup Codex's Round-1 review left open (both [P3], no AC gap): finish
stripping plan/workflow markers the Round-1 sweep missed, and correct a false native-DSA band
statement in the provenance doc. AC-8 was accepted in the Round-1 review (10/10 ACs).

## Work Completed

### [P3] Remaining plan/workflow markers — stripped
The Round-1 sweep targeted `AC-`/`DEC-`/`Milestone`/`Loop-N`/`[R-N]` but missed other workflow
labels. Reworded as durable technical language (no logic change):
- `Option B` → the DSA index-topk operating point / dropped: `config.py:8`,
  `page_table_adapter.py:9`, `validator.py` (×2, comment + user-facing error message).
- `Tier-2.A lifted-budget path` → `lifted-budget path`: `validator.py` (×2, comment + error message).
- `Round 3` → the actual fix description (flat-slicing would pick V/RoPE columns from later heads):
  `calibrate.py:428`.
- `round 1` → `first radix pass`: `topk_kernel.py:78` (radix-pass wording).
- Removed the stray EOF blank line in `metrics.py` (`git diff --check` was flagging it).

### [P3] Provenance doc native-DSA statement — corrected
`benchmarks/DOUBLE_SPARSITY.md` had claimed "both DS and DSA meet the band once the workload shape is
fixed", but native DSA's 46.50 s P99 TTFT is **not** in the ≤30.1 s band and there is no
corrected-shape DSA evidence. Fixed: the native-DSA column is now labelled **same-base context only**
(an earlier run made before the wrapper pinned the GSP grouping, explicitly "not in band"); the
**accepted** result is the DS run (256/256 completed, `request_shape_ok=true`, 35.05 TPS / 22.90 s).
Removed the false claim.

## Files Changed (v2 clone, R2)
- `python/sglang/srt/layers/attention/double_sparsity/{config,page_table_adapter,validator,calibrate,topk_kernel,metrics}.py` (comment/marker rewording + EOF blank line; no logic change)
- `benchmarks/DOUBLE_SPARSITY.md` (native-DSA note corrected)
- v2 commit `323cb7802`; branch re-pushed.

## Validation
- Final marker sweep over the full `<BASE>...HEAD` diff: **0** plan/workflow markers in added lines
  (DeepSeek-R1 model names + base-code Step/Phase preserved by design).
- `git diff --check` clean; AC-1 diff still 42 files, **0** dev-scaffolding, **0** dropped-module refs.
- `import sglang` OK; **114 unit tests pass** (no logic touched).
- Branch re-pushed to `Jiminator/sglang`; HEAD == remote `323cb7802`.

## Remaining Items
None. All 10 ACs pass; both Codex reviews' findings (R0 [P1] AC-8 workload shape + [P3] markers; R1
[P3] remaining markers + [P3] doc) are resolved. PR:
`https://github.com/Jiminator/sglang/pull/new/double-sparsity-v2`.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: R2 applied existing requirements (plan:417 code-style marker ban) and corrected doc prose;
  no new reusable technical lesson. The relevant lessons are already recorded
  (`BL-20260619-perf-parity-pin-request-shape`, `BL-20260619-ds-selector-width-ladder`,
  `BL-20260619-latest-main-base-drift`).
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
c6e3e943e [loop11b] R0: task10 DONE (UX pass); checkpoint — awaiting DSA matched re-run
d672d962f [loop11b] R0: matched-op-point verdict (task7/8/9 done) — DS meets SLO to conc32
425cdbcef [loop11b] R0: close-out — regenerate results.md (M-A+M-B+M-C complete)
65997cb4c [loop11b] R0: close-out evidence preflight — residual probe evidence + ignore .pt dumps
5df030348 [loop11b] R0: task11 close-out complete — all 11 tasks done; queue finalized
9af9d7835 [loop11b] R1: bench_serving emits prefix-reuse + DS no-op evidence (AC-5/AC-9)
8cde27faa [loop11b] R1: clean M-B re-run orchestrator (both op-points, tax probe, evidence)
73338e539 [loop11b] R1: fix mb_v2 tax_probe local-var bug; task10 serve-script de-plan
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
bec9348c7 [loop12] add gen-plan output + draft; record pensieve doctor state
879532819 [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
6fee2b0ce [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-18_13-17-09/round-1-summary.md
- @.humanize/rlcr/2026-06-18_13-17-09/round-1-review-result.md
- @.humanize/rlcr/2026-06-18_13-17-09/round-0-summary.md
- @.humanize/rlcr/2026-06-18_13-17-09/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-2-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
