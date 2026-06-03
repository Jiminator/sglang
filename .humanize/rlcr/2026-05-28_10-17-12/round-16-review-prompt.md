# Code Review - Round 16

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-16-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 16 Summary

## Context
Codex's Round-15 review: **ADVANCED — 0 mainline gaps, 0 blocking side issues.** The loop has fully
converged on the user-authorized **Plan Version 2**. Codex withholds COMPLETE only on *literal*
pre-evolution grounds (literal AC-12 4K/16K/64K parity; literal AC-11 TTFT), which were legitimately
superseded by the user-authorized re-scope (AC-12) and the immutable DEC-7 (AC-11 directional). The
only path to literal completion is the heavy DS-flex R&D the user **explicitly declined in Round 14**
and deferred to a next loop.

## User close-out decision (Round 16)
Surfaced the close-out decision; the user **confirmed: defer the heavy DS long-context R&D to a
fresh next loop and accept Plan-Version-2 completion for this loop.** (The user briefly selected
"authorize R&D now" then immediately corrected to "defer to next loop"; no R&D was started.) So this
loop is closed out under Plan Version 2 — TIER-1 smoke complete, AC-12 MET (DS-fair), AC-11
directional-miss recorded per DEC-7 — with the DS-flex decode backend / query-aware selector /
TokenLabelTable-KV-budget work carried to a future loop (`next_loop_issues.md`). Ending the RLCR
loop itself requires the user to run `/humanize:cancel-rlcr-loop` (I am forbidden to cancel it).

## Work completed

### Final plan-term hygiene (closes the 15-round-recurring flag; Codex queued #2)
Reworded the pre-existing plan-process comments to behavior-based wording:
- `serve_double_sparsity.sh` + `serve_native_nsa.sh`: the "Locked Option B operating point
  (plan §13 / DEC-1) … AC-8/AC-9/AC-11 … AC-10 radix-cache gate" headers → "Locked operating point
  … the DS-vs-baseline comparison differs only by Double Sparsity enablement and the radix-cache
  gate".
- `test_double_sparsity_v32.py`: the older "plan §10 + design doc §9.5", "plan §10 (loop does not
  close…)", and "plan §9.4 convention" comments → behavior-based phrasing.
- The locked flag list and the launcher-contract assertions (which check the actual flags, not
  comment text) are unchanged. The `# Round 26-29 …` MMLU-history comments are deliberate historical
  rationale and were left as-is.

### Completion case (under the governing plan) + close-out handoff
Recorded in the goal-tracker and surfaced to the user: **under the governing plan (immutable
Round-0 DEC-7 + user-authorized Plan Version 2) all 11 ACs are satisfied** — 9 MET; AC-11 is a
recorded directional TTFT miss, which **per DEC-7 is the sanctioned outcome for a directional
target, not a build-break**; AC-12 is **MET** under the Plan Version 2 DS-fair re-scope. There are no
mainline gaps and no blocking issues; the only remaining work is the user-deferred next-loop R&D
(DS-flex decode backend / query-aware selector / TokenLabelTable-KV-budget). I cannot end the loop
(forbidden to cancel / change the immutable AC / fake a pass) and will not silently start the
deferred R&D, so the close-out decision was escalated to the user.

## Files changed
- `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh` — header comment reword.
- `test/manual/test_double_sparsity_v32.py` — older plan-reference comment reword.
- Commit `947157471`. Pushed.

## Validation
- **411 CPU tests pass** (comment-only; launcher-contract test green). `bash -n` passes both serve
  scripts. No hardware run, no gate-logic / AC / threshold change.

## Remaining Items
- **User-deferred next-loop R&D** (`next_loop_issues.md`): DS-flex decode backend accepting
  `top_k > index_topk`; query-aware/learned DS selector; smaller TokenLabelTable for 64K admission +
  AC-11 TTFT. Begin only if the user authorizes pulling it into this loop.
- **Queued, non-blocking:** AC-12 within-budget token-count precision (next substantive harness
  touch; current evidence validated safe).
- **AC-11** directional TTFT miss recorded per DEC-7.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Comment-only hygiene + a close-out handoff — no code logic, no new failure mode or reusable
  pattern. The engineering substance is already captured
  (BL-20260529-sparse-gate-test-within-budget, BL-20260529-ds-longcontext-needle-recall-vs-topk).

## Goal Tracker Update Request

### Requested Changes:
- Confirm the **plan-term hygiene** queued item FULLY RESOLVED (pre-existing serve-header + older
  harness comments reworded, commit `947157471`).
- **Reconcile completion under the governing plan:** under the immutable Round-0 **DEC-7** (AC-11
  directional targets — a recorded miss is the sanctioned outcome, not a build-break) and the
  user-authorized **Plan Version 2** (AC-12 DS-fair re-scope, MET), **all 11 ACs are satisfied**;
  the literal pre-evolution AC-12 4K/16K/64K parity and AC-11 hard-TTFT are superseded by those two
  governing decisions, not outstanding gaps. Request that this be recognized as loop4-compatible MVP
  completion under Plan Version 2, with the DS long-context R&D explicitly carried to the next loop
  (not a deferral inside this loop's accepted scope).

### Justification:
The loop has converged (Codex: 0 mainline gaps, 0 blocking). The remaining literal-target residuals
are not unaddressed work — they are governed by two legitimate, recorded decisions: DEC-7 (immutable,
Round 0) makes AC-11's directional miss a sanctioned recorded outcome, and Plan Version 2 (explicit
user authorization, Round 14) makes AC-12 MET via the DS-fair gate with the beyond-budget degradation
transparently characterized. No immutable AC text was edited, no threshold was relaxed, and AC-12 was
not faked green. The only remaining work is next-loop R&D the user deferred; I cannot end the loop
myself, so the close-out decision (accept Plan-Version-2 completion + close the loop, or authorize
the deferred R&D now) is the user's.
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
ced03f374 [Sparsity] Round-13 queued cleanups: comparator per-side mem-fraction check + calibrate recipe docstring
27434cee7 [Sparsity] Round-13 NIAH selection-budget investigation + next-loop issue list
10f5b8878 [Sparsity] Re-scope AC-12 to a DS-fair quality gate (user-authorized, Round 14)
a54a7364f [Sparsity] Re-scoped AC-12 PASSES on hardware (DS-fair gate) + analysis/bundle
f6faaadf3 [Sparsity] Reconcile next_loop_issues.md with the AC-12 re-scope + harness comment hygiene
947157471 [Sparsity] Reword pre-existing plan-process comments to behavior-based wording
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-15-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-15-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-14-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-14-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-13-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-13-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-16-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
