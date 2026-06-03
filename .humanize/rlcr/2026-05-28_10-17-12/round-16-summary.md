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
