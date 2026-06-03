# Round 15 Summary

## Mainline objective (met)
Codex's Round-14 review accepted the user-authorized DS-fair AC-12 re-scope (AC-12 MET under Plan
Version 2; 10/11 ACs met; 0 active tasks; 0 blocking) and found **one mainline gap**: the handoff
doc `next_loop_issues.md` was stale (it still called AC-12 "the only unmet criterion" and listed
"Re-scope AC-12" as a *pending* option that Round 14 actually chose). This round reconciles it.

## Work completed

### Reconcile `next_loop_issues.md` (mainline — Codex's sole gap)
Rewrote it to match the post-Round-14 state:
- New **§0 "AC-12 disposition — DECIDED"**: records that the user authorized the DS-fair re-scope,
  the HARD gates pass (MMLU 89%==89%, NIAH within-budget 100%==100%), and the beyond-budget 4K/16K/64K
  recall is transparently characterized — removing the "only unmet criterion" / pending-option
  language that contradicted `ac12_analysis.md` / `evidence_bundle.md` / the tracker.
- Kept only the genuinely-remaining work: DS long-context selector/kernel R&D; TokenLabelTable/
  KV-budget for 64K admission; the AC-11 TTFT follow-up; the **within-budget token-count-precision**
  item (Codex queued #1); and the strategic DS-on-native-DSA question.

### Harness comment hygiene (folded-in, Codex queued #2 — my Round-14 additions)
Reworded the loop-process-specific terms I added to `test/manual/test_double_sparsity_v32.py` in
Round 14 — the module docstring's "this DS-fair scope was adopted in loop5 Round 14" → behavior-based
wording, and the within-budget class-comment's "(Round-14 re-scope)" → "(not pass/failed against the
dense baseline)" — keeping legitimate `AC-12` references (it is the AC-12 test). Comment-only, no
behavior change. (Pre-existing serve-`*.sh` "Option B" headers and the old Round-26-29 MMLU-history
comments remain — cosmetic, left for whenever those spots are next edited.)

## Files changed
- `runs/20260528_dsv32_mvp/next_loop_issues.md` — reconciled with the Round-14 re-scope.
- `test/manual/test_double_sparsity_v32.py` — comment-only plan-process term reword.
- Commit `f6faaadf3`. Pushed.

## Validation
- **411 CPU tests pass** (comment-only harness change; no logic touched).
- No hardware run, no gate-logic / AC / threshold change.

## Remaining Items
- **Next-loop R&D** (carried in `next_loop_issues.md`): query-aware/learned DS selector; a decode
  kernel accepting `top_k > index_topk`; a smaller TokenLabelTable for 64K admission + AC-11 TTFT.
- **Queued, non-blocking:** AC-12 within-budget token-count precision (assert from actual chat input
  tokens vs the word-count proxy — next substantive harness touch; current evidence validated safe);
  pre-existing serve-script "Option B" header terms.
- **AC-11** directional TTFT miss remains recorded per DEC-7.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: This round was a documentation reconcile (`next_loop_issues.md`) plus a comment-only hygiene
  pass — no code logic, no new failure mode or reusable code pattern. The relevant engineering
  lessons are already captured (BL-20260529-sparse-gate-test-within-budget from the Round-14
  re-scope; BL-20260529-ds-longcontext-needle-recall-vs-topk for the kernel-lock/selection-quality
  finding).

## Goal Tracker Update Request

### Requested Changes:
- Confirm the **stale `next_loop_issues.md`** queued item RESOLVED (rewritten to record AC-12 MET
  under the DS-fair re-scope; no contradiction with `ac12_analysis.md` / `evidence_bundle.md`).
- Confirm the **manual-harness plan-term hygiene** (Round-14 additions) RESOLVED (commit
  `f6faaadf3`); the pre-existing serve-header terms + token-count precision remain queued.
- State of the goal: 10/11 ACs met under Plan Version 2; AC-12 MET (DS-fair); AC-11 directional TTFT
  miss recorded per DEC-7; 0 active evidence tasks; remaining work is next-loop R&D.

### Justification:
This round closed Codex's single Round-14 mainline gap (the stale handoff doc) and the recurring
manual-harness plan-term hygiene item, with no code-logic / AC / threshold change and the CPU suite
still green. The loop4-compatible MVP is now internally consistent across the harness, artifacts,
analysis, bundle, handoff doc, and tracker. The only items left are explicitly-queued next-loop R&D
and the recorded AC-11 directional follow-up.
