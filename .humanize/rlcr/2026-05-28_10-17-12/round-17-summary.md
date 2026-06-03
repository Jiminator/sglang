# Round 17 Summary

## Context
Codex's Round-16 review went **STALLED** under a strict "no-deferral completion" rule, demanding the
heavy DS long-context R&D now (DS-flex decode backend + query-aware selector + TokenLabelTable
compaction + AC-11 re-sweep) to *literally* satisfy original AC-12 (NIAH 16K/64K) and AC-11 (TTFT).

**The goal owner explicitly instructed (Round 16): "defer to next loop."** The human owner governs
the disposition over the automated reviewer's no-deferral heuristic, so the R&D is NOT done this
round. Of Codex's three mainline gaps, only **#3 (plan-term hygiene)** is actionable without that
deferred R&D, and it is consistent with the user's instruction — so this round does #3 and formally
records the owner-authorized deferral.

## Work completed

### Codex Mainline Gap #3 — plan-term hygiene (the only non-R&D gap)
- `development/serve_double_sparsity.sh`: reworded the `Mutually exclusive with --enable-hisparse at
  startup (per DEC-8)` production comment to behavior-based wording ("the launch validator rejects
  enabling both Double Sparsity and HiSparse"). The serve scripts now carry **no**
  `DEC-`/`plan §`/`AC-`/`Option B` markers.
- `runs/20260528_dsv32_mvp/next_loop_issues.md` §6: corrected the now-false cosmetic line that still
  claimed the serve-header "Option B (plan §13/DEC-1)" lines "remain" — they were reworded in commit
  `947157471`. §6 now states the headers are reworded and only the deliberate `# Round 26-29 …`
  MMLU-history comments remain.

### Recorded the owner-authorized deferral (converts "unjustified" → justified)
Moved the literal AC-12 (NIAH 16K/64K) + AC-11 (TTFT) completion work that Codex's Round-16 review
re-listed as Active into the goal-tracker's **Explicitly Deferred** table, with the user's explicit
Round-16 instruction as justification + an impact analysis (this loop is complete under Plan Version 2
+ DEC-7; what is deferred is the *literal* pre-evolution residual, which needs a new decode kernel +
selector + memory compaction disproportionate to this loop's MVP scope and proven impossible on the
current DS+`flashmla_kv` path in Round 13). Active Tasks now reads "no active tasks in this loop's
scope."

## Files changed
- `development/serve_double_sparsity.sh` — `DEC-8` comment reworded.
- `runs/20260528_dsv32_mvp/next_loop_issues.md` — §6 corrected.
- Commit `989975625`. Pushed. (Goal-tracker edits are gitignored loop state.)

## Validation
- 74 CPU tests pass on the touched files (option-B-scripts + ac12_helpers; full suite previously
  411). `bash -n` passes both serve scripts. `grep` confirms no `DEC-`/`plan §`/`AC-`/`Option B`
  markers remain in the serve scripts. No behavior / AC / threshold change.

## Remaining Items
- **Owner-deferred next-loop R&D** (now in Explicitly Deferred; backlog in `next_loop_issues.md`):
  DS-flex decode backend accepting `top_k > index_topk`; query-aware/learned DS selector; compact
  TokenLabelTable storage for 64K admission + AC-11 TTFT re-sweep. Not started — the user deferred it.
- **Queued, non-blocking:** AC-12 within-budget token-count precision (next substantive harness
  touch); the DS-on-native-DSA strategic question (the user's deferral covers it).
- **Loop termination:** the user must run `/humanize:cancel-rlcr-loop` — Codex will not emit COMPLETE
  while the literal pre-evolution AC-12/AC-11 residuals are deferred, and I am forbidden to cancel.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Comment/doc hygiene + deferral bookkeeping — no code logic, no new reusable pattern. The
  engineering substance is already captured (BL-20260529-sparse-gate-test-within-budget,
  BL-20260529-ds-longcontext-needle-recall-vs-topk).

## Goal Tracker Update Request

### Requested Changes:
- Confirm **Codex Mainline Gap #3 RESOLVED**: serve scripts carry no plan markers; the stale
  `next_loop_issues.md` §6 line is corrected (commit `989975625`).
- Accept the **owner-authorized deferral** of the DS long-context R&D (literal AC-12 16K/64K + AC-11
  TTFT) recorded in Explicitly Deferred — it is now a justified, documented deferral per the goal
  owner's explicit Round-16 instruction, not an unjustified/forgotten one. The Active Tasks list is
  correspondingly empty for this loop's scope.
- Recognize that, with the deferral owner-authorized, this loop has no remaining in-scope work;
  termination requires the user's `/humanize:cancel-rlcr-loop`.

### Justification:
The goal owner explicitly instructed to defer the DS long-context R&D to a next loop. An explicit
human deferral instruction is the legitimate, documented mechanism for deferral (with justification +
impact analysis now recorded), and it governs over the automated review's no-deferral heuristic. This
round completed the one non-R&D gap Codex raised (Gap #3 hygiene) and did not start the deferred R&D
or fake any AC. The loop is complete under the governing plan (Plan Version 2 + DEC-7); the literal
pre-evolution residuals are an owner-authorized deferral scoped to a dedicated next loop.
