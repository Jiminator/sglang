# Round 17 Contract

## Situation
Codex's Round-16 review: **STALLED** under a strict "no-deferral completion" rule — it demands the
heavy DS long-context R&D now (DS-flex decode backend + TokenLabelTable compaction + AC-11 re-sweep)
to literally satisfy original AC-12 (NIAH 4K/16K/64K parity) and AC-11 (TTFT).

**The user has explicitly instructed (Round 16, just now): "defer to next loop."** The user owns the
goal; an explicit human deferral instruction governs over the automated reviewer's no-deferral
heuristic. Therefore the R&D (Codex Mainline Gaps #1 and #2's enabling work) is **NOT** done this
round — doing it would directly violate the user's instruction. It is recorded as an
owner-**authorized, justified** deferral.

Of Codex's three mainline gaps, only **#3 (plan-term hygiene)** is actionable without the deferred
R&D, and it is consistent with the user's instruction.

## Mainline Objective (exactly one)
**Resolve Codex Mainline Gap #3 (finish the plan-term hygiene) and formally record the user's
authorized deferral of the DS long-context R&D**, so the loop state is internally consistent and the
deferral is documented as justified — leaving the loop at a clean, owner-decided close-out that only
the user can terminate (`/humanize:cancel-rlcr-loop`).

## Target ACs
- None require code changes. AC-12/AC-11 literal completion is the user-deferred R&D (documented as
  an authorized deferral); this round is hygiene + deferral bookkeeping.

## Blocking Side Issues
- **None** (Codex: 0 blocking).

## In-scope work (Gap #3, non-R&D)
1. `development/serve_double_sparsity.sh` — reword the `startup (per DEC-8)` production comment to
   behavior-based wording (drop the `DEC-8` plan marker).
2. `runs/20260528_dsv32_mvp/next_loop_issues.md` §6 — fix the now-false cosmetic line claiming the
   serve-header "Option B (plan §13/DEC-1)" lines "remain" (they were reworded in commit `947157471`).
3. Record the **DS long-context R&D** in the goal-tracker's **Explicitly Deferred** table with the
   user's explicit authorization as justification + impact analysis (literal AC-12 NIAH 16K/64K +
   AC-11 TTFT remain unmet at the literal level; resolved under DEC-7 + Plan Version 2; deferred to a
   dedicated next loop because the enabling work — a new decode kernel + selector + memory
   compaction — exceeds this loop's MVP scope).

## Queued / OUT of scope (per the user's deferral)
- Codex Gap #1 (DS-flex decode backend + literal AC-12 re-run) and Gap #2 (TokenLabelTable compaction
  + AC-11 re-sweep) — the **user-deferred next-loop R&D**. NOT started this round.
- AC-12 within-budget token-count precision (next substantive harness touch).
- DS-on-native-DSA strategic question (owner decision; the user's deferral defers this too).

## Success Criteria
1. `serve_double_sparsity.sh` has no bare `DEC-8` plan marker in production comments; `bash -n` +
   the launcher-contract test stay green; 411 CPU tests pass.
2. `next_loop_issues.md` §6 no longer falsely says the serve-header lines "remain".
3. The DS long-context R&D is recorded in **Explicitly Deferred** with the user's authorization +
   impact analysis (so it is a justified deferral, not a forgotten/unjustified one).
4. Commit (NO AI authorship) + push. No immutable-AC/threshold change, no fake pass, no loop-state
   edit-to-exit, no unauthorized R&D.

## Out-of-Scope Confirmation
No hardware run, no DS-flex/TokenLabelTable code, no AC/threshold change. This round honors the
user's explicit defer-to-next-loop instruction; it does the one non-R&D hygiene gap and documents
the authorized deferral. The loop's termination is the user's (`cancel-rlcr-loop`).
