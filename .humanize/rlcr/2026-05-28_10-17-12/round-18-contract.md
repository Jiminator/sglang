# Round 18 Contract — Drift Recovery

## Honest root-cause of the "drift" (NOT engineering stagnation)
The last two rounds were marked STALLED, but this is **not** circular engineering failure or
avoidance. The diagnosis is complete and the path is known (Round 13: DS's selection budget is
kernel-locked to the model's DSA `index_topk=2048`; DS decode is sound; the long-context gap is
selection quality vs V3.2's trained DSA indexer). The "drift" is a **governance conflict**:

- Codex's recovery prompt enforces a **no-deferral** termination rule: literal original AC-12 (NIAH
  4K/16K/64K parity) and AC-11 (TTFT) must be forced to completion via heavy R&D now.
- The **goal owner explicitly instructed, last turn**: *"Actually I change my mine, defer to next
  loop."* (Reversing an initial "authorize R&D now" selection.) This is a direct, recent,
  unambiguous owner decision to **defer** that exact R&D to a dedicated next loop.

The owner's explicit instruction is the higher authority. I will not implement days of speculative
R&D (a new DS-flex decode kernel + a learned-selector distillation pipeline + int8 TokenLabelTable
quantization + multi-node hardware sweeps) against the owner's decision, nor fabricate mainline
movement on owner-deferred work, nor cheat to exit.

## Recovered Mainline Objective (honest fallback per the recovery prompt)
The recovery prompt allows: *"If you cannot produce a credible recovered mainline objective, say so
… with concrete blockers."* **That is this round's outcome.** I cannot credibly recover the
literal-AC mainline this round, because the only remaining mainline work is the heavy DS long-context
R&D that the goal owner explicitly deferred — pursuing it would violate the owner's instruction.

So the recovered objective is the only honest, owner-consistent one available:
**document the governance blocker precisely and re-surface the loop-termination decision to the
owner**, keeping the deferral recorded as owner-authorized. No code/R&D this round.

## Target ACs
- AC-12 / AC-11 literal completion are the deferred R&D (owner-authorized deferral). No AC can be
  advanced this round without contradicting the owner; this is stated as the blocker, not faked.

## The concrete blocker (the thing that must change to return to ADVANCED)
**Owner-deferral vs. no-deferral-rule conflict.** Mainline movement on AC-12/AC-11 requires the
heavy R&D; the owner deferred it. Only the owner can break the impasse, by either:
- **(a)** running `/humanize:cancel-rlcr-loop` to close this loop (accepting Plan Version 2
  completion: TIER-1 smoke + AC-12 MET DS-fair + AC-11 DEC-7 directional-miss; R&D → dedicated next
  loop), since I am forbidden to cancel; OR
- **(b)** explicitly **reversing the defer instruction** and authorizing the R&D now, at which point
  I would execute Codex's directive plan (DS-flex backend + learned selector + int8 TokenLabelTable
  + staged hardware sweeps).

## Blocking Side Issues
- The governance conflict above. No *technical* blocker remains (all CPU tests pass; the diagnosis
  is complete).

## Queued (OUT of scope, per the owner's deferral)
- Codex Gaps #1/#2 R&D (DS-flex decode backend; query-aware/learned selector; int8 TokenLabelTable;
  AC-11 re-sweep) — owner-deferred to a next loop (`next_loop_issues.md`; Explicitly Deferred table).
- AC-12 within-budget token-count precision; DS-on-native-DSA strategic question.

## Success Criteria (for this recovery round, given the blocker)
1. The recovery contract + summary honestly state the blocker (owner-deferral vs no-deferral rule)
   rather than fabricating mainline movement or doing owner-deferred R&D.
2. The goal-tracker accurately reflects the conflict: owner-authorized deferral recorded; Codex's
   no-deferral position noted; loop blocked on the owner's cancel/re-authorize decision.
3. No immutable-AC/threshold change, no fake pass, no loop-state edit-to-exit, no unauthorized R&D.
4. The owner is clearly informed that closing the loop requires their action.

## Note
A "return to ADVANCED" this round is only achievable by reversing the owner's deferral (option b),
which I will not do unilaterally. The honest recovery action is to surface this for the owner's
decision — not to override them.
