# Round 25 Contract

## Mainline Objective (exactly one)
**Finalize Loop-6 at its met Minimum Acceptable Scope: correct the one flagged evidence-hygiene nit and
record the terminal state.** Codex R24 confirmed the Loop-6 Lower Bound is met — AC-1..AC-9 landed (AC-5
reconciled directional-complete under DEC-3 with a fail-closed verifier + measured attribution; AC-7/AC-8
characterized per DEC-9), AC-10 owner-deferred to its own loop — with **0 active tasks and 0 blocking side
issues**. The only remaining concrete item is a stale note in `topk_design_microbench.json` ("C is the
no-context-cap win" while the measured rows show C is worse than monolithic). This round fixes that and
documents that no in-scope Loop-6 implementation work remains; the strict all-concurrency SLO and AC-10 are
explicitly downstream / own-loop per the plan + the owner's R24 directional close.

## Target AC(s)
- **AC-5 / AC-10 finalization** — housekeeping only (no new implementation); the loop is at its Lower-Bound
  terminal state.

## Truly Blocking This Objective
- **None.** Codex R24: 0 blocking side issues for the Loop-6 Lower Bound. The only item is the evidence-hygiene
  JSON note (does not block, but is the one concrete correction in scope).

## Queued / Explicitly Out of Scope This Round
- **Strict all-concurrency DS SLO** (`P99 TTFT <22s` AND `≥30 TPS/req` at every conc) — downstream hard
  blocker; structurally unattainable in Loop-6 (R24 microbench: no full-context top-k design reaches conc-16
  ≥30; DS ≤ DSA; conc-64 unattainable even for DSA). Its own future loop.
- **AC-10 (Tier-2 recall R&D)** — owner-deferred to its own loop (plan Lower Bound).
- **Cross-node wrapper smoke** — future remote-host artifacts. **DSA-default conc-64 TPS ~29.4** — separate
  DSA/client tension.

## Concrete Success Criteria
1. `topk_design_microbench.json` `note` corrected so it matches the measured rows + the (correct) markdown
   finding (C / blocked is WORSE than monolithic; only the context-capping design B reaches conc-16 ≥30).
2. Goal-tracker terminal state confirmed (AC-5 directional-complete; AC-10 deferred; strict SLO queued
   downstream) — consistent with Codex's R24 reconciliation; no Active mainline tasks.
3. A terminal round summary stating the Loop-6 Minimum Acceptable Scope is met with 0 in-scope work
   remaining, and that the COMPLETE gate (all ACs, no deferrals) is the upper bound the plan defers downstream
   — i.e. continuing the RLCR loop yields no further in-scope progress.
4. Surface the terminal fork to the owner (AskUserQuestion): stop the loop at the met Lower Bound (accept the
   MVP), or open the deferred Tier-2 (AC-10) / strict-SLO loop — since the loop cannot self-emit COMPLETE for
   a Lower-Bound close and there is no in-scope work left.
5. GPUs free (data/doc round). Commit + push to `jimmy`; goal-tracker + round-25-summary + BitLesson Delta.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260530-durable-tracked-acceptance-evidence` (committed evidence notes must match the measured data —
  the JSON-note correction).

## Out-of-bounds reminders
No production code change (finalization round). No ABI-lock change. No plan-process tokens in code/comments.
Do not change the DS-fair AC-12 gate. **Must NOT run `cancel-rlcr-loop`** (only the owner may stop the loop).
Must not exit by lying / editing loop state.
