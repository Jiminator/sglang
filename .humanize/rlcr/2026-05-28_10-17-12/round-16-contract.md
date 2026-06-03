# Round 16 Contract

## Situation
Codex's Round-15 review: **ADVANCED — 0 mainline gaps, 0 blocking side issues.** The loop has fully
converged on the **user-authorized Plan Version 2**:
- AC-12 **MET** (DS-fair re-scope; HARD gates pass; beyond-budget transparently characterized);
- AC-11 executed with a **recorded directional TTFT miss** — which, per the **immutable DEC-7**
  (Round 0), is the *sanctioned* satisfied outcome for a directional target, NOT a build-break;
- all other ACs MET (10/11 met + AC-11 directional).

Codex still withholds the COMPLETE sentinel solely because the **literal original** AC-12 4K/16K/64K
parity and AC-11 TTFT are not *literally* green — but those were legitimately resolved by the
user-authorized re-scope (AC-12) and DEC-7 (AC-11). The only path to literal completion is Codex's
conditional heavy-R&D directive (a new DS-flex decode backend + TokenLabelTable reduction + sweeps),
which the **user explicitly declined in Round 14** (chose "re-scope" over "start the heavy R&D now")
and deferred to a next loop. I am forbidden to cancel the loop, change the immutable AC, fake a pass,
or silently start the user-deferred R&D.

## Mainline Objective (exactly one)
**Bring Loop 5 to a clean converged close-out: clear the last flagged cosmetic plan-term comments,
make the explicit completion case (under the governing immutable DEC-7 + authorized Plan Version 2,
all 11 ACs are satisfied), and surface the close-out decision to the user** — because there are 0
mainline gaps / 0 blocking issues and the only remaining work is the user-deferred next-loop R&D,
which only the user can authorize (or they close the loop, which only they can do).

## Target ACs
- None require code changes (all met under Plan Version 2). AC-11/AC-12 disposition is documented;
  the round is a close-out, not new AC work.

## Blocking Side Issues
- **None** (Codex: 0 blocking, 0 mainline gaps).

## In-scope cleanup (the only concrete code touch — cosmetic, closes a 15-round-recurring flag)
- Reword the pre-existing plan-process terms Codex flagged: the "Locked Option B operating point
  (plan §13 / DEC-1)" headers in `development/serve_double_sparsity.sh` + `serve_native_nsa.sh`, and
  the older round-reference comments in `test/manual/test_double_sparsity_v32.py` (lines ~47/863-864/933
  — reword only if trivially round-references; leave substantive historical rationale). Comment-only,
  zero behavior change; the launcher contract test asserts on flags, not comment text. Re-run the
  CPU suite + `bash -n`.

## Queued (explicitly OUT of scope)
- DS long-context R&D (DS-flex decode backend; query-aware selector; TokenLabelTable/KV-budget for
  64K admission + AC-11 TTFT) — user-deferred to a next loop; begin only if the user authorizes it.
- AC-12 within-budget token-count precision — next substantive harness touch (needs a hardware
  re-run; current evidence validated safe).

## Success Criteria
1. The flagged pre-existing plan-term comments are reworded to behavior-based wording (or explicitly
   left with rationale); CPU suite stays green; `bash -n` passes.
2. The goal-tracker mutable section records the converged state and a completion case under
   DEC-7 + Plan Version 2; the plan-term hygiene item is fully resolved.
3. The close-out decision is surfaced to the user (close the loop via cancel vs authorize the
   deferred R&D now), making clear the re-scope succeeded (AC-12 MET) and only the user can end the
   loop or redirect it.
4. No immutable-AC/threshold change, no fake pass, no loop-state edit/cancel, no unauthorized R&D.

## Out-of-Scope Confirmation
No hardware run, no gate-logic / AC / threshold change. A comment-only hygiene pass plus the
converged-state close-out handoff/escalation.
