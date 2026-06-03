# Round 15 Contract

## Situation
Codex's Round-14 review: **ADVANCED**; the user-authorized DS-fair AC-12 re-scope is **accepted**
(AC-12 MET under Plan Version 2; 10/11 ACs met; AC-11 directional miss recorded per DEC-7; 0 active
tasks; 0 blocking issues). Codex identified **one mainline gap**: `next_loop_issues.md` is **stale**
— it still describes AC-12 as the only unmet criterion and lists "Re-scope AC-12" as a *pending
option*, which Round 14 actually chose, so it now contradicts `ac12_analysis.md`,
`evidence_bundle.md`, and the goal-tracker. (Codex withholds the COMPLETE sentinel because the
*literal* original AC-11 TTFT target and literal beyond-budget AC-12 NIAH parity were not literally
satisfied — AC-11 is a recorded directional miss and AC-12 was re-scoped — but it confirms no
remaining mainline implementation work beyond this doc reconcile.)

## Mainline Objective (exactly one)
**Reconcile `runs/20260528_dsv32_mvp/next_loop_issues.md` with the Round-14 AC-12 re-scope** so the
handoff is internally consistent: record that the re-scope was chosen and AC-12 is MET under the
DS-fair gate, and list only the genuinely-remaining work (DS long-context R&D, KV-budget/64K
admission, AC-11 TTFT follow-up, the strategic DS-on-native-DSA question). This is Codex's sole
mainline gap.

## Target ACs
- AC-12 (documentation consistency of the re-scoped gate's handoff). No code/threshold/AC change.

## Blocking Side Issues
- **None** (Codex: 0 blocking).

## In-scope cleanup (folded in — cheap, comment-only, addresses a recurring Codex flag)
- Reword the plan-process-specific terms I added to the manual harness in Round 14
  (`test/manual/test_double_sparsity_v32.py` docstring + within-budget gate comments) toward
  behavior-based wording — keep legitimate "AC-12" test references, drop loop-process references
  (e.g. "loop5 Round 14"). Comment-only, zero behavior change; re-run CPU suite to confirm green.

## Queued (explicitly OUT of scope this round)
- **Token-count precision** (Codex queued #1): record actual chat input token counts and assert the
  within-budget gate from them (vs the current word-count proxy). Deferred to the next *substantive*
  harness touch — it needs a hardware re-run to repopulate artifacts, and current evidence is
  validated safe (tokenizer sanity: 1024w→1097t, 1536w→1658t, both < index_topk=2048).
- Pre-existing "Option B" serve-script header terms (cosmetic; predate Round 11).
- DS long-context R&D (query-aware selector; kernel accepting `top_k > index_topk`; smaller
  TokenLabelTable for 64K admission + AC-11 TTFT) — next loop.

## Success Criteria
1. `next_loop_issues.md` no longer contradicts the post-Round-14 state: it records AC-12 MET under
   the DS-fair re-scope and lists only the remaining R&D / follow-up items.
2. Harness plan-process terms reworded (comment-only); 411 CPU tests still green.
3. Goal-tracker mutable section updated (stale-doc queued item resolved). Commit (NO AI authorship)
   + push. No immutable-AC/threshold change; no fake pass.

## Out-of-Scope Confirmation
No hardware run, no gate-logic change, no AC/threshold change. Pure documentation reconcile plus a
comment-only hygiene pass.
