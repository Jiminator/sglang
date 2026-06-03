# Round 13 Contract

## Situation
Codex's Round-12 review: **ADVANCED, 11/11 ACs addressed, 0 active plan tasks, 0 mainline gaps,
0 blocking issues.** AC-12 is evidence-complete but **NOT met** — an inherent DS limit (NIAH recall
bounded by `top_k=2048`; 64K unservable at mem 0.6). The loop cannot mechanically emit `COMPLETE`
while a hard AC is unmet, and that is not fixable by autonomous code work that preserves DS, so the
disposition was **escalated to the user** (AskUserQuestion).

**User decision (verbatim intent):** "Do step 4 first, and then look into whether if we
significantly increase selection budget we can pass NIAH. if not, then there is a serious issue.
regardless. we should knock out other items first. then after that niah test. document all issues
which we can knock out in the next rlcr loop."

## Mainline Objective (exactly one)
**Execute the user-directed close-out: clear the queued cleanups, then empirically determine whether
a significantly larger DS selection budget (`top_k`) recovers NIAH recall — separating the expected
`top_k` budget tradeoff from a genuine DS recall bug — and document the remaining work for the next
RLCR loop.**

## Target ACs
- **AC-12** (the `top_k` selection-budget investigation directly characterizes the unmet NIAH gate).
- AC-11 tooling (the comparator per-side `mem_fraction` cleanup hardens the AC-11 comparator).

## Plan (ordered per the user)
1. **Queued cleanups first** ("knock out other items first"):
   a. **Comparator per-side `mem_fraction_static` hole** (Codex queued #1): keep `mem_fraction`
      ignored for cross-side (DSA 0.85 vs DS 0.6) agreement, but compare it WITHIN each side; add a
      regression that per-side mem-fraction drift refuses (exit 2) while DSA 0.85 vs DS 0.6 proceeds.
   b. **Stale `calibrate.py` `--tp 1` recipe docstring** (Codex queued #3): correct it to the
      validated recipe recorded in `calibration_provenance.md`.
   c. Pre-existing "Option B" header terms in `serve_*.sh` — reword if cheap (lowest priority).
2. **NIAH selection-budget investigation** ("after that niah test"):
   - Boot DS at significantly larger `top_k` (sweep e.g. 2048 → 8192 → 16384) + a DSA baseline, run
     NIAH at a servable long context (16K; 16K ≫ 2048 so the budget effect is strong, and 16K fits
     the mem-0.6 KV pool). The channel mask is `top_k`-independent and the radix fixture fingerprint
     does not include `top_k`, so only the config knob changes.
   - **Decision rule:** if DS NIAH recall climbs toward DSA (≈100%) as `top_k` → full selection
     (16384 = dense at 16K), the AC-12 NIAH failure is the **expected `top_k` budget tradeoff** (not
     a bug; a larger `top_k` is the lever, at a perf cost). If recall stays low even at
     `top_k`=full-selection, that is a **serious DS recall bug** to flag prominently.
   - Record artifacts under `runs/20260528_dsv32_mvp/` (e.g. `ac12_topk_sweep_*`).
3. **Document remaining issues** for the next RLCR loop (the AC-12 disposition decision, the
   DS KV-budget/TokenLabelTable R&D for 64K admission, any newly-discovered items).

## Blocking Side Issues
- None known. (If the top_k sweep reveals a genuine recall bug, that becomes a finding for the next
  loop, not a blocker of this round's investigation/documentation objective.)

## Queued (still OUT of scope this round)
- AC-11 / DS KV-budget (TokenLabelTable) R&D for 64K admission — heavy; documented for next loop.
- Any AC-12 re-scope or operating-point change — the user's future decision (informed by the sweep).

## Success Criteria
1. Comparator per-side `mem_fraction` hole fixed with a passing regression; full CPU suite green.
2. `calibrate.py` recipe docstring corrected to the validated command.
3. The top_k sweep runs on hardware and yields a clear verdict: expected budget tradeoff vs. genuine
   bug, with committed artifacts and an analysis note.
4. A "next RLCR loop" issue list is documented (in the summary and/or a committed note).
5. Goal-tracker mutable section updated. Commit(s) (NO AI authorship) + push. No immutable-AC or
   threshold change; AC-12 is not faked green.

## Out-of-Scope Confirmation
No immutable AC or threshold is changed. The top_k sweep is a diagnostic characterization; it does
not alter the locked AC-12 operating point (`top_k=2048`) or claim AC-12 met.
