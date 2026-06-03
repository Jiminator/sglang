# Round 14 Summary

## Mainline objective (met)
At the Round-14 disposition the user chose **"Re-scope AC-12 to a DS-fair gate now."** Implemented
the re-scope with integrity (principled, transparent, user-authorized) and verified it on hardware:
**re-scoped AC-12 PASSES.**

## Why (rationale, from Rounds 12-13)
DS is dense-prefill / sparse-decode with a fixed per-decode-step selection budget = the model's
native DSA `index_topk` = **2048 on V3.2, kernel-locked** (the `flashmla_kv` decode kernel asserts
`indices.shape[-1] == dsa_index_topk`; the budget cannot be raised on this backend). The original
AC-12 tested needle recall at 4K/16K/64K — **beyond DS's selection budget**, where an arbitrary
needle is information-theoretically unrecallable from 2048 selected tokens. Round 13 proved this is
a **selection-quality** limit vs V3.2's trained DSA indexer at the same budget, **not a decode bug**
(DS recalls 100% when its selection is dense). Testing recall beyond the budget tested DS outside
its design envelope.

## Work completed

### Re-scoped AC-12 harness (`test_double_sparsity_v32.py`)
- **HARD gates:** MMLU 5-shot within 1 pp of DSA (unchanged) **+** NIAH **within the selection
  budget** — context lengths whose tokenized length ≤ `INDEX_TOPK` (dense DS selection; word counts
  1024/1536) within 5 pp of DSA. This measures DS recall inside its design envelope.
- **CHARACTERIZATION (recorded, NOT a DSA-parity pass/fail):** NIAH 4K/16K/64K recall-vs-length +
  any admission limit; only a monotone-non-increase sanity assertion among servable points. The
  beyond-budget artifacts keep `verdict=FAIL` so the degradation stays **transparent, not hidden**.
- Kept the Round-12 error-aware `_run_niah`/`_record` path; module docstring updated to the DS-fair
  scope. The immutable AC text was **not** edited — the re-scope is logged as a Plan Evolution.

### CPU regressions (`test_ac12_helpers.py`)
- within-budget hard gate **passes** when DS==DSA and **FAILS** when DS misses (teeth retained);
- beyond-budget characterization **records** a DS rejection (durable `verdict=FAIL` artifact)
  **without hard-failing** (an admission limit is characterized, not gated).
- **411 CPU tests pass.**

### Hardware verification (two H200 nodes, same locked Option B point)
`3 passed, 2 skipped, 5 subtests` (exit 0):

| Gate | class | DSA | DS | verdict |
|------|-------|-----|-----|---------|
| MMLU 5-shot (200) | HARD | 89.00% | 89.00% (Δ0.00) | **PASS** |
| NIAH @1024 (≤ budget) | HARD | 100% | 100% (Δ0) | **PASS** |
| NIAH @1536 (≤ budget) | HARD | 100% | 100% (Δ0) | **PASS** |
| NIAH 4K | characterization | 100% | 75% | recorded (FAIL) |
| NIAH 16K | characterization | 100% | 5% | recorded (FAIL) |
| NIAH 64K | characterization | 100% | 0% (HTTP 400, ds_served 0/20) | recorded (FAIL) |

DS preserves recall within its 2048-token budget (= dense) and on MMLU, matching DSA → decode is
sound; beyond the budget recall degrades as the inherent top_k tradeoff and 64K is unservable at
mem 0.6 — both recorded, neither a bug. **AC-12 (DS-fair) is MET.**

### Docs / tracker
- `ac12_analysis.md` + `evidence_bundle.md` rewritten to the DS-fair gate (PASS; beyond-budget
  characterization kept transparent; original-AC run preserved under
  `ac12_results/superseded_prerescope/`).
- Goal-tracker: Plan Version → 2; Round-14 Plan Evolution row (user-authorized re-scope); AC-12 row
  updated to MET under the DS-fair gate (pending Codex reconcile of the immutable definition).

## Files changed
- `test/manual/test_double_sparsity_v32.py` — re-scoped gate (within-budget hard + beyond-budget
  characterization); docstring.
- `test/registered/unit/manual/test_ac12_helpers.py` — re-scope regressions (`import re`).
- `runs/20260528_dsv32_mvp/` — fresh `ac12_results/` (within-budget 1024/1536 + MMLU +
  characterization 4K/16K/64K; pre-rescope under `superseded_prerescope/`), `ac12_analysis.md`,
  `evidence_bundle.md`, re-captured `ac12_{ds,dsa}_server_info.json`.
- Commits `10f5b8878` (re-scope + regressions), `a54a7364f` (hardware PASS + docs). Both pushed.

## Validation
- 411 CPU tests pass. Hardware re-scoped gate: 3 passed / 2 skipped / exit 0; all HARD gates green;
  beyond-budget characterization recorded. Both servers shut down; both nodes' GPUs freed; the
  pre-existing router untouched.

## Remaining Items
- **DS long-context R&D (carried to next loop, `next_loop_issues.md`):** a query-aware/learned DS
  selector; a decode kernel accepting `top_k > index_topk`; a smaller TokenLabelTable for 64K
  admission. These would lift the beyond-budget limits but are out of this loop's scope.
- **AC-11** directional TTFT miss remains recorded per DEC-7 (admission-bound, mem 0.6).
- Queued cosmetic: pre-existing "Option B" serve-script header terms.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-sparse-gate-test-within-budget
- Notes: Added the reusable gate-design principle behind the re-scope — a budget-limited sparse
  mechanism (DS/DSA/any top_k-bounded sparse attention) should be HARD-gated on quality WITHIN its
  selection budget (short-context parity + needle recall at lengths ≤ budget, where selection is
  dense), and its beyond-budget recall should be CHARACTERIZED (recorded, verdict kept visible),
  NOT pass/failed against the dense/native baseline — because beyond the budget the gate measures
  the inherent sparsity tradeoff, not decode quality. Apply only after ruling out a masked bug
  (dense recall == baseline + same-budget baseline succeeds); re-scoping an immutable AC needs owner
  authorization + a logged Plan Evolution + a teeth-checking regression. Builds on the R13
  kernel-lock + selection-quality finding (BL-20260529-ds-longcontext-needle-recall-vs-topk).

## Goal Tracker Update Request

### Requested Changes:
- **Reconcile AC-12 to the DS-fair gate (Plan Evolution, Round 14)** and mark **AC-12 MET**: HARD
  gates (MMLU within 1pp + NIAH within the selection budget within 5pp of DSA) pass on hardware;
  beyond-budget NIAH is characterized (recorded, not gated), with the degradation transparently kept.
- Record that this re-scope was **explicitly authorized by the user** (Round-14 AskUserQuestion:
  "Re-scope AC-12 to a DS-fair gate now"); the immutable AC text was not edited — logged as Plan
  Evolution per the tracker's change mechanism.
- Note the loop4-compatible MVP is now substantially complete (AC-10/AC-6/AC-1b done; AC-12 MET
  DS-fair; AC-11 directional TTFT miss recorded per DEC-7); DS long-context R&D carried to the next
  loop.

### Justification:
DS's selection budget is kernel-locked to the model's DSA `index_topk=2048` and DS decode is sound
(within-budget recall 100% = DSA; MMLU = DSA); testing recall beyond the budget measured the
inherent sparsity tradeoff, not a defect. The DS-fair re-scope tests DS within its design envelope
(the theoretically-correct measurement) while keeping the beyond-budget degradation transparently
recorded — it is not a threshold relaxation to hide a bug, and the within-budget gate retains teeth
(a CPU regression proves a within-budget DS miss FAILS). The user, who owns the goal, authorized the
re-scope; per the tracker's rules the change is logged as a Plan Evolution for Codex to reconcile
into the immutable AC-12 definition. No threshold was loosened within the budget and AC-12 was not
faked green.
