# Round 5 Summary — DRIFT RECOVERY

Codex marked R3–R4 STALLED (2 consecutive). Recovered mainline: **begin AC-6 production-path
single-variable bisection on the REAL sparse workload.** Both of Codex's stated acceptable outcomes
were delivered (pruning-valid AC-2.3 AND a GSM8K-measured AC-6 arm), and the AC-6 arm produced a new,
verdict-refining result.

## Work Completed
- **AC-2.3 RESOLVED on real pruning rows** (retires AC-6 radix+width legs, suspicion-order 5–6).
  Rewrote `verify_ac2_3.py` to record the seq_len distribution + a `pruning_rows` count, evaluate
  radix==`torch.topk` on the **pruning subset** (seq_len > top_k), split the width check by the 5120
  boundary, and **fail (exit 2) if `pruning_rows==0`** — so it can never pass on smoke captures again
  (verified: it now exits 2 on the old seq_len=13 set). Captured the **SPARSE** regime (24-shot, eager
  production DS): on **4992** real rows (median seq_len **4280**, ~2048 of ~4280 pruned), the
  production blocked/radix algorithm == exact `select_topk_sequence_order` **4992/4992**, and
  selector-width [5120] == full **4992/4992**. The sparse capture run also re-confirmed production DS
  sparse = **0.000**.
- **AC-6 single-variable bisection arm (measured GSM8K).** New `serve.sh ref_cosine_noinc` mode =
  `ref_cosine` with the ONE variable flipped: `reference_include_current` true→false (production
  current-slot exclusion); cosine scorer, `head_agg=max`, exact fp32, TF32-off all held fixed
  (config-only; no selection/adapter fix lands). **Result: dense 0.940→0.625 (= production 0.620),
  sparse 0.940→0.313.** This completes the **scorer × current-slot 2×2**:

  | scorer \ current-slot | EXCLUDED (production) | INCLUDED (faithful) |
  |---|---|---|
  | raw-dot | production 0.620 / **0.000** | ref_faithful 0.950 / **0.013** |
  | cosine | ref_cosine_noinc 0.625 / **0.313** | ref_cosine 0.940 / **0.940** |

  **New finding:** sparse ≈0.94 needs **BOTH** the cosine scorer AND current-slot inclusion (neither
  alone: cosine+excl 0.313, rawdot+incl 0.013); the two regressions **interact**, and current-slot
  exclusion (H3) is a culprit in **both** regimes, not dense-only — refining the R1 reference-ceiling
  verdict. Corroborated by the `ds_anchor` arms (current-slot forced back on raw-dot stays 0.000/0.007).
- **Provenance single-source-of-truth (blocking).** `build_ledger.py` now patches `run_meta.json`'s
  generator blob + `git_sha_current` from the same `GEN_BLOB`/`GEN_HEAD` it stamps into per-arm JSONs,
  and asserts per-arm JSON == table header == run_meta blob (fails loud otherwise). The Codex-R4
  mismatch (run_meta `1391f0e` vs arms `f8771c7f2`) is closed; all three now agree.
- Downgraded the AC-2.3 "RESOLVED" over-claim, then re-resolved it from the pruning-valid artifact;
  `cheap_controls.json._status` no longer contradicts the stale 81/546 join summary. ROOT_CAUSE.md /
  findings.md updated with the 2×2 and the refined attribution.

## Files Changed (committed `c7b66f04b`)
- `development/loop13/verify_ac2_3.py` — pruning-aware, fail-closed on `pruning_rows==0`, width split.
- `development/loop13/serve.sh` — new `ref_cosine_noinc` single-variable bisection mode.
- `development/loop13/build_ledger.py` — run_meta provenance patch + consistency assertion; new arm;
  refined verdict line.
- `development/loop13/ROOT_CAUSE.md`, `evidence/findings.md` — scorer×current-slot 2×2, refined verdict.
- `evidence/ac2_3_radix_width_equivalence.json` (4992/4992 pruning-valid), `cheap_controls.json`,
  `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json` (+ new
  `ref_cosine_noinc.json`), `.gitignore` (new sparse capture dirs; raw .pt stay on disk per convention).

## Validation
- `verify_ac2_3.py` on sparse captures → **4992/4992** radix and width identical, exit 0; on the old
  seq_len=13 set → **exit 2** (`pruning_rows=0`).
- `build_ledger.py` → "provenance consistent" assertion passes; run_meta blob == per-arm == table.
- `test_reference_selectors.py` → **all 5 pass**.
- GSM8K `ref_cosine_noinc`: dense **0.625**, sparse **0.313** (batched, `--api completion`, temp 0).
- Discipline: one TP=8 server at a time — capture server torn down to 0 MiB before the measurement
  arm; both torn down at end (all 8 GPUs 0 MiB). No `.pt`/`.humanize` committed.

## Remaining Items (next mainline)
- **AC-6 production-NUMERIC legs** (fp8-absorbed vs exact fp32, bf16 vs fp32 reduce, head_agg cross-TP)
  — **untestable via config toggle** (the reference selector is exact fp32); a production-numerics
  cosine needs a production-path cosine **kernel** = code change, out of scope under "no fix". Documented
  as second-order (production raw-dot 0.000 ≈ exact raw-dot 0.013), not hand-waved.
- AC-2.4 recall-oracle@2048 corroboration for the arms; AC-2.1 `forced_all_assertions.json`; AC-4
  sample IDs/order + garbage counters; AC-3.1 captured-row materialized-K; AC-2.2 head-agg semantics.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-ds-control-must-exercise-pruning, BL-20260621-ds-bisection-interaction
- Notes: Added two lessons — (1) a top-k/selection equivalence control is vacuous unless the captured
  rows exercise the branch under test (seq_len > top_k); capture the risky regime and fail-closed on a
  zero pruning-row count. (2) A "faithful reference" that carries multiple non-production crutches can
  hide an interaction; peel one variable per arm and measure the full 2×2 rather than reading a
  single-variable cost off a multi-variable reference. Also UPDATED BL-20260621-ds-capture-step-alignment
  so its evidence cites the R5 pruning-valid 4992/4992 (the R4 624/624 was the invalid seq_len=13 set).

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-2.3 RESOLVED** (task4): radix==torch.topk 4992/4992 + width [5120]==full 4992/4992 on real
  sparse pruning rows (median seq_len 4280); verifier fails if `pruning_rows==0`.
  Evidence: `evidence/ac2_3_radix_width_equivalence.json`. Radix + selector-width suspects retired.
- Mark **AC-6 ADVANCED** (task11): first GSM8K-measured single-variable arm run (`ref_cosine_noinc`,
  0.625/0.313); scorer×current-slot 2×2 complete; verdict refined (sparse needs both fixes; H3 hurts
  both regimes). The remaining numeric legs require a production-path cosine kernel (code) and are out
  of scope under "no fix" — request they be reclassified from "blocking" to "documented out-of-scope".
- Close **Blocking: ledger provenance inconsistency** — build_ledger.py single-source-of-truth +
  consistency assertion; run_meta == per-arm == table.
- Close **Blocking: AC-2.3 does not exercise pruning** — resolved on real sparse rows.
- Plan Evolution Round-5 row already added (drift cause + recovery).

### Justification:
The recovery round delivered both Codex-stated acceptable outcomes (pruning-valid AC-2.3 and a
measured AC-6 arm) and produced a new, verdict-refining bisection result on the actual sparse workload
— directly countering the drift pattern (cheap CPU work while AC-6 GSM8K arms never ran). The two
remaining AC-6 numeric legs are genuinely blocked by the "no fix" constraint (they need a new
production-path cosine kernel), so they belong as documented out-of-scope, not as open blockers.
