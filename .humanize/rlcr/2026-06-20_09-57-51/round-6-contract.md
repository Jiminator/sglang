# Round 6 Contract

Round 5 was ADVANCED (stall broken: AC-2.3 verified on pruning rows; first GSM8K AC-6 arm run).
Codex accepts those but blocks loop close-out on AC-6 corroboration + the remaining bisection legs,
plus two evidence-integrity fixes.

## Mainline Objective (exactly one)
**Close out AC-6 — make the production-path single-variable bisection complete and corroborated:**
(a) attach the required selected-index/current-slot corroboration to the measured `ref_cosine_noinc`
arm, and (b) produce a per-leg bisection matrix where EVERY AC-6 leg is either measured (with a config
toggle that affects the path) or carries an explicit per-leg blocker citing the exact production code
path and why no non-fix diagnostic route exists. No blanket "out of scope".

## Target ACs
- **AC-6** (primary): corroboration of the measured delta + the complete per-leg matrix.
- **AC-4** (secondary): the arm's measured-source provenance + corroboration fields wired into the
  per-arm ledger; build_ledger fails closed if an AC-6 arm has scores but no corroboration artifact.

## Blocking Side Issues (truly block AC-6 close-out / its evidence integrity)
- **`ref_cosine_noinc` measured SHA is wrong.** build_ledger stamps `measured_sha=R1_SHA (fea920c06)`,
  but the `ref_cosine_noinc` serve mode did not exist at fea920c06 — it was added in Round 5. Record a
  truthful measured-run identity: HEAD at measurement (`393966c02`, dirty worktree) + the `serve.sh`
  blob hash that defined the mode (so the run is replayable). This is AC-1/AC-4 provenance.
- **`cheap_controls.json` still exposes the stale AC-2.3 failure as the machine-readable `summary`.**
  `summary` still reads `AC_2_3_radix_eq_torch_topk_all=false`, `81/546`, `min_jaccard=0.0909`. Move
  that old join result under a clearly-named `superseded_round3_join_summary` key and put the
  pruning-valid `4992/4992` result in `summary`, so a script reading `summary` sees the current verdict.
- **build_ledger has no AC-6 corroboration guard.** It must fail loud if an AC-6 arm (here
  `ref_cosine_noinc`) records scores but no `corroboration_artifact` path that exists on disk.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-2.4 recall-oracle@2048 (the `recall_oracle` instrument is NIAH-only per DEC; GSM8K has no oracle —
  the selected-index/current-slot corroboration is the accepted alternative for the GOOD branch).
- AC-2.1 `forced_all_assertions.json`; AC-4 per-example sample IDs/order + length-cap garbage counters;
  AC-3.1 captured-row materialized-K; AC-2.2 head-agg `pre_reduce` semantics.
- Plan-term (`AC-*`, `H3`) comment cleanup; reference-mode fail-closed outside the guarded eager harness.

## Approach (per-leg plan for the AC-6 matrix)
The reference→production differing variables and their route:
- **scorer** (cosine vs raw-dot): MEASURED — 2×2 (rawdot 0.013/0.000 vs cosine 0.940/0.313).
- **current-slot** (incl vs excl): MEASURED — `ref_cosine_noinc` (0.625/0.313) + corroboration below.
- **radix top-k** & **selector width**: RETIRED — AC-2.3 pruning-valid (4992/4992).
- **head_agg**: NOT a differing variable — production and the reference both use `head_agg="max"`; the
  cross-TP `sum-of-per-rank-max` question is AC-2.2 (separate), not an AC-6 reference→production step.
- **fp8-absorbed** & **bf16-reduce**: per-leg BLOCKER — these live only in the production Triton scoring
  kernel (`_select_topk_indices` → kernel at deepseek_v2.py ~2589) which hard-locks raw-dot
  (`scorer_norm="off"`); the reference path computes exact fp32 and has no config to make THAT kernel
  compute cosine. Isolating fp8/reduce with cosine needs a new production-path cosine kernel = a
  selection-path code change = a FIX (forbidden). Second-order evidence recorded: production raw-dot
  (fp8+bf16) 0.000 ≈ exact-fp32 raw-dot 0.013.

Corroboration (CPU, no new GPU run): the only difference between `ref_cosine` and `ref_cosine_noinc` is
the `reference_include_current` flag in `_select_topk_with_optional_current`. Replay that exact function
on the 4992 real captured sparse score rows (include=True vs False) → show the selected sets differ by
EXACTLY the current decode index (Jaccard, symmetric-diff, current-slot membership/rank). Transparently
labeled: the include/exclude mechanism is scorer-independent; captured scores provide real
seq_len/position inputs.

## Concrete Success Criteria
1. `evidence/ac6_ref_cosine_noinc_corrob.json` exists, built by a committed script replaying the REAL
   `_select_topk_with_optional_current` on the 4992 captured pruning rows: per-row id, selected-index
   Jaccard, fraction of rows whose symmetric difference == {current index}, and the current-slot
   rank/masked summary. Fail-closed on zero rows.
2. `evidence/ac6_bisection_matrix.json` (generated) lists every AC-6 leg with `base_arm`,
   `changed_variable`, `config_diff`, dense/sparse GSM8K (or null), `corroboration`, and `verdict`
   ∈ {measured, retired, not-a-differing-variable, blocked}. Each `blocked` leg cites the exact code
   path + the no-non-fix-route reason; no leg is left silently deferred.
3. `ref_cosine_noinc.json` records a truthful measured source identity (HEAD `393966c02` dirty +
   serve.sh blob), not `fea920c06`.
4. `cheap_controls.json.summary` carries the pruning-valid `4992/4992` AC-2.3 verdict; the old
   `81/546` join result is moved under `superseded_round3_join_summary`.
5. `build_ledger.py` fails loud if an AC-6 arm has scores but no existing `corroboration_artifact`.
6. ROOT_CAUSE.md / findings.md reflect the matrix + corroboration (no blanket "out of scope"; per-leg
   blockers instead). Discipline: no selection/adapter fix; if any GPU arm is run, one TP=8 server at a
   time; commit; round-6-summary with BitLesson Delta + Goal Tracker Update Request. No exit by lying /
   editing loop state / cancel-rlcr-loop.
