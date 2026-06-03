# Round 0 Summary — Loop 7 (DSv3.2 long-context recall R&D)

## Outcome

Round 0 completed **M0 (the measure-first milestone)** and the **first M1 (Tier-2.B) measurement** on real 8×H200 hardware. The loop's central question — is the DS long-context recall gap budget- or scorer-limited — is answered with evidence, and the first selector lever (cosine scoring) is measured and **works at the target regime**.

## What Was Implemented

1. **M0 instrumentation**: a recall-oracle diagnostic (needle rank / score-only recall@K on the live all-reduced score tensor), a dedicated flag-gated oracle sink, the AC-1.1 post-topK force, and capture-guarded hooks on both decode score paths. All flag-gated + off by default (production byte-identical).
2. **Served-recall baseline** re-measured DS-vs-DSA at mem 0.7 (N=20, Clopper–Pearson CIs), separating served-miss from admission failure.
3. **Oracle budget-vs-scorer sweep** + Codex A-vs-B adjudication.
4. **M1 cosine scorer** (`scorer_norm=cosine`, direction-only scoring), made **config-borne** so it reaches the TP workers, and measured live.

## Key results

- **Baseline (corrected)**: served **100/100/75/5/5** (1K/1.5K/4K/16K/64K), all served, 0 admission failures. **64K now served** (old "0%" was admission failure at mem 0.6) → AC-5/task18 effectively met. `development/loop7/m0_baseline.md`.
- **Oracle**: 4K budget-limited (needle rank ~2208, recall@4096=100%); 16K/64K scorer-limited (rank ~10K+). `m0_oracle_finding.md`.
- **Decision (task7, Codex)**: lead Tier-2.B; Tier-2.A bounded-secondary; supersedes the gate's Tier-2.A-primary ordering. `m0_decision.md`.
- **M1 cosine — MEASURED**: 16K recall **5% → 40%** (8×, clears the 24.9% bar) but 4K **75% → 25%** — regime-dependent. Confirms the long-context gap is scorer-limited AND fixable. `m1_cosine_finding.md`.

## Files Changed

- New: `double_sparsity/selection_recall_oracle.py`, `double_sparsity/oracle_artifact_sink.py`.
- Modified: `double_sparsity/selection_kernel.py` (cosine scorer in `compute_token_scores` + `_compute_logical_token_scores`; oracle hooks; `scorer_norm` threading), `double_sparsity/config.py` (`scorer_norm` field), `double_sparsity/selector.py` (passes config value), `models/deepseek_v2.py` (routes to eager logical scorer when `scorer_norm != off`), `development/serve_double_sparsity.sh` (`SCORER_NORM` knob).
- Tests (CPU): `test_selection_recall_oracle.py`, `test_oracle_sink_and_force.py`, `test_scorer_norm.py`.
- Artifacts/drivers under `development/loop7/`: `niah_ds_baseline.py`, `niah_oracle_sweep.py`, baseline/oracle/cosine JSON, 5 finding docs.
- Commits on `dev/double-sparsity-standalone`: `9914a3004`, `8074cb1cf`, `c6ffcdea6`, `78f6b5d17`, `a1e2c72dc`, `599d7cc99`, `e2674f4f4` (+ the planning commit `9ca1f5133`).

## Validation

- **295–316 DS unit tests pass** (depending on sub-suite); new oracle math, sink/force, wired-hook, and cosine-scorer tests green.
- Live 8×H200 runs: baseline (N=20), oracle sweep, cosine sweep (N=20) — real measured numbers above.
- Production selection byte-identical when oracle + scorer flags off.

## Remaining Items

- **M1-next**: length/budget-conditional **hybrid scorer** (keep 4K's 75%, capture 16K's 40%); firm 16K to N≥50; MMLU non-regression; TP cross-rank determinism; port the winning scorer to the graph-safe Triton path (currently eager research path, DEC-6).
- **AC-2/AC-3 closure**: DSA re-confirm (100%/length, documented reference) + MMLU re-anchor at mem 0.7.
- **M2 (Tier-2.A)**: bounded/opt-in for the 4K-class budget-limited regime only (per the decision).
- **M4**: task20 decision record (gate supersession — text ready in `m0_decision.md`); consolidation.
- Queued: oracle recording on TP workers (env→config, partially understood); standalone graph-replay alloc test.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260602-ds-flag-must-be-config-borne-not-env
Notes: serve-time DS flags consumed inside the TP worker (selector/model-forward) must be **config-borne** (`--double-sparsity-config`), NOT env vars — SGLang TP workers don't inherit arbitrary `SGLANG_*` env. This cost a wrong "cosine doesn't help" reading (cosine hadn't actually run; env was absent on the workers). Fixed by adding `scorer_norm` to `DoubleSparsityConfig`; the real measurement then showed 16K 5%→40%. Verify a flag reached the workers via the launch cmdline / worker `/proc/<pid>/environ`, never from latency.

## Honest caveats

N=20: the 16K cosine win is large (8×) but its lower CI (0.19) dips just under the 24.9% bar — firm to N≥50 before binding. Cosine runs on the eager research path (DEC-6); production needs the graph-safe Triton port + MMLU non-regression. DSA re-confirm + MMLU re-anchor not yet run. The recall oracle's per-worker env propagation (distinct from the scorer fix) remains queued.
