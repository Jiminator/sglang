# Round 5 Contract — DRIFT RECOVERY

Codex marked Rounds 3–4 STALLED (2 consecutive). This is a recovery round: prove **mainline**
movement on the actual sparse workload, not more GPU-light evidence-integrity polish.

## Recovered Mainline Objective (exactly one)
**Begin AC-6 — production-path single-variable bisection — and execute its first legs on the REAL
sparse workload (seq_len > top_k), not on smoke captures.** Walking from the reference toward
production, retire the radix/selector-width legs on real pruning rows AND run at least one
GSM8K-measured single-variable arm from the cosine reference toward production.

## Target ACs (prove mainline progress this round)
- **AC-6** (primary): one clean GSM8K-measured single-variable bisection arm (dense+sparse), with the
  one changed variable named and corroborated.
- **AC-2.3** (the radix & width legs of AC-6, suspicion-order items 5–6): retire on REAL pruning-valid
  sparse captures. This is mainline bisection work, not a side control.

## Root Cause of the Drift
Every round (1–4) I delivered GPU-**light** CPU/evidence-integrity work — analyzer fail-closed,
ledger SHA provenance, capture row-identity joins — which is completable in a single turn **without
launching a server**, and each round I labeled AC-6 "next mainline" but never spent GPU on a measured
production-path bisection arm. Worse, Round-4's AC-2.3 "RESOLVED" used committed captures that are all
`seq_len=13` (a tiny smoke prompt, never a GSM8K regime), so with `top_k=2048` both top-k methods
select all positions — pruning was never exercised, and the width `[5120]` check was vacuous. The
recovery is to **actually spend GPU this round**: a sparse-regime capture for pruning-valid AC-2.3 and
at least one measured AC-6 arm.

## Blocking Side Issues (directly block the mainline's evidence integrity)
- **`run_meta.json` blob mismatch.** Per-arm JSONs/table record generator blob `f8771c7f2…` but
  `run_meta.json` records `1391f0e…`. The per-arm ledger that will record the AC-6 arm must be
  self-consistent. Fix `run_meta.json` from the same generator source and add a consistency assertion
  in `build_ledger.py` so per-arm JSON, table header, and run metadata cannot diverge again.
- **Active false claim: "AC-2.3 RESOLVED".** `findings.md`, `cheap_controls.json._status`, and
  `ac2_3_radix_width_equivalence.json` still say RESOLVED on the seq_len=13 captures. Downgrade to
  PARTIAL/INSUFFICIENT until pruning-valid evidence lands this round, then upgrade with pruning-row
  counts. (`cheap_controls.json` also still carries the stale 81/546 join summary — regenerate or
  remove so it cannot contradict the new truth source.)

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-2.1 `forced_all_assertions.json` (physical-slot/no-dup/no-`-1`/unwritten/out-of-range/adapter-error).
- AC-4 per-example sample IDs/order + per-step length-cap garbage counters; missing serial cells.
- AC-3.1 captured-row materialized fp32 `K_label` vs absorbed raw-dot selected-index equality.
- AC-2.2 head-agg `pre_reduce_scores` semantics (`served_sum_matches_post_reduce_all=false`).
- Plan-term (`AC-*`, `H3`) comment cleanup in retained diagnostics; reference-mode fail-closed outside
  the guarded eager harness.

## Concrete Success Criteria (flip the verdict to ADVANCED)
1. **Pruning-valid AC-2.3.** Captures generated from the **sparse** regime (24-shot) so rows have
   `seq_len > top_k=2048`. `verify_ac2_3.py` records seq_len min/median/max and a `pruning_rows`
   count, **fails (nonzero exit) if `pruning_rows == 0`**, and reports radix==`torch.topk` selected-set
   identity over the **pruning subset** (or promotes a real discrepancy with numbers). The width check
   is split explicitly: for `top_k < seq_len ≤ 5120` require full == 5120-window equality; for
   `seq_len > 5120` assert the production full-fallback policy and compare full-vs-full (or count
   overflow rows explicitly) — no vacuous `w=min(5120,seq_len)` on tiny rows.
2. **One GSM8K-measured AC-6 arm.** At least one clean single-variable bisection arm (exactly one
   variable changed vs the cosine reference, e.g. production-style cosine via an existing config
   toggle), with **dense + sparse** GSM8K numbers recorded in `evidence/meta/arms/*.json` +
   `evidence_table.md`, the changed variable named, and selected-index/recall or score-rank
   corroboration. If a variable has no existing toggle and would require a non-diagnostic code change,
   it is documented and the next clean toggle arm is run instead (no new selection/adapter fix lands).
3. **Provenance consistent.** `run_meta.json` generator blob == per-arm JSON/table blob; a consistency
   assertion exists in `build_ledger.py`.
4. **No standing overclaim.** No "AC-2.3 RESOLVED" text survives without pruning-valid backing.
5. **Discipline.** One TP=8 server at a time (teardown + all 8 GPUs ~0 MiB between arms); track the
   launched PID; `--disable-radix-cache` for DS; GSM8K via `--api completion`; **land no
   selection/adapter fix**; commit; write `round-5-summary.md` with a BitLesson Delta and a Goal
   Tracker Update Request. Do NOT exit by lying, editing loop state, or running `cancel-rlcr-loop`.
