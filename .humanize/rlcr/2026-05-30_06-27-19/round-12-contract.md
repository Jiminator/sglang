# Round 12 Contract

## Mainline Objective (exactly one)
**Honestly resolve AC-6**: fix the benchmark-script `--host` bug, commit *recomputable*
DSA-default SLO evidence (AC-5-grade), correct the AC-6 doc/tracker so nothing is
overclaimed, and surface the one genuine open question — **DSA-default's conc-64 per-req
TPS is ~29.4, a pre-existing ~2% miss of the `≥30 TPS` SLO** (also 29.5 in the Loop-5
baseline; not introduced by DS) — as a decision, since no re-run changes it and it
determines whether AC-6's literal "DSA meets the SLO" test is achievable. AC-5's
directional verdict + the open DS strict-SLO blocker stay tracked, not this round's objective.

## Target ACs (1)
- **AC-6** (`coding`, hardware-derived) — opt-in DS / DSA-default-unchanged product property, evidenced recomputably and reported truthfully.

## Blocking Side Issues in Scope (Codex R11 review)
1. **Benchmark `--host` propagation bug.** `development/benchmark.sh` + `benchmark_baseline.sh`
   pass only `--port` to `bench_serving`, never `--host`, so a cross-node run silently hits
   localhost. Fix: add `--host "${HOST}"` to both; cross-node smoke proving `bench_serving`
   and `/get_server_info` hit the same host. (Blocks safe AC-7.)
2. **DSA-default SLO evidence is summary-only.** Commit a recomputable artifact (per-request
   TTFTs + per-request TPOT/TPS source + errors + completed + output lens + source JSONL
   SHA256 + a **fail-closed** recompute verifier), AC-5-grade, for the steady-state
   `num_prompts=64` DSA run (my R11 runs already targeted node1 via direct `--host`).
3. **No overclaim.** AC-6 doc/tracker must not claim "met with ≥30 TPS at every conc": the
   conc-64 TPS is ~29.4. Record the **opt-in toggle + DSA-default non-regression** (vs the
   tracked Loop-5 baseline, byte-identical operating point) as the proven property, and the
   conc-64 ≥30-TPS as a pre-existing DSA/client-SLO tension (not a DS regression).
4. **`NUM_PROMPTS=320` vs `=64`.** The 320 run is the cold-ramp (epoch>warmup, documented R11);
   present it as a methodology datapoint, and either get the `num_prompts=64` steady-state
   methodology approved or keep AC-6 partial. Do not silently substitute.

## Decision to surface (genuinely the user's)
The `≥30 TPS/req` SLO (CLIENT_SLOS.md) is **marginally missed by DSA-default itself at conc-64
(~29.4, pre-existing)**. Ask the user how AC-6 / the loop should treat this (accept as within
noise / non-regression-is-the-real-AC-6-test / record as a real miss / revise the SLO framing),
since it is not DS-fixable and determines AC-6's verdict.

## Queued / Out of Scope (NOT downgraded)
- **AC-5 DS strict-SLO miss** stays the open mainline blocker.
- **AC-7** (3-trial DS+DSA re-sweep — now via the `--host`-fixed scripts, `num_prompts` per the steady-state finding), **AC-8** (~70K probe), gated **AC-10** — later rounds. No FlashMLA decode-assert changes; DS-fair thresholds unchanged (AC-9 done).

## Round Success Criteria
1. `--host "${HOST}"` added to `benchmark.sh` + `benchmark_baseline.sh`; cross-node smoke proves same-host targeting.
2. Tracked recomputable DSA-default SLO artifact (np64) + fail-closed verifier; numbers recompute from committed files.
3. AC-6 doc/tracker truthful: opt-in toggle (radix-on) + DSA non-regression proven; conc-64 TPS ~29.4 recorded honestly as a pre-existing marginal miss, not claimed as ≥30.
4. The conc-64 TPS-SLO interpretation surfaced to the user (AskUserQuestion); AC-6 finalized per the answer (or left partial pending it).
5. `git diff --check` clean; commit + push; goal-tracker updated; `round-12-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- No fabrication: do not claim ≥30 TPS at conc-64; do not substitute a passing-looking smaller workload without disclosure.
- No new serve/bench scaffolding beyond the `--host` fix. Kill stale `sglang::router` before any boot.
