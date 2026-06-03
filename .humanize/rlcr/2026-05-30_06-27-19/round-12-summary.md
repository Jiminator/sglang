# Round 12 Summary — AC-6 honestly resolved (non-regression, user decision) + benchmark `--host` fix

## Mainline objective (round contract)
Honestly resolve AC-6: fix the benchmark `--host` bug, commit recomputable DSA-default SLO
evidence, correct the AC-6 doc/tracker (no overclaim), and surface the one genuine open
question — DSA-default's conc-64 per-req TPS ~29.4 is a ~2% pre-existing miss of `≥30 TPS`
(not DS-introduced) — as a user decision, since no re-run changes it and it determines
AC-6's verdict. Codex R11 verified AC-9 (MET) and raised these as the AC-6 blockers.

## What landed
1. **`--host` benchmark-script fix (commit d0cc9fdc9).** `benchmark.sh` + `benchmark_baseline.sh`
   now pass `--host "${HOST}"` to `bench_serving` (was `--port`-only → silent localhost
   targeting; this had mislabeled R10's "DSA" bench, which actually hit node 0). Both the
   load path and the `/get_server_info` sidecar now use the same `${HOST}`. (R11 runtime
   evidence `bench_serving --host node1` → "Server ready 0.0s" proves the mechanism; the
   full script-level cross-node smoke is the AC-7 bring-up gate.) Recorded as BitLesson
   `bench-host-targeting`.
2. **Recomputable DSA-default SLO evidence (commit f9bc51b13).** `ac6_product_proof/dsa_slo_metrics_tool.py`
   + `dsa_slo_arrays.json` — exact per-request `ttfts`/`tpots`/`input_lens`/`output_lens`,
   errors-all-empty, source JSONL SHA256; `--verify` recomputes P99 TTFT + per-req TPS from
   the committed JSON alone and is **fail-closed** (exit 1 on mismatch). Verify PASS: conc
   16/32/64 P99 TTFT 0.89/1.49/2.18 s (all <22), per-req TPS 46.1/37.0/29.4.
3. **Honest AC-6 verdict + USER DECISION (R12).** The user ruled AC-6 is a **non-regression /
   opt-in product test**: DSA-default is byte-identical to the pre-DS Loop-5 baseline and
   reproduces it (0.89/1.49/2.18 s, 46.1/37.0/29.4 ≈ 0.97/1.39/2.02 s, 46.7/37.6/29.5), so
   enabling the DS opt-in code leaves DSA-default **unchanged**; the DS opt-in flag toggles
   the compact int8 path at the radix-on locked point. **AC-6 = MET.** The conc-64 per-req
   TPS ~29.4 (<30) is a **pre-existing DSA + H200 decode-batch-64 limit** (29.5 in Loop-5),
   **not introduced by DS** — recorded as a separate Queued client-SLO-vs-DSA tension that
   (per the user decision) does not block this non-regression AC.

## Result
AC-6 MET (per the R12 user decision: non-regression/opt-in test). The **AC-5 DS strict-SLO
miss remains the open mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30). Remaining:
AC-7 (3-trial DS+DSA re-sweep via the `--host`-fixed scripts), AC-8 (~70K probe), gated AC-10.

## Files Changed
- `development/benchmark.sh`, `development/benchmark_baseline.sh` (`--host "${HOST}"`).
- `runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md` (AC-6 verdict = non-regression MET; conc-64 TPS tension recorded; recomputable-evidence reference).
- `runs/20260530_dsv32_loop6/ac6_product_proof/dsa_slo_metrics_tool.py`, `dsa_slo_arrays.json` (new, recomputable + fail-closed verifier).
- `.humanize/bitlesson.md` (+1 lesson `bench-host-targeting`), goal-tracker (R12 row, plan-evolution = AC-6 non-regression grading per user decision; AC-6/`--host` blockers → RESOLVED; conc-64 TPS → Queued), round-12 contract/summary (gitignored loop state).

## Validation
- `dsa_slo_metrics_tool.py --verify`: recomputed == stored + sanity PASS; prints the honest SLO verdict (TTFT <22 all conc; TPS ≥30 conc 16/32; conc-64 ~29.4 marginal miss).
- `--host` fix: both scripts pass `--host "${HOST}"`; both sites use the same `${HOST}` (static-verified); R11 runtime banner confirmed `--host` targets the named node.
- `git diff --check` clean; commits d0cc9fdc9 + f9bc51b13 pushed to `jimmy`; no servers left running (GPUs were freed at R11 end; none booted this round).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc).
- **Queued (not blocking):** DSA-default conc-64 per-req TPS ~29.4 < 30 — pre-existing DSA/H200 limit, a client-SLO-vs-DSA tension independent of DS.
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, radix-on both, `num_prompts=64` per R11's steady-state finding, via the `--host`-fixed scripts + a cross-node host-targeting smoke), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-bench-host-targeting
Notes: Added BL-20260530-bench-host-targeting: a cross-node benchmark wrapper that passes only `--port` (not `--host`) to `sglang.bench_serving` silently load-tests localhost while its `/get_server_info` sidecar points at the remote host — this mislabeled loop6 R10's "DSA" run (it hit node0 DS). Fix: thread `--host "${HOST}"` to bench_serving so the load path and the sidecar target the same host, or run the wrapper on the server node; smoke-test that bench_serving's readiness banner names the intended host before trusting a cross-node artifact. Validated R10 (bug)/R11 (direct --host workaround)/R12 (script fix, commit d0cc9fdc9). The R12 AC-6 grading (non-regression/opt-in test; conc-64 TPS is a pre-existing DSA limit, not DS-introduced) is a USER DECISION recorded in the goal-tracker Plan Evolution Log, not a BitLesson. Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (AC-5-grade recomputable arrays + fail-closed verifier for the DSA SLO) and BL-20260530-cold-flood-not-steady-state-slo (num_prompts=64 steady-state vs the 320 cold-ramp).
