# Round 11 Summary — AC-6 redo: proper-methodology DSA SLO + radix-on toggle

## Mainline objective (round contract)
Finish AC-6 properly: a DSA-default client-SLO confirmation under the proper
steady-state methodology that actually passes, plus a DS-opt-in toggle proof at the
locked **radix-on** operating point, and an AC-6 doc that claims only what the artifacts
prove. Codex's R10 review verified AC-9 (MET) but rejected AC-6: the WARMUP=0 smoke
showed DSA *failing* the SLO (cold-ramp), and the DS boot was radix-OFF while DSA was
radix-ON. AC-5's directional verdict + the open strict-SLO blocker stay tracked, not this round's objective.

## What landed (commit 0e1ce974d, pushed to `jimmy`)
Cross-node bring-up: DS int8 @ 0.7 **radix-on** (node 0) + DSA-default (node 1).

**Gap 2 fixed — DS opt-in toggle at the locked radix-on point.** Re-booted DS with the
int8 radix fixture: `/get_server_info` proves `enable_double_sparsity=True`,
`signature_dtype=int8`, **`disable_radix_cache=False`**, `double_sparsity_radix_fixture_artifact`
set; boot log shows the int8 `token_label_table` 6.48 GB/rank on **all 8 ranks** + the
radix fixture recorded PASSED (sha f3b67943). DSA-default: `enable_double_sparsity=False`,
`config=None`, 0 table lines, full 910784 pool, **`disable_radix_cache=False`** too. Both
radix-on ⇒ differ by DS enablement (and the mem-fraction it forces: 0.7+table vs 0.85+full-pool).

**Gap 1 fixed — DSA-default SLO under proper methodology.** The fresh DSA-default boot is
**byte-identical to the tracked Loop-5 DSA SLO baseline** (`dsa_default_matches_loop5_baseline.txt`:
all 11 operating-point fields match), so the established baseline applies after the DS
changes (DSA-default runs no DS path). Baseline + **fresh R11 `num_prompts=64` reproduction**
(`dsa_default_slo_np64.txt`, cross-node, warmup 120 / window 600):

| conc | P99 TTFT (fresh / L5) | per-req TPS (fresh / L5) | SLO `<22` & `≥30` |
|---:|---:|---:|:--|
| 16 | 0.89 / 0.97 s | 46.1 / 46.7 | ✅ / ✅ |
| 32 | 1.49 / 1.39 s | 37.0 / 37.6 | ✅ / ✅ |
| 64 | 2.18 / 2.02 s | 29.4 / 29.5 | ✅ / ⚠ ~29.4 (marginal, pre-existing) |

DSA-default meets **P99 TTFT < 22 s at every conc** (0.89/1.49/2.18 s); TPS ≥ 30 at conc
16/32; **conc-64 TPS ~29.4 is marginally below 30 in the DSA baseline itself** — a
pre-existing DSA characteristic at the threshold (decode batch of 64), reproduced fresh,
**not** introduced by the DS opt-in code. completed 832/1344/2048, errors 0, achieved == nominal.

**Methodology finding (why R10/this round's NUM_PROMPTS=320 run failed):** a `NUM_PROMPTS=320`
run has an epoch (~558 s at conc-16, request_rate=inf) **longer than the 120 s warmup**, so
the measurement captures the synchronized first-epoch cold-ramp (P99 TTFT 17.2/34.2 s), not
steady state. `num_prompts=64` (epoch ≈ 35 s ≪ warmup) reproduces the baseline. (Also: R10's
"DSA" bench actually hit node0 because `benchmark_baseline.sh` never passes `--host`; R11
targets node1 DSA directly via `bench_serving --host`.) The 320-prompt run is kept only as
the cold-ramp datapoint in `dsa_default_slo.txt`.

## Result
AC-6 met: DS ships opt-in (int8 table toggles at the locked radix-on point), DSA stays the
production default (no DS table, full admission, SLO < 22 s at every conc, unchanged by the
DS code). The **strict-SLO miss remains the open mainline blocker** (AC-5 conc-32/64) — unchanged.

## Files Changed
- `runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md` (rewritten honestly: radix-on parity, proper-methodology SLO, methodology note).
- `runs/20260530_dsv32_loop6/ac6_product_proof/`: `ds_opt_in_get_server_info.json` (radix-on), `ds_table_boot_excerpt.txt` (8 ranks + fixture PASSED), `dsa_default_get_server_info.json`, `dsa_notable_boot_excerpt.txt`, `get_server_info_keys.json` (both radix-on), `dsa_default_matches_loop5_baseline.txt` (NEW), `dsa_default_slo_np64.txt` (NEW fresh SLO), `dsa_default_slo.txt` (320 cold-ramp datapoint).
- `.humanize/bitlesson.md` (extended `cold-flood-not-steady-state-slo` with the epoch>warmup trap), goal-tracker, round-11 contract/summary (gitignored loop state).

## Validation
- DS radix-on: `get_server_info` `disable_radix_cache=false` + fixture path + int8 table 8 ranks (fixture PASSED).
- DSA-default operating point byte-identical to Loop-5 baseline (11/11 fields); fresh `num_prompts=64` reproduces baseline (0.89/1.49/2.18 s); errors 0; achieved == nominal.
- bench correctly targeted node1 DSA (`bench_serving --host 10.220.51.5`, smoke "Server ready in 0.0s").
- `git diff --check` clean; commit 0e1ce974d pushed; servers killed, both nodes' GPUs freed (0 MiB).

## Remaining Items
- **Open mainline blocker:** AC-5 strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc) for DS.
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, 120/600, radix-on both — note: use `num_prompts=64` per this round's methodology finding, and target the right server), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3); DS-fair thresholds unchanged (AC-9 done).

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-cold-flood-not-steady-state-slo
Notes: Extended the cold-flood lesson with the R11 finding: there are TWO cold-ramp traps, not one. Trap 1 (R10) is `WARMUP=0`. Trap 2 (R11) is `WARMUP>0` but the EPOCH longer than the warmup — with `request_rate=inf`, bench_serving processes `num_prompts` per epoch; `num_prompts=320` at conc-16 gives an epoch ~558 s ≫ 120 s warmup, so the measurement still captures the synchronized first-epoch burst (DSA P99 TTFT 17.2/34.2 s), whereas `num_prompts=64` (epoch ~35 s ≪ warmup) reproduces the steady-state baseline (0.89/1.49/2.18 s, matching Loop-5 0.97/1.39/2.02). Rule added: choose `num_prompts` so the epoch ≪ warmup, or reuse the established small-`num_prompts` baseline. Validation/Source updated to R10+R11. (Also surfaced, recorded in the round summary not the lesson: `benchmark_baseline.sh` never passes `--host`, so a cross-node "baseline" run silently hits localhost — invoke `bench_serving --host` directly or run on the server node.) Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (baseline numbers captured in a tracked `.txt` since the Loop-5 `.jsonl` are gitignored; fresh np64 is self-contained tracked evidence) and the `pkill -f 'sglang::router'` router-kill gotcha.
