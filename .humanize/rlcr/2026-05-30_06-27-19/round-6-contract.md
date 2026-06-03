# Round 6 Contract

## Mainline Objective (exactly one)
**AC-5 — the headline directional client-SLO result.** Run the full client workload (`benchmark.sh`, `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL / ~55% cache) against DS with the **compact int8 table at the lifted 0.7 operating point, radix-on proven**, and write `client_slo_report.md` recording the **absolute** P99 TTFT and per-request TPS vs the strict SLO (`< 22.0 s`, `≥ 30 TPS/req`), with a **measured admission-wait vs prefill-compute attribution** and a directional-improvement statement vs the Loop-5 baseline (57.7/132.9/292.0 s). Graded **directional** (DEC-3): movement toward the strict numbers is accepted progress, explicitly **not** a shippable pass.

## Target ACs (1–2)
- **AC-5** (`coding`, hardware-run) — primary.

## Blocking Side Issues in Scope
1. **Radix-on needs an int8 fixture artifact.** `serve_double_sparsity.sh` is radix-off by default; radix-on requires `RADIX_FIXTURE_ARTIFACT`. My R2 fingerprint binds `signature_dtype`, so the Loop-5 fp16 state (`runs/20260528_dsv32_mvp/ds_radix_fixture_state.json`, no `signature_dtype`) fails closed. **Regenerate** the state for the int8/0.7 config: boot with `SIGNATURE_DTYPE=int8 SGLANG_DS_RADIX_OVERRIDE=1 SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, run the two M3-B fixtures (`test_dsv32_radix_label_capture_fixture.py` + `test_dsv32_fp8_scale_stability.py`), confirm both pass (the label-capture is now scale-aware), and `write_radix_fixture_state` → an int8 artifact. Then boot radix-on authorized by that artifact and prove radix-on from `/get_server_info` (`disable_radix_cache=false`) + `.meta.json` sidecars.

## Queued / Out of Scope
- **AC-7 (AC-11 DS+DSA 3-trial re-sweep)** — needs the live DSA too; the *next* round. AC-5 is DS-only absolute-vs-SLO.
- AC-6 hardware proof, AC-8 (~70K probe), AC-9 (within-budget harness edit), gated AC-10 — later. No FlashMLA decode-assert changes (AC-3.3).
- A strict **all-trials** pass (3 trials, 120/600) is reserved for downstream (DEC-3); this MVP measures the **directional** result.

## Round Success Criteria
1. A tracked int8 radix fixture state under `runs/20260530_dsv32_loop6/` authorizes a radix-on DS-int8/0.7 boot (`disable_radix_cache=false` proven); both M3-B fixtures pass for the int8 config.
2. `benchmark.sh MODE=double_sparsity` runs the client workload at conc 16/32/64 with the int8/0.7 radix-on server (`--enable-request-time-stats-logging`); JSONL + `.meta.json` sidecars copied into `runs/20260530_dsv32_loop6/` with radix-on proven from the sidecars/server args. Trial count and window are **disclosed** (directional MVP; a faster window is acceptable for the directional claim, no failed run hidden).
3. `client_slo_report.md` records, per conc: **absolute** P99 TTFT (asserted strict `< 22.0`) and per-request TPS (`≥ 30`), a **measured admission-wait (queue) vs prefill-compute (forward) attribution** from the request-time-stat logs + bench JSONL, and a **directional-improvement** statement vs Loop-5 (57.7/132.9/292.0 s). If attribution data is genuinely unavailable, the run is recorded but the spine is **not** called validated (no root-cause claim without data).
4. Servers killed cleanly; commit + push; `round-6-summary.md` with BitLesson Delta; tracker (task6/AC-5).

## Out-of-Scope Guards
- The directional verdict is **not** a shippable pass (DEC-3); "shippable"/"done" needs an all-trials strict pass at every conc downstream.
- A genuine miss (TTFT ≥ 22 s or < 30 TPS at some conc) is recorded honestly **with** the attribution breakdown — not hidden, and not a loop failure for the MVP. If conc-64 is prefill-bound, surface it as a chunked-prefill/scheduling follow-up.
- Reuse the serve/bench scripts; no new bench scaffolding.
