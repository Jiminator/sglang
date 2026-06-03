# Round 4 Summary — AC-4 mem-fraction lift + no-OOM validation (int8 compact table)

## Mainline objective (round contract)
AC-4 — boot DS with the compact int8 table, sweep `MEM_FRACTION_STATIC=0.6→0.7→0.8`, record `max_total_num_tokens` rising + the full HBM budget at each point, and survive a sustained long `/generate` at the lifted fraction with **no generation-time OOM and no monotonic memory growth**. This is the spine's payoff — the footprint reduction exists to lift the no-OOM operating point.

## Result — PASS (lifted operating point = 0.7; commit `75e68053f`)
Real single-node TP=8 H200, `serve_double_sparsity.sh SIGNATURE_DTYPE=int8`, Loop-5 mask, Option B.

| `f` | int8 table | KV pool | `max_total_num_tokens` | post-graph headroom | result |
|---:|---:|---:|---:|---:|:--|
| 0.6 | 0.87 GB | 2.38 GB | 53056 | 38.34 GB | serves |
| 0.7 | 6.48 GB | 17.73 GB | **396096** | 17.56 GB | **serves + sustained-gen no-OOM** |
| 0.8 | 12.10 GB | 33.09 GB | 739200 (attempted) | — | **boot OOM (cuda-graph capture)** |

- `max_total_num_tokens` **rises** with `f` (53056 → 396096 → 739200). The int8 table at 0.7 is **6.48 GB vs fp16's 11.52 GB** → **17.56 GB** post-cuda-graph headroom vs fp16's 12.29 GB that gen-OOM'd in Loop-5.
- **No-generation-OOM validation at the lifted fraction (0.7):** a sustained stress — **32 concurrent** `/generate`, ~4096-ISL, 256 new tokens, 3 rounds + a ~30K-token long-context request — completed **97/97 OK, 0 failed**, **no generation-time OOM**. NVML over the run rose to the generation working set then **plateaued** (last sample == max; min-free 17.9→11.9 GB steady) — **no monotonic growth**. This directly refutes fp16's Loop-5 0.7 generation-OOM: same fraction, same workload, int8 survives.
- Full HBM budget (NVML per-GPU + torch avail + `/get_server_info` + log components: weights / KV / table+scales / cuda-graph pool / headroom) captured per fraction under `runs/20260530_dsv32_loop6/memfraction_sweep_int8/`.

## The 0.8 ceiling — honest, AC-2-framed
`f=0.8` **boot-OOMs during cuda-graph capture** (verbatim: `Capture cuda graph failed: ... Tried to allocate 146.00 MiB ... 132.12 MiB free`): the int8 table (12.10 GB) + the 739K-token KV pool (33 GB) leave only 22.68 GB pool-end headroom, and the fixed ~11.6 GB cuda-graph capture pool doesn't co-fit. This is a **boot-time** OOM, **not** the AC-4 generation-time negative test, and **not** "the table is still too big". Per the verified AC-2 budget the target is *admitted KV capacity*, not `f=0.8` as a number, and the plan calls 0.7 "acceptable as a more conservative first step": **0.7's `max_total=396096` exceeds the conc-64 admission target (~114K) by ≈3.5×**, so the admission goal is met with large margin and the page-level escalation is **not** triggered. Reaching 0.8 would require trimming the fixed Option-B cuda-graph batch set (a productionization pass, out of scope) and is unnecessary.

## Files changed
- `runs/20260530_dsv32_loop6/memfraction_sweep_int8.md` + `memfraction_sweep_int8/` (per-fraction NVML/`get_server_info`/log captures, the 0.7 NVML time series, boot excerpts) — commit `75e68053f`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-4 contract/summary (gitignored loop state).

## Validation
- 3 clean TP=8 boots (0.6/0.7/0.8) with `SIGNATURE_DTYPE=int8` confirmed (`token_label_table ... dtype=torch.int8`).
- 0.7 sustained generation: 97/97 OK, no gen-OOM, NVML plateau.
- 0.8 boot-OOM captured verbatim. GPUs freed after each boot (stale-`sglang::router` cleared via explicit-PID kill).

## Remaining items
**AC-4 done** (lifted operating point 0.7, no-OOM validated, full budget recorded). Nothing in AC-4 deferred.
- **AC-5 (next mainline):** full client-SLO benchmark — `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL / ~55% cache, radix-on proven, at the lifted operating point (0.7, int8). `client_slo_report.md` with strict `<22.0` TTFT + `≥30 TPS/req`, the pre-declared trial-aggregation rule, and the **required measured admission-wait vs prefill-compute attribution**. Then AC-6 hardware opt-in/DSA-default, AC-7/AC-8/AC-9, gated AC-10.
- Note: at 0.7 DS now admits ~396K tokens (vs Loop-5's 53K at 0.6) — admission should be largely restored; AC-5 measures whether that actually moves P99 TTFT toward <22s and attributes any residual to prefill compute.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-int8-memfraction-ceiling-is-cudagraph-capture
Notes: Added a lesson capturing the AC-4 finding: after the footprint lever relieves the generation-OOM, the binding HBM constraint at high mem_fraction MOVES from generation headroom to the BOOT-time cuda-graph capture pool — so 0.8's OOM is a cuda-graph-capture boot failure, not "the table is too big". The rule: don't escalate the footprint lever (page-level) on such an OOM; check the admission target first (0.7 already over-admits conc-64 ~3.5×) and define the operating point as "highest fraction that boots AND survives a sustained generate", targeting admitted KV capacity not the mem-fraction number. Applied existing lessons as context: BL-20260528-dsv32-ds-serving-boot-chain (fp16 0.7 gen-OOM baseline) and BL-20260530-verify-hardware-before-deferring (8-GPU TP=8 confirmed before the serve).
