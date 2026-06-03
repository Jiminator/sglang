# Round 5 Summary — AC-4 evidence addendum (durable, complete)

## Mainline objective (round contract)
Close the two AC-4 acceptance-completeness gaps Codex's R4 review found, so task5/AC-4 is durably verifiable (it gates AC-5). **No verdict change** — AC-4 still PASSES at the lifted operating point 0.7; this round makes the *evidence* complete and tracked (an evidence/packaging round; no production code).

## What was done (commit `91e9c20a3`)

### 1. Full per-rank HBM budget (`ac4_hbm_budget_addendum.md`)
Assembled the complete torch-tracked budget for f=0.6/0.7/0.8 from the server's own memory stage-deltas (`Init torch distributed` / `Load weight end mem usage=80.63 GB` / `KV Cache` / pool-end / `Capture cuda graph` / headroom):

| component (GB/rank) | 0.6 | 0.7 | 0.8 |
|---|---:|---:|---:|
| weights | 80.63 | 80.63 | 80.63 |
| KV pool | 2.38 | 17.73 | 33.09 |
| table+scales (int8) | 0.87 | 6.48 | 12.10 |
| written+scratch+flashmla-meta+bind | 2.63 | 2.66 | (OOM in capture) |
| cuda-graph pool | 11.61 | 11.59 | partial→OOM |
| headroom | 38.43 | 17.65 | — |
| torch_used (=139.80−avail) / NVML used | 101.4 / 101.4 | 122.2 / 122.2 | OOM @ `134.41 GiB alloc` |

The budget **closes** (Σnamed + residual ≈ 139.80 GiB to ~0.5 GiB driver reserve) — *not only named tensors*. `torch.memory_reserved/allocated` per rank aren't HTTP-exposed, so the addendum provides: torch-tracked stage deltas, `torch_used`, NVML used (== per-process — confirmed `nvidia-smi --query-compute-apps` = per-GPU since the server is the sole process), a labeled residual bucket, and the 0.8 OOM's literal `134.41 GiB allocated by PyTorch`. `written` per rank = `bool[L, max_tokens]` = 0.023 GB (negligible).

### 2. Durable, tracked no-OOM proof at 0.7
The R4 NVML series was a **gitignored `.csv`** (never committed) and the 97/97 was only summarized. Re-ran the sustained stress (one re-boot) and tracked everything as `.txt`:
- `stress_0.7_client.txt` — `SUMMARY: 97/97 ok, 0 failed, 92.7s` (32-conc 4096-ISL × 3 rounds + 30K long-context).
- `stress_0.7_server_excerpt.txt` — server scheduler log: prefill 8192-chunks → **Decode batch `#running-req: 32`**, token-usage 0.39, gen ~380 tok/s; **generation-time OOM line count = 0**.
- `nvml_timeseries_0.7.txt` — used **1,005,832 → plateau 1,041,136 MiB** (last == max), min-free steady — **no monotonic growth**.
- `get_server_info_0.7.json` — `mem_fraction_static=0.7`, `max_total_num_tokens=396096`.

### 3. Hygiene
Stripped trailing whitespace from `mf_*.txt` + the sweep `.md` (`git diff --check` now **clean**); removed the gitignored `nvml_*.csv`.

## Files changed
- New/updated under `runs/20260530_dsv32_loop6/`: `ac4_hbm_budget_addendum.md`, `memfraction_sweep_int8/{get_server_info_0.7.json, nvml_timeseries_0.7.txt, stress_0.7_client.txt, stress_0.7_server_excerpt.txt}`, whitespace-fixed `mf_*.txt` + `memfraction_sweep_int8.md` — commit `91e9c20a3`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-5 contract/summary (gitignored loop state).

## Validation
- `git diff --check` clean on the new artifacts.
- Re-boot @0.7 confirmed int8 table (`dtype=torch.int8`), `/health` 200, sustained stress 97/97 with 0 server-side OOM lines, NVML plateau. GPUs freed after.

## Remaining items
**AC-4 evidence complete and tracked.** Nothing in AC-4 deferred.
- **AC-5 (next mainline):** the headline result — full client-SLO benchmark at the lifted 0.7 int8 operating point: `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL / ~55% cache, radix-on proven from `.meta.json` sidecars, `client_slo_report.md` asserting strict `< 22.0` TTFT + `≥ 30 TPS/req`, the pre-declared trial-aggregation rule, and the **required measured admission-wait vs prefill-compute attribution**. Then AC-6 hardware opt-in/DSA-default, AC-7/AC-8/AC-9, gated AC-10.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Added a lesson capturing the R4→R5 packaging failure: a hardware AC's durable artifact summarized the result and referenced a `.csv` that the repo gitignores (so it never committed), and the HBM "budget" listed big named tensors without closing. The lesson: embed acceptance evidence as tracked `.txt`/`.md` (never a gitignored format — check `git ls-files`/`status --ignored` + `git diff --check`), and CLOSE the HBM budget from the server's own torch memory stage-deltas (`torch_used = total − avail`, NVML == per-process for a sole-process GPU, + a labeled residual bucket summing to total). Applied existing lessons as context: BL-20260530-int8-memfraction-ceiling-is-cudagraph-capture (the 0.8 cuda-graph boot-OOM framing) and BL-20260530-ds-... router-kill gotcha (explicit-PID kill for the re-boot).
