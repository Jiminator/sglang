# Round 6 Summary — AC-5 client-SLO directional result (the loop's headline)

## Mainline objective (round contract)
AC-5 — run the full client workload against DS with the compact int8 table at the lifted 0.7 operating point, **radix-on proven**, and write `client_slo_report.md` with the absolute P99 TTFT + per-request TPS vs the strict SLO, a **measured admission-wait vs prefill-compute attribution**, and a directional-improvement statement vs Loop-5. Graded directional (DEC-3): accepted progress, **not** a shippable pass.

## Blocking prereq landed: int8 radix fixture (commit `8883848e9`)
`serve_double_sparsity.sh` is radix-off by default; radix-on needs a fixture artifact, and my R2 `signature_dtype` fingerprint makes the Loop-5 fp16 state fail closed for int8. **Regenerated** it: booted with `SIGNATURE_DTYPE=int8 SGLANG_DS_RADIX_OVERRIDE=1 SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, ran both M3-B fixtures — **label-capture PASSED** (cold==warm DS label SHAs bit-equal, confirming the R2 scale-aware radix capture works in the real fixture) and **fp8-scale-stability PASSED** — and `write_radix_fixture_state` → `ds_radix_fixture_state_int8.json` (fingerprint includes `signature_dtype: int8`). The benchmark server then booted **radix-on authorized** by that artifact (validator: "fixture recorded as PASSED ... artifact_sha256=f3b67943"; `disable_radix_cache=false`), proven in every `.meta.json` sidecar.

## Result — DIRECTIONAL: accepted progress, NOT shippable (DEC-3)
DS int8/0.7, radix-on, `--enable-request-time-stats-logging`, gsp 4096 ISL (median input_len ≈ 4280) / 512 OSL, conc 16/32/64, 320 prompts, **1 directional trial** (`WARMUP=0/WINDOW=60` → one full 320-prompt epoch per conc; disclosed).

| conc | achieved (vs L5) | **P99 TTFT** | `<22`? | L5 TTFT | **per-req TPS** | `≥30`? | L5 TPS |
|---:|---:|---:|:--:|---:|---:|:--:|---:|
| 16 | **16.0** / 14.5 | **12.8 s** | ✅ | 57.7 | 17.6 | ❌ | 34.0 |
| 32 | **32.0** / 24.6 | 25.5 s | ❌ | 132.9 | 11.5 | ❌ | 33.9 |
| 64 | **60.1** / 35.7 | 111.2 s | ❌ | 292.0 | 9.3 | ❌ | 33.9 |

**The spine is validated:** admission restored (achieved ≈ nominal vs Loop-5's queue-starved 14.5/24.6/35.7); **P99 TTFT collapsed 4.5×/5.2×/2.6×**; **conc 16 MEETS the strict `< 22 s`** (12.8 s). The footprint→pool→admission→TTFT chain works on the real client workload.

**Attribution (required, measured):** from `ReqTimeStats` (`queue_duration` vs `forward_duration`) + the per-conc TTFT floor: prefill-compute floor ≈ **1.3 s** (an un-queued request prefills 4096 ISL in ~1.3 s); the **residual TTFT is queue/throughput-bound** (`queue_duration` p99 ≈ 98.5 s at the high-load tail), and it is **NOT KV-pool-admission-bound** (64×4608 = 295K < the 396K pool). So conc-32/64's residual is throughput contention from the 320-request flood → the follow-up is **chunked-prefill / scheduling**, not more footprint (the plan's anticipated "prefill-bound at conc 64" risk, confirmed with data).

**NEW FINDING — the TPS/TTFT tradeoff:** per-request TPS is **below 30 at every conc** (17.6/11.5/9.3), below Loop-5's 34. Cause: restoring admission grows the decode batch (Loop-5's 53K pool decoded only ~2–3 of these 4608-token requests at conc 64 → 34 tok/s/req; the 396K pool decodes ~19–20 → server log gen ~277 tok/s ⇒ ~14 tok/s/req). The loop's premise that "DS already beats 30 TPS, only TTFT is the problem" held **only** at the queue-starved operating point; once admission is restored the `≥30 TPS/req` SLO is in genuine tension with high concurrency on this 671B MoE. Captured as `BL-20260530-admission-restore-tps-tradeoff`.

**Verdict:** DIRECTIONAL accepted progress (spine validated, TTFT collapsed, conc-16 SLO met) — **explicitly not shippable** (strict SLO not met at every conc; honestly recorded with attribution per DEC-3). Two surfaced downstream blockers, with data, **neither a footprint problem**: (1) conc-32/64 TTFT = prefill/throughput-bound (chunked-prefill follow-up); (2) per-request TPS-vs-admission tradeoff (decode optimization / operating-point choice).

## Files changed
- `runs/20260530_dsv32_loop6/`: `ds_radix_fixture_state_int8.json`, `client_slo_report.md`, `client_slo_int8/` (3× `.meta.json` sidecars [radix-on proof], `client_slo_metrics.txt`, `reqtimestats_excerpt.txt`). Raw 4 MB `.jsonl` are gitignored (`*.jsonl`); metrics embedded as tracked `.txt` (per BL-20260530-durable-tracked-acceptance-evidence). commit `8883848e9`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-6 contract/summary (gitignored loop state).

## Validation
- int8 radix fixture: both M3-B fixtures PASS; state written with `signature_dtype: int8`; server booted radix-on authorized (no override), proven in sidecars.
- Benchmark: 3 conc, 320/320 completed each; `git diff --check` clean; GPUs freed after.

## Remaining items
- **AC-6 (next, partial→hardware):** the DSA-default product property on hardware (DSA-default boot meets SLO unchanged, allocates **no** DS table; DS opt-in toggles the compact path). Then:
- **AC-7** (AC-11 DS+DSA 3-trial re-sweep at the lifted point, radix-on both), **AC-8** (~70K-token 64K servability probe at 0.7), **AC-9** (within-budget harness edit to real `usage.prompt_tokens` + live re-run), then gated **AC-10**.
- Carry forward to the report/roadmap: the conc-32/64 chunked-prefill follow-up and the per-request-TPS-vs-admission tradeoff are downstream items (not Loop-6 footprint scope).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-admission-restore-tps-tradeoff
Notes: Added a lesson capturing the AC-5 finding: a footprint/pool lever that restores admission to fix TTFT grows the decode batch, which lowers per-request decode TPS (every in-flight request advances one token per forward step; step time grows with batch). The ">=30 TPS/req" measured at the queue-starved Loop-5 point (34 tok/s, ~2-3 decoding) does NOT hold at the restored-admission point (~14 tok/s, ~19-20 decoding). The rule: measure per-request TPS at the actual target operating point, and always pair the TTFT win with the per-request TPS + the queue-vs-forward attribution so a TTFT improvement that hides a TPS regression is caught. Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (raw .jsonl gitignored → embedded metrics as tracked .txt + verified with git check-ignore) and the router-kill gotcha for the server re-boots.
