# Loop 8 / task9 — GLM-5.1 DS-vs-DSA-native gate record (8×H200, 2026-06-08)

Accuracy + client-SLO gates for the opt-in Double-Sparsity path on GLM-5.1-FP8, DS vs DSA-native on the
**same node/op-point** (only DS enablement + the required DS mem-fraction differ). Landing policy per DEC-2.

> **Status of this record:** the **SLO gates (iii decode-TPS, iv P99-TTFT)** are measured live with real
> DS-vs-DSA numbers below, but at a **PRELIMINARY op-point** (concurrency 32, 64 prompts, single trial, no
> 120 s/600 s window) — **NOT the final locked landing sweep** (conc 16/32/64 × 3 trials × 600 s). The
> **accuracy gates (i MMLU, ii NIAH)** are **PENDING** (the harness needs both DS+DSA servers live, which
> two TP=8 servers cannot do on 8 GPUs — execution plan below). Final landing numbers require the locked
> sweep; nothing here is marked "final/passed-to-land".

## Production landing mask (256-sample, AC-3)
- Path `/models/glm51-fp8-channel-mask-s256.safetensors`, `content_sha256=35155ac46ad7…`,
  `num_samples=256`, `calibration_source=real`, `label_dim=32`, `head_dim=192`, `layers=78`,
  `page_size=64`, `dtype=fp8_e4m3`, indices ∈ [0,191]. `load_channel_mask` re-verifies the hash;
  `verify_bind_shapes(qk_nope=192, label_dim=32, layers=78, TP=8)` PASS. DS server binds it (bind shape
  gate PASS all 78 layers/8 ranks) and serves. This is the production landing artifact (supersedes the
  32-sample bring-up mask `e7dbf4c9308f` used for R0 smoke).

## Op-point (both columns, same node — R5 preliminary, parity caveat)
Both columns booted via the **paired locked-op-point launchers** (`serve_native_nsa.sh` DSA-native /
`serve_double_sparsity.sh` DS) with the GLM model + 256 mask. From each JSONL's `server_info`, the columns
match on **TP=8, page 64, kv_cache_dtype fp8_e4m3, disable_radix_cache=True, disable_piecewise_cuda_graph=
True, disable_overlap_schedule=True, dsa_prefill/decode_backend=flashmla_kv** (this **fixes the R4
op-point mismatch** — R4's DSA had `disable_piecewise_cuda_graph=False`). **Caveat (R6 review):** the
recorded R5 artifacts still differ on `server_info.random_seed` (DSA 515248618 / DS 689475326) because the
launchers did not yet pin a seed; the intended config diffs are `enable_double_sparsity` (False/True) and
`mem_fraction_static` (DSA 0.8 / DS 0.7 — the inherent DS TokenLabelTable reservation). **R6 added a fixed
`RANDOM_SEED` (20260607, `--random-seed`) to BOTH launchers**, so the **final locked sweep** will be
seed-matched; this R5 curve is **preliminary** and not seed-matched. Workload: gsp 4096 ISL (sys 2253 +
q 1843, ~55% prefix cache) / 512 OSL, seed 431.

## SLO gates (iii) decode TPS + (iv) P99 TTFT — parity conc curve (R5, PRELIMINARY window)
Decode TPS = `output_tokens / (e2e_latency − ttft)` per request (DEC-4). Via `bench_serving` →
`benchmark_compare.py` (decode-TPS primary, strict `P99 TTFT < 22 s`). **PRELIMINARY:** `num_prompts =
concurrency`, single pass, NO 120 s/600 s window — short cold runs, not the locked landing sweep (esp.
conc 64 TTFT is cold-burst-inflated, per BL-20260530-cold-flood-not-steady-state-slo).

| conc | DSA decode P50 | DSA P99 TTFT | DSA verdict | DS decode P50 | DS P99 TTFT | DS verdict |
|-----:|---------------:|-------------:|:-----------:|--------------:|------------:|:----------:|
| 16 | **38.69** | 7.24 s | **PASS** | **23.16** | 3.68 s | FAIL (TPS) |
| 32 | **31.52** | 14.19 s | **PASS** | **17.09** | 37.17 s | FAIL |
| 64 | 24.35 | 28.32 s | FAIL (cold-burst) | 17.11 | 74.33 s | FAIL |

(DS achieved concurrency 16.0 / 22.6 / 40.2 — admission-bound at conc 32/64 by the smaller DS KV pool.)

**Findings (authoritative for the preliminary window):**
- **DS-on FAILS the decode-TPS ≥ 30 bar at EVERY concurrency** (best case conc-16 = 23.16 < 30), and the
  P99-TTFT bar at conc 32/64. Confirmed via the parity-matched comparator (DS SLO verdict: **fail**).
- **DSA-native PASSES the SLO at conc 16 and 32** (decode ≥ 30 + TTFT < 22 s) and fails only at conc 64
  (24.35 TPS / 28 s TTFT — a cold-burst short run; re-confirm under the locked steady-state window).
- DS-on is consistently ~1.4–1.7× slower decode than DSA-native — the expected posture (GLM ships a strong
  trained DSA indexer; DS is the default-off long-context-recall fallback, not a throughput win here).
- Artifacts: `runs/20260607_glm51_loop8/parity_{dsa,ds}_c{16,32,64}.jsonl`.

### Reading (characterization) — R5 parity numbers
At the standard client op-point, **DS-on is slower than DSA-native** (decode P50 17.1 vs 31.5 tok/s at
conc 32) and its P99 TTFT is admission-bound by the smaller DS KV pool (mem 0.7 + per-rank
TokenLabelTable → achieved conc 22.6 < 32). **DSA-native PASSES** the SLO at conc 16 (38.7 tok/s / 7.2 s)
and conc 32 (31.5 tok/s / 14.2 s), and fails only at conc 64 (24.4 / 28.3 s — a cold-burst short run).
**DS-on fails the decode-TPS ≥ 30 bar at EVERY concurrency** (23.2 / 17.1 / 17.1). Decode is **coherent**
(not degenerate). This is the **expected** posture from the plan: GLM ships a strong trained DSA indexer,
so DS is the **reversible default-OFF opt-in fallback**, valuable where the indexer underperforms
(long-context recall), **not** a throughput win on the standard workload.

## Accuracy gates (i) MMLU + (ii) NIAH — PENDING (executable offline path hardened R9)
`test/manual/test_double_sparsity_v32.py` requires `DS_BASE_URL` **and** `DSA_BASE_URL` live
simultaneously (skipUnless); two TP=8 servers cannot co-reside on 8×H200 **and GLM-5.1 cannot run at TP=4**
(weights ~2× exceed a single H200), so the only viable path is **sequential collect + offline compare**.
`development/loop8/accuracy_gate.py` provides it (reusing the harness's tuned MMLU parser + deterministic
NIAH prompt-gen + recall scorer): `AC12_MODE=collect` scores ONE live server (`AC12_SIDE=dsa|ds`,
`AC12_BASE_URL=…`) and writes a per-side artifact (run_id + prompt-set hashes + hits/totals + index_topk);
`AC12_MODE=compare` (offline, no server) validates the two sides used the same prompt set and applies the
mandatory thresholds (MMLU DS within 1.0 pp of DSA; within-budget NIAH DS within 5.0 pp; beyond-budget =
characterization-only), failing closed on any mismatch. **Schema v3; publication-safe (R7 fail-closed +
R8 denominator/graph-flags + R9 full-op-point + exact-mask provenance).**

**Op-point (R9): no hand whitelist.** The gate reuses the EXACT stable launch-arg projection
`development/benchmark_compare.py` derives from `dataclasses.fields(ServerArgs)` (shared via
`_bench_compare()` so the accuracy and SLO gates cannot drift and a new sglang launch flag is
auto-protected). It compares the FULL stable ServerArgs set minus only the DS knobs
(`enable_double_sparsity`/`double_sparsity_config`/radix-fixture) and the DS-vs-DSA memory reservation
(`mem_fraction_static`). The exact contract: every locked Option-B field (`model_path`, `tp_size`,
`page_size`, `kv_cache_dtype`, DSA backends, radix, the three CUDA-graph/overlap/piecewise flags) MUST be
present + non-null on both sides (so a `None == None` "agreement" cannot pass); and the captured stable
fields are compared by **union**, so any field present on one side only, or differing on both (e.g. `dtype`,
`max_total_tokens`), fails closed. (A non-locked stable field omitted from BOTH synthetic artifacts is not
itself rejected — but `collect()` captures the full stable `/get_server_info` projection, so real artifacts
always carry it.) Unlike the throughput sweep, the accuracy gate **requires `random_seed` parity**
(greedy/deterministic scoring + this loop's seed-parity mandate).

**Mask provenance (R9): exact path + content hash.** The DS column must prove it served the exact 256-sample
GLM landing mask: `compare()` fails closed unless `double_sparsity_config.channel_mask_path` equals
`/models/glm51-fp8-channel-mask-s256.safetensors` **and** the recorded safetensors `content_sha256` equals
`35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00` (collect reads the header at score time;
overridable via `AC12_EXPECTED_DS_MASK_PATH`/`AC12_EXPECTED_DS_MASK_SHA256`). The verdict records both. A
non-empty-but-wrong path, a wrong/missing hash, and a malformed DS config all fail closed.

**Mandatory accuracy = MMLU within 1.0 pp + within-budget NIAH NON-regression (R10).** The immutable AC-4
makes NIAH "characterization-only / uplift-or-gap", and the mandatory clause is DS-vs-DSA **non-regression**.
So within-budget NIAH uses a **one-sided** check — DS must not be *worse* than DSA by more than 5.0 pp (a
regression fails closed) — and DS *uplift* (DS better) is NOT penalized; the symmetric `within_tolerance` is
still reported for characterization. Beyond-budget NIAH recall is `hits/num_prompts` (harness-consistent:
unserved prompts count as misses), pure characterization (DS's `top_k` index budget bounds long-context
recall by design). Offline-compare unit tests: `test/registered/unit/test_accuracy_gate_compare.py`
(**30 pass** — incl. within-budget regression fail-closed, within-budget uplift NOT penalized,
locked-field-missing-from-both, `dtype`-mismatch, wrong/missing-mask-path + wrong/missing-mask-sha +
malformed-config, per-graph-flag mismatch, partial-service beyond-budget 10%-not-100%). `AC12_INDEX_TOPK=2048`.

### Accuracy RESULT — RUN on 8×H200 (R10), DS-vs-DSA-native on the same node
Sequential collect → collect → offline compare; matched op-point (TP=8, page 64, fp8 KV, radix-off,
`--disable-overlap-schedule --disable-piecewise-cuda-graph --disable-custom-all-reduce`, seed 20260607; only
DS enablement/config + mem-fraction differ — DSA 0.8 / DS 0.7). DS column proven to have served the exact 256
mask (`channel_mask_path` + `content_sha256=35155ac4…2a9b00`). Artifacts + verdict in
`development/loop8/runs/20260608_ac4/` (`dsa_artifact.json`, `ds_artifact.json`, `verdict.json`; the original
symmetric-tolerance verdict preserved as `verdict_symmetric_tolerance_prefix.json`). `run_id=3df21daeae7a7db0`.

| gate | DSA-native | DS (256 mask) | Δ | verdict |
|------|-----------|---------------|---|---------|
| **MMLU** (200 ex, 5-shot, served 200/200 both) | **87.5 %** (175/200) | **87.5 %** (175/200) | 0.0 pp | **PASS** (≤1.0 pp, mandatory) |
| **NIAH within-budget** L=1024 (20 prompts) | 40.0 % | 65.0 % | DS +25 pp | **non-regression PASS** (DS uplift) |
| **NIAH within-budget** L=1536 (20 prompts) | 45.0 % | 70.0 % | DS +25 pp | **non-regression PASS** (DS uplift) |
| NIAH beyond-budget L=4096 (char-only) | 70.0 % | 0.0 % | DS −70 pp | characterization (beyond `top_k`=2048 budget) |
| NIAH beyond-budget L=16384 (char-only) | 30.0 % | 0.0 % | DS −30 pp | characterization |
| NIAH beyond-budget L=65536 (char-only) | 5.0 % | 0.0 % | DS −5 pp | characterization |

**Mandatory accuracy verdict: PASS** (`mandatory_pass=true`). DS-on **matches MMLU exactly** and does **not
regress** within-budget NIAH (it is actually higher within budget). Beyond the 2048-token index budget DS
recall drops to 0 — the **expected, by-design DS long-context limitation** (the mask indexes only `top_k`
tokens), recorded as characterization, NOT a mandatory failure. Repro: `AC12_MODE=collect AC12_SIDE=dsa|ds
AC12_BASE_URL=… AC12_INDEX_TOPK=2048 python development/loop8/accuracy_gate.py` (one server at a time), then
`AC12_MODE=compare AC12_DSA_ARTIFACT=… AC12_DS_ARTIFACT=…`.

### SLO RESULT — locked sweep on 8×H200 (R10), DS-vs-DSA-native, same node
Seed-matched paired launchers (radix-off, `--disable-overlap-schedule --disable-piecewise-cuda-graph
--disable-custom-all-reduce`, seed 20260607; DSA mem 0.8 / DS mem 0.7), gsp 4096 ISL / 512 OSL, conc
16/32/64 × **3 trials × 120 s warmup × 600 s window** (the locked window — supersedes the R5 preliminary
`num_prompts=conc` numbers). `development/benchmark.sh` → `benchmark_compare.py --ac11`. Artifacts:
`development/results/{native_nsa,double_sparsity}_gsp_isl4096_osl512_c{16,32,64}_t{1,2,3}.jsonl`; AC-11 report
`development/loop8/runs/20260608_ac4/slo_ac11_report.txt`.

| conc | DSA decode-TPS (p50) | DS decode-TPS (p50) | DSA P99 TTFT | DS P99 TTFT | DSA SLO | DS SLO |
|------|----------------------|---------------------|--------------|-------------|---------|--------|
| 16 | **41.82** | **23.10** | 7.18 s | 3.67 s | **PASS** | **FAIL** (TPS < 30) |
| 32 | **31.60** | **17.18** | 14.15 s | 42.95 s | **PASS** | **FAIL** (TPS + TTFT) |
| 64 | 26.12 | 17.16 | 28.23 s | 79.42 s | FAIL (TPS + TTFT) | **FAIL** (TPS + TTFT) |

Client SLO bars: decode-TPS ≥ 30 (new def `output_tokens/(e2e−ttft)`), P99 TTFT < 22 s. **DS-on FAILS
decode-TPS at every concurrency** (23/17/17 ≪ 30) and TTFT at conc 32/64; trials are tight (±0.1 TPS).
DSA-native passes conc 16/32 and is throughput-bound at conc 64. AC-11 directional verdict: **FAIL** (DS/DSA
TPS ratio 0.55/0.54/0.66 < 0.95 at all conc). Effective-vs-nominal concurrency shows DS achieves 100/97/93 %
of nominal, so part of the conc-32/64 DS TTFT gap is queue/admission-bound (mem-0.7 KV-pool), not solely
per-request latency. Per the comparator's profiling obligation, a captured DS profile is the documented
directional follow-up for the failing rows (does not change the SLO verdict).

## DEC-2 landing-policy assessment — FINAL (R10, all mandatory gates measured)
- **AC-1 DS-off byte-identical (mandatory): PASS** (task7, R3 — GLM DSA-native byte-identical vs d018026f9).
- **MMLU within tolerance of DSA (mandatory): PASS** — DSA 87.5 % == DS 87.5 % (Δ 0.0 pp ≤ 1.0).
- **DS-vs-DSA non-regression (mandatory): PASS** — served default is DSA-native (unregressed by AC-1); and
  DS-on does not regress within-budget NIAH (DS is higher: 65/70 % vs 40/45 %).
- **SLO decode-TPS ≥ 30 + P99 TTFT < 22 s (mandatory): DS-on FAILS at every concurrency** (locked 3×600 s
  sweep above; 23.10/17.18/17.16 tok/s ≪ 30). DSA-native passes conc 16/32.
- **NIAH / long-context recall (characterization-only):** DS uplift within budget; DS recall 0 beyond the
  2048-token index budget (by design) — characterized, not gated.

**Landing status per DEC-2 (SLO mandatory):** DS-on meets every mandatory accuracy gate but **does not meet
the mandatory SLO**. DEC-2 makes the SLO mandatory-to-land, so **DS-on cannot be the served default**. The
shipped default remains **DSA-native** (AC-1 byte-identical, SLO-passing at conc 16/32). The Loop-8
deliverable is the **reversible, default-OFF DS opt-in** — coherent, accuracy-parity, mask-validated — whose
value is long-context recall scenarios where the trained indexer underperforms, NOT a throughput win on this
standard workload (GLM's trained DSA indexer is strong). Whether the default-OFF opt-in code may LAND despite
the DS-on SLO miss is the explicit DEC-2 plan-evolution decision (see Plan Evolution Log / round-10 summary).

## V3.2-vs-GLM shape matrix
| dim | DeepSeek-V3.2 | GLM-5.1 |
|-----|---------------|---------|
| qk_nope_head_dim | 128 | **192** |
| v_head_dim | 128 | **256** |
| qk_rope_head_dim | 64 | 64 |
| kv_lora_rank | 512 | 512 |
| q_lora_rank | 1536 | **2048** |
| num_hidden_layers | 61 | **78** |
| num_attention_heads | 128 | 64 |
| DSA index_topk | 2048 | 2048 |
| DSA index_head_dim | 128 | 128 |
| DSA index_n_heads | 64* | 32 |
| DS label_dim (calibrated) | 16 | **32** (DEC-3) |
| channel mask `content_sha256` | 36d8bf573091 (16-sample regen) | 35155ac46ad7 (256-sample landing) |

The same inherited DS wiring + bind-time `verify_bind_shapes` gate serve both shapes live (V3.2 128/128,
GLM 192/256) — see task6_serving_smoke.md.

## Repro
**SLO — sequential per column via the PAIRED launchers (launch-arg parity; seed-matched as of R6):**
```bash
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
# DSA-native column:
MODEL_PATH=$GLM MEM_FRACTION_STATIC=0.8 DISABLE_RADIX_CACHE=1 RANDOM_SEED=20260607 PORT=30000 \
  bash development/serve_native_nsa.sh   # then bench_serving (gsp 2253+1843/512) per conc -> parity_dsa_c{16,32,64}.jsonl
# DS column (shut DSA first):
MODEL_PATH=$GLM CHANNEL_MASK_PATH=/models/glm51-fp8-channel-mask-s256.safetensors \
  MEM_FRACTION_STATIC=0.7 RANDOM_SEED=20260607 PORT=30000 \
  bash development/serve_double_sparsity.sh   # then the same bench_serving per conc -> parity_ds_c{16,32,64}.jsonl
python development/benchmark_compare.py --baseline parity_dsa_c32.jsonl --ds parity_ds_c32.jsonl
```
**Full locked landing sweep (next round):** `development/benchmark_baseline.sh` (MODE=native_nsa) +
`benchmark.sh` (MODE=double_sparsity) with `CONCURRENCIES="16 32 64" NUM_PROMPTS=320 TRIALS=3
WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 RANDOM_SEED=20260607`, then `benchmark_compare.py --ac11`.
**Accuracy — sequential collect + offline compare (the executable path; both-URLs-at-once is infeasible):**
```bash
# boot DSA-native, then:
AC12_MODE=collect AC12_SIDE=dsa AC12_BASE_URL=http://127.0.0.1:30000 AC12_INDEX_TOPK=2048 \
  python development/loop8/accuracy_gate.py
# shut DSA, boot DS (256 mask), then:
AC12_MODE=collect AC12_SIDE=ds  AC12_BASE_URL=http://127.0.0.1:30000 AC12_INDEX_TOPK=2048 \
  python development/loop8/accuracy_gate.py
# offline (no server):
AC12_MODE=compare AC12_DSA_ARTIFACT=<dsa.json> AC12_DS_ARTIFACT=<ds.json> \
  python development/loop8/accuracy_gate.py   # exit 0 iff MMLU within 1.0pp AND within-budget NIAH within 5.0pp
```

## Remaining for the final landing record
1. **Accuracy gates** — RUN the collect→collect→compare flow on hardware (path landed R6, scoring run next).
2. **Full locked SLO sweep** (conc 16/32/64 × 3 trials × 600 s) for landing-grade steady-state numbers.
3. **DEC-2 landing decision (user)** — DS-on fails the mandatory SLO at all concurrencies; whether a
   default-off DS opt-in may land anyway (plan framing) vs literal DEC-2 SLO-mandatory is the user's call,
   recorded as plan evolution if relaxed.
