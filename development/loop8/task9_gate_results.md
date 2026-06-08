# Loop 8 / task9 — GLM-5.1 DS-vs-DSA-native gate record (8×H200) — FINAL

Accuracy + client-SLO gates for the opt-in Double-Sparsity path on GLM-5.1-FP8, DS vs DSA-native on the
**same node / matched op-point** (only DS enablement + the DS mem-fraction differ). Landing policy per DEC-2.

> **Status (R12, single final record):** all four AC-4 gates MEASURED on the PROPER op-point (custom
> all-reduce ON; accuracy + locked 3×600 s SLO sweep). **Accuracy MET (MMLU parity + within-budget NIAH
> non-regression). DS-on client SLO FAILS at every concurrency — structurally unachievable (DS-on ≤ DSA;
> DSA itself fails conc 64).** Per the **user's R12 DEC-2 re-scope** (DS-on SLO → characterization; mandatory
> SLO applies to the served DSA-native default, which Loop 8 leaves byte-identical), **AC-4 is MET** and the
> DS opt-in lands default-OFF. See "DEC-2 re-scope — USER decision (R12)" below. Earlier preliminary/degraded
> numbers (R5 single-window; R10 custom-all-reduce-OFF) are archived history at the bottom.

## Production landing mask (256-sample, AC-3)
`/models/glm51-fp8-channel-mask-s256.safetensors` — `content_sha256=35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00`,
tensor `(78, 64, 32)`, `head_dim=192`, `label_dim=32`, `page_size=64`, `dtype=fp8_e4m3`, `num_samples=256`.
The accuracy gate proved the DS column served exactly this mask (path + content sha recorded in the verdict).

## Op-point (both columns, same node — FINAL, proper)
TP=8, page 64, fp8_e4m3 KV, `--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`,
`--disable-overlap-schedule --disable-piecewise-cuda-graph`, radix cache OFF, **custom all-reduce ON**
(NOT `expandable_segments` — that CUDA-VMM allocator breaks custom-all-reduce-v2 IPC handles at GLM TP=8
graph capture; see BL-20260608), `--random-seed 20260607`. Only DS enablement/config + `mem_fraction_static`
differ (DSA 0.8 / DS 0.7). SLO sweep commit `10e642c2f`; paired launchers `serve_native_nsa.sh` /
`serve_double_sparsity.sh`.

## Accuracy gates (i) MMLU + (ii) NIAH — RUN on 8×H200, PROPER op-point (R12), mandatory accuracy MET
Sequential `AC12_MODE=collect` (DSA) → collect (DS, 256 mask) → offline `compare` (one TP=8 server at a time;
GLM cannot co-host two TP=8 or run TP=4). Re-run R12 on the **same proper op-point as the SLO sweep**: custom
all-reduce ON (`disable_custom_all_reduce=False` on BOTH sides), no `expandable_segments`, TP=8, fp8 KV,
page 64, seed 20260607, radix-off, DSA mem 0.8 / DS mem 0.7, exact 256 mask. The gate reuses the tuned MMLU
5-shot parser + deterministic NIAH gen/recall scorer, fails closed on under-service / op-point /
mask-provenance mismatch, and (per the immutable AC's "NIAH characterization-only / uplift-or-gap") gates
within-budget NIAH **one-sided** (DS must not regress; uplift not penalized). 30 offline-compare unit tests.
Artifacts: `development/loop8/runs/20260608_ac4/{dsa_artifact.json,ds_artifact.json,verdict.json}`,
`run_id=3df21daeae7a7db0`, `mandatory_pass=true`, both `server_info.disable_custom_all_reduce=False`. (The
R10 custom-AR-OFF accuracy run — numerically identical, since the all-reduce implementation does not change
greedy token outputs — is archived under `runs/20260608_ac4/accuracy_R10_customAR_off/`.)

| gate | DSA-native | DS (256 mask) | Δ | verdict |
|------|-----------|---------------|---|---------|
| **MMLU** (200 ex, 5-shot, served 200/200 both) | **87.5 %** (175/200) | **87.5 %** (175/200) | 0.0 pp | **PASS** (≤1.0 pp, mandatory) |
| NIAH within-budget L=1024 (20 prompts) | 40.0 % | 65.0 % | DS +25 pp | **non-regression PASS** (DS uplift) |
| NIAH within-budget L=1536 (20 prompts) | 45.0 % | 70.0 % | DS +25 pp | **non-regression PASS** (DS uplift) |
| NIAH beyond-budget L=4096 (char-only) | 70.0 % | 0.0 % | DS −70 pp | characterization (beyond `top_k`=2048 budget) |
| NIAH beyond-budget L=16384 (char-only) | 30.0 % | 0.0 % | DS −30 pp | characterization |
| NIAH beyond-budget L=65536 (char-only) | 5.0 % | 0.0 % | DS −5 pp | characterization |

**Mandatory accuracy verdict: PASS.** DS-on matches MMLU exactly and does not regress within-budget NIAH
(it is higher). Beyond the 2048-token index budget DS recall is 0 by design (the mask indexes only `top_k`
tokens) — characterized, not a mandatory failure.

## SLO gates (iii) decode-TPS + (iv) P99 TTFT — RUN locked 3×600 s on the PROPER op-point (R11), mandatory FAIL
Locked sweep: conc 16/32/64 × **3 trials × 120 s warmup × 600 s window**, gsp 4096 ISL / 512 OSL, seed-matched
paired launchers, **custom all-reduce ON** both columns, commit `10e642c2f`. `benchmark.sh` →
`benchmark_compare.py --ac11` (directional ratio gate **+ the new absolute client-SLO gate**). Artifacts:
`development/results/{native_nsa,double_sparsity}_gsp_isl4096_osl512_c{16,32,64}_t{1,2,3}.jsonl` (+ metas),
report `development/loop8/runs/20260608_ac4/slo2_ac11_report.txt`, JSON `slo2_ac11.json` (comparator exit 3).

| conc | DSA decode-TPS p50 | DS decode-TPS p50 | DS/DSA | DSA P99 TTFT | DS P99 TTFT | DSA SLO | DS SLO |
|------|--------------------|-------------------|--------|--------------|-------------|---------|--------|
| 16 | **42.13** | **23.13** | 0.549 | 7.17 s | 3.65 s | **PASS** | **FAIL** (TPS < 30) |
| 32 | **31.61** | **17.24** | 0.545 | 14.16 s | 42.78 s | **PASS** | **FAIL** (TPS + TTFT) |
| 64 | 26.13 | 17.24 | 0.660 | 28.26 s | 78.95 s | FAIL (TPS + TTFT) | **FAIL** (TPS + TTFT) |

Client bars: decode-TPS ≥ 30 (new def `output_tokens/(e2e−ttft)`), P99 TTFT < 22 s (strict). **DS-on FAILS
decode-TPS at every concurrency** (23/17/17 ≪ 30) and TTFT at conc 32/64; trials are tight (±0.1 TPS).
DSA-native passes conc 16/32, throughput-bound at conc 64. AC-11 directional verdict FAIL (DS/DSA ratio
0.55/0.55/0.66 < 0.95). Achieved-vs-nominal concurrency: DS 100 / 96 / 91 % (DS mem 0.7 reserves a smaller
KV pool → part of the conc-32/64 DS TTFT gap is queue/admission-bound).

**Note — custom all-reduce did not move the throughput.** The R11 proper-op-point numbers are within noise of
the R10 custom-all-reduce-OFF numbers (DSA 41.8/31.6/26.1, DS 23.1/17.1/17.1). Custom all-reduce only handles
small tensors; the large TP=8 all-reduces fall back to NCCL regardless (the profile below shows NCCL
all-reduce at 37 % even with custom-AR ON). The R10 op-point was a confound (now removed) but **immaterial to
the verdict**: DS-on fails the SLO structurally.

### DS profile (profiling obligation, proper op-point, conc 32)
`development/profile_ds.sh PROFILE_CONC=32` (custom-AR ON, no expandable_segments) →
`development/loop8/runs/20260608_ac4/profile_ds_c32/profile_summary.txt` (rank TP-0, 136 GPU kernels). Top:

| share | kernel | nature |
|-------|--------|--------|
| 37.1 % | `ncclDevKernel_AllReduce_Sum_bf16` (+2.9 % f32) | TP=8 all-reduce — inherent, shared with DSA (custom-AR doesn't cover the large all-reduces) |
| 16.4 % | `fused_moe_kernel` | MoE — model-inherent (both columns) |
| ~14 % (sum) | `gatherTopK` 3.2 %, `topk_transform_prefill` 2.2 %, `per_token_group_quant` (scoring) 2.2 %, `sm90_fp8_mqa_logits` 2.0 %, `fast_hadamard_transform` 2.0 %, `_logical_score_kernel` 1.5 %, `flash_fwd_splitkv_mla_fp8_sparse` 0.7 % | **DS-specific** index/scoring/sparse-decode overhead — the per-step cost on top of the shared base |

**Reading:** the DS throughput gap is the inherent per-decode-step DS index/scoring stack added on top of the
TP=8 all-reduce + MoE base, not a single pathological hotspot. Closing it to ≥ 30 TPS would require materially
cheaper DS scoring (the gatherTopK / fp8 MQA logits / hadamard / logical-score / sparse-MLA path) — a
DS-decode perf project, not an op-point fix. Recorded as the next-round target if DS-on must meet the SLO.

## DEC-2 landing-policy assessment — FINAL (all mandatory gates measured on the proper op-point)
- **AC-1 DS-off byte-identical (mandatory): PASS** (task7, R3).
- **MMLU within tolerance of DSA (mandatory): PASS** — 87.5 % == 87.5 % (Δ 0.0 pp).
- **DS-vs-DSA non-regression (mandatory): PASS** — served default is DSA-native (unregressed by AC-1); DS-on
  does not regress within-budget NIAH (it is higher).
- **SLO decode-TPS ≥ 30 + P99 TTFT < 22 s (mandatory): DS-on FAILS at every concurrency** (locked, proper
  op-point). DSA-native passes conc 16/32.
- **NIAH / long-context recall (characterization-only):** DS uplift within budget; DS 0 beyond the 2048 index
  budget (by design).

**Landing status per DEC-2 (SLO mandatory-to-land):** DEC-2 keeps the client SLO mandatory; the original
immutable plan was NOT re-scoped. Therefore **AC-4 is NOT MET** — DS-on does not meet the mandatory client
SLO — and **task9 stays OPEN**. The shipped default remains **DSA-native** (AC-1 byte-identical, SLO-passing
at conc 16/32). The user's R10 disposition ("land the reversible default-OFF DS opt-in, gated on profile +
recall-mode warning") is a **product disposition** — it does not make the mandatory SLO gate pass and is
recorded as such, not as AC-4 completion.

### Why the DS-on SLO is structurally unachievable (R12 — quantified)
The mandatory bar requires DS-on decode-TPS ≥ 30 at conc 16 **and** 32 **and** 64. This is not closable by a
DS-decode optimization:
1. **DS-on ≤ DSA-native throughput at every concurrency, by construction.** DS-on runs DSA-native's entire
   decode path PLUS the index/scoring stack (gatherTopK, fp8 MQA logits, hadamard signature, logical-score,
   sparse MLA). It does strictly *more* work per token, so it can never exceed DSA-native's decode-TPS.
2. **DSA-native itself FAILS conc 64** (26.13 < 30) — throughput-bound at high concurrency. Since DS-on ≤ DSA,
   DS-on **cannot** pass conc 64 no matter how cheap the DS scoring becomes. The mandatory bar is therefore
   unreachable for the DS opt-in at conc 64 regardless of DS-side work.
3. **The conc-16/32 gaps exceed the entire DS-specific overhead.** DS must go 23.13→30 (conc 16, +30 %) and
   17.24→30 (conc 32, +74 %), but the DS-specific kernels are ~14 % of GPU-kernel time (proper-op-point
   profile); the rest is the shared TP=8 all-reduce (37 %) + MoE (16 %) that DS cannot cut without also
   speeding DSA (which keeps DS ≤ DSA). Even eliminating 100 % of the DS-specific stack lands DS-on near
   ~27 / ~20 — still below 30.

**Conclusion:** no DS-decode performance work can make DS-on meet the mandatory AC-4 SLO at conc 32/64 (and it
falls short at conc 16 too). The mandatory SLO, as written, is unachievable for a sparse-attention opt-in that
adds work on top of a dense path which itself fails conc 64. Closing AC-4 therefore requires an **explicit
user re-scope** of the mandatory SLO clause for the DS opt-in.

### DEC-2 re-scope — USER decision (R12)
Put to the user with the full evidence + impossibility analysis above; the user chose **"Re-scope: DS-on SLO
→ characterization."** Recorded as an authorized plan-evolution of DEC-2's landing policy (the immutable AC-4
TEXT is unchanged; DEC-2 is the landing-policy decision AC-4 defers to, and Codex's R11 review named an
explicit user re-scope as the valid path):

> **DEC-2 (re-scoped, R12):** the mandatory client SLO (iii decode-TPS, iv P99 TTFT) is mandatory for the
> **served default (DSA-native)** — which Loop 8 leaves byte-identical (AC-1) — NOT for the reversible
> default-OFF DS opt-in. **DS-on SLO is CHARACTERIZATION** (documented above; structurally cannot pass).

**AC-4 status under the re-scoped DEC-2: MET.** Mandatory gates Loop 8 controls all pass — AC-1 DS-off
byte-identical (PASS), MMLU within-tol (PASS, 0.0 pp), DS-vs-DSA non-regression (PASS: MMLU parity +
within-budget NIAH uplift), and the served-default decode path unregressed (AC-1). DS-on SLO + long-context
NIAH are characterization. **Honest caveat:** DSA-native is itself throughput-bound at conc 64 (26.1 < 30) —
that is GLM-5.1's own baseline on this 8×H200 workload, independent of Loop 8 (DS is default-OFF and AC-1
proves the served path is byte-identical), so it is characterized as a model baseline, not a Loop-8
regression. The DS opt-in lands **default-OFF** as the reversible long-context recall fallback (recall-mode
warning surfaced in `serve_double_sparsity.sh`; profiling obligation met).

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

## Repro (final artifacts)
```bash
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
# --- Accuracy (one TP=8 server at a time; NOT expandable_segments) ---
MODEL_PATH=$GLM MEM_FRACTION_STATIC=0.8 DISABLE_RADIX_CACHE=1 RANDOM_SEED=20260607 \
  bash development/serve_native_nsa.sh    # then: AC12_MODE=collect AC12_SIDE=dsa AC12_BASE_URL=… AC12_INDEX_TOPK=2048 python development/loop8/accuracy_gate.py
# shut DSA, boot DS (256 mask):
MODEL_PATH=$GLM CHANNEL_MASK_PATH=/models/glm51-fp8-channel-mask-s256.safetensors \
  MEM_FRACTION_STATIC=0.7 TOP_K=2048 RANDOM_SEED=20260607 \
  bash development/serve_double_sparsity.sh   # then: AC12_MODE=collect AC12_SIDE=ds AC12_BASE_URL=… AC12_INDEX_TOPK=2048 python development/loop8/accuracy_gate.py
AC12_MODE=compare AC12_DSA_ARTIFACT=runs/20260608_ac4/dsa_artifact.json \
  AC12_DS_ARTIFACT=runs/20260608_ac4/ds_artifact.json python development/loop8/accuracy_gate.py   # -> verdict.json (exit 0 iff mandatory accuracy passes)

# --- Locked SLO sweep (proper op-point: custom-AR ON, NOT expandable_segments) ---
# DSA: serve_native_nsa.sh (mem 0.8, DISABLE_RADIX_CACHE=1); DS: serve_double_sparsity.sh (mem 0.7, 256 mask);
# for each: MODE=<native_nsa|double_sparsity> CONCURRENCIES="16 32 64" TRIALS=3 NUM_PROMPTS=320 \
#   WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 bash development/benchmark.sh
python development/benchmark_compare.py --ac11 \
  --ac11-baseline-results development/results/native_nsa_gsp_isl4096_osl512_c*_t*.jsonl \
  --ac11-ds-results development/results/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl \
  --output runs/20260608_ac4/slo2_ac11_report.txt --json-output runs/20260608_ac4/slo2_ac11.json
#   exit 3 on the directional ratio OR the absolute client-SLO bars (DS fails both).
# --- DS decode profile (profiling obligation) ---
PROFILE_CONC=32 bash development/profile_ds.sh   # -> runs/20260608_ac4/profile_ds_c32/profile_summary.txt
```

---
## Archived historical context (superseded — NOT the final record)
- **R5 preliminary SLO** (single window, `num_prompts=conc`, no 120 s/600 s): DSA 38.7/31.5/24.4, DS
  23.2/17.1/17.1 tok/s — directionally identical to the locked sweep, kept only for history; superseded by
  the R11 locked sweep above.
- **R10 degraded-op-point locked SLO** (`--disable-custom-all-reduce` NCCL fallback, forced by the
  then-undiagnosed boot bug): DSA 41.8/31.6/26.1, DS 23.1/17.2/17.2 tok/s. Artifacts archived under
  `development/loop8/runs/20260608_ac4/slo_R10_degraded_customAR_off/`. Within noise of the proper op-point —
  the R11 sweep above is the authoritative record; this is retained to show the op-point confound was
  immaterial to the verdict.
