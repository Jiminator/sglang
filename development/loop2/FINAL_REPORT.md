# GLM-5.1-FP8 Profile-Driven Flags-Only Hill-Climb — Loop 2 Final Report

> **TL;DR — final recommended config (safe, no accuracy risk): `combo` ≈ 24.06 ± 0.15 TPS.**
> ```
> SGLANG_ENABLE_SPEC_V2=1 sglang serve --model-path zai-org/GLM-5.1-FP8 --tp 8 \
>   --reasoning-parser glm45 --tool-call-parser glm47 \
>   --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
>   --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 \
>   --chunked-prefill-size 4096 --schedule-policy lpm
> ```
> Best **achievable** (accuracy-risk, needs an accuracy eval): add `--json-model-override-args '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'` → ≈ 26.43 ± 0.38 TPS.
> **30 TPS is not reachable flags-only** on this build (binding cost is MoE/deep-GEMM compute; needs expert parallelism = out of scope). Fixed at TP8; attention-backend grid = the DSA prefill×decode 16-cell matrix (exhausted).

**Goal:** drive `zai-org/GLM-5.1-FP8` (MLA + DeepSeek-Sparse-Attention / DSA, 256-expert MoE) on one node of 8× H200 (TP8, CUDA) as close as possible to the rebased client SLO, using **only** `sglang serve` CLI flags + `SGLANG_*` env vars (no perf-affecting source/kernel/test/benchmark/SLO edits), measured by the fixed `development/benchmark.sh`. Unlike Loop 1, Loop 2 is **profile-driven**: every candidate is paired with a decode-phase torch-profiler trace, and the profile (not blind sweeping) decides the next knob.

**Workload (fixed):** generated-shared-prefix, 4096 ISL (2253 shared system prompt + 1843 question) / 512 OSL, 320 prompts, max-concurrency 64, ~55% prefix-cache hit, greedy, seed 31234.

## Official metric (rebased)
- **Selection metric: client TPS** `= Σ output_tokens / Σ (latency − ttft) ≈ 1000/mean_tpot_ms`, target ≥ 30.
- **P99 TTFT < 22 s: report-only** (owner decision DEC-2 — recorded per candidate, not a disqualifier). All candidates met it with wide margin (~12–17 s).
- `median ITL` / `1000/ITL`: speculation-inflated cross-check ONLY (never the verdict).

## Result summary (client TPS = Σtok/Σdecode)

| config | client TPS (3-run mean ± range) | mean_tpot | p99_ttft | accept | 320/0err | note |
|---|---:|---:|---:|---|---|---|
| **combo (incumbent, recommended SAFE)** | **24.06 ± 0.15** (24.08/23.89/24.20) | 41.6 ms | ~12–15 s | 3.14 | ✓ | loop1 safe winner; bf16, DSA prefill=flashmla_sparse/decode=fa3 |
| DSA fa3/fa3 (safe, alt) | 24.17 ± 0.27 (24.35/23.83/24.32) | 41.2 ms | ~12 s | 3.1 | ✓ | statistically indistinguishable from combo |
| **combo+IndexCache (recommended BEST-ACHIEVABLE, ⚠ ACCURACY-RISK)** | **26.43 ± 0.38** (26.12/26.87/26.30) | ~38 ms | ~11.4 s | 3.0–3.12 | ✓ | only knob that moves the metric; +2.4 TPS via DSA-indexer reuse |

**Verdict: 30 TPS is not achievable flags-only on this build.** Best **safe** (no accuracy risk) = **combo ≈ 24.1 TPS**; best **achievable** = **combo+IndexCache ≈ 26.4 TPS** (accuracy-risk, gap to 30 ≈ 3.6 / ~12%). No DSA-backend, comms-fusion, topk-backend, scheduling, or speculative flag beats these. P99 TTFT met with wide margin throughout (report-only).

### Safe finalist resolution (AC-2.1)
`combo` (24.06 ± 0.15) and `fa3/fa3` (24.17 ± 0.27) overlap within run-to-run variance — **statistically indistinguishable**. We recommend **combo** as the stable default-safe config (it is the loop-1-established incumbent and uses default DSA backends); `fa3/fa3` is an equivalent alternative, not a distinct win.

### Best-achievable (accuracy-risk) — combo+IndexCache
3 fresh-server gate repeats: **26.12 / 26.87 / 26.30 → 26.43 ± 0.38 TPS**, p99_ttft ~11.4 s, accept ~3.0–3.12, 320/0 err. Mechanism confirmed by profile (`profiling/indexcache_loop2.md`): the DSA-indexer category drops from 3.0% → **1.6%** and indexer-logits kernel launches **halve** (6640 → 3600) because IndexCache reuses the DSA indexer result across layers — total decode-loop kernel time 2436 ms vs 2507 ms incumbent. This is the only knob that cut the *binding decode-path* compute. **⚠ Accuracy-risk:** IndexCache reuses the indexer across layers (`index_topk_pattern`); SGLang docs state "negligible accuracy loss," but this latency benchmark **cannot verify quality** — an accuracy eval must gate any production use. Still short of 30 TPS.

## Central question — answered with profiler evidence

**Is the ~24–27 TPS decode ceiling hard MoE-GEMM compute, or is there flags-only overlap/fusion/scheduling/attention headroom?**

**Answer: the ceiling is MoE/deep-GEMM-compute-dominated, and the residual material slices (comms, attention) are NOT flags-only-addressable on this fixed TP8 path.** Decode-loop kernel rollup (incumbent, TP-0, steady-state conc-64 window grouping `DECODE+TARGET_VERIFY+DRAFT_EXTEND`; CUDA-graph-on so summed ≈ exposed, overlap <1%):

| category | share | flags-only headroom? |
|---|---:|---|
| MoE (`fused_moe_kernel` + sum) | 38.3% | No — no flag removes it without expert parallelism (EP/a2a, out of scope) |
| dense/other GEMM (`deep_gemm` fp8) | 10.7% | No — q/kv/o + gate/up/down + lm_head compute |
| Attn MLA/DSA (`FlashAttnFwdSm90`) | 17.8% | No — DSA backend matrix exhausted, all decode∈{fa3,sparse} ≈ identical; flashmla_kv regresses |
| Comms (all-reduce) | 16.5% | No — `--enable-fused-moe-sum-all-reduce` is neutral (23.33); all-reduce is critical-path TP8 cost |
| indexer/topk + DSA-indexer | ~8.5% | No — `--dsa-topk-backend flashinfer` regresses (20.15), `torch` fails to launch |
| quantize/elementwise/other | ~8% | No |

Compute (MoE + dense GEMM) is ~49% of decode time and is the binding cost; it needs expert parallelism (out of scope) or a smaller/faster model. The two material non-MoE slices were each directly probed flags-only and **none helped** (comms-fusion neutral; topk backends regress/fail; the profile's <1% exposed idle rules out scheduling/overlap knobs). EAGLE adds ~2× MoE launches (verify+draft); it stays ON because it improves slot turnover / P99 TTFT, but it does not lift the decode rate (consistent with loop 1).

## DSA prefill × decode cross-product (bf16) — exhausted (no pruning, DEC-3)

Client TPS, every cell launch-attempted (12 launchable fully gate+profile measured, 4 rejected):

| prefill ＼ decode | flashmla_sparse | flashmla_kv | flashmla_auto | fa3 |
|---|---:|---:|:---:|---:|
| flashmla_sparse | 24.10 | 15.18 | ❌ | 24.08 |
| flashmla_kv | 20.74 | 13.55 | ❌ | 20.76 |
| flashmla_auto | 23.57 | 14.72 | ❌ | 23.71 |
| fa3 | 24.19 | 14.74 | ❌ | **24.35** |

- **decode ∈ {fa3, flashmla_sparse}:** ~24 TPS, flat — identical decode kernel profiles (FA3-class cost); no backend shifts the bottleneck.
- **decode = flashmla_kv:** severe regression (~14 TPS) — profile attributes it: the **Quantize category explodes to 52%** (3007 ms vs 63 ms), total kernel time 2.3× — `_forward_flashmla_kv` re-quantizes the whole bf16 cache every step (`dsa_backend.py:1846-1848`).
- **prefill = flashmla_kv:** ~20.7 TPS — decode profile is normal; the regression is prefill-side quantize bleeding into the conc-64 interleaved decode.
- **decode = flashmla_auto (all 4):** startup-reject — `ValueError: Unsupported dsa_impl = 'flashmla_auto' for forward_extend` (`dsa_backend.py:1567`); `flashmla_auto` is prefill-auto-select only.

## EAGLE-tree axis (topk>1) — infeasible flags-only
`--speculative-eagle-topk 2` → hard `ValueError` at launch (`speculative_hook.py:388`): topk>1 + page_size>1 is only supported for `flashinfer`/`fa3`, but DSA forces `attention_backend=dsa` + `page_size=64`. Closed by citation; the incumbent topk=1 verify/draft cost is characterized from the profile (above).

## Attention-backend coverage (note)
This model is MLA + DeepSeek-Sparse-Attention, so the top-level `attention_backend` is **forced to `dsa`** by SGLang (`server_args.py:1828-1854`) — flashinfer/fa3/triton as a *top-level* attention backend are not applicable. The attention-backend search therefore lives in the **DSA prefill×decode sub-kernel grid above** (`flashmla_sparse / flashmla_kv / flashmla_auto / fa3`), which was exhausted (16 cells). No top-level attention-backend axis remains.

## Parallelism axis (TP / DP-attention / EP) — fixed at TP8, by constraint
Loop 2 did **not** grid-search model parallelism; the parallelism axis is settled by scope + prior evidence, not re-swept here:
- **TP8** is the only viable full-node configuration — the FP8 weights (~756 GB on cluster storage; ~89 GB/GPU resident) require all 8× H200; a smaller TP would not fit and would reduce decode compute. All runs are `tp_size=8, pp_size=1`.
- **DP-attention** (`--enable-dp-attention`): **not re-run** — loop 1 already measured it as a clear regression at this concurrency (per-rank batch collapses to 64/8≈8 and DP-attn↔TP-MoE adds all-gather/reduce-scatter every layer; TPOT 48.6 ms vs ~41). It is a high-concurrency/KV-bound optimization; this workload is neither. The loop-2 profile corroborates the mechanism: batch-splitting (TBO/SBO, same family) was closed because the decode is compute-saturated (<1% idle), so shrinking per-rank GEMMs only hurts.
- **EP / MoE all-to-all** (`--moe-a2a-backend`, deepep): **explicitly out of scope** (draft exclusion). The profile shows this is exactly where the real headroom is (MoE GEMM ~38%, ~49% with dense GEMM), so EP is the *recommended next step beyond flags-only* — but it requires the out-of-scope code/parallelism path.
Resolved parallelism in every reported config: `tp_size=8, ep_size=1, dp_size=1, moe_a2a_backend=none` (verified, AC-8).

## Profile-directed follow-ups (flags-only) — none beat incumbent (all profile-backed)
Each launchable candidate has a decode-phase profile (`profiling/t6_*.md`); profile-backed attribution:
| candidate | client TPS | profile evidence | verdict |
|---|---:|---|---|
| `--enable-fused-moe-sum-all-reduce` | 23.33 | Comms stays **16.4%** (≈ baseline 16.5%), total 2572 ms ≈ 2507 | no help — fusing MoE-sum into all-reduce doesn't cut exposed comms |
| `--dsa-topk-backend flashinfer` | 20.15 | total kernel time **rises to 3093 ms** (topk/indexer path slower) | regress |
| `--dsa-topk-backend torch` | launch-fail | — | startup-reject `RuntimeError dsa_topk_backend.py:167` (incompatible with fused-topk CUDA-graph) |
| `--num-continuous-decode-steps 2` | 24.30 | profile ≈ incumbent (total 2572 ms) | neutral (confirms <1% idle — no scheduling gap) |
`--enable-two/single-batch-overlap` not run: profile shows <1% exposed idle (compute-saturated) so no overlap gap to fill, and batch-split shrinks MoE GEMMs at conc 64 (loop-1 DP-attn mechanism) — closed with profiler evidence.

## Recommended config (safe, no accuracy risk)
```
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path zai-org/GLM-5.1-FP8 --tp 8 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.85 --max-running-requests 64 \
  --chunked-prefill-size 4096 --schedule-policy lpm
```
Gate result (unprofiled fresh server): client TPS **24.08**, p99_ttft 15.2 s, accept 3.14, 320/0 err, conc 60.6, `max_total_num_tokens=300352`, bf16 KV, DSA prefill=flashmla_sparse/decode=fa3, page 64.
Finalist stability (AC-2.1): 3 fresh-server gate repeats = **24.08 / 23.89 / 24.20 → 24.06 ± 0.15 TPS** (all 320/0 err, p99_ttft ~12 s). Stable; consistent with loop-1's combo ×3 (24.2/24.3/24.5). `fa3/fa3` (24.17 ± 0.27 over 3 repeats) is statistically indistinguishable — combo is recommended as the stable default-safe config.

## Recommended config (best-achievable — ⚠ ACCURACY-RISK)
Same as the safe config plus:
```
  --json-model-override-args '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
```
Gate (3 repeats): **26.12 / 26.87 / 26.30 → 26.43 ± 0.38 TPS**, p99_ttft ~11.4 s, accept ~3.0–3.12, 320/0 err. Closest to 30 (gap ~3.6). **Must be gated by an accuracy eval before production use** (this latency benchmark cannot verify quality).

## Scope (AC-8) — flags-only preserved
`git diff 7800740ad..HEAD -- python sgl-kernel test development/benchmark.sh` is **empty**; `benchmark.sh` byte-unchanged. Resolved args: `tp_size=8, ep_size=1, dp_size=1, moe_a2a_backend=none, enable_torch_compile=False`, no NGRAM/pdmux/deepep/aiter/trtllm. All gate numbers from the unmodified harness.

## Reproducibility
- **SGLang** `0.0.0.dev1+g64e2b54a8`, branch `perf/sglang-hillclimb-c64`, loop2 setup base `7800740ad`.
- **Model** `zai-org/GLM-5.1-FP8`, snapshot `f396cf805182f4ca10fa675e1a99815b3ca384db` (fp8 e4m3), served from `/cluster-storage/models`.
- **Stack** torch 2.11.0+cu130, CUDA 13.0, driver 580.105.08, Ubuntu 24.04.3.
- **GPU** 8× NVIDIA H200 (143771 MiB, SM90); SM clock max 1980 MHz.
- **Harness** `run_candidate.sh` (fresh-server gate), `parse_result.py` (client TPS, TTFT report-only), `profile_candidate.sh` (non-scoring profile-only, steady-state window, no `--profile-by-stage`), `dsa_matrix.sh` / `profile_matrix.sh` / `task6_*` drivers. Profiler insights in `profiling/<tag>.md`; raw traces deleted after extraction (DEC-4). Full gate rows: `sweep_table.md`; coverage: `coverage_log.md`.

## Bottom line
- **Best SAFE ≈ 24.06 ± 0.15 TPS** (combo; fa3/fa3 indistinguishable). **Best ACHIEVABLE ≈ 26.43 ± 0.38 TPS** (combo+IndexCache, accuracy-risk). **30 TPS not reachable flags-only** (safe gap ~5.9, best-achievable gap ~3.6); P99 TTFT met throughout.
- **Profiler-grounded verdict:** MoE/deep-GEMM compute (~49% of decode) is the binding ceiling; the material comms (16.5%) and attention (~26%) slices were each probed flags-only and none helped (fused-comms neutral, topk-backends regress/fail, scheduling inert). The **only** knob that cut the binding decode cost is IndexCache (+2.4 TPS, via DSA-indexer reuse 3.0%→1.6% — profile-confirmed), accuracy-risk and still short of 30. Closing the remaining gap requires expert parallelism (EP/a2a — out of scope) or a faster model.
