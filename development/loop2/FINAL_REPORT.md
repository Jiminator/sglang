# GLM-5.1-FP8 Profile-Driven Flags-Only Hill-Climb — Loop 2 Final Report

**Goal:** drive `zai-org/GLM-5.1-FP8` (MLA + DeepSeek-Sparse-Attention / DSA, 256-expert MoE) on one node of 8× H200 (TP8, CUDA) as close as possible to the rebased client SLO, using **only** `sglang serve` CLI flags + `SGLANG_*` env vars (no perf-affecting source/kernel/test/benchmark/SLO edits), measured by the fixed `development/benchmark.sh`. Unlike Loop 1, Loop 2 is **profile-driven**: every candidate is paired with a decode-phase torch-profiler trace, and the profile (not blind sweeping) decides the next knob.

**Workload (fixed):** generated-shared-prefix, 4096 ISL (2253 shared system prompt + 1843 question) / 512 OSL, 320 prompts, max-concurrency 64, ~55% prefix-cache hit, greedy, seed 31234.

## Official metric (rebased)
- **Selection metric: client TPS** `= Σ output_tokens / Σ (latency − ttft) ≈ 1000/mean_tpot_ms`, target ≥ 30.
- **P99 TTFT < 22 s: report-only** (owner decision DEC-2 — recorded per candidate, not a disqualifier). All candidates met it with wide margin (~12–17 s).
- `median ITL` / `1000/ITL`: speculation-inflated cross-check ONLY (never the verdict).

## Result summary (client TPS = Σtok/Σdecode)

| config | client TPS | mean_tpot | p99_ttft | accept | 320/0err | note |
|---|---:|---:|---:|---|---|---|
| **combo (incumbent, recommended safe)** | **24.08** | 41.6 ms | 15.2 s | 3.14 | ✓ | loop1 safe winner; bf16, DSA prefill=flashmla_sparse/decode=fa3 |
| DSA fa3/fa3 (matrix best) | 24.35 | 41.2 ms | 12.1 s | 3.12 | ✓ | ≈ incumbent (within ~1% noise) |
| best flags-only achievable | **~24.1–24.4** | — | — | — | — | **30 TPS NOT reached flags-only** (gap ~5.6, ~19%) |

**Verdict: 30 TPS is not achievable flags-only on this build.** No flag (DSA backend, comms-fusion, topk-backend, scheduling, or speculative) beats the ~24.3 TPS incumbent. P99 TTFT is met with wide margin throughout. The accuracy-risk IndexCache knob (loop1's ~26.5 TPS path) was not re-run in loop 2 (loop 1 already characterized it; out of the profile-driven flags scope here unless requested) — it remains the only knob that moved the binding metric, and only by ~2 TPS, still short of 30.

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

## Profile-directed follow-ups (flags-only) — none beat incumbent
| candidate | client TPS | verdict |
|---|---:|---|
| `--enable-fused-moe-sum-all-reduce` | 23.33 | neutral/slightly worse (comms fusion no help) |
| `--dsa-topk-backend flashinfer` | 20.15 | regress |
| `--dsa-topk-backend torch` | launch-fail | `RuntimeError dsa_topk_backend.py:167` (incompatible with fused-topk CUDA-graph) |
| `--num-continuous-decode-steps 2` | 24.30 | neutral (confirms <1% idle) |
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
Finalist stability (AC-2.1): 3 fresh-server gate repeats = **24.08 / 23.89 / 24.20 → 24.06 ± 0.15 TPS** (all 320/0 err, p99_ttft ~12 s). Stable; consistent with loop-1's combo ×3 (24.2/24.3/24.5).
(`fa3/fa3` 24.35 is within noise and an equally valid pick.)

## Scope (AC-8) — flags-only preserved
`git diff 7800740ad..HEAD -- python sgl-kernel test development/benchmark.sh` is **empty**; `benchmark.sh` byte-unchanged. Resolved args: `tp_size=8, ep_size=1, dp_size=1, moe_a2a_backend=none, enable_torch_compile=False`, no NGRAM/pdmux/deepep/aiter/trtllm. All gate numbers from the unmodified harness.

## Reproducibility
- **SGLang** `0.0.0.dev1+g64e2b54a8`, branch `perf/sglang-hillclimb-c64`, loop2 setup base `7800740ad`.
- **Model** `zai-org/GLM-5.1-FP8`, snapshot `f396cf805182f4ca10fa675e1a99815b3ca384db` (fp8 e4m3), served from `/cluster-storage/models`.
- **Stack** torch 2.11.0+cu130, CUDA 13.0, driver 580.105.08, Ubuntu 24.04.3.
- **GPU** 8× NVIDIA H200 (143771 MiB, SM90); SM clock max 1980 MHz.
- **Harness** `run_candidate.sh` (fresh-server gate), `parse_result.py` (client TPS, TTFT report-only), `profile_candidate.sh` (non-scoring profile-only, steady-state window, no `--profile-by-stage`), `dsa_matrix.sh` / `profile_matrix.sh` / `task6_*` drivers. Profiler insights in `profiling/<tag>.md`; raw traces deleted after extraction (DEC-4). Full gate rows: `sweep_table.md`; coverage: `coverage_log.md`.

## Bottom line
- **Best flags-only ≈ 24.1–24.4 TPS** (incumbent combo 24.08 / matrix best fa3/fa3 24.35); **30 TPS not reachable flags-only**, P99 TTFT met throughout.
- **Profiler-grounded verdict:** MoE/deep-GEMM compute (~49% of decode) is the binding ceiling; the material comms (16.5%) and attention (~26%) slices were each probed flags-only and none helped. Closing the gap requires expert parallelism (EP/a2a — out of scope) or a faster model.
