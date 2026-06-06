# Decode-phase profile — `combo_baseline` (incumbent)

**Config:** cookbook EAGLE (steps3/topk1/draft4) + `--mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm`, bf16 KV, `SGLANG_ENABLE_SPEC_V2=1`. DSA prefill=flashmla_sparse / decode=fa3 (defaults). TP8, page 64.

**Paired gate result (unprofiled fresh server):** client TPS **24.08** (Σtok/Σdecode), mean_tpot 41.6 ms, accept_length 3.14, p99_ttft 15.2 s (sub-22s, info), 320/0 err, conc 60.6 / max 84, `max_total_num_tokens=300352`.

## Capture method
- **Tool:** torch profiler (the required floor; NVTX/Nsight not used this run). Captured via `profile_candidate.sh` — a non-scoring profile-only run replaying the identical conc-64 generated-shared-prefix workload (separate server, identical flags). Gate number above is from a separate unprofiled run.
- **Window:** `--profile-start-step 150 --profile-num-steps 40` (skips warmup/cold prefill; bounded steady-state window). **No `--profile-by-stage`** — so `DECODE + TARGET_VERIFY + DRAFT_EXTEND` are grouped in one window (avoids the `is_extend()` misclassification of `TARGET_VERIFY` as prefill, `forward_batch_info.py:109-118`).
- **Rank:** TP-0 (8 per-rank traces captured; TP-0 analyzed). Single-rank → cannot prove MoE/routing rank imbalance; aggregate decode-loop attribution only.
- Total kernel GPU time in window (TP-0): **2506.8 ms**.

## Kernel category rollup (summed kernel GPU time, decode loop)
| Category | Share | Note |
|---|---:|---|
| MoE (`fused_moe_kernel` 36.6% + `moe_sum_reduce`) | **38.3%** | core expert compute; 6240 launches |
| Attn MLA/DSA (`FlashAttnFwdSm90` fa3) | **17.8%** | decode attention |
| Comms (`trtllm_allreduce_fusion` + `all_reduce_two_shot`) | **16.5%** | TP8 all-reduce |
| GEMM dense/other (`deep_gemm` fp8 sm90) | 10.7% | q/kv/o + gate/up/down + lm_head; some MoE-adjacent |
| topk/indexer (`gatherTopK`, `topk_transform_decode`, `bitonicSort`) | 5.5% | DSA indexer + EAGLE topk |
| DSA-indexer (`sm90_fp8_paged_mqa_logits`) | 3.0% | sparse-attn logits |
| Quantize (`per_token_group_quant_8bit`) | 2.5% | fp8 activation quant |
| elementwise/norm + other | 5.7% | rmsnorm/rotary/sort/misc |

**Top-3 kernels by time:** `fused_moe_kernel` 36.6% (919 ms) · `FlashAttnFwdSm90` 16.7% (418 ms) · `trtllm_allreduce_fusion` 13.9% (348 ms).

## Summed vs exposed (critical-path) share
This is a **CUDA-graph-ON** decode trace (graph replay → essentially one serialized stream). The single-trace overlap analysis found **no** kernels above the 1% overlap-attribution bar, i.e. negligible inter-kernel overlap. Under graph replay the decode loop is serialized, so **summed kernel time ≈ exposed/critical-path time** here — the rollup above is a credible exposed-time proxy (caveat: TP-0 only; a mapping+formal two-trace pass would be needed to attribute cross-stream overlap precisely, not pursued since overlap is negligible).

## EAGLE verify/draft share
`fused_moe_kernel` launches ≈ **6240 vs ≈3240** attention launches (~2×), and many kernels carry a 2–26% site-share on `eagle_draft_cuda_graph_runner.py:248 (_replay)`. So the EAGLE draft model + `TARGET_VERIFY` forwards add a large fraction of the MoE/GEMM invocations. With accept_length 3.14 this amortizes real decode steps, but at conc 64 the extra speculative MoE/GEMM work is a **major** part of decode cost — consistent with loop 1's finding that lighter-EAGLE regresses (it sheds acceptance faster than it sheds the expensive forwards). Verify/draft cost is **not** negligible; it is woven through the MoE/GEMM buckets.

## Bottleneck verdict (profiler-grounded)
The incumbent is **MoE/deep-GEMM dominated but NOT purely so.** MoE alone is 38.3% (compute approaches half the window with dense GEMM), but **attention/MLA-DSA (17.8%), all-reduce comms (16.5%), and indexer/topk (~8.5%) are each too large to dismiss.** This is more nuanced than loop 1's "hard MoE compute" hypothesis: the ≥30-TPS gap will not close from generic flags alone (no flag removes the 38% MoE bucket without expert parallelism, which is out of scope), but the profile points to **plausible flags-only headroom** in: (a) all-reduce/comms overlap or backend choice (16.5% is large for decode), (b) attention/DSA-indexer behavior (justifies exhausting the DSA prefill×decode matrix — mandated no-prune anyway), (c) scheduling/batching that reduces verify/draft fragmentation, (d) EAGLE acceptance-vs-cost tuning.

## Fuse-pattern candidates (from analyzer — all CODE fusions = OUT OF SCOPE flags-only)
Recorded as evidence, not actionable here (flags-only): CUTLASS FP8 scaled-MM replacing nvjet (3.2%, PR #22392); Qwen-style shared-expert append into routed top-k (9.0%); DSA fused quantize + indexed K-cache store (7.1%). These would require source changes (out of scope) — they corroborate that real headroom exists but lies in kernel/fusion territory, not flags.

## Follow-ups this profile directs (in-scope, flags-only)
1. **Comms (16.5%)** — probe any all-reduce/overlap-scheduling flags; check whether overlap scheduling is already on.
2. **Attention/indexer (~26% incl. indexer)** — exhaust the DSA prefill×decode matrix (no-prune); attribute each cell's delta to the FA3/indexer kernels.
3. **EAGLE acceptance-vs-cost** — characterize, but topk>1 tree is launch-rejected on DSA (see EAGLE-tree note).

_Raw traces (8× per-rank, 160 MB) deleted after this extraction (disk hygiene). Analyzer triage saved in `_work/combo_baseline_TP0_triage.md`._
