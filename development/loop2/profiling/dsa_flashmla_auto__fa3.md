# Decode-phase profile — `dsa_flashmla_auto__fa3`  (prefill=flashmla_auto, decode=fa3)

## Capture method (AC-3.1/3.3/3.4)
- Tool: torch profiler (required floor). Non-scoring profile-only run (`profile_candidate.sh`) replaying the identical conc-64 generated-shared-prefix workload; separate server, identical flags.
- Window: `--profile-start-step 150 --profile-num-steps 40` (warmup/cold-prefill excluded). **No `--profile-by-stage`** → the EAGLE decode loop `DECODE + TARGET_VERIFY + DRAFT_EXTEND` is grouped in one window.
- Rank: TP-0 (8 per-rank traces captured; TP-0 analyzed). Steady-state decode (kernel launch counts ~6240 MoE / ~3240 attn confirm verify+draft+decode forwards).
- Raw traces deleted after extraction (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **23.71** | mean_tpot 42.26 ms | p99_ttft 12090.2 ms (sub-22s=True, info) | accept 3.039 | completed 320/err 0 | conc 61.9 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2502838  (2502.8 ms)
category                       ms   share%   launches
MoE                        950.40    38.0%      12480
Attn(MLA/DSA)              445.68    17.8%       9720
Comms                      418.42    16.7%       6800
GEMM(dense/other)          268.73    10.7%      39840
topk/indexer               138.97     5.6%      18840
other                       88.11     3.5%      37200
DSA-indexer                 75.29     3.0%       6640
Quantize                    63.40     2.5%      25920
elementwise/norm            53.85     2.2%      20200
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE experts) — moe, 909.98 ms (36.3%)
2. FlashAttnFwdSm90 (FA3 MLA/DSA attention) — gemm, 418.74 ms (16.7%)
3. trtllm_allreduce_fusion (TP all-reduce) — communication, 346.83 ms (13.8%)

## Summed vs exposed (critical-path) share
CUDA-graph-ON decode (graph replay → ~single serialized stream); no kernels cleared the 1% overlap-attribution bar (single-trace). Under graph replay there is negligible inter-kernel overlap, so **summed kernel time ≈ exposed/critical-path time** here (the category rollup is a credible exposed-time proxy). Caveat: TP-0 only.

## Overlap / fuse notes
- Overlap: no kernels cleared the 1% overlap-attribution bar (single-trace) → no exposed idle gap for overlap/scheduling flags to reclaim.
- Fuse candidates (from analyzer; all CODE fusions = out-of-scope flags-only): CUTLASS FP8 scaled-MM (≈3.2%), Qwen shared-expert append (≈9%), DSA fused quantize+indexed-store (≈7%). Recorded as evidence; not actionable flags-only.

## Delta attribution vs incumbent `combo_baseline` (24.08 TPS)
**No bottleneck shift vs incumbent.** Category profile is within noise of `combo_baseline` (MoE ~38% vs 38.3%, total 2503 ms vs 2507 ms); gate 23.71 TPS ≈ 24.08. `decode=fa3` is FA3-class cost — swapping prefill/decode among {fa3, flashmla_sparse} does not move the binding MoE/comms/attention mix.
