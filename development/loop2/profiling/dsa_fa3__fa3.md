# Decode-phase profile — `dsa_fa3__fa3`  (prefill=fa3, decode=fa3)

## Capture method (AC-3.1/3.3/3.4)
- Tool: torch profiler (required floor). Non-scoring profile-only run (`profile_candidate.sh`) replaying the identical conc-64 generated-shared-prefix workload; separate server, identical flags.
- Window: `--profile-start-step 150 --profile-num-steps 40` (warmup/cold-prefill excluded). **No `--profile-by-stage`** → the EAGLE decode loop `DECODE + TARGET_VERIFY + DRAFT_EXTEND` is grouped in one window.
- Rank: TP-0 (8 per-rank traces captured; TP-0 analyzed). Steady-state decode (kernel launch counts ~6240 MoE / ~3240 attn confirm verify+draft+decode forwards).
- Raw traces deleted after extraction (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **24.35** | mean_tpot 41.15 ms | p99_ttft 12138.2 ms (sub-22s=True, info) | accept 3.121 | completed 320/err 0 | conc 61.9 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2535254  (2535.3 ms)
category                       ms   share%   launches
MoE                        924.91    36.5%      12480
Comms                      479.50    18.9%       6800
Attn(MLA/DSA)              444.50    17.5%       9720
GEMM(dense/other)          267.44    10.5%      39840
topk/indexer               138.72     5.5%      18840
other                       87.91     3.5%      37200
DSA-indexer                 74.96     3.0%       6640
Quantize                    63.44     2.5%      25920
elementwise/norm            53.88     2.1%      20200
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE experts) — moe, 884.57 ms (34.9%)
2. FlashAttnFwdSm90 (FA3 MLA/DSA attention) — gemm, 416.95 ms (16.4%)
3. trtllm_allreduce_fusion (TP all-reduce) — communication, 350.70 ms (13.8%)

## Summed vs exposed (critical-path) share
CUDA-graph-ON decode (graph replay → ~single serialized stream); no kernels cleared the 1% overlap-attribution bar (single-trace). Under graph replay there is negligible inter-kernel overlap, so **summed kernel time ≈ exposed/critical-path time** here (the category rollup is a credible exposed-time proxy). Caveat: TP-0 only.

## Overlap / fuse notes
- Overlap: no kernels cleared the 1% overlap-attribution bar (single-trace) → no exposed idle gap for overlap/scheduling flags to reclaim.
- Fuse candidates (from analyzer; all CODE fusions = out-of-scope flags-only): CUTLASS FP8 scaled-MM (≈3.2%), Qwen shared-expert append (≈9%), DSA fused quantize+indexed-store (≈7%). Recorded as evidence; not actionable flags-only.

## Delta attribution vs incumbent `combo_baseline` (24.08 TPS)
**No bottleneck shift vs incumbent.** Category profile is within noise of `combo_baseline` (MoE ~36% vs 38.3%, total 2535 ms vs 2507 ms); gate 24.35 TPS ≈ 24.08. `decode=fa3` is FA3-class cost — swapping prefill/decode among {fa3, flashmla_sparse} does not move the binding MoE/comms/attention mix.
