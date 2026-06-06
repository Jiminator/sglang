# Decode-phase profile — `dsa_flashmla_sparse__flashmla_sparse`  (prefill=flashmla_sparse, decode=flashmla_sparse)

## Capture method (AC-3.1/3.3/3.4)
- Tool: torch profiler (required floor). Non-scoring profile-only run (`profile_candidate.sh`) replaying the identical conc-64 generated-shared-prefix workload; separate server, identical flags.
- Window: `--profile-start-step 150 --profile-num-steps 40` (warmup/cold-prefill excluded). **No `--profile-by-stage`** → the EAGLE decode loop `DECODE + TARGET_VERIFY + DRAFT_EXTEND` is grouped in one window.
- Rank: TP-0 (8 per-rank traces captured; TP-0 analyzed). Steady-state decode (kernel launch counts ~6240 MoE / ~3240 attn confirm verify+draft+decode forwards).
- Raw traces deleted after extraction (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **24.10** | mean_tpot 41.58 ms | p99_ttft 12116.2 ms (sub-22s=True, info) | accept 3.143 | completed 320/err 0 | conc 61.4 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2583767  (2583.8 ms)
category                       ms   share%   launches
MoE                        965.22    37.4%      12480
Comms                      479.24    18.5%       6800
Attn(MLA/DSA)              404.10    15.6%       3240
GEMM(dense/other)          266.84    10.3%      39840
topk/indexer               138.73     5.4%      18840
elementwise/norm            96.26     3.7%      33160
other                       94.64     3.7%      40440
DSA-indexer                 75.02     2.9%       6640
Quantize                    63.71     2.5%      25920
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE experts) — moe, 924.23 ms (35.7%)
2. void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::Kern… — gemm, 404.10 ms (15.6%)
3. trtllm_allreduce_fusion (TP all-reduce) — communication, 345.08 ms (13.3%)

## Summed vs exposed (critical-path) share
CUDA-graph-ON decode (graph replay → ~single serialized stream); no kernels cleared the 1% overlap-attribution bar (single-trace). Under graph replay there is negligible inter-kernel overlap, so **summed kernel time ≈ exposed/critical-path time** here (the category rollup is a credible exposed-time proxy). Caveat: TP-0 only.

## Overlap / fuse notes
- Overlap: no kernels cleared the 1% overlap-attribution bar (single-trace) → no exposed idle gap for overlap/scheduling flags to reclaim.
- Fuse candidates (from analyzer; all CODE fusions = out-of-scope flags-only): CUTLASS FP8 scaled-MM (≈3.2%), Qwen shared-expert append (≈9%), DSA fused quantize+indexed-store (≈7%). Recorded as evidence; not actionable flags-only.

## Delta attribution vs incumbent `combo_baseline` (24.08 TPS)
**No bottleneck shift vs incumbent.** Category profile is within noise of `combo_baseline` (MoE ~37% vs 38.3%, total 2584 ms vs 2507 ms); gate 24.10 TPS ≈ 24.08. `decode=flashmla_sparse` is FA3-class cost — swapping prefill/decode among {fa3, flashmla_sparse} does not move the binding MoE/comms/attention mix.
