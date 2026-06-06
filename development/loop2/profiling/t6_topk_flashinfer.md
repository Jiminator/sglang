# Decode-phase profile — `t6_topk_flashinfer`

**Knob:** `--dsa-topk-backend flashinfer`

## Capture method (AC-3.1/3.3/3.4)
- torch profiler, non-scoring profile-only run (`profile_candidate.sh`) replaying conc-64 workload, identical flags. Window `start-step 150 / num-steps 40`, **no `--profile-by-stage`** (DECODE+TARGET_VERIFY+DRAFT_EXTEND grouped). TP-0 analyzed. Raw traces deleted (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **20.15** | mean_tpot 49.72 ms | p99_ttft 12963.7 ms (sub-22s=True, info) | accept 3.040 | completed 320/err 0 | conc 61.0 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0; total 3093 ms (~1.23× incumbent 2507 ms))
```
total_kernel_us=3093159  (3093.2 ms)
category                       ms   share%   launches
MoE                        931.78    30.1%      12480
Comms                      521.46    16.9%       6800
elementwise/norm           521.37    16.9%      23432
Attn(MLA/DSA)              450.23    14.6%       9720
GEMM(dense/other)          269.38     8.7%      39840
topk/indexer               172.27     5.6%      18840
other                       88.85     2.9%      37200
DSA-indexer                 74.21     2.4%       6640
Quantize                    63.61     2.1%      25920
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE) — moe, 891.49 ms (28.8%)
2. void at::native::elementwise_kernel<128, 2, at::native… — memory, 467.58 ms (15.1%)
3. FlashAttnFwdSm90 (FA3 attn) — gemm, 423.19 ms (13.7%)

## Summed vs exposed
CUDA-graph-ON decode (graph replay); single-trace overlap analysis found no kernels above the 1% bar → summed ≈ exposed/critical-path here. TP-0 only.

## Verdict (delta vs incumbent combo_baseline 24.08 TPS)
The indexer/topk path is slower with the flashinfer backend; total decode time rises and gate drops to 20.15 TPS (regression). The default `sgl-kernel` topk backend is the better choice.
