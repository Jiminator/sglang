# Decode-phase profile — `t6_fused_moe_sum_ar`

**Knob:** `--enable-fused-moe-sum-all-reduce`

## Capture method (AC-3.1/3.3/3.4)
- torch profiler, non-scoring profile-only run (`profile_candidate.sh`) replaying conc-64 workload, identical flags. Window `start-step 150 / num-steps 40`, **no `--profile-by-stage`** (DECODE+TARGET_VERIFY+DRAFT_EXTEND grouped). TP-0 analyzed. Raw traces deleted (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **23.33** | mean_tpot 42.95 ms | p99_ttft 13478.1 ms (sub-22s=True, info) | accept 3.070 | completed 320/err 0 | conc 61.6 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0; total 2572 ms (~1.03× incumbent 2507 ms))
```
total_kernel_us=2571953  (2572.0 ms)
category                       ms   share%   launches
MoE                       1004.62    39.1%       9360
Attn(MLA/DSA)              444.51    17.3%       9720
Comms                      422.64    16.4%       6800
GEMM(dense/other)          268.57    10.4%      39840
topk/indexer               138.93     5.4%      18840
other                       88.00     3.4%      37200
DSA-indexer                 74.62     2.9%       6640
elementwise/norm            66.63     2.6%      26440
Quantize                    63.41     2.5%      25920
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE) — moe, 994.96 ms (38.7%)
2. FlashAttnFwdSm90 (FA3 attn) — gemm, 417.29 ms (16.2%)
3. trtllm_allreduce_fusion (comms) — communication, 348.95 ms (13.6%)

## Summed vs exposed
CUDA-graph-ON decode (graph replay); single-trace overlap analysis found no kernels above the 1% bar → summed ≈ exposed/critical-path here. TP-0 only.

## Verdict (delta vs incumbent combo_baseline 24.08 TPS)
Comms remains a material slice in the profile and total decode time is unchanged vs incumbent — fusing the MoE-sum into the all-reduce does **not** reduce exposed comms/decode time at conc 64 (gate 23.33 ≈ incumbent, slightly worse). The 16.5% all-reduce is critical-path TP8 cost that no flag removes without expert parallelism.
