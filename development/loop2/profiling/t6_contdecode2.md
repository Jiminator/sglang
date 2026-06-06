# Decode-phase profile — `t6_contdecode2`

**Knob:** `--num-continuous-decode-steps 2`

## Capture method (AC-3.1/3.3/3.4)
- torch profiler, non-scoring profile-only run (`profile_candidate.sh`) replaying conc-64 workload, identical flags. Window `start-step 150 / num-steps 40`, **no `--profile-by-stage`** (DECODE+TARGET_VERIFY+DRAFT_EXTEND grouped). TP-0 analyzed. Raw traces deleted (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **24.30** | mean_tpot 41.24 ms | p99_ttft 12081.1 ms (sub-22s=True, info) | accept 3.127 | completed 320/err 0 | conc 61.3 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0; total 2572 ms (~1.03× incumbent 2507 ms))
```
total_kernel_us=2572309  (2572.3 ms)
category                       ms   share%   launches
MoE                        924.26    35.9%      12480
Comms                      513.73    20.0%       6800
Attn(MLA/DSA)              445.48    17.3%       9720
GEMM(dense/other)          269.74    10.5%      39840
topk/indexer               138.88     5.4%      18840
other                       87.88     3.4%      37200
DSA-indexer                 75.17     2.9%       6640
Quantize                    63.29     2.5%      25920
elementwise/norm            53.87     2.1%      20200
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE) — moe, 883.54 ms (34.3%)
2. FlashAttnFwdSm90 (FA3 attn) — gemm, 418.46 ms (16.3%)
3. trtllm_allreduce_fusion (comms) — communication, 346.60 ms (13.5%)

## Summed vs exposed
CUDA-graph-ON decode (graph replay); single-trace overlap analysis found no kernels above the 1% bar → summed ≈ exposed/critical-path here. TP-0 only.

## Verdict (delta vs incumbent combo_baseline 24.08 TPS)
Profile is ~identical to incumbent and gate (24.30) is within noise — consistent with the baseline's <1% exposed idle: there is no scheduling/CPU gap for continuous-decode to reclaim.
