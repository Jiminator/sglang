# Decode-phase profile — `indexcache_loop2`

**Knob:** `--json-model-override-args {index_topk_pattern:...}` (IndexCache, **ACCURACY-RISK**)

## Capture method (AC-3.1/3.3/3.4)
- torch profiler, non-scoring profile-only run (`profile_candidate.sh`) replaying conc-64 workload, identical flags. Window `start-step 150 / num-steps 40`, **no `--profile-by-stage`** (DECODE+TARGET_VERIFY+DRAFT_EXTEND grouped). TP-0 analyzed. Raw traces deleted (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **26.12** | mean_tpot 38.36 ms | p99_ttft 11466.9 ms (sub-22s=True, info) | accept 3.004 | completed 320/err 0 | conc 61.7 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0; total 2436 ms (~0.97× incumbent 2507 ms))
```
total_kernel_us=2435868  (2435.9 ms)
category                       ms   share%   launches
MoE                        952.58    39.1%      12480
Comms                      476.53    19.6%       6800
Attn(MLA/DSA)              447.31    18.4%       9720
GEMM(dense/other)          227.26     9.3%      33760
topk/indexer               117.52     4.8%      17320
other                       69.04     2.8%      29720
elementwise/norm            53.21     2.2%      20080
Quantize                    52.81     2.2%      21360
DSA-indexer                 39.61     1.6%       3600
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE) — moe, 912.19 ms (37.4%)
2. FlashAttnFwdSm90 (FA3 attn) — gemm, 420.28 ms (17.2%)
3. trtllm_allreduce_fusion (comms) — communication, 345.27 ms (14.2%)

## Summed vs exposed
CUDA-graph-ON decode (graph replay); single-trace overlap analysis found no kernels above the 1% bar → summed ≈ exposed/critical-path here. TP-0 only.

## Verdict (delta vs incumbent combo_baseline 24.08 TPS)
IndexCache reuses the DSA indexer result across layers, cutting decode-path indexer compute (the only knob that moved the binding metric in loop 1). Gate TPS and the DSA-indexer category share vs incumbent are the evidence. **Accuracy-risk: this latency benchmark cannot verify output quality — an accuracy eval must gate any production use.**
