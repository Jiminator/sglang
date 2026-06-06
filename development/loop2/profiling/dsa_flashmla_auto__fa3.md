# Decode-phase profile — dsa_flashmla_auto__fa3 (prefill=flashmla_auto decode=fa3)

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
