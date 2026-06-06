# Decode-phase profile — dsa_flashmla_kv__fa3 (prefill=flashmla_kv decode=fa3)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2546809  (2546.8 ms)
category                       ms   share%   launches
MoE                        943.13    37.0%      12480
Comms                      472.44    18.6%       6800
Attn(MLA/DSA)              445.00    17.5%       9720
GEMM(dense/other)          267.13    10.5%      39840
topk/indexer               138.87     5.5%      18840
other                       88.05     3.5%      37200
DSA-indexer                 74.87     2.9%       6640
Quantize                    63.43     2.5%      25920
elementwise/norm            53.90     2.1%      20200
```
