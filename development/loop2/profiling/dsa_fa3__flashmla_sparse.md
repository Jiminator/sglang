# Decode-phase profile — dsa_fa3__flashmla_sparse (prefill=fa3 decode=flashmla_sparse)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2573315  (2573.3 ms)
category                       ms   share%   launches
MoE                        963.92    37.5%      12480
Comms                      470.62    18.3%       6800
Attn(MLA/DSA)              404.51    15.7%       3240
GEMM(dense/other)          265.47    10.3%      39840
topk/indexer               138.94     5.4%      18840
elementwise/norm            96.06     3.7%      33160
other                       94.71     3.7%      40440
DSA-indexer                 75.20     2.9%       6640
Quantize                    63.88     2.5%      25920
```
