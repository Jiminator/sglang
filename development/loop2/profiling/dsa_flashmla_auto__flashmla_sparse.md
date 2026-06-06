# Decode-phase profile — dsa_flashmla_auto__flashmla_sparse (prefill=flashmla_auto decode=flashmla_sparse)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2631759  (2631.8 ms)
category                       ms   share%   launches
MoE                        953.38    36.2%      12480
Comms                      539.24    20.5%       6800
Attn(MLA/DSA)              404.55    15.4%       3240
GEMM(dense/other)          266.56    10.1%      39840
topk/indexer               138.79     5.3%      18840
elementwise/norm            96.05     3.6%      33160
other                       94.65     3.6%      40440
DSA-indexer                 74.86     2.8%       6640
Quantize                    63.68     2.4%      25920
```
