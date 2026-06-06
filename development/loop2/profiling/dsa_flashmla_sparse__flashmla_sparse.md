# Decode-phase profile — dsa_flashmla_sparse__flashmla_sparse (prefill=flashmla_sparse decode=flashmla_sparse)

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
