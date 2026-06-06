# Decode-phase profile — dsa_fa3__fa3 (prefill=fa3 decode=fa3)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2535254  (2535.3 ms)
category                       ms   share%   launches
MoE                        924.91    36.5%      12480
Comms                      479.50    18.9%       6800
Attn(MLA/DSA)              444.50    17.5%       9720
GEMM(dense/other)          267.44    10.5%      39840
topk/indexer               138.72     5.5%      18840
other                       87.91     3.5%      37200
DSA-indexer                 74.96     3.0%       6640
Quantize                    63.44     2.5%      25920
elementwise/norm            53.88     2.1%      20200
```
