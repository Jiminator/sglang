# Decode-phase profile — dsa_flashmla_kv__flashmla_kv (prefill=flashmla_kv decode=flashmla_kv)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=5792841  (5792.8 ms)
category                       ms   share%   launches
Quantize                  3010.31    52.0%      29160
MoE                        946.09    16.3%      12480
GEMM(dense/other)          940.69    16.2%      46320
Comms                      496.07     8.6%       6800
topk/indexer               138.69     2.4%      18840
other                       99.38     1.7%      40560
elementwise/norm            86.79     1.5%      26680
DSA-indexer                 74.81     1.3%       6640
```
