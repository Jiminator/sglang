# Decode-phase profile — dsa_flashmla_auto__flashmla_kv (prefill=flashmla_auto decode=flashmla_kv)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=5821309  (5821.3 ms)
category                       ms   share%   launches
Quantize                  3009.46    51.7%      29160
MoE                        952.04    16.4%      12480
GEMM(dense/other)          943.14    16.2%      46320
Comms                      517.22     8.9%       6800
topk/indexer               138.64     2.4%      18840
other                       99.46     1.7%      40560
elementwise/norm            86.68     1.5%      26680
DSA-indexer                 74.68     1.3%       6640
```
