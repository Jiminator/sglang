# Decode-phase profile — dsa_flashmla_sparse__flashmla_kv (prefill=flashmla_sparse decode=flashmla_kv)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=5793054  (5793.1 ms)
category                       ms   share%   launches
Quantize                  3007.52    51.9%      29160
GEMM(dense/other)          942.57    16.3%      46320
MoE                        940.47    16.2%      12480
Comms                      502.82     8.7%       6800
topk/indexer               138.56     2.4%      18840
other                       99.39     1.7%      40560
elementwise/norm            86.73     1.5%      26680
DSA-indexer                 74.98     1.3%       6640
```
