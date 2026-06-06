# Decode-phase profile — dsa_fa3__flashmla_kv (prefill=fa3 decode=flashmla_kv)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=5835277  (5835.3 ms)
category                       ms   share%   launches
Quantize                  3010.49    51.6%      29160
GEMM(dense/other)          941.12    16.1%      46320
MoE                        933.84    16.0%      12480
Comms                      550.36     9.4%       6800
topk/indexer               138.55     2.4%      18840
other                       99.36     1.7%      40560
elementwise/norm            86.70     1.5%      26680
DSA-indexer                 74.87     1.3%       6640
```
