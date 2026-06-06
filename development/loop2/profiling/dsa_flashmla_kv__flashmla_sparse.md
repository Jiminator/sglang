# Decode-phase profile — dsa_flashmla_kv__flashmla_sparse (prefill=flashmla_kv decode=flashmla_sparse)

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2574062  (2574.1 ms)
category                       ms   share%   launches
MoE                        946.78    36.8%      12480
Comms                      491.27    19.1%       6800
Attn(MLA/DSA)              402.76    15.6%       3240
GEMM(dense/other)          265.51    10.3%      39840
topk/indexer               138.76     5.4%      18840
elementwise/norm            95.90     3.7%      33160
other                       94.68     3.7%      40440
DSA-indexer                 74.56     2.9%       6640
Quantize                    63.83     2.5%      25920
```
