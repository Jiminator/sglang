# Decode profile (main + best config) — bottleneck identification

## GPU time breakdown (decode, 1->512)
- **NCCL communication: 17.4%** (ReduceScatter 12.5% + AllGather 4.9%, 244+249 launches)
- FP8 GEMMs (deep_gemm sm90 fp8): 30.6% across 4 shape variants
- Fused MoE kernel: 16.8% (580 launches)
- FP8 per-token quantize: 2.9% (2135 launches — high per-op overhead)
- MLA attention (flash_fwd): 5.3%
- MoE align + topk: 2.6%

## Hill-climb plan
1. NCCL collectives are 17.4% of decode GPU time → biggest avoidable cost.
   - Try `--enable-nccl-nvls` (H200 NVLink 4 supports NVLS multicast)
   - Try `--enable-flashinfer-allreduce-fusion` (folds AllReduce into GEMM)
   - Try `--enable-symm-mem` (low-overhead symmetric memory for collectives)
2. FP8 GEMMs at 30.6% split across 4 shape variants — suggests no single-shape batching win; would need kernel-level work.
3. fused_moe_kernel and DSA fused store are already "Confirmed fused" — limited fusion room.
