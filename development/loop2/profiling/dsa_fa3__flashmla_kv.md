# Decode-phase profile — `dsa_fa3__flashmla_kv`  (prefill=fa3, decode=flashmla_kv)

## Capture method (AC-3.1/3.3/3.4)
- Tool: torch profiler (required floor). Non-scoring profile-only run (`profile_candidate.sh`) replaying the identical conc-64 generated-shared-prefix workload; separate server, identical flags.
- Window: `--profile-start-step 150 --profile-num-steps 40` (warmup/cold-prefill excluded). **No `--profile-by-stage`** → the EAGLE decode loop `DECODE + TARGET_VERIFY + DRAFT_EXTEND` is grouped in one window.
- Rank: TP-0 (8 per-rank traces captured; TP-0 analyzed). Steady-state decode (kernel launch counts ~6240 MoE / ~3240 attn confirm verify+draft+decode forwards).
- Raw traces deleted after extraction (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **14.74** | mean_tpot 67.95 ms | p99_ttft 12201.5 ms (sub-22s=True, info) | accept 3.078 | completed 320/err 0 | conc 60.1 | max_total_num_tokens 300352

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

## Top-3 kernels by GPU time
1. _quantize_k_cache_fast_kernel — quantize, 2947.08 ms (50.5%)
2. fused_moe_kernel (MoE experts) — moe, 893.46 ms (15.3%)
3. void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_f… — attention, 661.87 ms (11.3%)

## Summed vs exposed (critical-path) share
CUDA-graph-ON decode (graph replay → ~single serialized stream); no kernels cleared the 1% overlap-attribution bar (single-trace). Under graph replay there is negligible inter-kernel overlap, so **summed kernel time ≈ exposed/critical-path time** here (the category rollup is a credible exposed-time proxy). Caveat: TP-0 only.

## Overlap / fuse notes
- Overlap: no kernels cleared the 1% overlap-attribution bar (single-trace) → no exposed idle gap for overlap/scheduling flags to reclaim.
- Fuse candidates (from analyzer; all CODE fusions = out-of-scope flags-only): CUTLASS FP8 scaled-MM (≈3.2%), Qwen shared-expert append (≈9%), DSA fused quantize+indexed-store (≈7%). Recorded as evidence; not actionable flags-only.

## Delta attribution vs incumbent `combo_baseline` (24.08 TPS)
**Regression attributed to the decode quantize tax.** Total decode-loop kernel time balloons to **5835 ms (~2.3× the incumbent's 2507 ms)** and the **Quantize category jumps to ~52%** (vs ~2.5% baseline): `_forward_flashmla_kv` re-quantizes the whole bf16 KV cache every decode step (`dsa_backend.py:1846-1848`). This is the direct cause of the gate drop to 14.74 TPS vs incumbent 24.08.
