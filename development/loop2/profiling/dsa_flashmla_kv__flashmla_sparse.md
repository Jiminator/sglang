# Decode-phase profile — `dsa_flashmla_kv__flashmla_sparse`  (prefill=flashmla_kv, decode=flashmla_sparse)

## Capture method (AC-3.1/3.3/3.4)
- Tool: torch profiler (required floor). Non-scoring profile-only run (`profile_candidate.sh`) replaying the identical conc-64 generated-shared-prefix workload; separate server, identical flags.
- Window: `--profile-start-step 150 --profile-num-steps 40` (warmup/cold-prefill excluded). **No `--profile-by-stage`** → the EAGLE decode loop `DECODE + TARGET_VERIFY + DRAFT_EXTEND` is grouped in one window.
- Rank: TP-0 (8 per-rank traces captured; TP-0 analyzed). Steady-state decode (kernel launch counts ~6240 MoE / ~3240 attn confirm verify+draft+decode forwards).
- Raw traces deleted after extraction (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **20.74** | mean_tpot 48.31 ms | p99_ttft 16679.2 ms (sub-22s=True, info) | accept 3.083 | completed 320/err 0 | conc 61.9 | max_total_num_tokens 300352

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

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE experts) — moe, 905.87 ms (35.2%)
2. void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::Kern… — gemm, 402.76 ms (15.6%)
3. trtllm_allreduce_fusion (TP all-reduce) — communication, 354.78 ms (13.8%)

## Summed vs exposed (critical-path) share
CUDA-graph-ON decode (graph replay → ~single serialized stream); no kernels cleared the 1% overlap-attribution bar (single-trace). Under graph replay there is negligible inter-kernel overlap, so **summed kernel time ≈ exposed/critical-path time** here (the category rollup is a credible exposed-time proxy). Caveat: TP-0 only.

## Overlap / fuse notes
- Overlap: no kernels cleared the 1% overlap-attribution bar (single-trace) → no exposed idle gap for overlap/scheduling flags to reclaim.
- Fuse candidates (from analyzer; all CODE fusions = out-of-scope flags-only): CUTLASS FP8 scaled-MM (≈3.2%), Qwen shared-expert append (≈9%), DSA fused quantize+indexed-store (≈7%). Recorded as evidence; not actionable flags-only.

## Delta attribution vs incumbent `combo_baseline` (24.08 TPS)
**Decode-window profile is ~identical to the incumbent** (MoE ~37%, total 2574 ms ≈ baseline 2507 ms) — the decode kernels are unchanged. The gate regression to 20.74 TPS is therefore **prefill-side**: `flashmla_kv` prefill re-quantizes the cache (`dsa_backend.py:1846-1848`), and that cost bleeds into the conc-64 chunked-prefill-interleaved decode (not the steady-state decode kernels captured here).
