# Decode-phase profile — `dsa_flashmla_kv__fa3`  (prefill=flashmla_kv, decode=fa3)

## Capture method (AC-3.1/3.3/3.4)
- Tool: torch profiler (required floor). Non-scoring profile-only run (`profile_candidate.sh`) replaying the identical conc-64 generated-shared-prefix workload; separate server, identical flags.
- Window: `--profile-start-step 150 --profile-num-steps 40` (warmup/cold-prefill excluded). **No `--profile-by-stage`** → the EAGLE decode loop `DECODE + TARGET_VERIFY + DRAFT_EXTEND` is grouped in one window.
- Rank: TP-0 (8 per-rank traces captured; TP-0 analyzed). Steady-state decode (kernel launch counts ~6240 MoE / ~3240 attn confirm verify+draft+decode forwards).
- Raw traces deleted after extraction (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **20.76** | mean_tpot 48.27 ms | p99_ttft 16654.9 ms (sub-22s=True, info) | accept 3.089 | completed 320/err 0 | conc 61.5 | max_total_num_tokens 300352

## Category rollup (summed kernel GPU time, TP-0)
```
total_kernel_us=2546809  (2546.8 ms)
category                       ms   share%   launches
MoE                        943.13    37.0%      12480
Comms                      472.44    18.6%       6800
Attn(MLA/DSA)              445.00    17.5%       9720
GEMM(dense/other)          267.13    10.5%      39840
topk/indexer               138.87     5.5%      18840
other                       88.05     3.5%      37200
DSA-indexer                 74.87     2.9%       6640
Quantize                    63.43     2.5%      25920
elementwise/norm            53.90     2.1%      20200
```

## Top-3 kernels by GPU time
1. fused_moe_kernel (MoE experts) — moe, 902.77 ms (35.4%)
2. FlashAttnFwdSm90 (FA3 MLA/DSA attention) — gemm, 417.92 ms (16.4%)
3. trtllm_allreduce_fusion (TP all-reduce) — communication, 348.58 ms (13.7%)

## Summed vs exposed (critical-path) share
CUDA-graph-ON decode (graph replay → ~single serialized stream); no kernels cleared the 1% overlap-attribution bar (single-trace). Under graph replay there is negligible inter-kernel overlap, so **summed kernel time ≈ exposed/critical-path time** here (the category rollup is a credible exposed-time proxy). Caveat: TP-0 only.

## Overlap / fuse notes
- Overlap: no kernels cleared the 1% overlap-attribution bar (single-trace) → no exposed idle gap for overlap/scheduling flags to reclaim.
- Fuse candidates (from analyzer; all CODE fusions = out-of-scope flags-only): CUTLASS FP8 scaled-MM (≈3.2%), Qwen shared-expert append (≈9%), DSA fused quantize+indexed-store (≈7%). Recorded as evidence; not actionable flags-only.

## Delta attribution vs incumbent `combo_baseline` (24.08 TPS)
**Decode-window profile is ~identical to the incumbent** (MoE ~37%, total 2547 ms ≈ baseline 2507 ms) — the decode kernels are unchanged. The gate regression to 20.76 TPS is therefore **prefill-side**: `flashmla_kv` prefill re-quantizes the cache (`dsa_backend.py:1846-1848`), and that cost bleeds into the conc-64 chunked-prefill-interleaved decode (not the steady-state decode kernels captured here).
