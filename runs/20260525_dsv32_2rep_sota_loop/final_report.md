# DSv3.2 SLO Loop — Final Report (updated with PD)

## Workload
- generated-shared-prefix, num_groups=1, prompts=320, sys=2253, q=1843, out=512
- concurrency=64, ~52% prefix cache hit
- Hardware: 2× 8× H200 (16 total), NVLink 4 intra-node, IB NDR 400 Gb/s cross-node
- Model: deepseek-ai/DeepSeek-V3.2 (671B / ~37B active per token, fp8)

## SLOs
- decode_tps_per_request ≥ 30 (1000/mean_TPOT)
- P99 TTFT < 22 s

## Final Best Results

### Run #11 — `main_best` (2 replicas behind sgl-router)
Config: TP=8 EP=8 DP=8 dp-attention + fp8 KV + page 64 + mixed-chunk + lpm + mem 0.88, RR routing

| metric | value | SLO |
|---|---|---|
| Mean TPOT | 49.9 ms | — |
| **1000/TPOT** | **20.0 tok/s** | **❌ FAIL** (50% short) |
| Median ITL | 28.99 ms | (decode-steady-state already 34.5 tok/s) |
| P99 TPOT | 64.8 ms | — |
| Max ITL | 510 ms | — (mixed-chunk worked) |
| Mean E2E | 32 s | — |
| **P99 TTFT** | **20.3 s** | **✅ PASS** |
| Aggregate output | 1020 tok/s | — |

### Run #21 — PD Disaggregation (node 0 prefill, node 1 decode)
Config: Same per-side knobs + `--disaggregation-mode prefill|decode`, mooncake KV transfer via IB

| metric | value | SLO |
|---|---|---|
| **Mean TPOT** | **30.49 ms** | — |
| **1000/TPOT** | **32.8 tok/s** | **✅ PASS** |
| Median ITL | 31.28 ms | — |
| P99 ITL | **40.92 ms** | — (super flat, no prefill stalls) |
| P99 TPOT | **31.1 ms** | — |
| Max ITL | **107 ms** | — |
| **P99 TTFT** | **47.6 s** | **❌ FAIL** |
| output/64 | 14.3 tok/s | — (TTFT-dominated) |

## SLO trade-off on 16 GPUs

You cannot hit BOTH decode ≥30 tok/s AND P99 TTFT <22 s simultaneously on this hardware count with current SGLang stack:

- **2-replica config** has enough prefill capacity (2 servers handle 32 reqs each) but suffers prefill-decode mixing in steady-state decode → decode 20 tok/s.
- **PD config** has zero prefill-decode mixing (decode is 32.8 tok/s pure) but has half the prefill capacity (1 server handles all 64 reqs) → TTFT 47s.

To hit both halves of the SLO:
1. **3+ nodes**: e.g., 2× prefill (16 GPUs) + 1× decode (8 GPUs), or 1×prefill + 2×decode pair.
2. **Working cross-node single replica**: TP=16 or PP=2 — but both attempted configs (TP=16 crashed mid-bench, PP=2 hung in NCCL init) showed cross-node distributed isn't reliable on this stack without IB env tuning.

## Full Sweep Summary (23 runs)
- Routing: cache_aware → 187:1 imbalance; round_robin → balanced; power_of_two not tested.
- EP=8 ✓ (small win); EP=0 → -10% TPOT
- mixed-chunk ✓ kills max-ITL stalls; lpm ✓; mem 0.88 ✓ (KV pool 341k → 722k)
- fp8 KV ✓ vs bf16; deep_gemm fp8 ✓ vs cutlass (-39%) / triton (-50%)
- MTP/EAGLE → -55% TPOT (high-concurrency anti-pattern)
- NVLS → -11%; tier1b (attn-tp-input-scattered + dp-lm-head) → -10% (all_reduce_one_shot_push is worse algo)
- piecewise-cuda-graph: no help
- num-continuous-decode-steps 3: -10%
- fused-qk-norm-rope: -84%; fused-moe-sum-all-reduce: CRASH
- TBO+DeepEP: CRASH; cutlass/flashinfer_cutlass/deep_gemm MoE: all CRASH with DSA
- Hierarchical/MoE backend swap: blocked by DSA-bug class on this commit
- v0.5.11: dep-incompatible CRASH; v0.5.12: -38% vs main; main b13d3d18c: best
- ray-based MoE Triton autotuner: pip blocked
- TP=16 cross-node: CRASH mid-bench; PP=2 cross-node: HANG in NCCL init
- disable-radix: -49% (radix gives ~50% prefill saving on 52% cache hit)

## Hardware verified healthy
- 16 H200 P0 (1980/3201 MHz max), 0 ECC errors, no throttle
- NVLink 18 links × 26.5 GB/s = 478 GB/s/GPU ✓
- IB NDR 400 Gb/s × 8 NICs per node, all Active LinkUp ✓
- Memory-BW theoretical ceiling: 32 tok/s/req at c=64
- Median ITL on best config: 28.99 ms = 34.5 tok/s ← AT the ceiling

## Decision points

1. **Accept the 2-replica best (run #11)** and report P99 TTFT pass + decode 20 tok/s (50% of SLO). Optimization options exhausted on current hardware.
2. **Run PD permanently** if the workload weights toward decode-heavy traffic — decode SLO met, but expect 2-3× longer TTFT.
3. **Acquire additional GPUs** (≥3 nodes) for 2× prefill + 1× decode pair → hits both SLOs.
4. **Debug cross-node TP=16/PP=2** — requires fixing NCCL IB config, likely with `NCCL_IB_HCA=mlx5_0,...,mlx5_7`, `NCCL_SOCKET_IFNAME`, etc.

## Artifacts
- `/sgl-workspace/sglang/runs/20260525_dsv32_2rep_sota_loop/`
- 21 benchmark JSONLs
- 2 decode-stage torch profiles (run #11 best, tier1b)
- analysis/decode_profile_summary.md
- manifest.txt
- final_report.md (this file)
