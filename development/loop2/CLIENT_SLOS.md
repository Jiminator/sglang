Here are the immediate client requirements (loop 2 — rebased SLO):

- Model: zai-org/GLM-5.1 (FP8)
- Inference SLOs:
  - **Per-user speed ≥ 30 TPS**, where **TPS is the client's ground-truth formula**:
    `TPS = total_output_tokens / (total_latency − TTFT)` — i.e. decode tokens ÷ decode
    wall-time, applied to the run totals: `Σ output_tokens / Σ (latency − ttft) ≈ 1000 / mean_tpot_ms`.
    This is the **official** per-user-speed metric.
    - `median ITL` / `1000-over-ITL` is **NOT** the official metric — under EAGLE/MTP speculation
      it is inflated ~2.3× by multi-token bursts. Report it only as a cross-check.
  - **P99 TTFT < 22 s.**
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, cache hit: ~55% (baked into development/benchmark.sh)
- Page size: **not a requirement** (no preference for 64; on CUDA the DSA backend pins the
  effective page size to 64 regardless of the `--page-size` flag).
- FP8 KV cache: fully on the table (use if it helps).
- Support for performant knobs: TP, CUDA graphs, radix cache.
