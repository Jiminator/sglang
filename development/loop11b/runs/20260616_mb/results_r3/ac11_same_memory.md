# AC-11 Directional Comparator — DS vs DSA

Gates: DS TPS ≥ 95% of DSA TPS; DS P99 TTFT ≤ DSA P99 TTFT × 1.10. At least 2 trials per concurrency, median.

| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99 | DS TTFT p99 | TTFT ratio | TTFT gate |
|------|-------------|------------|-----------|----------|--------------|-------------|------------|-----------|
| 16 | 41.360 | 40.696 | 0.984 | pass | 3.440 | 1.581 | 0.460 | pass |
| 32 | 33.299 | 34.050 | 1.023 | pass | 6.787 | 3.003 | 0.442 | pass |
| 64 | 26.187 | 26.912 | 1.028 | pass | 33.216 | 25.110 | 0.756 | pass |

## Effective vs nominal concurrency (#F)

| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |
|----------------|--------------|-------------|------------|
| 16 | 15.996 | 15.996 | 100% |
| 32 | 31.989 | 31.987 | 100% |
| 64 | 60.264 | 58.914 | 92% |

When DS achieved concurrency is below nominal while DSA tracks nominal, the DS P99 TTFT gap is partly queue/admission-bound (DS mem_fraction_static=0.8 reserves a smaller KV pool), not solely per-request latency.

## Absolute client-SLO gate (DEC-2 mandatory-to-land)

Bars: decode-TPS p50 ≥ 30 tok/s AND P99 TTFT < 22 s (strict).

| Conc | DSA decode-TPS | DSA P99 TTFT | DSA SLO | DS decode-TPS | DS P99 TTFT | DS SLO |
|------|----------------|--------------|---------|---------------|-------------|--------|
| 16 | 41.360 | 3.440 | pass | 40.696 | 1.581 | pass |
| 32 | 33.299 | 6.787 | pass | 34.050 | 3.003 | pass |
| 64 | 26.187 | 33.216 | FAIL | 26.912 | 25.110 | FAIL |

**DS client-SLO verdict: FAIL** (DEC-2 mandatory-to-land NOT met for DS-on):
- conc=64: DS decode-TPS 26.91 < 30; P99 TTFT 25.110 s >= 22 s

## AC-11 directional verdict (DS-vs-DSA ratios): PASS — REPORT-ONLY (DEC-6; does NOT gate)
