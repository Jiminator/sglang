# AC-11 Directional Comparator — DS vs DSA

Gates: DS TPS ≥ 95% of DSA TPS; DS P99 TTFT ≤ DSA P99 TTFT × 1.10. At least 2 trials per concurrency, median.

| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99 | DS TTFT p99 | TTFT ratio | TTFT gate |
|------|-------------|------------|-----------|----------|--------------|-------------|------------|-----------|
| 16 | 41.683 | 40.730 | 0.977 | pass | 3.481 | 1.587 | 0.456 | pass |
| 32 | 33.613 | 34.126 | 1.015 | pass | 6.771 | 2.987 | 0.441 | pass |
| 64 | 26.131 | 26.977 | 1.032 | pass | 33.217 | 25.080 | 0.755 | pass |

## Effective vs nominal concurrency (#F)

| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |
|----------------|--------------|-------------|------------|
| 16 | 15.996 | 15.996 | 100% |
| 32 | 31.989 | 31.986 | 100% |
| 64 | 60.292 | 58.937 | 92% |

When DS achieved concurrency is below nominal while DSA tracks nominal, the DS P99 TTFT gap is partly queue/admission-bound (DS mem_fraction_static=0.8 reserves a smaller KV pool), not solely per-request latency.

## Absolute client-SLO gate (DEC-2 mandatory-to-land)

Bars: decode-TPS p50 ≥ 30 tok/s AND P99 TTFT < 22 s (strict).

| Conc | DSA decode-TPS | DSA P99 TTFT | DSA SLO | DS decode-TPS | DS P99 TTFT | DS SLO |
|------|----------------|--------------|---------|---------------|-------------|--------|
| 16 | 41.683 | 3.481 | pass | 40.730 | 1.587 | pass |
| 32 | 33.613 | 6.771 | pass | 34.126 | 2.987 | pass |
| 64 | 26.131 | 33.217 | FAIL | 26.977 | 25.080 | FAIL |

**DS client-SLO verdict: FAIL** (DEC-2 mandatory-to-land NOT met for DS-on):
- conc=64: DS decode-TPS 26.98 < 30; P99 TTFT 25.080 s >= 22 s

## AC-11 verdict: PASS
