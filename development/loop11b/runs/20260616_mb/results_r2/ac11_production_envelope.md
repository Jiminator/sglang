# AC-11 Directional Comparator — DS vs DSA

Gates: DS TPS ≥ 95% of DSA TPS; DS P99 TTFT ≤ DSA P99 TTFT × 1.10. At least 2 trials per concurrency, median.

| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99 | DS TTFT p99 | TTFT ratio | TTFT gate |
|------|-------------|------------|-----------|----------|--------------|-------------|------------|-----------|
| 16 | 41.640 | 40.654 | 0.976 | pass | 3.456 | 1.596 | 0.462 | pass |
| 32 | 33.409 | 34.057 | 1.019 | pass | 6.814 | 3.004 | 0.441 | pass |
| 64 | 26.199 | 26.916 | 1.027 | pass | 33.219 | 25.098 | 0.756 | pass |

## Effective vs nominal concurrency (#F)

| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |
|----------------|--------------|-------------|------------|
| 16 | 15.996 | 15.996 | 100% |
| 32 | 31.989 | 31.986 | 100% |
| 64 | 60.263 | 58.909 | 92% |

When DS achieved concurrency is below nominal while DSA tracks nominal, the DS P99 TTFT gap is partly queue/admission-bound (DS mem_fraction_static=0.8 reserves a smaller KV pool), not solely per-request latency.

## Absolute client-SLO gate (DEC-2 mandatory-to-land)

Bars: decode-TPS p50 ≥ 30 tok/s AND P99 TTFT < 22 s (strict).

| Conc | DSA decode-TPS | DSA P99 TTFT | DSA SLO | DS decode-TPS | DS P99 TTFT | DS SLO |
|------|----------------|--------------|---------|---------------|-------------|--------|
| 16 | 41.640 | 3.456 | pass | 40.654 | 1.596 | pass |
| 32 | 33.409 | 6.814 | pass | 34.057 | 3.004 | pass |
| 64 | 26.199 | 33.219 | FAIL | 26.916 | 25.098 | FAIL |

**DS client-SLO verdict: FAIL** (DEC-2 mandatory-to-land NOT met for DS-on):
- conc=64: DS decode-TPS 26.92 < 30; P99 TTFT 25.098 s >= 22 s

## AC-11 verdict: PASS
