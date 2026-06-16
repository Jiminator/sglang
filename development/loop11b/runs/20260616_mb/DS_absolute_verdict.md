# loop11b M-B — DS-vs-DSA locked sweep verdict (production-envelope, ld32 mask)

Production-envelope op-point: DS mem 0.8 (radix-on via minted fixture, max_running_requests=64,
cuda_graph_max_bs=64) vs DSA-native. gsp 4096 ISL / 512 OSL, ~55% prefix, 2 trials/conc (median),
600s window, conc 16/32/64. Block-scheduled (labeled unpaired). SLOS.md absolute bars (DEC-6):
decode-TPS p50 >= 30, P99 TTFT < 22 s, judged regardless of DSA.

## AC-2 (P99 TTFT) + AC-3 (decode-TPS p50) — absolute verdict

| op-point | conc | decode-TPS p50 | P99 TTFT (s) | achieved conc | verdict |
|----------|------|----------------|--------------|---------------|---------|
| DS  | 16 | 40.75 | 1.59  | 16.0 | **PASS** |
| DS  | 32 | 34.12 | 3.20  | 32.0 | **PASS** |
| DS  | 64 | 26.98 | 25.12 | 58.9 (admission-capped <64) | **FAIL** (TPS<30 AND TTFT>22) |
| DSA | 16 | 41.48 | 3.45  | 16.0 | PASS |
| DSA | 32 | 33.61 | 6.79  | 32.0 | PASS |
| DSA | 64 | 26.28 | 13.53 | 64.0 | FAIL (TPS<30; TTFT ok) |

## DS/DSA ratios (REPORTED, not gated — DEC-6)

| conc | decode-TPS ratio (DS/DSA) | P99 TTFT ratio (DS/DSA) |
|------|---------------------------|--------------------------|
| 16 | 0.982 | 0.462 (DS much better) |
| 32 | 1.015 | 0.472 (DS much better) |
| 64 | 1.027 | 1.856 (DS worse) |

## AC-4 per-step tax (TPOT p50, conc-64): DS 37.13 ms vs DSA 38.13 ms, ratio **0.974 ≤ 1.10 → PASS**
(the loop-10 per-step win is preserved; DS is marginally faster per decode step than DSA.)

## Honest verdict
DS table-free meets the GLM-5.1-FP8 client SLO at concurrency 16 and 32, but FAILS at concurrency 64
(decode-TPS p50 26.98 < 30 AND P99 TTFT 25.12 s > 22 s). Native DSA also misses the 30-TPS decode floor
at conc 64 (26.28) — the throughput floor is hard for BOTH at high concurrency on this node/workload.
DS decode throughput is competitive with DSA (within +-3%, ratio >= 0.95); the per-step decode tax is
preserved (TPOT ratio 0.974). DS's distinct weakness is TTFT at high concurrency (c64 1.86x DSA), where
DS is also admission-capped below 64 (achieved 58.9).

## Caveat (being fixed)
serve_native_nsa.sh did not cap DSA to the locked op-point (max_running_requests=64, cuda_graph_max_bs=64);
DSA booted at cuda_graph_max_bs=512 / max_running_requests=None, so benchmark_compare.py --ac11 REFUSED the
cross-side op-point match (the DS absolute verdict above is valid — DS ran at the correct 64/64). DSA is
being re-run at the matched 64/64 op-point for the official comparator verdict + a fair ratio.
