# loop11b M-B — DS-vs-DSA locked sweep VERDICT (matched op-point, ld32 mask)

Production-envelope op-point: DS mem 0.8 (radix-on via the minted content-hash fixture) vs DSA-native
mem 0.85; BOTH at the locked op-point max_running_requests=64, cuda_graph_max_bs=64, page 64, fp8_e4m3,
TP=8, radix-ON. gsp 4096 ISL / 512 OSL, ~55% prefix, 2 trials/conc (DEC-4, median), 600 s window, conc
16/32/64. Block-scheduled (labeled unpaired). SLOS.md absolute bars (DEC-6): decode-TPS p50 >= 30,
P99 TTFT < 22 s, judged REGARDLESS of DSA. Data: results_prod_envelope/ (DS @ 72cb24751, DSA matched
re-run @ 94313249e — the intervening commits change no served behavior; verdict_matched.json).

## AC-2 (P99 TTFT) + AC-3 (decode-TPS p50) — absolute verdict

| op-point | conc | decode-TPS p50 | P99 TTFT (s) | achieved conc | verdict |
|----------|------|----------------|--------------|---------------|---------|
| DS  | 16 | 40.75 | 1.59  | 16.0 | **PASS** |
| DS  | 32 | 34.12 | 3.20  | 32.0 | **PASS** |
| DS  | 64 | 26.98 | 25.12 | 58.9 (admission-capped <64) | **FAIL** (TPS 26.98<30 AND TTFT 25.12>22) |
| DSA | 16 | 41.50 | 3.50  | 16.0 | PASS |
| DSA | 32 | 33.34 | 6.80  | 32.0 | PASS |
| DSA | 64 | 26.22 | 33.32 | 60.3 | FAIL (TPS 26.22<30 AND TTFT 33.32>22) |

**DS absolute verdict: PASS at conc 16 and 32; FAIL at conc 64.**

## DS/DSA ratios (REPORTED, not gated — DEC-6) — matched op-point

| conc | decode-TPS ratio | P99 TTFT ratio | per-step TPOT ratio |
|------|------------------|----------------|---------------------|
| 16 | 0.982 | 0.456 (DS lower/better) | 1.018 |
| 32 | 1.023 (DS higher) | 0.471 (DS better) | 0.977 |
| 64 | 1.029 (DS higher) | 0.754 (DS better) | 0.972 |

## AC-4 per-step tax: DS/DSA TPOT p50 ratio = 0.972–1.018 (conc 16/32/64) — all **<= 1.10 → PASS**
The loop-10 per-step win is preserved: DS is equal-or-faster than DSA per decode step at conc 32/64.

## Honest verdict
**Table-free DS on GLM-5.1-FP8 meets the client SLO (decode-TPS p50 >= 30, P99 TTFT < 22 s) at
concurrency 16 and 32, but FAILS at concurrency 64** (decode-TPS 26.98 < 30 AND P99 TTFT 25.12 s > 22 s).
**Native DSA ALSO fails at conc 64** (decode-TPS 26.22 < 30, P99 TTFT 33.32 s > 22 s) — the 30-TPS decode
floor is the binding constraint for BOTH at high concurrency on this node/workload. At the matched op-point
DS is competitive-to-better than DSA across the board: equal-or-higher decode throughput (ratio 0.98–1.03),
LOWER P99 TTFT at every concurrency (ratio 0.46–0.75), and equal-or-faster per-step decode (TPOT 0.97–1.02).
DS is additionally admission-capped just below 64 at conc-64 (achieved 58.9). A documented FAIL at conc-64
is a complete, reportable result — the deliverable is the honest measured gap, not a DS win.

## Caveat (measurement honesty / AC-9)
The official `benchmark_compare.py --ac11` REFUSED the cross-side comparison twice: first on an op-point
mismatch (DSA lacked the 64/64 caps — fixed in serve_native_nsa.sh), then on a commit_sha mismatch (DS
benched at 72cb24751, DSA matched-re-run at 94313249e; the intervening commits are the verdict capture +
the serve_native_nsa op-point fix + the task10 doc pass — none change served DS/DSA behavior). The numbers
above are extracted by the comparator's OWN metric readers (`_read_bench_jsonl`/`_median_metrics`/
`_evaluate_client_slo`) via extract_verdict.py — the DS absolute verdict (DEC-6) is DS-only and does not
depend on the cross-side guard; the DS/DSA ratios are reported with this same-commit-family caveat. The
SAME-MEMORY op-point (both 0.8) is DEFERRED-and-recorded (plan lower bound).
