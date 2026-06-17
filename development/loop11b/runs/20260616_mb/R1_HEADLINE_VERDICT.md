# loop11b M-B — HEADLINE VERDICT: table-free Double Sparsity vs native DSA on GLM-5.1-FP8

**Round 1 (publishable).** Supersedes the Round-0 `DS_absolute_verdict.md`. All numbers from comparator-
ACCEPTED artifacts at one frozen HEAD (`commit_sha 99ac584ac`, repo HEAD 8fbe848ed): DS verdict sweep
`results_v2/ds080/`, DSA baselines `results_v2/dsa080,dsa085/`. Op-point: GLM-5.1-FP8, TP=8, page 64,
kv fp8_e4m3, CUDA graph ON, flashmla_kv both phases, max_running_requests=64, cuda_graph_max_bs=64.
Workload: gsp 4096 ISL / 512 OSL, ~54% prefix reuse (measured, per trial), 2 trials/conc (DEC-4, median),
600 s window, conc 16/32/64. DS radix-ON (DEC-12 content-hash fixture). DS mem 0.8.

## The verdict — DS absolute client SLO (DEC-2/DEC-6: decode-TPS p50 ≥ 30 AND P99 TTFT < 22 s, judged regardless of DSA)

| conc | DS decode-TPS p50 | DS P99 TTFT | DS SLO |
|------|-------------------|-------------|--------|
| 16 | 40.73 | 1.59 s | **PASS** |
| 32 | 34.13 | 2.99 s | **PASS** |
| 64 | 26.98 | 25.08 s | **FAIL** (TPS 26.98 < 30 AND TTFT 25.08 ≥ 22) |

**Table-free DS on GLM-5.1-FP8 meets the client SLO at concurrency 16 and 32, and FAILS at 64.** This is the
honest, reportable result (the plan accepts a throughput FAIL as a complete deliverable). The 30-TPS decode
floor is the binding constraint at conc-64 — and **native DSA ALSO fails at conc-64** (26.13–26.20 TPS,
33.2 s TTFT), so the failure is the node/workload hitting its decode-throughput ceiling at high concurrency,
not a DS-specific regression.

## DS vs DSA — directional (REPORTED, not gating; DEC-6). Both op-points comparator-ACCEPTED.

### Production envelope (DS mem 0.8 vs DSA mem 0.85) — `ac11_production_envelope` rc=3
| conc | DSA TPS | DS TPS | TPS ratio | DSA P99 TTFT | DS P99 TTFT | TTFT ratio |
|------|---------|--------|-----------|--------------|-------------|------------|
| 16 | 41.68 | 40.73 | 0.977 | 3.48 | 1.59 | 0.456 |
| 32 | 33.61 | 34.13 | 1.015 | 6.77 | 2.99 | 0.441 |
| 64 | 26.13 | 26.98 | 1.032 | 33.22 | 25.08 | 0.755 |

### Same memory (DS mem 0.8 vs DSA mem 0.8) — `ac11_same_memory` rc=3
| conc | DSA TPS | DS TPS | TPS ratio | DSA P99 TTFT | DS P99 TTFT | TTFT ratio |
|------|---------|--------|-----------|--------------|-------------|------------|
| 16 | 41.42 | 40.73 | 0.983 | 3.44 | 1.59 | 0.462 |
| 32 | 33.46 | 34.13 | 1.020 | 6.86 | 2.99 | 0.436 |
| 64 | 26.20 | 26.98 | 1.030 | 33.25 | 25.08 | 0.754 |

At BOTH op-points DS is **competitive-to-better than DSA**: equal-or-higher decode throughput (ratio
0.98–1.03) and **lower P99 TTFT at every concurrency** (ratio 0.44–0.76). All AC-11 directional gates pass.

## AC-4 per-step decode tax (dedicated controlled probe, distinct-prefix, GRAPH, mem 0.8 both sides)
| fixed bs | DS median ITL | DSA median ITL | DS/DSA ratio | gate ≤ 1.10 |
|----------|---------------|----------------|--------------|-------------|
| 64 | 39.83 ms | 37.70 ms | 1.056 | **PASS** |
| 30 | 31.85 ms | 30.14 ms | 1.057 | **PASS** |

bs30 per-step window 31 850 µs ≪ 380 000 µs bound. The loop-10 per-step parity is preserved — DS pays ≤ 6 %
per decode step vs DSA at fixed batch. (The sweep steady-state at conc-64 is even closer: DS 37.1 ms vs
DSA 38.3 ms.) The probe uses distinct prefixes deliberately: a 100 %-identical-prefix burst trips a DS
selector reuse-edge that crashed an earlier run — see `R1_DS_CRASH_FINDING.md`.

## Concurrency admission (AC-2/AC-3)
Peak running-req **63** (≥ 61 ⇒ nominal conc-64 reached; the workload was not admission-capped below target).
Time-averaged achieved concurrency 58.9 reflects DS's smaller KV pool at mem 0.8 (queue-bound, partly
explaining the conc-64 TTFT gap) — a real DS property, not a measurement artifact.

## No-op / reuse (AC-5/AC-9)
Every published trial ran at ~54 % measured prefix reuse (`*.evidence.json`). DS sparse selection is real:
0 dense_fallback events (serve log, all 6 trials), top_k 2048 < 4096 context (selected < total by
construction), 4303 DS decode batches. Per-request meta_info aggregate is unwired for GLM — see
`ac5_no_op_evidence.md` (observability gap + recommended backend-side fix).

## Bottom line
Table-free Double Sparsity is a **viable, competitive** attention path on GLM-5.1-FP8 — at parity-to-better
than native DSA on throughput and TTFT, at ≤ 6 % per-step tax — but on this node/workload **neither** DS nor
DSA clears the 30-TPS decode floor at concurrency 64. DS serves the SLO at concurrency ≤ 32.
