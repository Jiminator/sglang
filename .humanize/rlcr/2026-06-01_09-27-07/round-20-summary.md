# Round 20 Summary — Loop 7

## Mainline objective (round-20-contract.md)
**task19 (AC-6 completion) — record the missing fresh conc-1/conc-16 TTFT guardrails
(plus a clean streaming decode-TPS) for DS-default, DS-hybrid, and DSA/native-NSA at the
Loop-7 op-point under CUDA graph, and update `m11_perf_consolidation.md` so AC-6 is claimed
only after TTFT is present.**

## Outcome: ACHIEVED — AC-6 MET; task19 done. 5/6 ACs MET.

## Why this round (R19-review gap)
The R19 review (ADVANCED) rejected full AC-6 closure: the plan
(`refined_plan_v1.md:80`) requires **`TTFT, decode TPS/req, GPU memory, graph-replay
success, admission` at conc-1/16**, and `perf_closed_batch.py` was a non-streaming
`/generate` probe that records only request wall time + completion tokens — it
structurally cannot measure time-to-first-token. The reviewer's authorized fix: "the
existing `bench_serving` path or **an equivalent streaming probe that records first-token
timestamps**."

## Work Completed (`coding`, live measurement; no production-code change)
1. **Extended `perf_closed_batch.py` with a `--stream` SSE mode.** It mirrors the
   canonical SGLang streaming parser (`data: {"text": <cumulative>, "meta_info":
   {"completion_tokens": N}}`), records per-request **TTFT** = first-streamed-token arrival
   − submit and a **clean post-first-token decode TPS** = `(completion_tokens − 1) /
   (t_last − t_first)`, and — per BL-20260531-bench-empty-stream-failclosed — **fails
   closed** on an HTTP-200 empty stream (raises rather than recording a no-token response
   as a completion). The non-streaming R19 path is preserved for reproducibility.
2. **Re-measured all three variants** (DS-default, DS-hybrid Tier-2.B, DSA/native-NSA) at
   conc-1/16 under CUDA graph at the Loop-7 op-point (int8 / mem 0.7 / fp8-KV / TP=8 / page
   64 / radix-off), two prompt regimes: a SHORT prompt (the R19 decode cross-check) and a
   ~770-token prompt (a prefill-bound TTFT guardrail, dense-prefill regime).

### TTFT (ms; conc-16 reported p50 / p99 across 16 concurrent reqs)
| variant | c1 short | c16 short (p50/p99) | c1 ~770-tok | c16 ~770-tok (p50/p99) |
|---|---|---|---|---|
| DSA (native-NSA) | 150.8 | 307.1 / 309.2 | 150.9 | 1161.5 / 1322.1 |
| DS-default | 183.3 | 371.7 / 374.0 | 180.4 | 1210.9 / 1400.2 |
| DS-hybrid (Tier-2.B) | 178.4 | 363.3 / 365.1 | 177.7 | 1218.1 / 1405.2 |

### Streaming decode-TPS cross-check (clean, post-first-token)
DSA **87.3 / 58.7**, DS-default **40.8 / 28.5**, DS-hybrid **41.1 / 28.5** (c1 / c16) —
reproduces the R19 closed-batch ordering and the DS ≈ 0.48–0.49× DSA structural ratio
(slightly higher than R19's e2e number because it excludes prefill+first-token — the
theoretically correct pure-decode rate). All decode batches `cuda graph: True`; served
16/16 every run.

## Key findings (non-regression)
- **DS-hybrid TTFT ≈ DS-default TTFT at every point** (178 vs 183 ms c1; 363 vs 372 ms
  c16-short p50; 1218 vs 1211 ms c16-p770 p50 — within run-to-run noise). The Tier-2.B
  hybrid scorer adds **no material TTFT cost** — the same decode-free result the R19
  decode-TPS table showed, now confirmed on first-token latency too.
- **DS TTFT is modestly above DSA** (~+30 ms c1, ~+60 ms c16-short) — the small per-step
  cost of the DS selection + logical→physical adapter, the same structural overhead as
  the decode-TPS gap; NOT a Loop-7 regression. In the prefill-bound c16-p770 case TTFT is
  prefill-dominated and DS ≈ DSA + ~5%.
- **Every measured TTFT is far below the Loop-6 directional P99 22 s ceiling** — heaviest
  point (DS conc-16, ~770-tok prefill) is P99 ≈ **1.4 s**. The Loop-6 directional P99
  13.13 s is retained only as the historical full-context reference (that path is
  unchanged because all Loop-7 work is opt-in/default-off).

## Files Changed
- `development/loop7/perf_closed_batch.py` (added `--stream` SSE mode + fail-closed guard;
  dev probe, not imported by tests).
- `development/loop7/m11_perf_consolidation.md` (TTFT table + streaming cross-check +
  TTFT findings + corrected conclusion-3 + AC-6-MET restated to include TTFT + provenance).
- `development/loop7/ttft_{ds_default,ds_hybrid,dsa}_c{1,16}{,_p770}.json` (12 artifacts).
- Commit `30173f08b` (local — loop hook). **No production-code change.**

## Validation
- TTFT + streaming decode-TPS per variant (above); GPU mem from `nvidia-smi` (DS 125 GB /
  DSA 133 GB, matching R19); graph from `cuda graph: True` decode batches (27 each);
  admission served 16/16 all variants; the probe fail-closes on empty streams so a
  completed run guarantees real per-request latency.
- Full DS unit suite → **350 passed + 9 subtests** (identical to R19 — confirms no
  production-code regression).
- GPUs freed (0 MiB) + all three servers stopped at round end.

## AC status after R20
- **AC-6 → MET** (the full guardrail set — TTFT + decode TPS + mem + graph + admission at
  conc-1/16 — is now recorded; landed deliverable non-regressing on both TPS and TTFT).
  With AC-1/3/4/5 (prior), **5/6 ACs MET**.
- **AC-2 PARTIAL** — only task20 (the final strategic-gate supersession decision record)
  remains, now **unblocked** (its corrected AC-6 source artifact exists). After task20,
  all 6 ACs are met and Loop 7 can close.

## Remaining Items (active mainline)
- **task20 (AC-2, next mainline + loop close)** — the final gate-supersession decision
  record: cite M0 regime attribution, AC-1 closure, AC-3 hybrid scorer, AC-4 production-ready
  lifted, AC-5 servability, AC-6 perf guardrails (this `m11`, now with TTFT); explicitly
  state what measured evidence superseded the Loop-6 Tier-2.A-primary ordering; cite/preserve
  the R8 oracle-sink provenance before relying on it.
- Evidence-hygiene queued (fold into task20): R8 stride/oracle provenance citation;
  plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: applied the selected lessons — BL-20260531-bench-empty-stream-failclosed (the
  `--stream` probe adopts the same fail-closed empty-stream guard and the standard SGLang
  SSE format), BL-20260528-dsv32-ds-serving-boot-chain + BL-20260529-dsv32-bench-smoke-sizing
  (TP=8 boot + mem-fraction op-point hygiene for the three measurement servers), and
  BL-20260527-shell-json-into-python-source (the probe consumes server JSON as data, never
  source). No NEW reusable pitfall surfaced: the streaming probe re-uses the established
  SSE parser + fail-closed pattern, so the findings (Tier-2.B is TTFT-free; DS ≈ 0.48–0.49×
  DSA) are project evidence recorded in `m11_perf_consolidation.md`, not a cross-round
  engineering lesson.

## Goal Tracker
Updated directly (Plan Version 29): R20 plan-evolution row; task19 → Completed (TTFT
recorded, AC-6 MET) and added to Completed-and-Verified with Verified = pending (R20
Review); Active = task20 only (marked unblocked). No Goal Tracker Update Request needed.
