# Round 1 Contract

## Mainline Objective (ONE)
Make the M-B M4 verdict PUBLISHABLE and close out the loop honestly. Round 0 delivered M-A + a directional
verdict, but the M-B verdict is not publishable per the strict bars (Codex review): the comparator REFUSED
the cross-side comparison, only the production-envelope op-point was attempted, conc-64 was admission-capped,
the benchmark emits no per-trial reuse/no-op evidence, AC-4 was not the spec'd tax guard, and AC-8 ledger/
evidence/push are incomplete. This round produces the publishable verdict + close-out.

## Target ACs (focused)
- **AC-9** (measurement honesty): comparator ACCEPTS each matched op-point; BOTH production-envelope (DS0.8/
  DSA0.85) AND same-memory (both 0.8) published; per-trial prefix-reuse recorded; run order logged.
- **AC-2 / AC-3** (publishable absolute verdict): DS decode-TPS p50 / P99 TTFT at conc 16/32/64 from a sweep
  that is NOT admission-capped below nominal (conc-64 running-req peak ≥ 61), comparator-accepted.

## Blocking issues in scope (must fix to reach the objective)
- **B1 (AC-5/AC-9 evidence):** `bench_serving.py` does not emit per-request `cached_tokens` (prefix reuse) or
  DS no-op counters (`dense_fallback`, `selected_tokens`, `total_tokens`, `sparsity_rate`). Extend
  RequestFuncOutput + the JSONL/sidecar emission; add a fail-closed per-trial summary. Without this, AC-5
  blocks publication.
- **B2 (AC-9 comparator acceptance):** the `--ac11` comparator refused (op-point caps, then commit_sha). Re-run
  DS + DSA from ONE clean HEAD with the final serve_native_nsa.sh caps already present; produce TWO artifact
  trees (production_envelope, same_memory_080) + comparator-accepted `ac11_report.md`/`ac11_verdict.json` each.
- **B3 (AC-2/AC-3 admission):** conc-64 DS achieved 58.9 < 64. Capture per-trial running-request peak; re-run
  conc-64 and verify peak ≥ 61, else report as an AC-2/AC-3 measurement failure (not a complete verdict).
- **B4 (AC-4 tax guard):** the sweep TPOT ratio is NOT the spec'd guard. Build a controlled same-batch decode-
  window probe (serving-backed, chunked prefill so the 4096-context bs64 fits; the DS validator runs), GRAPH
  mode, mem 0.8, bs64 + bs30; commit the raw + parsed summary (bs64 ratio, bs30 window, radix/shape/warmup/mem).
- **B5 (AC-8):** queue.md statuses match reality; fix the stale `a4be98c4` capacity claim → `35155ac4`; preserve
  raw evidence (committed lossless summaries + hashes for jsonl/log/mask/fixture/server_info/run-order/reuse);
  push to an owner-approved remote (origin is the PUBLIC sgl-project upstream — record owner-direction-needed if no fork).

## Queued / out of scope this round
- Production-facing plan-vocabulary leakage in serve scripts (DEC/loop identifiers) — clean during the UX pass
  but do NOT let it displace the measurement fixes.
- The deferred 128k-ISL second op-point — permanently OUT OF SCOPE.

## Round success criteria
1. `bench_serving.py` emits per-request cached_tokens + DS counters; benchmark.sh sidecar carries them + the
   per-trial running-request peak + boot order; a fail-closed per-trial summary records reuse distribution,
   `dense_fallback_total == 0`, `selected_tokens_mean < total_tokens_mean`.
2. AC-4 controlled tax guard run (bs64 ratio ≤ 1.10, bs30 ≤ 380k µs, GRAPH, both mem 0.8); raw + summary committed.
3. Clean sweep from one HEAD: production_envelope + same_memory_080, comparator ACCEPTS each (rc 0/3, not 2);
   conc-64 peak ≥ 61 (or labeled measurement failure); `ac11_report.md` + `ac11_verdict.json` committed for both.
4. Headline report re-written from the ACCEPTED comparator output + the reuse/no-op evidence.
5. AC-UX cleanup: plan vocabulary removed from serve scripts.
6. AC-8: queue.md current, a4be98c4 fixed, evidence package committed, pushed (or owner-direction recorded).

Implementation order (Codex): B1 (bench evidence) → B4 (task7 tax) → B2/B3 (task8 sweep) → task9 report →
task10 cleanup → B5 (task11 close-out). An honest documented FAIL remains a complete result.
