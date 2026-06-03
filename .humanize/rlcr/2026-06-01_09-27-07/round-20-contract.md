# Round 20 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task19 (AC-6 completion) — record fresh conc-1/conc-16 TTFT (the missing guardrail
column) alongside a clean streaming decode-TPS for DS-default, DS-hybrid, and DSA /
native-NSA at the Loop-7 op-point under CUDA graph, and update
`m11_perf_consolidation.md` so AC-6 is claimed only after TTFT is present.**

The R19 review (ADVANCED) rejected full AC-6 closure: the plan (`refined_plan_v1.md:80`)
requires **`TTFT, decode TPS/req, GPU memory, graph-replay success, admission` at
conc-1/16**, and `perf_closed_batch.py` is a non-streaming `/generate` probe that records
only request wall time + completion tokens — it structurally cannot measure
time-to-first-token. The reviewer's authorized fix: "the existing `bench_serving` path or
**an equivalent streaming probe that records first-token timestamps**."

Approach (theoretically-correct, owner preference): extend the already-accepted
`perf_closed_batch.py` to a streaming mode (SSE `/generate` with `"stream": true`) that
captures per-request `t_submit`, `t_first_token`, `t_last`, so:
- **TTFT** = `t_first_token − t_submit` (the missing metric),
- **pure decode TPS** = `(completion_tokens − 1) / (t_last − t_first_token)` — cleaner than
  the R19 e2e approximation (which folded prefill into decode), and a direct cross-check
  that reproduces the accepted R19 decode-TPS within noise.
Re-run all three variants at conc-1/16 at the same Loop-7 op-point under CUDA graph (R19
servers are down; TTFT requires fresh streaming runs). No production-code change.

## Target AC(s)
- **AC-6** (perf guardrails: TTFT + decode TPS/req + GPU memory + graph-replay + admission
  at conc-1/16; Tier-1 non-regression). This is the only AC still PARTIAL.

## Blocking issues (truly block the mainline)
- **None.** Measurement-only; no production-code change to the default path. The streaming
  extension is to an existing dev probe (accepted R19), not new serve/bench scaffolding.

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task20 (AC-2)** — the final strategic-gate supersession decision record. The R19
  review explicitly requires it to **wait for the corrected AC-6 source artifact**; it is
  the next mainline after this round.
- Evidence-hygiene queued (fold into task20): cite/preserve the R8 oracle-sink provenance;
  plan-marker cleanup (pre-existing).
- Learned/distilled selector (DEC-5) — out of scope.

## Concrete success criteria
1. **Streaming TTFT probe**: `perf_closed_batch.py` gains a streaming mode that records,
   per request, TTFT (first-token arrival − submit) and pure decode TPS (post-first-token),
   at conc-1 and conc-16 for **DS-default**, **DS-hybrid (Tier-2.B)**, and **DSA/native-NSA**
   at the Loop-7 op-point (int8 / mem 0.7 / fp8-KV / TP=8 / page 64 / radix-off), under
   CUDA graph.
2. **Fresh TTFT artifacts committed** per variant×conc: TTFT mean + p50 + p99 (or the
   per-request array), the cross-checked streaming decode-TPS, GPU mem, graph status
   (`cuda graph: True`), admission/served counts, exact launch args/config, commit SHA,
   and GPU type.
3. **Decode-TPS cross-check**: the streaming decode-TPS reproduces the accepted R19
   closed-batch numbers within noise (validating both methods).
4. **`m11_perf_consolidation.md` corrected**: a TTFT column/section is added to the conc-1/16
   table; the AC-6 MET claim is restated to include TTFT; the stale "old Loop-6 conc-16
   TTFT" citation is replaced/augmented by the fresh R20 conc-1/16 TTFT (the Loop-6 number
   may remain only as the historical full-context directional reference).
5. Full DS unit suite still passes (no code regression); GPUs freed + servers stopped at
   round end.
6. `goal-tracker.md` updated (task19 → done with TTFT evidence; AC-6 MET; task20 unblocked
   as the next mainline); commit.

## Tag routing
- task19 is a **`coding`** task → Claude executes directly (probe extension + live
  measurement + report correction).
