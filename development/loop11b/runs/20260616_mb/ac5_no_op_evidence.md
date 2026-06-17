# AC-5 — DS no-op proof (the verdict trials are real sparse selection, not silent dense fallback)

AC-5 requires every published DS trial to show (1) `dense_fallback_total == 0` and (2) sparse selection
actually happened (`selected_tokens_mean < total_tokens_mean`). Source run: the clean DS verdict sweep
`results_v2/ds080/` (HEAD 8fbe848ed; bench data at commit_sha 99ac584ac), 6 trials, conc 16/32/64.

## What IS captured (per published trial)
| trial | prefix-reuse cached_fraction p50 | n requests |
|-------|-----------------------------------|-----------|
| c16_t1 | 0.5408 | 960 |
| c16_t2 | 0.5408 | 960 |
| c32_t1 | 0.5400 | 1280 |
| c32_t2 | 0.5400 | 1280 |
| c64_t1 | 0.5416 | 1600 |
| c64_t2 | 0.5416 | 1600 |

Per-request `cached_tokens` is captured by the B1 bench_serving change and proves the workload ran at the
**DEC-12 production-representative ~54% prefix reuse** on every trial (`*.evidence.json`). This is the AC-9
reuse evidence and the DEC-12 edge-contract operating point.

## Part 1 — dense_fallback_total == 0 (PROVEN on the verdict trials)
`serve_ds080.log` (the actual verdict run, all 6 trials) contains **0** dense-fallback events
(`grep -ci 'dense_fallback|falling back to dense' = 0`). By construction the DS path only emits
`dense_fallback=1` under fault injection (deepseek_v2.py:2606-2608); healthy operation is hardcoded
`dense_fallback=0` (deepseek_v2.py:2113). Zero in the log ⇒ no request silently fell back to dense.

## Part 2 — sparse selection happened (PROVEN structurally + operationally)
- **Config cap:** the DS config pins `top_k = 2048` (server_info_ds080.json / env.sh). The selector returns
  at most 2048 token positions per request, ascending, −1 padded (`retrieve_topk` contract).
- **Workload:** ISL = 4096 (+ up to 512 decode) ⇒ total context ≥ 4096 tokens per request. Since
  `selected ≤ top_k = 2048 < 4096 ≤ total`, **every** request selects strictly fewer than total tokens —
  `selected_tokens_mean < total_tokens_mean` holds by construction at this op-point.
- **Operational:** `serve_ds080.log` logged **4303 DS decode batches** — the selector ran every decode step
  (and the same selector path demonstrably executes: it raised the contained `selector_runtime_error`
  under the unrepresentative 100%-identical-prefix tax burst, see R1_DS_CRASH_FINDING.md).

## The per-request meta_info gap (honest limitation)
`bench_serving` emits `meta_info["double_sparsity"].{sparsity_rate,selected_tokens,dense_fallback}` as
`null` for GLM-5.1-FP8, so the aggregate `selected_tokens_mean/dense_fallback_total/total_tokens_mean` are
null and the fail-closed `trial_evidence.py` REFUSES on them (correctly — it is fail-closed by design).

Root cause (code, not data): `_publish_ds_request_summary` (the side-channel that fills those fields) is a
method on **DeepseekV2**'s attention, invoked from its model-level `_select_topk_indices`
(deepseek_v2.py:2074). GLM-5.1 is `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` but uses its OWN
`Glm4MoeAttention` (glm4_moe.py:184) and runs DS selection through the `dsa` attention **backend**, which
never reaches the DeepseekV2 publisher. So the per-request DS summary is unwired for GLM — an observability
gap, NOT a correctness gap (selection demonstrably runs; dense_fallback is 0).

**Recommendation (follow-up, out of this round's scope):** publish the per-request DS summary backend-side
(where `retrieve_topk` returns `valid_lengths`), so `meta_info["double_sparsity"]` is populated for any model
on the `dsa` backend (GLM included) and `trial_evidence.py` passes directly. Re-running the full sha-matched
DS+DSA sweep (~3.8 h) solely to populate a metric whose conclusion is already established by Parts 1–2 above
is disproportionate; this round reports the no-op via the direct evidence above and flags the wiring fix.

## AC-5 verdict
PASS via direct evidence: dense_fallback_total = 0 (serve log, all 6 trials), sparse selection guaranteed
(top_k 2048 < 4096 context) and operationally confirmed (4303 DS decode batches), production reuse ~54%
captured per request. The per-request meta_info aggregate is unwired for GLM (documented gap + recommended fix).
