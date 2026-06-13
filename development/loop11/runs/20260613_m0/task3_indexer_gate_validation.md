# task3 — DS-mode indexer-cache gate: validation (R5)

The designed gate (`DSATokenToKVPool.gate_index_k_cache`, DS-only, default byte-compatible)
replaces the R0 `SGLANG_DS_PROBE_SKIP_INDEXER` preview. Driver `stage_r5_task3_v2.sh`; durable
evidence `r5v2_*_evidence.txt` (serve.logs gitignored). Robust readiness (serve.log "fired up"
marker + bounded-curl /health) after a flaky `wait_ready` mis-reported healthy boots.

## A guard caught a path the static audit missed (then fixed)
The first boot of the gated DS server raised the fail-loud guard during PREFILL:
`forward_mha.py:164` calls `self.indexer(..., return_indices=False)` to STORE index-k — a path
separate from the `_select_topk_indices` selection the audit traced. Under DS that stored cache is
never read (DS decode uses query-signature selection; the decode/selection indexer call at
`deepseek_v2.py:2586` is already skipped by the `:2191` DS early-return). So the complete gate also
skips that dead prefill store under DS (`forward_mha.py`: `if not self.use_double_sparsity`). The
guard doing its job is the point — a silent skip-allocation would have corrupted prefill.

## AC-1.1 — gate capacity payoff (DS-only)
| config | max_total_num_tokens | bs_cap | capture | smoke |
|---|---:|---:|---|---|
| DS @0.7 **ungated** (frozen p01) | 142,208 | 30 | yes | OK |
| DS @0.7 **gated** (R5, designed) | **174,848** | 37 | yes | OK: "Paris. The city is located on the River Seine…" |

+23% tokens at the same fraction — exactly matching task0's indexer-off probe (p14 = 174,848),
now produced end-to-end by the committed cell-size + pool path (KV pool 8.34 GB, indexer sidecar
gone). The gate is a **contributing** capacity lever, not sufficient alone for mem 0.8: DS @0.8
gated (fp16, default envelope) hits `graph_capture_oom` — consistent with task0's fp16/off/default
ceiling 0.75 (the fp16 table grows with the pool; sustainable 0.8 needs task4's int8 +
table-aware sizing + right-sized envelope, where task0 measured int8/off/rs = bs109/11.5 GB-ready).

## AC-7 — DSA-native un-regressed (shared-surface change)
The gate is DS-only (`use_ds_selector_width_keys`-style gating: set only when
`enable_double_sparsity and not enable_hisparse`); DSA-native keeps the index-k buffer AND the
cell-size term.
| config | max_total_num_tokens | smoke |
|---|---:|---|
| DSA @0.8 (DS off) | 410,560 | OK (coherent) |
| DSA @0.8 radix-ON | 410,560 | OK (coherent), disable_radix_cache=False |

410,560 == the frozen case3 baseline — DSA capacity byte-unchanged. The default-policy DS path
(`full_fallback` selector-width from R2) was separately re-confirmed against the frozen anchor in
R2's AC-7 run; this round's change is the indexer gate, gated entirely behind DS.

## Unit coverage
`test_double_sparsity_unit.py::TestDSIndexerCacheGate` (6 tests): gated pool skips the alloc +
data accessors raise + management methods (clear/offload/state/size) are None-safe; cell-size
drops exactly the indexer term (132 B/token/layer) when DS-gated and keeps it for DSA-native +
HiSparse. Full DS unit suite 401 passed / 4 skipped; adjacent pool + hisparse suites 11 passed.
