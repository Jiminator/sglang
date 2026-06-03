<<<BEGIN ds_on_v32_decision.md>>>
# DS on DeepSeek-V3.2: Recall R&D Gate

## Decision

Pursue DS long-context-recall R&D on DeepSeek-V3.2, but only after the Loop-6 engineering spine has landed and passed validation:

1. Reduce TokenLabelTable footprint.
2. Lift admitted KV capacity without generation-time OOM.
3. Re-run the client SLO benchmark with admission-wait vs prefill-compute attribution.
4. Harden the selected footprint path.

The selected recall-R&D direction is a custom sparse-matmul DECODE kernel that mirrors the native NSA/DSA sparse decode path but exposes an adjustable `top_k`, removing the current hard cap that requires `indices.shape[-1] == dsa_index_topk`. A learned or query-aware DS selector is the secondary alternative.

DSA remains the production default. The DS compact-table path remains opt-in and reversible.

## Rationale

The current V3.2 DS recall failure is not evidence that sparse decode is mathematically broken. It is evidence that DS is selecting the wrong 2048 positions.

Evidence:

- V3.2’s sparse decode budget is kernel-locked to the model-native DSA `index_topk = 2048`.
- The shared `flashmla_kv` decode path asserts `indices.shape[-1] == self.dsa_index_topk` during the decode path, including CUDA graph capture. `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` does not bypass that kernel contract.
- DS NIAH recall at `top_k=2048` is poor: 4K = 75%, 16K = 5%, 64K = 0%.
- DSA recall is 100% at every tested length with the same 2048 budget and same decode kernel.
- Dense DS for `seq <= 2048` recalls 100%, which proves the DS decode path can return the needle when selection includes it.

Therefore the DS-vs-DSA recall gap is selection quality against V3.2’s trained DSA indexer, not raw sparse attention budget alone.

## Why Adjustable `top_k` Needs a Kernel

Increasing DS `top_k` above 2048 is not a config-only change on V3.2. The current `flashmla_kv` decode kernel and its metadata path are shaped around the model-native DSA top-k. The backend asserts the final indices dimension equals `dsa_index_topk`; CUDA graph capture, metadata sizing, scheduling, and kernel assumptions all depend on that shape.

A valid top-k relaxation therefore requires a new decode kernel path, or a kernel variant, with explicit support for adjustable top-k and corresponding performance validation. The correct R&D target is a custom sparse-matmul DECODE kernel that preserves the native sparse path’s serving semantics while making top-k a real runtime/configurable parameter.

The learned/query-aware selector alternative is still credible because DSA achieves 100% recall at the same 2048 budget. If DS can learn or infer a better selector, it may close recall without widening top-k. It is secondary because it adds data/training/integration complexity and does not remove the kernel cap.

## Sequencing Consequence

Recall R&D is gated behind this decision and behind the landed engineering spine. It must not block, destabilize, or regress the footprint-reduction, memory-lift, client-SLO, or hardening work.

That sequencing is intentional. Loop 6 exists because DS currently misses P99 TTFT through admission and queueing, not because per-request generation speed is below the TPS SLO. Long-context recall R&D is legitimate, but it belongs in its own loop after DS can admit the target concurrency with HBM headroom.

## Non-V3.2 Note

On a model without a trained sparse indexer, DS has a stronger value proposition. The V3.2 comparison is unusually harsh because DSA already has a trained selector that places the needle inside the same 2048 decode budget. This matters for deferred GLM-5.1 / 128k work, where DS may be more attractive if no equivalent native trained sparse indexer exists.
<<<END ds_on_v32_decision.md>>>

<<<BEGIN footprint_feasibility.md>>>
# TokenLabelTable Footprint Feasibility Budget

Units below are GiB when derived from code/log formulas; SGLang logs label these as GB.

## Empirical Basis

TokenLabelTable signature footprint per rank:

```text
table_bytes_fp16 =
  num_layers_local * max_tokens * num_heads_local * label_dim * elem_size

V3.2 TP=8:
  L = 61
  H_local = 128 / 8 = 16
  label_dim = 16
  elem_size = 2 bytes
  max_tokens = max_total_num_tokens + page_size(64)

=> table_bytes_fp16 = 31,232 * max_tokens
=> table_GiB_fp16 = 2.9087067e-5 * max_tokens
```

The `written` tensor is `bool[L, max_tokens]` and is small relative to signatures.

Real anchors:

| anchor | status | max_total_num_tokens | table T | fp16 table | observed headroom |
|---|---:|---:|---:|---:|---:|
| A: `mem_fraction_static=0.6` | serves | 53,056 | 53,120 | 1.545 GiB | memory-pool-end 37.78 GiB, after-table ≈ 36.23 GiB |
| B: `mem_fraction_static≈0.77-0.8` | boots, gen OOM | 396,096 | 396,160 | 11.523 GiB | runtime headroom ≈ 12.29 GiB; later 248 MiB alloc fails |
| C: `mem_fraction_static=0.897` | boot OOM | 1,072,000 | 1,072,064 | 31.183 GiB | memory-pool-end 7.20 GiB, table alloc fails |

Raw KV cost from Anchor C:

```text
47.99 GiB / 1,072,000 tokens = 4.4767e-5 GiB/token
                              ≈ 46.94 KiB/token
```

The table is allocated after weights and KV pool allocation. It does not reduce the static budget directly; it competes with runtime/generation headroom. Raising `mem_fraction_static` increases the KV pool, which increases `max_total_num_tokens`, which increases TokenLabelTable bytes. This is the fixed point.

Approximate A/C static fit, using 139.80 GiB/rank:

```text
f(pool) ≈ (81.718 + 4.0749e-5 * pool) / 139.80
```

This maps 95K tokens to `f≈0.612`, 114K to `f≈0.620`, 396K to `f≈0.700`, and 1.072M to `f=0.897`. Anchor B’s stated `f≈0.77-0.8` at 396K is not consistent with that fit, so pool size and measured headroom are treated as authoritative; exact `mem_fraction_static` must be swept on hardware.

Approximate pre-table headroom fit from A/C:

```text
H_pre(pool) ≈ 39.372 - 3.0011e-5 * pool GiB
H_after_table(pool, lever) = H_pre(pool) - table_bytes(pool, lever)
```

## Admission Target

Anchor A admits 35.7 requests at nominal concurrency 64 with `max_total_num_tokens=53,056`.

```text
effective pool / admitted request = 53,056 / 35.7
                                  = 1,486.16 tokens/request

minimum pool for 64 admitted requests = 1,486.16 * 64
                                      = 95,114 tokens

20% margin target = 95,114 * 1.20
                  = 114,137 tokens
```

The admission target is therefore 95K tokens minimum, with 114K tokens as the working target. `f≈0.625` is a practical sweep point because the A/C fit predicts ≈139K tokens, or ≈93 admitted-request equivalents, while still far below the known 396K-token OOM region.

## Lever Budget

| lever | storage ratio vs fp16 token table | table @95K pool | table @114K pool | `f` needed for 95K / 114K | predicted after-table headroom @114K | B-pool cross-check @396K | predicted conc-64 admission |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 table + raise `mem_fraction_static` | 1.0000x | 2.768 GiB | 3.322 GiB | ≈0.612 / ≈0.620 | ≈32.63 GiB | actual ≈12.29 GiB and gen OOM; model ≈15.96 GiB | 64 admitted at 95K; 20% margin at 114K |
| int8 same `label_dim` + fp16 scale | 0.5625x | 1.557 GiB | 1.869 GiB | ≈0.612 / ≈0.620 | ≈34.08 GiB | actual-adjusted ≈17.33 GiB; model ≈21.00 GiB | 64 admitted at 95K; 20% margin at 114K |
| page-level / two-stage fp16 | ≈1/64x | ≈0.043 GiB | ≈0.052 GiB | ≈0.612 / ≈0.620 | ≈35.89 GiB | actual-adjusted ≈23.63 GiB; model ≈27.30 GiB | 64 admitted at 95K; 20% margin at 114K |

Int8 math:

```text
fp16 vector per (layer, slot, head) = 16 dims * 2 bytes = 32 bytes
int8 signature = 16 dims * 1 byte = 16 bytes
scale = 1 fp16 per vector = 2 bytes
int8+scale = 18 / 32 = 0.5625x fp16
net table reduction = 1 / 0.5625 = 1.78x smaller
```

At the 114K working pool, int8 saves:

```text
3.322 - 1.869 = 1.453 GiB/rank
```

At the 396K B-pool, int8 saves:

```text
11.523 - 6.482 = 5.041 GiB/rank
```

Page-level math:

```text
one signature per 64-token page => approximately fp16_token_table / 64
```

At the 114K working pool, page-level saves ≈3.270 GiB/rank vs fp16 token-level. At the 396K B-pool, it saves ≈11.343 GiB/rank. This is structurally stronger, but it changes selector granularity and must be held to NIAH non-regression rather than bitwise or top-k equivalence.

## Binding Lever Decision

Paper budget result: int8 same-`label_dim` is predicted sufficient to restore nominal conc-64 admission with generation headroom at the actual admission target. It includes scale-storage overhead and the larger-pool feedback. The compaction implementation path should therefore be int8, not page-level/two-stage. Page-level/two-stage is reserved for failed hardware confirmation or a later requirement to operate at much larger pools.

However, the no-code fp16 baseline is not ruled out. On paper, an fp16 table with a smaller `mem_fraction_static` bump already reaches the admission target:

```text
95K pool:  f≈0.612, after-table headroom ≈33.75 GiB
114K pool: f≈0.620, after-table headroom ≈32.63 GiB
f≈0.625:   pool≈139K, after-table headroom ≈31.17 GiB
```

The known 0.7-region generation OOM is at a much larger ≈396K-token pool with an 11.5 GiB fp16 table. It does not by itself rule out an fp16 operating window around `f≈0.612-0.650`.

Cheapest hardware action first: sweep fp16 DS at approximately `f=0.612`, `0.625`, `0.650`, and stop once conc-64 admission, no-OOM long generation, and sufficient residual headroom are confirmed. If that sweep passes, it is the true minimum deployment lever. If it fails due to allocator residuals, fragmentation, or unmodeled generation memory, build the int8 compact table. Do not build page-level first.

## Selection Equivalence Gate

Primary binding metric for int8:

```text
top-k overlap@2048 >= 0.99 vs fp16 baseline
```

Measure on synthetic V3.2-shaped selector inputs with the same query/channel-mask/table contents, comparing int8-dequantized scoring against fp16 scoring. Failure is overlap below 0.99.

Recorded-only diagnostics:

- selected-token recall
- score-error distribution
- rank displacement around the 2048 cutoff
- NIAH recall trend

## Minimum-Reversible-Opt-In Justification

The compact TokenLabelTable path is justified only to recover DS admission headroom; it is flag-gated, fp16 remains the default, and DSA remains the production default.

## Caveats and Hardware Confirmation

This is a predicted HBM budget, not a TTFT guarantee. Confirmation requires the mem-fraction sweep, full HBM accounting through NVML plus `torch` reserved/allocated residuals, and no-OOM long-generate traffic at the target workload.

Even after admission is fixed, conc-64 P99 TTFT may become prefill-compute-bound for 4096-ISL x 64 traffic. The downstream client-SLO benchmark must attribute TTFT into admission wait vs prefill compute before declaring the SLO resolved or missed.
<<<END footprint_feasibility.md>>>
