# Loop 2 coverage ledger (no silent skips)

Every axis is either measured (gate + profile) or closed with a cited launch/runtime outcome.

## Infeasible / non-launchable axes (closed with citation)

| axis / knob | attempt | outcome (taxonomy) | citation |
|---|---|---|---|
| EAGLE tree `--speculative-eagle-topk 2` (draft8/steps3) | launched on combo base | **startup-reject** (~15s, before weight load) | `ValueError` at `python/sglang/srt/arg_groups/speculative_hook.py:388` (`_handle_eagle_family`): "speculative_eagle_topk > 1 with page_size > 1 ... only supported for the 'flashinfer' backend." DSA forces `page_size=64` + `attention_backend=dsa`. Server log also notes the v2→v1 fallback ("spec v2 topk > 1 currently requires page_size == 1") preceding the hard reject. |

EAGLE tree (topk>1) is therefore **infeasible flags-only on this DSA path** — it cannot be benchmarked without an out-of-scope attention-backend swap. The incumbent topk=1 verify/draft cost is instead characterized from the `combo_baseline` profile (see `profiling/combo_baseline.md`): EAGLE adds ~2× MoE launches; verify/draft is a major, non-negligible share of decode cost at conc 64 (consistent with loop 1's "lighter EAGLE regresses").

## DSA prefill × decode cross-product (bf16) — taxonomy

Base = combo (EAGLE steps3/topk1/draft4, mem0.85, mrr64, chunked-prefill 4096, lpm) + `--dsa-prefill-backend P --dsa-decode-backend D`.
Expected (source): `decode=flashmla_auto` → first-request runtime failure (no `auto` branch in decode dispatch, `dsa_backend.py:1726`); `flashmla_kv` under bf16 → launchable, quantizes whole cache (`dsa_backend.py:1846-1848`), expected slow. Owner decision DEC-3: **no pruning** — every launchable cell fully gate-benchmarked AND profiled.

Gate-sweep **COMPLETE** (all 16 cells launch-attempted). Client TPS (selection metric), bf16:

| prefill ＼ decode | flashmla_sparse | flashmla_kv | flashmla_auto | fa3 |
|---|---|---|---|---|
| **flashmla_sparse** | 24.10 | 15.18 | ❌ reject | **24.08 (combo_baseline)** |
| **flashmla_kv** | 20.74 | 13.55 | ❌ reject | 20.76 |
| **flashmla_auto** | 23.57 | 14.72 | ❌ reject | 23.71 |
| **fa3** | 24.19 | 14.74 | ❌ reject | **24.35 (best)** |

All launchable cells: 320/0 err, conc ≈ 60–62, p99_ttft 12–17 s (all sub-22s, info), `max_total_num_tokens=300352`.

**Failure taxonomy — `decode=flashmla_auto` (all 4 cells): startup-reject.** Observed (not the `:1726` decode assert the plan predicted): during server warmup forward, `ValueError: Unsupported dsa_impl = 'flashmla_auto' for forward_extend. Consider using an other attention backend.` at `python/sglang/srt/layers/attention/dsa_backend.py:1567`. `flashmla_auto` resolves only on the prefill auto-select path (`dsa_backend.py:335,2273`); it has no decode/extend implementation, so selecting it for decode rejects at warmup before serving. (The `flashmla_auto__flashmla_auto` cell raised twice — prefill+decode both auto.)

**Three regimes (kernel attribution pending per-cell profiles):**
1. `decode ∈ {fa3, flashmla_sparse}` → ~24 TPS, flat with incumbent (best = fa3/fa3 24.35 ≈ baseline 24.08, within ~1% noise). `prefill ∈ {flashmla_sparse, flashmla_auto, fa3}` all equivalent here.
2. `decode = flashmla_kv` → **severe regression ~13.5–15.2 TPS** (mean_tpot ~66–74 ms). Confirms `_forward_flashmla_kv` "inefficiently quantize[s] the whole cache" (`dsa_backend.py:1846-1848`) under bf16 — slow, as predicted.
3. `prefill = flashmla_kv` (with fast decode) → ~20.7 TPS (the prefill-side quantize tax dents TTFT/throughput even when decode is fa3/sparse).

**Verdict so far:** no DSA backend swap beats the incumbent meaningfully; fa3/fa3 (24.35) ≈ combo (24.08). Matrix exhausted (12 launchable measured, 4 rejected — no pruning, DEC-3). Per-cell decode profiles for the 11 non-incumbent launchable cells follow (profile→rollup→delete-raw); they attribute each delta to the responsible kernel.

## Profile-directed follow-up candidates (task6, flags-only, in-scope)

Directed by the `combo_baseline` profile (comms 16.5%, attn/indexer ~26% — both material). To be gate+profile measured (or closed with cited evidence — no silent skips):

| candidate | flag(s) | targets profile slice | accuracy-risk? | status |
|---|---|---|---|---|
| fused MoE-sum + all-reduce | `--enable-fused-moe-sum-all-reduce` | comms 16.5% + moe_sum_reduce | no | pending |
| DSA topk backend = flashinfer | `--dsa-topk-backend flashinfer` | indexer/topk ~8.5% | no | pending |
| DSA topk backend = torch | `--dsa-topk-backend torch` | indexer/topk ~8.5% | no | pending |
| continuous decode steps | `--num-continuous-decode-steps 2` | scheduling/CPU gap | no | pending (profile shows <1% idle → likely inert) |

Already-on (not headroom): FlashInfer all-reduce fusion (`enable_flashinfer_allreduce_fusion=True`, auto on SM90), overlap schedule (`disable_overlap_schedule=False`). Two/single-batch-overlap (`--enable-two-batch-overlap`/`-single-batch-overlap`): candidate but high regression risk at conc 64 (batch split shrinks MoE GEMMs; aligns with loop-1 DP-attn per-rank-collapse finding) — will probe or close with the profile's negligible-idle evidence.
