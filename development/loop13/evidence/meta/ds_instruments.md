# DS diagnostic-instrument recipe (loop13 reference)

Config fields (all in `--double-sparsity-config` JSON; `_ALLOWED_FIELDS` whitelist in config.py):

| field | type | default | role |
|---|---|---|---|
| `score_capture` | bool | false | per-(rank,req,layer) fp32 absorbed score row → `.sglang_ds_scorecap/rank{tp}_req{idx}_layer{id}.pt`; record has `scores` (post cross-TP reduce) AND `pre_reduce_scores` (per-rank local-max-over-heads, before cross-TP SUM). **Requires `--disable-cuda-graph`.** |
| `selection_capture` | bool | false | per-decode-step selected indices `[num_layers,bs,max_top_k]` int32 + valid lengths → `.sglang_ds_selcap/rank{tp}_step{step}.pt`. **CUDA-graph-safe** (works under graph). |
| `latent_capture` | bool | false | per-(rank,req,layer) prompt-slot fp8-latent SHA → `.sglang_ds_latentcap/...pt`. Eager only. |
| `recall_oracle` | bool | false | NIAH needle recall@K vs score-only top-k → `.sglang_ds_oracle/sink.jsonl`. Needs `.sglang_ds_oracle/trial.json` with needle_positions; **fails closed (no_active_trial) for non-NIAH prompts like GSM8K**. Requires `--disable-cuda-graph`. |
| `score_reduce_dtype` | "bf16"\|"fp32" | "bf16" | cross-TP score SUM-reduce transport dtype (scoring + topk stay fp32). Bisection knob. |
| `head_agg` | "max"\|"mean" | "max" | cross-head reduction. "max" = local-max-per-rank then cross-TP SUM. |
| `selector_width_buckets` | List[int] | [5120] | compact selector prefix-window widths; `[]` = full-width only. |
| `scorer_norm` | "off" | "off" | HARD-LOCKED off (absorbed-latent identity only holds for raw dot). |

Capture-dir env overrides: `SGLANG_DS_SCORE_CAPTURE_DIR`, `SGLANG_DS_SELECTION_CAPTURE_DIR`, `SGLANG_DS_LATENT_CAPTURE_DIR`, `SGLANG_DS_RECALL_ORACLE_DIR`. Default = `os.getcwd()/.sglang_ds_*`.

Per-request meta_info: `meta_info["double_sparsity"]` = {sparsity_rate, selected_tokens, total_tokens, dense_fallback}. (recall-oracle does NOT ride in meta_info.)

## AC-2 capture-arm recipe (eager, small request count)
Boot DS with `--disable-cuda-graph` and a config that adds:
`"score_capture": true, "selection_capture": true` (optionally `latent_capture`).
Send a handful of long-context (>2048) + dense (<2048) requests, then read:
- `.sglang_ds_scorecap/*` for the TP head-agg micro-test (AC-2.2: SUM(pre_reduce per rank) vs MAX vs MEAN) and selected-index equivalence (AC-2.3).
- `.sglang_ds_selcap/*` for the actual served selected sets.

## TP head-agg derivation (AC-2.2)
head_agg="max" computes, per rank, local-max over that rank's heads = `pre_reduce_scores`.
Cross-TP reduce SUMs across the 8 ranks → served score = Σ_rank pre_reduce[rank].
- served (current): `Σ_rank pre_reduce[rank]`
- global-max-over-all-heads: `max_rank pre_reduce[rank]`  (max of local maxes = global max)
- global-mean: needs per-head counts; approx `mean_rank pre_reduce[rank]` as a coarse proxy.
Compare the top-2048 index sets under each rule on captured rows.
