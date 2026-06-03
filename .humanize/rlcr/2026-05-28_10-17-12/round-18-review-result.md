Mainline Progress Verdict: STALLED

Goal Alignment Summary:
ACs: 11/11 addressed; 9/11 literally met under the immutable AC text, with AC-11 executed as a recorded directional FAIL/follow-up and original AC-12 NIAH 4K/16K/64K still failing | Forgotten items: 0 | Unjustified deferrals: 1 attempted/rejected

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-18-prompt.md`, `round-18-contract.md`, `round-18-summary.md`, Round 15-17 summaries/reviews, `goal-tracker.md`, recent commit history, `next_loop_issues.md`, AC-11/AC-12 artifacts, and the relevant DS config/validator/backend/selector/TokenLabelTable code.

Verification rerun:

```bash
PYTHONPATH=python pytest \
  test/registered/unit/development/test_ac11_comparator.py \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py \
  test/registered/unit/development/test_option_b_scripts.py \
  test/registered/unit/manual/test_ac12_helpers.py -q
# 411 passed, 24 warnings, 28 subtests

bash -n development/serve_double_sparsity.sh development/serve_native_nsa.sh
# pass
```

## Implementation Review

Round 18 made no code change and produced no new hardware artifact. The recovery contract and summary explicitly state that the deferred DS long-context R&D was not started. That is honest bookkeeping, but it does not satisfy this review prompt: deferred tasks are incomplete, and the prompt requires forcing original-plan work to completion before the stop sentinel can be emitted.

The requested owner-authorized deferral is rejected for this review's stop condition. It may be a valid human loop-disposition preference outside this automated no-deferral review, but it cannot be counted as original-plan completion here.

## Mainline Gaps

1. Literal original AC-12 is still incomplete; Round 18 did not implement any of the required recovery work.

   Evidence:
   - `development/loop5/refined_plan_v1.md:79-82` defines AC-12 as NIAH 4K/16K/64K plus MMLU, with all gates passing.
   - `goal-tracker.md:40` preserves the same immutable AC-12 requirement.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_4096_20260529T190151Z.json:11-17`: DSA 100%, DS 75%, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_16384_20260529T190258Z.json:11-17`: DSA 100%, DS 5%, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_65536_20260529T190614Z.json:8-17`: DS served 0/20, HTTP 400 admission failure, `verdict=FAIL`.
   - `python/sglang/srt/layers/attention/double_sparsity/config.py:21-27` has no `decode_backend` field, so there is still no DS-owned flexible backend contract.
   - `python/sglang/srt/layers/attention/double_sparsity/validator.py:154-178` still treats `top_k != index_topk` as an environment-override ablation, not a production-safe backend-specific mode.
   - `python/sglang/srt/layers/attention/dsa_backend.py:1967-1979` routes DS decode through `_forward_flashmla_kv`, and `python/sglang/srt/layers/attention/dsa_backend.py:2146-2149` asserts `indices.shape[-1] == self.dsa_index_topk`, which keeps DS kernel-locked to 2048.
   - `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py:15-20` documents the fp16 table as roughly 8 GB/rank, and `token_label_table.py:68-98` still allocates fp16 signatures by default.
   - `runs/20260528_dsv32_mvp/next_loop_issues.md:19-32` still carries query-aware selector, wider decode, and TokenLabelTable footprint work forward instead of having completed it.

   Directive implementation plan:
   1. Extend `DoubleSparsityConfig` with `decode_backend`, defaulting to `"flashmla_kv"`, and accept `"ds_flex"` as the only new value. Keep unknown top-level fields rejected.
   2. Change `validator.py` so production serving cannot use `SGLANG_DS_ALLOW_TOPK_MISMATCH` to bypass the top-k contract. Permit `config.top_k != model index_topk` only when `decode_backend == "ds_flex"`; keep the current `flashmla_kv` path fail-closed.
   3. Add registered validator/config tests proving: unknown `decode_backend` rejects, `flashmla_kv` rejects `top_k > index_topk`, `ds_flex` accepts `top_k=8192` and `top_k=16384`, and the old env override cannot authorize production mismatch.
   4. Add a DS-owned flexible decode path in `dsa_backend.py`. After DS selection and logical-to-physical slot mapping, route `decode_backend=="ds_flex"` to a new `_forward_double_sparsity_flex` path that consumes `[bs, config.top_k]` physical slots directly and does not call `_forward_flashmla_kv`.
   5. Implement the flex path by gathering and dequantizing only the selected KV rows from the existing KV cache, then running attention over that gathered set. Reuse the existing page-table adapter, graph scratch ownership, and DS metadata; do not widen or mutate the native DSA indexer path.
   6. Add registered backend tests proving DS-flex metadata and scratch buffers size from `config.top_k`, sparse metadata reports the widened selection correctly, and CUDA-graph allocation remains bounded for `top_k=8192/16384`.
   7. Add a query-aware learned selector mode trained from V3.2's native DSA indexer on calibration prompts. Store the learned artifact as an explicit DS selector artifact and fall back to the current Method-1 channel mask only when the learned artifact is absent.
   8. Add compact TokenLabelTable storage before attempting 64K admission: `int8_symmetric` signatures plus per-layer/slot/head scale metadata; update `token_label_write.py` to quantize on write and `selection_kernel.py` to apply scales during scoring. Keep fp16 as the default until the compact path has unit and hardware evidence.
   9. Boot DS on H200 with compact labels and `decode_backend=ds_flex`, first at `top_k=8192`, then `top_k=16384`. Record `/get_server_info.max_total_num_tokens`; do not run 64K until the DS server admits more than the observed ~69,970-token prompt.
   10. Rerun the original AC-12 gate without the DS-fair re-scope as the pass condition. Update `ac12_analysis.md`, `evidence_bundle.md`, and `next_loop_issues.md` only from the new hardware artifacts.

2. AC-11 remains a recorded directional failure with follow-up work, not a satisfied target.

   Evidence:
   - `development/loop5/refined_plan_v1.md:72-77` says a TPS/TTFT miss is an AC-11 failure requiring follow-up tuning.
   - `runs/20260528_dsv32_mvp/mvp_compare_ac11.md:7-9` shows TTFT FAIL at conc 16/32/64 and TPS FAIL at conc 16/32.
   - `runs/20260528_dsv32_mvp/mvp_compare_ac11.md:21` says `AC-11 verdict: FAIL`.
   - `runs/20260528_dsv32_mvp/ac11_analysis.md:18-19` records the comparator miss as an AC-11 failure plus follow-up, and `ac11_analysis.md:40-47` lists the TokenLabelTable/KV-budget work before re-evaluating TTFT.

   Directive implementation plan:
   1. Complete the compact TokenLabelTable work from Mainline Gap 1 first.
   2. Reboot DS at the new admitted operating point and verify achieved concurrency tracks nominal at conc 16/32/64 before spending another full 3-trial run.
   3. Rerun the radix-on AC-11 sweep with the same DSA sidecar controls and the per-side `mem_fraction_static` consistency check enabled.
   4. Replace the AC-11 report only if the new comparator passes or explicitly records the updated miss with fresh effective-vs-nominal evidence and profile evidence.

## Blocking Side Issues

None technical in the Round-18 code state. The CPU and shell validation pass, and no Round-18 code diff exists.

## Queued Side Issues

1. `test/manual/test_double_sparsity_v32.py:752-754` still records word-count proxies as `length_tokens` and derives `within_budget` from the requested word count rather than actual tokenizer/chat input tokens. Current evidence was previously sanity-checked as safe, but the next substantive AC-12 harness edit should record actual prompt token counts and gate from those counts.

2. The DS-on-native-DSA strategic question remains valid: on V3.2, the current DS selector is inferior to the model's trained DSA indexer at the same 2048 budget. This should be an explicit owner decision, but it must not replace literal AC-12/AC-11 completion work while this no-deferral review loop continues.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added a Round-18 review correction row.
- Updated the Plan Version note to make the no-deferral completion state explicit.
- Kept literal original AC-12 and AC-11 completion work in Active Tasks.
- Removed Claude's re-added Explicitly Deferred row so the same work is not both Active and Deferred.

I reject Claude's requested tracker change that the owner-authorized deferral should govern this review's stop condition. The prompt explicitly says deferred work is incomplete and must be forced to completion.

Do not emit the stop sentinel: original AC-12 long-context quality, AC-11 TTFT follow-up, and the DS long-context/KV-budget work remain unfinished.
