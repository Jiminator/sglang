Mainline Progress Verdict: STALLED

Goal Alignment Summary:
ACs: 11/11 addressed; 9/11 literally met under the immutable AC text, with AC-11 executed as a recorded directional FAIL/follow-up and original AC-12 NIAH 4K/16K/64K still failing | Forgotten items: 0 | Unjustified deferrals: 1

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-17-prompt.md`, `round-17-contract.md`, `round-17-summary.md`, Round 14-16 summaries/reviews, `goal-tracker.md`, commit `989975625`, the touched serve script and handoff doc, AC-11/AC-12 artifacts, and the relevant DS config/validator/backend/selector/TokenLabelTable code.

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

No high-signal bug found in the Round-17 code/doc diff itself. `development/serve_double_sparsity.sh:6-7` now uses behavior-based HiSparse-exclusion wording, and `rg "DEC-|plan §|AC-|Option B" development/serve_double_sparsity.sh development/serve_native_nsa.sh` returns no matches. `runs/20260528_dsv32_mvp/next_loop_issues.md:52-56` no longer falsely says the already-reworded serve-header lines remain. Codex Gap #3 is resolved.

The close-out / deferral claim is not accepted under this review prompt. The prompt explicitly says deferred tasks are incomplete and must be forced to completion. Claude intentionally did not implement the literal original AC-12 and AC-11 follow-up work, and moved it to `Explicitly Deferred`; that is tracker drift for this review's stop condition.

## Mainline Gaps

1. Literal original AC-12 is still incomplete and was deferred.

   Evidence:
   - `development/loop5/refined_plan_v1.md` and the immutable tracker define AC-12 as NIAH 4K/16K/64K + MMLU, with any gate below threshold failing the loop4 MVP claim.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_4096_20260529T190151Z.json`: DSA 100%, DS 75%, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_16384_20260529T190258Z.json`: DSA 100%, DS 5%, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_65536_20260529T190614Z.json`: DSA 100%, DS served 0/20, HTTP 400 admission failure, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/next_loop_issues.md:19-36` still carries the required selector/kernel/TokenLabelTable work forward.

   Directive implementation plan:
   1. Extend `DoubleSparsityConfig` with a top-level `decode_backend` field. Default it to `flashmla_kv`; add `ds_flex` as the only new value; keep unknown fields rejected.
   2. Change `validator.py` so production no longer uses `SGLANG_DS_ALLOW_TOPK_MISMATCH` to bypass the top-k contract. Permit `config.top_k != model index_topk` only when `decode_backend == "ds_flex"`; keep the current `flashmla_kv` path fail-closed.
   3. Add a DS-owned flexible decode path in `dsa_backend.py`. After DS selection and `logical_to_physical`, route `decode_backend=="ds_flex"` to a new `_forward_double_sparsity_flex` path that consumes the selected physical slots with shape `[bs, config.top_k]` directly and does not call `_forward_flashmla_kv`, whose `indices.shape[-1] == self.dsa_index_topk` assertion is the current hard cap.
   4. Implement the flex path by gathering/dequantizing only the selected KV rows from the existing KV cache and running attention over that gathered set. Reuse the existing page-table adapter, graph-state allocation, and DS metadata; do not widen or mutate the model's native DSA indexer path.
   5. Add registered tests proving: `flashmla_kv` rejects `top_k > index_topk`; `ds_flex` accepts `top_k=8192` and `top_k=16384`; metadata and graph scratch sizes follow `config.top_k`; and the old env override cannot make production serving bypass the backend check.
   6. Add a query-aware learned selector mode that improves over the current offline channel-mask selector on long NIAH. Use the native V3.2 DSA indexer as the teacher on calibration prompts, distill the selected-token target into DS selector weights, and keep the existing Method-1 mask as the fallback only when the learned artifact is absent.
   7. Reduce TokenLabelTable HBM enough to admit the ~70K-token 64K prompt. Add an `int8_symmetric` signature storage mode: `TokenLabelTable.signatures` stores int8 labels, `signature_scales` stores per-layer/slot/head scale, `token_label_write.py` quantizes on write, and `selection_kernel.py` multiplies by the scale while scoring. Keep fp16 as the default until the compact path has unit and hardware coverage.
   8. Boot DS on H200 with compact labels and `decode_backend=ds_flex`, first at `top_k=8192`, then `top_k=16384`. Run staged NIAH 4K, then 16K, then 64K only after `/get_server_info.max_total_num_tokens` exceeds the observed ~69,970-token prompt length.
   9. Rerun the full original AC-12 gate without using the DS-fair re-scope as the pass condition. Update `ac12_analysis.md`, `evidence_bundle.md`, and `next_loop_issues.md` only from new hardware artifacts.

2. AC-11 remains a recorded directional failure with follow-up work, not a satisfied target.

   Evidence:
   - `runs/20260528_dsv32_mvp/mvp_compare_ac11.md:7-9` shows TTFT FAIL at conc 16/32/64 and TPS FAIL at conc 16/32.
   - `runs/20260528_dsv32_mvp/mvp_compare_ac11.md:21` says `AC-11 verdict: FAIL`.
   - `runs/20260528_dsv32_mvp/ac11_analysis.md:42-45` says TokenLabelTable footprint / KV budget must be reduced before re-running.

   Directive implementation plan:
   1. Complete the compact TokenLabelTable work from Gap 1 first.
   2. Reboot DS at the new admitted operating point and verify achieved concurrency tracks nominal at conc 16/32/64 before spending a full 3-trial run.
   3. Rerun the radix-on AC-11 sweep with the same sidecar controls and the per-side `mem_fraction_static` consistency check enabled.
   4. Replace the AC-11 report only if the new comparator passes or explicitly records the updated miss with a new effective-vs-nominal table and profile evidence.

## Blocking Side Issues

None. Round 17's actual file diff is comment/doc-only and the validation passes.

## Queued Side Issues

1. `test/manual/test_double_sparsity_v32.py` still records word-count proxies as `length_tokens` for the within-budget gate. Current evidence was previously sanity-checked as safe, but the next substantive harness edit should record actual tokenizer/chat input tokens and gate from those counts.

2. The DS-on-native-DSA strategic question remains valid: on V3.2, the current DS selector is inferior to the model's trained DSA indexer at the same 2048 budget, so further V3.2 DS investment needs an explicit owner decision. It must not replace the literal AC-12/AC-11 completion work while this no-deferral review loop continues.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added a Round-17 review correction row.
- Marked the Round-16 hygiene close-out issue resolved by commit `989975625`.
- Replaced the "no active tasks" text with active literal original AC-12 and AC-11 completion tasks, with `Tag=coding` and `Owner=Claude`.
- Rejected the requested `Explicitly Deferred` entry for this review's stop condition and left the deferral table empty.

I reject Claude's requested tracker change that the owner-authorized deferral should empty Active Tasks. That may be a valid owner preference outside this prompt, but this review prompt explicitly defines deferred work as incomplete.

Do not emit the stop sentinel: original AC-12 literal quality, AC-11 TTFT follow-up, and the DS long-context/KV-budget work remain unfinished.
