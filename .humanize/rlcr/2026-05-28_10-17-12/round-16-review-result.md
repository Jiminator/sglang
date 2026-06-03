Mainline Progress Verdict: STALLED

Goal Alignment Summary:
ACs: 11/11 addressed; 9/11 literally met under the immutable AC text, with AC-11 executed as a recorded directional FAIL/follow-up and AC-12 original 4K/16K/64K NIAH still failing | Forgotten items: 0 | Unjustified deferrals under this review's no-deferral completion rule: 3

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-16-prompt.md`, `round-16-contract.md`, `round-16-summary.md`, Round 13-15 summaries/reviews, `goal-tracker.md`, commit `947157471`, the changed serve scripts/manual harness, `next_loop_issues.md`, `ac12_analysis.md`, `evidence_bundle.md`, AC-11/AC-12 artifacts, and the relevant DS backend/config/TokenLabelTable code.

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

The Round-16 code diff itself is safe and comment-only. The serve-header comments and the older `plan §10` / `plan §9.4` / `design doc §9.5` comments were reworded without changing launcher flags or harness behavior. The reported CPU validation reproduces.

The close-out claim does not satisfy the strict review prompt. Claude explicitly carried DS long-context R&D to a future loop, while this review prompt says deferred tasks are incomplete and `COMPLETE` is allowed only when all original plan tasks and ACs are fully done with no deferrals. The original plan and immutable tracker still say AC-12 is NIAH 4K/16K/64K + MMLU with all gates passing; the current artifacts keep 4K/16K/64K as `verdict=FAIL`.

The AC-11 completion claim is also overstated. DEC-7 makes a TPS/TTFT miss non-fatal, but the plan text says the miss is recorded as an AC-11 failure requiring follow-up tuning, not that the performance target is satisfied. `mvp_compare_ac11.md` still records `AC-11 verdict: FAIL`, with TTFT failing at all three concurrencies.

## Mainline Gaps

1. Literal original AC-12 remains incomplete, and the required enabling work is deferred.

   Evidence:
   - `development/loop5/refined_plan_v1.md` defines AC-12 as NIAH 4K/16K/64K + MMLU, with any gate below threshold failing the loop4 MVP claim.
   - `goal-tracker.md` immutable AC-12 preserves the same requirement.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_4096_20260529T190151Z.json`: DSA 100%, DS 75%, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_16384_20260529T190258Z.json`: DSA 100%, DS 5%, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_65536_20260529T190614Z.json`: DSA 100%, DS served 0/20, HTTP 400 admission failure, `verdict=FAIL`.
   - `runs/20260528_dsv32_mvp/next_loop_issues.md:19-36` carries the required selector/kernel/TokenLabelTable work forward instead of completing it.

   Directive implementation plan:
   1. Add an explicit DS-flex decode backend to the Double Sparsity path. Extend `DoubleSparsityConfig` with a validated backend selector, e.g. `decode_backend="flashmla_kv"` by default and `decode_backend="ds_flex"` for the new path. Keep unknown top-level fields rejected.
   2. In `validator.py`, remove the production use of `SGLANG_DS_ALLOW_TOPK_MISMATCH` for serving. Allow `config.top_k != model index_topk` only when `decode_backend=="ds_flex"`; keep the current `flashmla_kv` path fail-closed for mismatched `top_k`.
   3. Implement the DS-flex decode call in `dsa_backend.py` after DS selection and `logical_to_physical`. It must consume the selector's `[bs, config.top_k]` physical-slot tensor directly, support `top_k=8192` and `top_k=16384`, and avoid `_forward_flashmla_kv`'s `indices.shape[-1] == self.dsa_index_topk` assertion. Reuse the existing page-table adapter and scratch buffers; do not widen the native DSA indexer path.
   4. Add registered tests proving: current `flashmla_kv` rejects `top_k > index_topk`; `ds_flex` accepts it; metadata/scratch allocation sizes use `config.top_k`; and the old env override cannot make production serving bypass the backend check.
   5. Reduce TokenLabelTable HBM usage enough to admit the ~70K-token 64K NIAH prompt. The concrete change is to add a compact signature storage mode for `TokenLabelTable.signatures` (fp8 or int8 with explicit scale metadata), update `token_label_write.py` and `selection_kernel.py` to dequantize/score from that representation, and preserve the existing fp16 mode as the default until the compact path passes unit tests.
   6. Boot DS on H200 with `top_k=8192` first and run a staged NIAH sweep: 4K, 16K, then 64K only after `/get_server_info.max_total_num_tokens` exceeds the observed ~69,970-token prompt length. If 16K fails at a near-dense/wider selection budget, treat it as a DS-flex decode bug, not a documentation issue.
   7. Rerun the full literal AC-12 gate without re-scope logic becoming the pass condition. Update `ac12_analysis.md`, `evidence_bundle.md`, and `next_loop_issues.md` only from new hardware artifacts.

2. AC-11 remains a recorded directional failure with follow-up work, not a fully satisfied target.

   Evidence:
   - `runs/20260528_dsv32_mvp/mvp_compare_ac11.md:7-9` shows TTFT FAIL at conc 16/32/64 and TPS FAIL at conc 16/32.
   - `runs/20260528_dsv32_mvp/mvp_compare_ac11.md:21` says `AC-11 verdict: FAIL`.
   - `runs/20260528_dsv32_mvp/ac11_analysis.md:10-18` records a directional MISS.
   - `next_loop_issues.md:26-36` carries TokenLabelTable/KV-budget and AC-11 TTFT follow-up forward.

   Directive implementation plan:
   1. Complete the TokenLabelTable compact-storage work from Mainline Gap 1 before rerunning AC-11.
   2. Reboot DS at the new admitted operating point and verify achieved concurrency is close to nominal at conc 16/32/64 before running the long comparator.
   3. Rerun the radix-on 3-trial AC-11 comparator with the same DSA sidecar controls and the per-side `mem_fraction_static` consistency check enabled.
   4. Replace the current AC-11 artifact only if the new run passes or explicitly records the updated miss with the new effective-concurrency table.

3. Round-16 plan-term hygiene is not fully resolved.

   Evidence:
   - `development/serve_double_sparsity.sh:7` still says `startup (per DEC-8)`, which is a production comment containing a plan marker.
   - `runs/20260528_dsv32_mvp/next_loop_issues.md:52-55` still says the pre-existing "Locked Option B operating point (plan §13 / DEC-1)" serve-header lines remain, even though commit `947157471` reworded those exact headers.

   Fix:
   1. Reword `serve_double_sparsity.sh:7` to behavior-based wording.
   2. Remove or update the stale cosmetic section in `next_loop_issues.md`.
   3. Keep the `Round 26-29` MMLU historical comments if they are intentionally historical, but do not call the hygiene item fully resolved while the stale DEC/handoff references remain.

## Blocking Side Issues

None. The Round-16 comment-only code change does not introduce a serving or test regression.

## Queued Side Issues

1. `test/manual/test_double_sparsity_v32.py:752-754` still records word counts as `length_tokens` and derives `within_budget` from the requested word count rather than actual tokenizer/chat input tokens. Current evidence was previously sanity-checked as safe, but the next harness edit should record actual prompt tokens and gate from them.

2. The DS-on-native-DSA strategic question in `next_loop_issues.md` remains valid: on V3.2, the current DS selector is inferior to the model's trained DSA indexer at the same 2048 budget, so further V3.2 DS investment needs an explicit owner decision.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added a Round-16 review correction row rejecting the literal completion claim under this review's no-deferral rule.
- Replaced the "no active work remains" Active Tasks text with active original-AC completion work for literal AC-12 and AC-11 follow-up.
- Added a queued hygiene correction for the remaining `DEC-8` production comment and stale `next_loop_issues.md` cosmetic row.

I reject Claude's requested tracker change that "all 11 ACs are satisfied" for the purpose of the stop condition. That statement is defensible only under Plan Version 2 close-out semantics; it is not true under the immutable original AC text plus this review prompt's explicit no-deferrals rule.

Do not emit the stop sentinel: original AC-12 literal quality, AC-11 follow-up, and the deferred DS long-context/KV-budget work remain unfinished.
