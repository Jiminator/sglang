Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 11/11 addressed (9 met, AC-11 executed with directional MISS, AC-12 executed/evidence-complete HARD FAIL and not met) | Forgotten items: 0 | Unjustified deferrals: 0, but AC-12 completion work is carried forward and blocks COMPLETE

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review knowledge, `round-13-prompt.md`, `round-13-contract.md`, `round-13-summary.md`, Round 10-12 summaries/reviews, and `goal-tracker.md`.

Reviewed commits `ced03f374` and `27434cee7`, the comparator changes, the new comparator regression, the calibration docstring update, the top-k investigation artifacts, `next_loop_issues.md`, and current goal-tracker state.

Verification rerun:

```bash
PYTHONPATH=python pytest \
  test/registered/unit/development/test_ac11_comparator.py \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py \
  test/registered/unit/development/test_option_b_scripts.py \
  test/registered/unit/manual/test_ac12_helpers.py -q
# 409 passed, 24 warnings, 28 subtests
```

## Acceptance Criteria Audit

| AC | Status | Evidence / Gap |
|----|--------|----------------|
| AC-0 | MET | Previously verified hardware capture + producer regression. |
| AC-4 | MET | Previously verified native-FP8 sharded calibration, mask validation, and SHA. Round 13 fixed only the stale operator recipe docstring. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, and invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on a >top_k prompt. |
| AC-1b | MET | Round 9 chunked-prefill probe passed at the radix-on operating point. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status. |
| AC-8 / AC-9 | MET | Round-4 smoke DS/DSA benchmark pair + comparator verified. |
| AC-10 | MET | Round-8 no-env-override radix flip verified; Round-11 bundle provenance note accepted. |
| AC-11 | EXECUTED, DIRECTIONAL MISS | Round 10 comparator artifact accepted; Round 13 fixed the queued per-side `mem_fraction_static` validation hole. |
| AC-12 | EXECUTED, HARD FAIL, EVIDENCE COMPLETE, NOT MET | MMLU passes; NIAH 4K/16K/64K fail thresholds. Round 13 characterizes the failure as a fixed-budget selection-quality limit plus 64K admission limit; it does not make AC-12 pass. |
| AC-Q | MET | Round-8 corrected AC-Q gate verified with `all_pass=true`. |

## Implementation Review

No high-signal Round-13 implementation bug found.

The comparator cleanup matches the queued directive. `_validate_per_side_agreement()` now compares raw `server_args.mem_fraction_static` within each side after the normalized cross-side projection still ignores it (`development/benchmark_compare.py:829-865`). The new regression covers both cases: DSA per-side drift refuses with exit 2, and constant DSA 0.85 vs DS 0.6 proceeds.

The `calibrate.py` docstring now matches the committed calibration provenance: `--tp 8`, local `--dataset`, `-v`, `torch_dtype="auto"` load, and `device_map="auto"` sharding (`python/sglang/srt/layers/attention/double_sparsity/calibrate.py:15-37`).

The top-k investigation is coherent with the code. The validator refuses `top_k=8192` unless the mismatch override is set, and the `flashmla_kv` decode path then asserts `indices.shape[-1] == self.dsa_index_topk` (`runs/20260528_dsv32_mvp/ac12_topk_sweep/boot_evidence_topk_locked.txt:3-12`, `python/sglang/srt/layers/attention/dsa_backend.py:2148`). The fresh DS recall artifact records 1024/1536/4096 rows at `top_k=2048`; the 16K and 64K points in the analysis are correctly the prior AC-12 artifacts, not fresh rows in `ds_recall_vs_length_topk2048.json` (`runs/20260528_dsv32_mvp/ac12_topk_sweep/ds_recall_vs_length_topk2048.json:6-28`).

## Mainline Gaps

1. **AC-12 remains not met; the loop4-compatible MVP is still incomplete.**

   This is not a Round-13 regression, but it is still the plan-derived completion gap. The original AC-12 positive criterion requires all NIAH 4K/16K/64K and MMLU gates to pass. The committed evidence still has NIAH failures, and `next_loop_issues.md` explicitly carries the AC-12 disposition/R&D forward (`runs/20260528_dsv32_mvp/next_loop_issues.md:7-31`). Therefore this review must not emit the stop word.

   Directive implementation plan if the loop continues toward the original AC-12 instead of accepting/re-scoping it:

   1. Keep the AC-12 harness and thresholds unchanged.
   2. Add a DS-owned decode backend that does not call `flashmla_kv` with the fixed V3.2 `index_topk`. Use the existing DS selector output shape `[bs, config.top_k]`, map logical positions to physical KV slots through the page-table adapter, gather the selected KV rows, and run attention over that gathered set. Wire it as an explicit DS decode backend; keep the current `flashmla_kv` path fail-closed for `top_k != index_topk`.
   3. Update validation so `top_k > index_topk` is allowed only with the new DS-flex decode backend and never via the current environment override. Add registered tests for validator behavior and a unit test that the new path accepts `top_k=8192/16384` while the old path rejects.
   4. Reduce the TokenLabelTable footprint so DS can admit the 64K NIAH prompt at a higher `mem_fraction_static`. The concrete target is to change `TokenLabelTable.signatures` storage from fp16 to a compact fp8/int8 representation with selection-time dequantization, then boot DS with enough KV budget for the ~70K-token prompt and record `/get_server_info.max_total_num_tokens`.
   5. Run a hardware sweep at `top_k=2048,8192,16384` on 16K first. If 16K still fails at full/near-full selection, treat it as a decode/backend bug and fix before moving on.
   6. Run the full AC-12 gate only after 16K passes and 64K is servable. Commit the new per-gate JSONs and update `ac12_analysis.md` and `evidence_bundle.md` without changing the immutable AC.

## Blocking Side Issues

None for the Round-13 close-out objective. The new comparator regression and CPU suite pass, and the top-k lock is a real backend constraint rather than an unverified claim.

## Queued Side Issues

1. AC-11 TTFT/effective-concurrency follow-up remains valid after the TokenLabelTable/KV-budget work; it should not be treated as green performance.
2. The pre-existing plan-specific header terms in `development/serve_*.sh` remain cosmetic only.
3. The top-k investigation artifacts are adequate for the conclusion, but if this evidence is used outside the RLCR context, copy the full boot logs/commands alongside the excerpted `boot_evidence_topk_locked.txt`.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added a Round-13 review correction entry accepting the queued cleanup resolutions and the top-k investigation characterization.
- Kept AC-12 marked NOT MET and the loop4-compatible MVP incomplete.
- Kept the Explicitly Deferred table empty, while noting the AC-12 disposition/R&D remains carried forward and blocks COMPLETE.

Do not output the stop word: the Round-13 close-out advanced the diagnosis, but the original AC-12 hard gate is still not satisfied.
