Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 11/11 addressed (9 met, AC-11 executed with directional MISS, AC-12 executed with HARD FAIL but not met and evidence incomplete) | Forgotten items: 0 | Unjustified deferrals: 0 | Active original-plan tasks remaining: 2

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline, `round-11-prompt.md`, `round-11-contract.md`, `round-11-summary.md`, Round 8-10 summaries/reviews, and `goal-tracker.md`. Reviewed commits `7478c27a0` and `1a1293f01`, the HOST launcher knob, the AC-12 harness transport change, AC-12 artifacts, server_info files, and `evidence_bundle.md`.

Verification rerun:

```bash
PYTHONPATH=python pytest \
  test/registered/unit/development/test_ac11_comparator.py \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py \
  test/registered/unit/development/test_option_b_scripts.py \
  test/registered/unit/manual/test_ac12_helpers.py -q
# 407 passed, 24 warnings, 28 subtests

bash -n development/serve_double_sparsity.sh development/serve_native_nsa.sh
# pass
```

Artifact sanity check:

```text
found:   ac12_mmlu_5shot, ac12_niah_4096, ac12_niah_16384
missing: ac12_niah_65536
```

## Acceptance Criteria Audit

| AC | Status | Evidence / Gap |
|----|--------|----------------|
| AC-0 | MET | Previously verified hardware capture + producer regression. |
| AC-4 | MET | Previously verified native-FP8 sharded calibration, mask validation, and SHA. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, and invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on a >top_k prompt. |
| AC-1b | MET | Round 9 chunked-prefill probe passed at the radix-on operating point. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status. |
| AC-8 / AC-9 | MET | Round-4 smoke DS/DSA benchmark pair + comparator verified. |
| AC-10 | MET | Round-8 no-env-override radix flip verified; Round-11 bundle adds the requested label-capture provenance note. |
| AC-11 | EXECUTED, DIRECTIONAL MISS | Round 10 comparator artifact accepted; performance target not green. |
| AC-12 | EXECUTED, HARD FAIL, NOT MET | MMLU/4K/16K artifacts exist and match the summary. 64K fails with DS HTTP 400, but the required per-gate JSON artifact is missing. |
| AC-Q | MET | Round-8 corrected AC-Q gate verified with `all_pass=true`. |

## Mainline Gaps

1. **AC-12 NIAH 64K loses its durable per-gate artifact when DS rejects the prompt.**

   In `test/manual/test_double_sparsity_v32.py`, `_run_niah()` calls `_generate()` directly for every DSA prompt and then every DS prompt. `_niah_assert()` records the artifact only after `_run_niah()` returns. For 64K, DS returns HTTP 400, so `urllib.error.HTTPError` escapes at `test/manual/test_double_sparsity_v32.py:630` before `_record_artifact()` at lines 636-648 runs. The pytest summary proves the test failed (`ac12_pytest_summary.txt:19-23`), and `ds_boot_log_excerpt.txt` proves the DS admission error, but there is no `ac12_niah_65536_*.json`.

   This violates the Round-11 contract success criterion that `ac12_niah_{4096,16384,65536}.json` artifacts be produced and copied. It also makes `evidence_bundle.md` overstate `ac12_results/` as covering NIAH 4K/16K/64K while the bundle separately admits 64K wrote no artifact.

   Directive implementation plan:

   1. Make the NIAH generation path error-aware instead of letting HTTP errors bypass artifact recording. Add a small dataclass or dict shape for each generation attempt: `text`, `ok`, `http_status`, `error`, and optional response body.
   2. In `_run_niah()`, query DSA first as now, but collect attempts rather than raw strings. Then query DS attempts. If DS rejects the first 64K prompt, return a result object with DSA served/hit counts if available, DS served count, DS error details, and `delta_pct`/`verdict` set to a hard failure. Do not silently convert server errors into a pass.
   3. In `_niah_assert()`, always call `_record_artifact()` before failing. The 64K payload must include `length_tokens=65536`, `num_prompts`, DSA served/hits if available, DS served/hits, the HTTP status/message/body from the rejection, threshold, and `pass=false` or `verdict="FAIL"`.
   4. Add a registered CPU regression in `test/registered/unit/manual/test_ac12_helpers.py` that patches `_generate` so DSA returns the needle and DS raises `urllib.error.HTTPError`; assert that `_niah_assert(65536)` fails cleanly and records exactly one `niah_65536` artifact with the error details.
   5. Rerun the reported CPU suite. Then rerun at least the AC-12 64K gate on hardware with the same locked DS/DSA operating point and copy the new `ac12_niah_65536_*.json` into `runs/20260528_dsv32_mvp/ac12_results/`.
   6. Update `ac12_analysis.md` and `evidence_bundle.md` to reference the 64K JSON artifact. Only claim DSA served 20/20 if that value is present in the new artifact; otherwise phrase it as an attempted/partial DSA result plus DS admission failure.

2. **task15 evidence bundle remains incomplete until the 64K artifact exists.**

   The bundle is directionally honest about the bottom line: AC-12 hard-fails and the loop4-compatible MVP is not complete. It is not yet a complete task15 bundle because the 64K gate has no structured artifact. Keep task15 active until the artifact and bundle are corrected.

## Blocking Side Issues

1. **#L: AC-12 64K HTTP-error path is not artifact-safe.**

   This blocks task14/task15 verification, not because the AC-12 failure is disputed, but because the plan requires durable evidence for each AC-12 subgate. Fix #L before moving task14/task15 to Completed and Verified.

## Queued Side Issues

1. **Comparator per-side `mem_fraction_static` validation hole remains queued.**

   The current AC-11 artifact is not invalidated, and AC-12 did not touch the comparator. Keep the existing queued fix: compare `mem_fraction_static` within each side while ignoring DSA-vs-DS cross-side differences.

2. **AC-11 performance follow-up remains queued.**

   TokenLabelTable/KV-budget tuning is now also relevant to AC-12 64K admission, but it is performance/design work, not a prerequisite to recording the hard AC-12 failure honestly.

3. **Stale `calibrate.py` operator recipe docstring remains queued.**

4. **Plan-specific terms were reintroduced in production comments/help.**

   Round 9 cleaned this up, but Round 11 added new references such as `AC-12`, `AC-Q`, `BL-...`, and "Option B" in `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`, and `test/manual/test_double_sparsity_v32.py`. This does not block AC-12 evidence, but should be reworded next time those files are touched.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Rejected Claude's requested move of task14/task15 to Completed and Verified for now.
- Kept task14/task15 Active, marked as executed/drafted but not verified complete.
- Added Round-11 review correction to the Plan Evolution Log.
- Added #L as a Blocking Side Issue for task14/task15 verification.
- Accepted the AC-10 label-capture provenance note as resolved by `evidence_bundle.md`.
- Added the plan-term reintroduction as a queued side issue.

Original plan work remains pending.
