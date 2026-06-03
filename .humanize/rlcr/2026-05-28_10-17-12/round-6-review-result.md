Mainline Progress Verdict: STALLED

Goal Alignment Summary:
ACs: 7/11 addressed (6/11 met) | Forgotten items: 0 | Unjustified deferrals: 0 accepted; 5 original-plan active tasks still pending

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline, taste-review knowledge, `goal-tracker.md`, `round-6-prompt.md`, `round-6-contract.md`, `round-6-summary.md`, round 3-5 summaries/reviews, commit `70bb52a15`, the changed quality-smoke code, and Round-6 artifacts under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python pytest test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 265 passed, 24 warnings in 11.78s

env -u DS_BASE_URL -u DSA_BASE_URL PYTHONPATH=python pytest test/manual/test_dsv32_quality_smoke.py -q
# 1 skipped, 1 warning in 0.02s
```

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Previously verified hardware capture + unit suite. |
| AC-4 | MET | Previously verified calibrated FP8 mask + loader validation. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on >top_k prompt. |
| AC-1b | NOT MET | Chunked-prefill probe has not run; must precede AC-11. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status. |
| AC-8 / AC-9 | MET | Round-4 smoke benchmark pair + comparator verified. |
| AC-10 | NOT MET | No no-env-override radix flip, no final radix-on DS launch, fixtures not run. |
| AC-11 | NOT MET | No 3-trial radix-on 120s/600s sweep; #F must be resolved or explicitly handled first. |
| AC-12 | NOT MET | Full NIAH 4K/16K/64K + MMLU 5-shot gate has not run. |
| AC-Q | ADDRESSED, NOT MET | `runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:10-27` still has `mean_rouge_l=0.726 < 0.85` and `all_pass=false`. |

## Mainline Gaps

1. **AC-Q is still failed, and the measurement-change request is not approved.**

   The immutable AC-Q definition is unchanged: any gate miss fails the quality smoke. The Round-5/6 result still has `mean_rouge_l` below threshold and `all_pass=false` (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:10-27`). Round 6 did not rerun AC-Q to a pass and did not receive approval to alter decoding, prompts, or thresholds.

2. **The Round-6 "not a DS bug" diagnosis does not satisfy its own contract.**

   `round-6-contract.md:23-35` required DS `meta_info["double_sparsity"]` for the short failing prompts, specifically to rule out context-dropping selection/label bugs before calling this inherent greedy numerics. The success criteria repeat this at `round-6-contract.md:46-55`.

   The committed graph/eager artifacts only contain OpenAI chat-completion payloads with `usage` and `metadata`; there is no `meta_info["double_sparsity"]` in `runs/20260528_dsv32_mvp/ds_diag_graph_chat_1723.json:1` or `runs/20260528_dsv32_mvp/ds_diag_eager_chat_1723.json:1`. That proves the loop is not regular-CUDA-graph-specific, but it does not prove DS selected all available context, nor that `dense_fallback` stayed 0 on the exact failing prompt.

   The concise-answer and sampling controls are also only summarized in markdown (`runs/20260528_dsv32_mvp/ac_q_diagnosis_round6.md:16-25`); no raw JSON outputs for those controls are committed. Those claims may be true, but they are not reviewable evidence.

3. **The original plan remains incomplete.**

   The Smoke MVP still lacks a passing AC-Q artifact. The Loop4-compatible tier remains active and unimplemented: task11 AC-10, task12 AC-1b, task13 AC-11, task14 AC-12, and task15 evidence bundle.

## Blocking Side Issues

1. **#H remains blocking: DS selection/label path is not exonerated for the short AC-Q failure.**

   Blocking AC: AC-Q / TIER-1 Smoke MVP.

   Required correction: capture reviewable raw artifacts for the exact failing chat-formatted prompts that include DS `meta_info["double_sparsity"]`. For seq <= top_k, the expected healthy shape is full-context selection (`selected_tokens == seq_len` or equivalent), `sparsity_rate == 0`, and `dense_fallback == 0`. If that does not hold, fix the DS selection/label path before discussing measurement changes.

## Queued Side Issues

1. **#I is resolved and verified.** `_validate_reference_artifact` now enforces the exact committed 20 smoke prompts and 5 NIAH prompt/needle pairs position-by-position (`test/manual/_dsv32_quality_smoke_lib.py:322-359`), and the new truncated/reordered/wrong-needle regressions pass (`test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py:159-180`).

2. **#F remains queued for AC-11.** DS effective concurrency at `mem_fraction_static=0.6` will make TTFT comparison dishonest unless resolved or explicitly accounted for before task13.

3. The stale `calibrate.py` operator recipe docstring remains queued cleanup.

## Goal Tracker Updates Applied

Updated only the mutable tracker section:

- Added a Round-6 review correction entry.
- Kept task9 active as `NOT MET; #H still blocking after incomplete diagnosis`.
- Rejected Claude's request to confirm #H as definitively "not a DS bug" because the required metadata/raw controls are missing.
- Marked #I resolved and verified.
- Left `Explicitly Deferred` empty.

## Directive Implementation Plan

1. Keep task9 / AC-Q as the next mainline objective. Do not start task11-task15 until AC-Q is either passing under the current immutable gate or the user explicitly approves a measurement change.
2. Re-run the DS `17 * 23` and primes controls on 8x H200 with raw artifacts saved. Use the exact chat-formatted prompt text that the AC-Q chat-completions path sends, and capture both output text and DS `meta_info["double_sparsity"]` through a metadata-capable path.
3. Save raw JSON for: DS graph temp-0 `17*23`, DS eager temp-0 `17*23`, DS graph temp-0 primes, DSA temp-0 `17*23`, DS concise `17*23`, DS concise primes, and DS temp-0.5 `17*23`. Include request bodies, server info, generated text, and DS metadata where applicable.
4. If the metadata shows less than full-context selection or any dense fallback on seq<=top_k, fix the DS selection/label/summary path and add the narrowest regression that covers the defect.
5. If the metadata proves full-context selection with no fallback and the raw controls prove only greedy decoding degenerates, file one concrete measurement-change request for user approval; do not alter harness defaults until approval.
6. After the fix or approved measurement change, rerun the sequential `capture` then `compare` workflow on hardware and require `all_pass=true`.
7. Then continue the original plan in order: task11 AC-10 radix flip with fixtures and no env override; task12 AC-1b; task13 AC-11 after #F; task14 AC-12; task15 evidence bundle.

Original plan work remains pending.
