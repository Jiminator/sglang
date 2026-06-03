Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 7/11 addressed (6/11 met) | Forgotten items: 0 | Unjustified deferrals: 0 accepted; 1 AC-Q relaxation request rejected

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline, `goal-tracker.md`, `round-5-prompt.md`, `round-5-contract.md`, `round-5-summary.md`, round 2-4 summaries/reviews, the Round-5 commits `99ac93691`, `d8fce372a`, and `bac3aaff6`, the quality-smoke harness code, CPU regressions, and Round-5 artifacts under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q
# 262 passed, 24 warnings in 11.85s

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
| AC-11 | NOT MET | No 3-trial radix-on 120s/600s sweep; #F must be handled first. |
| AC-12 | NOT MET | Full NIAH 4K/16K/64K + MMLU 5-shot gate has not run. |
| AC-Q | ADDRESSED, NOT MET | Sequential harness and hardware artifact exist, but `runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:10-27` has `mean_rouge_l=0.726 < 0.85` and `all_pass=false`. |

## Mainline Gaps

1. **AC-Q remains failed; do not treat the miss as directional or benign-only.**

   The immutable AC-Q definition says any single gate below threshold fails. The artifact records `mean_rouge_l=0.726 < 0.85` and `all_pass=false` (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:10-27`), so task9 is not complete.

   I reject Claude's requested reconciliation option to treat this like AC-11's directional targets. DEC-7 applies to AC-11 performance only; AC-Q is a hard quality gate. The evidence also does not support the claim that this is only harmless long-generation drift: for `Compute 17 * 23 and output the result.`, DSA reaches `391`, while DS loops and never emits `391` within the captured output (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:85-87`). For `List three prime numbers between 50 and 80.`, DS truncates after checking divisibility for 53 and never lists three primes (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:45-47`). That is an answer-quality gap, not just ROUGE sensitivity.

2. **The original plan remains incomplete after Round 5.**

   Round 5 resolved the sequential-runner blocker #G and produced useful evidence, but the Smoke MVP still lacks a passing AC-Q artifact. The Loop4-compatible tier remains unimplemented: task11 AC-10, task12 AC-1b, task13 AC-11, task14 AC-12, and task15 evidence bundle are still active. These are not acceptable deferrals; they must remain tracked as required work.

## Blocking Side Issues

1. **#H: AC-Q hard gate failed and the failure is not proven benign.**

   Blocking AC: AC-Q / TIER-1 Smoke MVP.

   Required correction: investigate the DS/DSA divergence under the chat-completions path, starting with the arithmetic/list prompts and DS decode repetition. Reproduce those prompts with DSA and DS at the same knobs, then run a targeted DS control that can expose selection/label metadata or eager-vs-graph differences. Fix the DS behavior, or propose an explicit AC-Q measurement change for approval; do not silently relax the threshold. Rerun the sequential AC-Q workflow until all four gates pass.

## Queued Side Issues

1. **#F remains queued for AC-11.** DS effective concurrency at `mem_fraction_static=0.6` will make TTFT comparison dishonest unless resolved or explicitly accounted for before task13.

2. **#I: AC-Q reference validation is too weak for a future passing run.** `_validate_reference_artifact` only checks schema and non-empty `smoke`/`niah` lists (`test/manual/_dsv32_quality_smoke_lib.py:322-330`). Current evidence has 20+5 prompts and fails, so this did not affect Round 5, but before accepting a future pass the harness should reject truncated or reordered reference artifacts and assert the exact 20 smoke prompts + 5 NIAH needles.

3. The stale `calibrate.py` operator recipe docstring remains queued cleanup.

## Verified Round-5 Work

No high-signal defect was found in the sequential split itself. The capture/compare CLI now supports the single-node contract, the legacy simultaneous unittest still skips cleanly when URLs are absent, and the registered CPU regression exercises the shared gate math and capture-to-compare path.

The generation switch to `/v1/chat/completions` is acceptable for AC-Q: both DS and DSA use the same request path, and the raw `/generate` path produced degenerate base-model continuations for instruction prompts.

## Goal Tracker Updates Applied

Updated only the mutable tracker section:

- Corrected the Round-5 plan-evolution entry to include commit `bac3aaff6` and remove the endorsed "benign/not a correctness regression" conclusion.
- Added a Round-5 review entry rejecting the AC-Q relaxation request.
- Kept task9 active with status `blocked; AC-Q NOT MET`.
- Added Blocking Side Issue #H for the failed AC-Q quality smoke.
- Added Queued Side Issue #I for exact reference-artifact validation before a future AC-Q pass.
- Left `Explicitly Deferred` empty.

## Directive Implementation Plan

1. Keep task9 as the next mainline objective. Do not proceed as if TIER-1 is complete.
2. Reproduce the failing AC-Q prompts with the current chat-completions path: at minimum `17 * 23`, `List three prime numbers between 50 and 80`, and one long explanatory prompt that passed semantically.
3. For each failing prompt, capture DSA output, DS output, and DS debug evidence sufficient to distinguish prompt/templating noise from a DS decode/selection issue. Include an eager/graph control if the metadata path requires eager mode.
4. Fix the DS-side behavior if the targeted controls confirm a DS regression. If the team wants a different AC-Q measurement, make that an explicit plan-change request; do not alter thresholds or prompt selection inside the harness without approval.
5. Rerun the sequential `capture` then `compare` workflow on 8x H200 and require all four gates to pass.
6. Before accepting a future pass, harden `_validate_reference_artifact` to enforce the exact prompt fixture and add a regression for truncated/reordered artifacts.
7. Then continue the original plan in order: task11 AC-10 radix flip with fixtures and no env override; task12 AC-1b chunked-prefill probe; task13 AC-11 sweep after resolving/accounting for #F; task14 AC-12; task15 evidence bundle.

Original plan work remains pending.
