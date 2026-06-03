# Round 9 Review Result

Mainline Progress Verdict: ADVANCED

Acceptance Criteria: 4/10 met (0 deferred) | Active Tasks: 6 remaining | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review knowledge, `goal-tracker.md`, Round 6-8 summaries/reviews, Round 9 summary, commit `57f86b66f`, the R9 AC-5 artifacts under `runs/20260530_dsv32_loop6/client_slo_int8/`, and design/SLO references in `development/CLIENT_SLOS.md`, `development/roadmap.md`, `development/loop6/runbook.md`, and `development/past_implementations/study/07-mvp-proposed-architecture.md`.

## Goal Alignment Check

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified in earlier reviews: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`. |
| AC-2 | MET | Feasibility budget and binding int8 same-`label_dim` lever verified in R1 review. |
| AC-3 | MET | Compact int8 `TokenLabelTable`, scale sidecar consumers, launcher `SIGNATURE_DTYPE`, CUDA-graph-safety, selection-equivalence, real-mask NIAH, and microbench evidence verified by R3 review. |
| AC-4 | MET | Mem-fraction lift to the 0.7 operating point, full HBM budget, and durable no-OOM stress proof verified by R5 review. |
| AC-5 | PARTIAL | R9 resolves the evidence/attribution verifier: TTFT/TPOT/TPS/ITL are now exact-recomputable from tracked files and fail-closed. The strict SLO still fails: conc 32/64 TTFT > 22 s and TPS < 30 at every conc, so this is directional only, not shippable. |
| AC-6 | PARTIAL | Dev checks exist from earlier rounds, but the required hardware DSA-default/no-DS-table/SLO product proof and DS opt-in toggle proof remain pending. |
| AC-7 | NOT MET | Lifted-point 3-trial DS+DSA AC-11 re-sweep remains pending. |
| AC-8 | NOT MET | Lifted-point ~70K-token servability probe remains pending. |
| AC-9 | NOT MET | `test/manual/test_double_sparsity_v32.py` still needs real `usage.prompt_tokens` budget assertion plus live rerun/copy of artifacts. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion; no Tier-2 work should start yet. |

Forgotten items: none. Every task in `development/loop6/refined_plan_v1.md` is represented in Active, Completed, or the gated AC-10 path. I found no task marked complete in the tracker without Codex verification.

Deferred items: none explicit. AC-10 is gated, not deferred; that remains valid and does not contradict the Ultimate Goal because the plan requires Tier-1 AC-3 through AC-9 first.

Goal completion summary:

```text
Acceptance Criteria: 4/10 met (0 deferred)
Active Tasks: 6 remaining (task6 strict-SLO resolution + task7-task11)
Estimated remaining rounds: 4-6 if hardware availability is good; more if strict SLO requires scheduler/decode work
Critical blockers: strict client SLO miss; AC-6/7/8/9 pending hardware/harness work; AC-10 gated
```

## Mainline Drift Audit

Round 9 had a clear and singular objective: close the two Round 8 AC-5 evidence residuals, namely exact ITL recomputation and fail-closed verification. That objective serves the original plan because AC-5 requires durable, attributable client-SLO evidence before the loop can safely proceed to product/default and hardening work.

Claude has been clearing a blocking side issue, not a queued cleanup. The AC-5 evidence repair took several rounds, but it was not circular: R7 added durable summaries, R8 fixed row reconciliation and TTFT/TPOT arrays, and R9 fixed the remaining ITL source plus fail-open verifier. This should now stop being the round objective; the next round must move to hardware AC-6 + AC-9 as already recorded.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 1
Queued Side Issues: 0
```

Blocking side issue:
- Strict client SLO still fails after admission restore: conc 32/64 TTFT remain above 22 s, and per-request TPS remains below 30 at all concurrencies. This blocks the Ultimate Goal and any strict/shippable claim.

Queued side issues:
- None newly added.

## Implementation Review

No high-signal Round 9 implementation defect found.

What I verified:

- `python3 runs/20260530_dsv32_loop6/client_slo_int8/ac5_metrics_tool.py --verify` exits 0 and recomputes TTFT, TPOT/TPS, ITL, and sanity checks from committed files alone.
- `runs/20260530_dsv32_loop6/client_slo_int8/ac5_itl_flat_ms.json` is tracked, not ignored, sorted per conc, and has counts 163323 / 163334 / 163304 matching `itl_flat_count`.
- The local source JSONLs still present under `development/results/` match the SHA256 and byte counts recorded in `ac5_metrics_arrays.json`.
- Reflattening the local source JSONL ITLs with the tool's `round(v * 1000.0, 4)` rule exactly matches the committed ITL arrays.
- Temporary tamper tests fail closed:
  - Mutated stored `median_ttft_ms`: verifier exit 1.
  - Shifted committed ITL values by +50 ms: verifier exit 1.
  - Dropped one `ttfts_s` entry: verifier exit 1.
- `client_slo_report.md` keeps the result directional and explicitly records the strict SLO miss; it does not claim a shippable pass.
- `git diff --check` is clean after the tracker update.

The only residual caveat is non-blocking: the ITL evidence source stores rounded 4-decimal millisecond values, so recomputed ITL percentiles differ from raw stored summaries by at most ~0.00004 ms and pass under the verifier's 0.01 ms tolerance. That is well below the report precision and does not undermine the published percentile evidence.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:

- Plan version moved to Round 9 Review.
- Added an R9-review plan-evolution row with the verifier evidence.
- Updated task6 from awaiting Codex re-verification to verified evidence, while keeping it Active/partial because the strict SLO still fails.
- Marked the AC-5 durable/recomputable evidence blocker as RESOLVED and VERIFIED.

No immutable section was modified.

## Action Items

Mainline Gaps:

1. Keep task6/AC-5 visible as partial because strict SLO is still not met. Before any strict/shippable claim, solve or characterize the scheduling/decode/admission operating point that leaves conc 32/64 TTFT and all-conc TPS failing.
2. Next round should move to hardware AC-6 + AC-9: prove DSA-default boot has `enable_double_sparsity=false`, no DS `TokenLabelTable`, and SLO unchanged; prove DS opt-in toggles compact int8; edit the NIAH harness to use real `usage.prompt_tokens`, fail closed, and rerun/copy artifacts.
3. Then complete AC-7 and AC-8. Start AC-10 only after AC-3 through AC-9 are verified.

Blocking Side Issues:

1. Strict SLO failure remains the sole active blocking side issue for the Ultimate Goal.

Queued Side Issues:

None.

## Stagnation Check

No STOP condition. There is a pattern of evidence-quality rework across R6-R9, but the same issue is not persisting unchanged: each review exposed a narrower, verified residual and Round 9 closed the exact Round 8 residuals. Development would become stagnant if Round 10 spends another round on AC-5 evidence instead of moving to AC-6/AC-9 hardware or a concrete strict-SLO plan.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline 57f86b66f`
- `git show --name-only --format=short 57f86b66f`
- `python3 runs/20260530_dsv32_loop6/client_slo_int8/ac5_metrics_tool.py --verify`
- `git ls-files runs/20260530_dsv32_loop6/client_slo_int8/ac5_itl_flat_ms.json ...`
- `git check-ignore -v runs/20260530_dsv32_loop6/client_slo_int8/ac5_itl_flat_ms.json ...`
- Independent JSON checks of array lengths, ITL sortedness, source SHA256s, and flattened ITL equality against local source JSONLs.
- Three temporary-copy verifier mutation tests.
- Inspected `client_slo_report.md`, `ac5_evidence_addendum.txt`, and `goal-tracker.md`.
- `git diff --check`

NOT COMPLETE
