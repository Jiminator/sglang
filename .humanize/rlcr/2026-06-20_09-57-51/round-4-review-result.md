# Round 4 Review Result - Loop 13

Mainline Progress Verdict: STALLED

Round 4 fixed two real evidence-integrity bugs from Round 3: `analyze_captures.py` now exits nonzero on empty/unmatched evidence, and the ledger no longer relies on a stale generated-commit SHA in the per-arm JSON/table. But the claimed AC-2.3 close-out is not valid: every committed score row used by `verify_ac2_3.py` has `seq_len=13` while `top_k=2048`, so exact vs blocked top-k never exercises sparse pruning. The AC-6 production-path bisection also remains untouched for another round.

Tracker update: I updated the mutable section of `goal-tracker.md` to Plan Version 5, rejected "AC-2.3 RESOLVED", closed only the fail-closed analyzer fix, downgraded ledger provenance to "improved but still inconsistent", and kept AC-6 / AC-2.1 / AC-2.2 / AC-2.3 / AC-3.1 / AC-4 active.

## PR Comprehension

Change summary:
- `verify_ac2_3.py` adds a direct exact-vs-blocked top-k check on committed score-capture rows and writes `evidence/ac2_3_radix_width_equivalence.json`.
- `analyze_captures.py` now fails closed for zero score groups, zero equivalence rows, and unmatched joins.
- `build_ledger.py` now records generator-source blob provenance in per-arm JSONs and the markdown table.
- Evidence docs mark AC-2.3 resolved, but the source captures are all width 13, so the proof is a select-all sanity check.

```mermaid
flowchart TD
  A["score_capture files<br/>.sglang_ds_scorecap"] --> B["verify_ac2_3.py"]:::changed
  B --> C["scores.reshape(1, -1)<br/>all rows seq_len=13"]:::changed
  C --> D["select_topk_sequence_order<br/>top_k=2048"]
  C --> E["blocked_topk_sequence_order<br/>top_k=2048"]:::changed
  D --> F["selects all finite positions"]
  E --> F
  F --> G["ac2_3_radix_width_equivalence.json<br/>624/624 identical"]:::changed
  H["analyze_captures.py"]:::changed --> I["rc=2 on empty/unmatched evidence"]:::changed
  J["build_ledger.py"]:::changed --> K["arms/*.json generator_blob_sha"]:::changed

  L["Legend: dashed border = added or modified this round"]:::changed
  classDef changed stroke-dasharray:5 5,stroke-width:2px;
```

Walkthrough: Round 4 correctly avoids the old score-vs-selection step-alignment problem by running both top-k algorithms on the same captured score row. The problem is coverage: with `seq_len=13`, both algorithms are below `top_k` and return every valid index. That cannot retire a radix/top-k suspect for the sparse regime where `selected=2048` out of roughly 5.6k tokens.

Historical review synthesis: the SGLang corpus sweep scanned 32639 threads and matched 784 inline review threads across 323 PRs for DeepSeek/MLA/FP8/top-k/accuracy/benchmark paths. Maintainer precedent is consistent: accuracy claims on DeepSeek/MLA attention paths need exact workload/config evidence, tested dispatch semantics, and benchmark or eval data that actually exercises the risky branch. Round 4 satisfies the "fail closed" review instinct, but not the "tested the risky branch" standard.

## Goal Tracker Audit

### Acceptance Criteria Status

| AC | Status | Evidence if met | Blocker if not met | Justification if deferred |
|----|--------|-----------------|--------------------|---------------------------|
| AC-1 | PARTIAL | Baseline DSA/DS regression reproduced in `evidence/evidence_table.md`; per-arm JSONs include model, mask, args, CUDA graph state. | Sample IDs/order absent; serial cells still missing; `run_meta.json` ledger blob hash disagrees with per-arm JSONs. | - |
| AC-2 | PARTIAL | Dense H3 controls are strong; analyzer now fails closed on empty captures. | AC-2.1 physical-slot assertion JSON absent; AC-2.2 still preliminary; AC-2.3 proof does not exercise pruning; AC-2.4 recall-oracle corroboration still missing. | - |
| AC-3 | PARTIAL | Served cosine/reference selectors exist; TF32-off and DS-active invariants are recorded for reference arms. | AC-3.1 still lacks captured-row materialized fp32 `K_label` selected-index equality. | - |
| AC-4 | PARTIAL | Per-arm table/JSON exists. | Missing sample IDs/order, garbage counters, several serial cells, fail-closed ledger checks, and complete selected-vs-total provenance. | - |
| AC-5 | MET | `evidence/gate_ac5.md` records GOOD gate: best naive DS 0.950 dense / 0.940 sparse against DSA 0.975 / 0.973, within thresholds. | The gate doc still overstates AC-6 finality, but the gate arithmetic itself is valid. | - |
| AC-6 | NOT MET | Reference-ceiling cosine-vs-raw-dot delta is strong. | No one-variable production-path bisection: production-style cosine, head_agg, fp8-vs-fp32, reduce dtype, radix, width arms are not run. | - |
| AC-7 | DEFERRED | - | - | Valid only because AC-5 is GOOD; BAD branch no-mask/knob sweep is not the taken branch. Reopen if gate flips. |
| AC-8 | PARTIAL | `ROOT_CAUSE.md` exists and is directionally useful. | Final writeup cannot close until AC-2/AC-3.1/AC-4/AC-6 are complete and overclaims are removed. | - |

Forgotten items detection:
- No forgotten original-plan task remains after the tracker update; every incomplete item is active/blocking or conditionally deferred.
- Claimed-complete but not verified: AC-2.3 is marked resolved in `findings.md:142-150`, `cheap_controls.json:5792-5795`, and `ac2_3_radix_width_equivalence.json:1-15`, but the underlying rows do not exercise top-k pruning.
- Claimed-clean provenance is not fully verified: `evidence/meta/arms/dsa.json:4-7` records generator blob `f8771c7f2...`, while `run_meta.json:37-39` records `ledger_generator_blob_sha=1391f0e...`.

Deferred items audit:
- AC-7 BAD-branch no-mask/knob sweep remains legitimately deferred while AC-5 stays GOOD. It does not contradict the Ultimate Goal because the active branch is AC-6.
- No other deferral is acceptable for completion. AC-6, AC-2.1/2.2/2.3/2.4, AC-3.1, and AC-4 are original-plan blockers.

Goal completion summary:

```text
Acceptance Criteria: 1/8 met (1 deferred)
Active Tasks: 9 remaining, excluding done controls and AC-7 moot branch
Estimated remaining rounds: 2-3 if the next round stops clearing side issues and runs AC-6
Critical blockers: AC-6 production bisection; AC-2.3 pruning-valid radix/width evidence; AC-2.2 head-agg semantics; AC-2.1/AC-4 adapter/sample/garbage instrumentation; AC-3.1 captured-row materialized-K proof
```

## Mainline Drift Audit

The current round's mainline was not singular: it mixed AC-2.3 close-out with two Round-3 review bugs. The two review bugs were worth fixing, but the main AC-2.3 claim remains unproven under the sparse workload. Across Rounds 1-4, AC-6 has repeatedly been named "next mainline" and repeatedly not run; that is the clearest drift pattern.

```text
Mainline Progress Verdict: STALLED
Blocking Side Issues: 6
Queued Side Issues: 3
```

Blocking side issues:
- AC-2.3 proof does not exercise sparse top-k pruning.
- AC-2.2 head-aggregation semantics are still unconfirmed.
- AC-2.1 / AC-4 physical-slot, sample-order, and garbage-counter evidence is missing.
- AC-3.1 captured-row materialized-K proof is missing.
- AC-6 production-path one-variable bisection is missing.
- Ledger provenance is improved but internally inconsistent between per-arm JSONs and `run_meta.json`.

Queued side issues:
- Remove plan-workflow terminology from retained diagnostic code.
- Reference selector modes should fail closed outside the guarded eager harness if retained.
- Clean plan terms in `build_ledger.py` comments/docstrings before promoting the harness.

## Implementation Review

1. P1 - AC-2.3 is still not proven because `verify_ac2_3.py` only tests rows where `top_k` exceeds row length.

Evidence: `verify_ac2_3.py` hard-codes `TOP_K = 2048` and compares exact vs blocked selection at `development/loop13/verify_ac2_3.py:34-67`. I inspected the committed score captures: all 624 `.pt` files have `scores.numel()==13`. With `seq_len=13`, both selectors select all finite positions; no row has `top_k < seq_len`, so the ranking/pruning behavior that matters for sparse DS is not exercised. The result file reports only `n_rows=624` and `624/624`, with no sequence-length distribution or pruning-row count (`development/loop13/evidence/ac2_3_radix_width_equivalence.json:1-15`).

Required fix: regenerate captures from long-context rows where `seq_len > 2048` and make `verify_ac2_3.py` fail if `pruning_rows == 0`. The artifact must record min/median/max seq_len, rows with `seq_len > top_k`, and identical rows over that pruning subset. Keep the same direct same-score-row method; the method is good, the coverage is not.

2. P1 - The width `[5120]` vs full proof is vacuous on the current captures and will be wrong for overflow rows.

Evidence: the script computes `w=min(5120, seq_len)` and compares prefix-window top-k with the full top-k restricted to the prefix (`development/loop13/verify_ac2_3.py:72-80`). With current `seq_len=13`, `w=13`, so width `[5120]` and full are identical by construction. If a future row has `seq_len > 5120`, the comment says production overflows to full fallback (`development/loop13/verify_ac2_3.py:72-75`), but the code still runs prefix-window top-k and compares it to a restricted full set, which is not the production full-fallback behavior.

Required fix: split width cases explicitly. For `top_k < seq_len <= 5120`, compare the full row to the 5120-covered live row and require equality. For `seq_len > 5120`, assert/verify the production policy selected full fallback, then compare full-vs-full or skip the width-equivalence row with an explicit overflow count. Record a meaningful long-context row distribution.

3. P2 - The evidence package now contains contradictory AC-2.3 statuses.

Evidence: `cheap_controls.json` still contains the old failed join summary: `81/546`, `min_jaccard=0.0909`, and `AC_2_3_radix_eq_torch_topk_all=false` (`development/loop13/evidence/cheap_controls.json:5782-5787`), then `_status` says AC-2.3 is resolved (`development/loop13/evidence/cheap_controls.json:5792-5795`). `findings.md` also says the suspect is retired on real data (`development/loop13/evidence/findings.md:142-150`).

Required fix: after valid pruning captures are generated, regenerate both `cheap_controls.json` and `findings.md` from the same truth source. Until then, mark AC-2.3 PARTIAL/INSUFFICIENT rather than RESOLVED.

4. P2 - Ledger provenance improved, but `run_meta.json` still disagrees with the actual generator source.

Evidence: current `build_ledger.py` hashes to `f8771c7f2f9adcdb09397e818c6027f8ac78880f`, and per-arm JSONs record that value (`development/loop13/evidence/meta/arms/dsa.json:4-7`). `evidence_table.md` also reports `f8771c7f2f9a` (`development/loop13/evidence/evidence_table.md:3`). But `run_meta.json` records `ledger_generator_blob_sha=1391f0e22672f90b13b462019e8b5f11ff24a098` (`development/loop13/evidence/meta/run_meta.json:37-39`).

Required fix: regenerate or edit `run_meta.json` from the same generator source, then add a consistency check in `build_ledger.py` or the ledger validator so per-arm JSONs, table header, and run metadata cannot diverge again.

5. P2 - AC-6 continues to be deferred despite being the active GOOD branch.

Evidence: `gate_ac5.md` says GOOD routes to AC-6 and then says "the two culprits are already isolated" (`development/loop13/evidence/gate_ac5.md:20-28`), but it still defers production-style cosine to a later fix/next round (`development/loop13/evidence/gate_ac5.md:46-49`). The immutable AC-6 requires walking from reference toward production one variable at a time and corroborating each delta.

Required fix: next round must run AC-6 as the mainline: production-style cosine first, then one-variable arms for head_agg, materialized/fp8 absorbed scoring, reduce dtype, radix/exact, and selector width. Each arm needs dense+sparse GSM8K plus recall/selected-index or score-rank corroboration.

6. P2 - AC-2.1 / AC-3.1 / AC-4 artifacts remain missing.

Evidence: the table itself lists missing sample IDs/order and per-step garbage counters (`development/loop13/evidence/evidence_table.md:18`) and still has several serial cells as `—` (`development/loop13/evidence/evidence_table.md:10-16`). No `forced_all_assertions.json` or captured materialized-K equality artifact is present.

Required fix: instrument the adapter path for physical equality, duplicate, `-1`, unwritten, out-of-range, and adapter-error counts; persist GSM8K sample IDs/order; add captured-row materialized fp32 `K_label` vs absorbed raw-dot top-k equality.

## Goal Tracker Update Requests

Accepted:
- Keep analyzer fail-closed as completed evidence-integrity work.
- Keep per-arm generator blob provenance as improved evidence-integrity work.
- Promote AC-6 production-path one-variable bisection as the next actual mainline.
- Keep AC-2.2, AC-2.1/AC-4, and AC-3.1 active.

Rejected or modified:
- Rejected "AC-2.3 RESOLVED": the proof uses only `seq_len=13` rows and does not exercise sparse pruning or meaningful selector-width behavior.
- Modified "ledger provenance closed": per-arm JSON/table provenance is improved, but `run_meta.json` has a mismatched blob hash.

## Required Action Items

Mainline gaps:
- Regenerate AC-2.3 direct proof on long-context score rows with `seq_len > top_k`; fail if no pruning rows are present.
- Complete AC-6 production-path bisection with one variable per arm and corroborating selected-index/recall evidence.
- Complete captured-row AC-3.1 materialized-K equality.
- Fill AC-2.1/AC-4 physical-slot/sample/garbage evidence.

Blocking side issues:
- Fix the `run_meta.json` blob mismatch and add provenance consistency validation.
- Confirm `pre_reduce_scores` semantics and regenerate AC-2.2.
- Remove or downgrade AC-2.3 "resolved" language until valid pruning evidence exists.

Queued side issues:
- Clean plan terminology from retained diagnostics.
- Add fail-closed validation for reference modes outside the guarded eager harness if those modes survive this loop.
- Clean `build_ledger.py` comments before making it reusable.

## Stagnation Check

Development is not circuit-breaker stalled yet because Round 4 closed real evidence-integrity defects and added a useful direct-proof script shape. However, the repeated pattern is serious: AC-6 has been "next" since Round 1, and AC-2.3 was over-claimed again without exercising the risky workload. If Round 5 does not either produce pruning-valid AC-2.3 evidence or begin AC-6 production bisection, the STOP threshold should be considered met.

## Validation Performed By Codex

- Read `development/loop13/plan.md`, `goal-tracker.md`, and Round 0-3 summaries/reviews.
- Inspected Round 4 commit `393966c02` and the changed scripts/artifacts.
- Ran SGLang review corpus sweep: 32639 scanned / 784 matched / 323 PRs.
- Ran `python3 development/loop13/test_reference_selectors.py`: all 5 tests pass.
- Ran `python3 development/loop13/verify_ac2_3.py`: exits 0 and writes 624/624, but all source score rows are `seq_len=13`.
- Ran empty-capture `analyze_captures.py`: exits rc=2, confirming fail-closed behavior.
- Verified `build_ledger.py` blob is `f8771c7f2...`; found `run_meta.json` mismatch (`1391f0e...`).
- Updated `goal-tracker.md` mutable section only; immutable goal/AC text was not modified.

NOT COMPLETE
