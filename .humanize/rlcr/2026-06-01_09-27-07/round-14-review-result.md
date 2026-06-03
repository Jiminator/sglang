# Round 14 Review Result

Mainline Progress Verdict: ADVANCED

Round 14 advanced the stated task15 contract. The lifted-budget path now has
binding served 4K recall evidence, a direct backend-level correctness test for the
wired `_forward_lifted_budget` method, and TP=8 lifted-width determinism at 4096
and 8192. This still does not complete Loop 7: AC-4 needs the task16/task17
production-hardening or explicit landing-disposition decision, AC-6 needs task19
perf consolidation, and AC-2 needs the task20 final strategic decision record.

## Implementation Review

Accepted R14 work:

1. The served recall claim is supported by committed artifacts and reproducible
   arithmetic.

   Evidence: `development/loop7/niah_ds_lifted4096.json` reports 19/20 served
   hits, 0 admission failures, prompt tokens 4368-4408; `niah_ds_default2048_eager.json`
   reports 15/20 served hits, 0 admission failures, same prompt-token range.
   Regenerating `development/loop7/lifted_recall_matrix.py` from those inputs
   reproduced `development/loop7/ds_lifted_vs_default_recall_4k.json` exactly:
   lifted recall 0.95 exceeds the default eager baseline CP-CI high 0.9134, so
   the +20pp uplift is material under the plan's directional rule. The finding
   correctly labels both numbers as eager-mode, same-node measurements.

2. The backend-level test now drives the real wired lifted method rather than only
   the helper.

   Evidence: `test/registered/unit/layers/attention/test_lifted_budget_decode.py:336`
   constructs a minimal `DeepseekSparseAttnBackend` and calls
   `be._forward_lifted_budget(...)` at widths 4096 and 8192. The test covers a
   request selecting 3000 > 2048 rows, prefix-shared physical slots, an in-row
   duplicate that becomes an interior invalid compact lane, and `valid_lengths`
   far below the padded width. It compares against an independent physical-slot
   reference after full dequant.

3. The lifted-width TP=8 determinism gap from Round 13 is closed.

   Evidence: `test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py:113`
   adds an 8-rank gloo worker that runs the production logical
   `retrieve_topk_via_labels` path with `max_top_k` 4096 and 8192, then asserts
   identical `selected_indices` and `valid_lengths` across ranks. The full-length
   request asserts `valid_lengths == max_top_k`.

4. The launcher knob is consistent with the eager-only lifted path.

   Evidence: `development/serve_double_sparsity.sh` emits
   `enable_lifted_budget_decode` and `lifted_budget_top_k` when `LIFTED_BUDGET=1`,
   emits the default-off pair when disabled, and adds `--disable-cuda-graph` for
   either `RECALL_ORACLE=1` or `LIFTED_BUDGET=1`. This matches the validator's
   current eager-only requirement.

5. Local validation reproduced the claimed test count.

   Commands run:
   - `python development/loop7/lifted_recall_matrix.py --ds-default development/loop7/niah_ds_default2048_eager.json --ds-lifted development/loop7/niah_ds_lifted4096.json --out /tmp/ds_lifted_vs_default_recall_4k.review.json` -> output matched committed JSON by `diff -u`.
   - `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py::TestTP8LiftedWidthDeterminism` -> `16 passed`.
   - `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py` -> `341 passed, 24 warnings, 9 subtests passed`.
   - `git diff --check 2ba4dafc1..0ad20774a` -> clean.

No high-signal implementation bug was found in the Round 14 task15 work.

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-1 | MET | R8 remains accepted: fail-closed oracle sink, live all-reduced score tensor records, oracle-off byte-equivalence and zero allocation under graph replay, separated served/admission baseline, stride=1 reference, and AC-1.1 force-in oracle. |
| AC-2 | PARTIAL | DS-vs-DSA recall and Tier-2.B material uplift evidence exists; task20 final gate-supersession decision record is still missing, and task19 consolidation feeds that closeout. |
| AC-3 | MET | R9/R10 graph-safe non-learned variants remain accepted: scorer/head/anchor flag-gated, default-off unchanged, TP=8 equality, dense/within-budget parity, and MMLU within tolerance. |
| AC-4 | PARTIAL / NOT MET | task13, task14, and task15 are now verified. Remaining blocker: task16 production-hardening decision and task17 Tier-2.A landing/disposition record. The current lifted path is explicitly eager-only because dequant allocates. |
| AC-5 | MET | 64K `/generate` servability at mem0.7 remains verified from the separated baseline: served 20/20 with 0 admission failures; 128k remains out of scope. |
| AC-6 | PARTIAL | Final perf guardrails at conc-1/16, graph replay status, decode TPS, memory, and Tier-1 spine non-regression report are still missing under task19. |

Acceptance Criteria: 3/6 met (0 deferred)
Active Tasks: 4 remaining
Estimated remaining rounds: 2-3
Critical blockers: task16/task17 AC-4 disposition, task19 perf consolidation, task20 final decision record.

## Forgotten Items Detection

No original task is forgotten. Every task from `development/loop7/refined_plan_v1.md`
is represented in `goal-tracker.md` as Completed/Verified or Active. R14 correctly
completed task15, but the tracker still had `task15` listed under Active as
`done`; I corrected that drift by moving task15 to Completed and Verified.

The old tracker plan-evolution note saying `development/loop7/refined_plan_v1.md`
was absent is now stale because the file exists and was read for this review. It
does not create missing task coverage; the task mapping still matches the original
plan.

## Deferred Items Audit

The Explicitly Deferred table is empty. Learned/distilled selector work remains a
queued out-of-scope follow-on under DEC-5, not a deferred original Loop-7 task.
There is no accepted deferral yet for task16; if Claude chooses not to harden the
eager lifted path in this loop, task17 must record the deferred-with-evidence
disposition explicitly before AC-4 can close.

## Mainline Drift Audit

The current round's objective was clear and singular: close task15 by proving
served lifted-budget 4K recall recovery and the lifted-width correctness/determinism
evidence requested in Round 13. Claude advanced mainline AC-4 rather than clearing
side issues. The remaining side issues are not blocking current progress unless
task20/final merge starts without handling them.

Blocking Side Issues: 0
Queued Side Issues: 3

Queued Side Issues:
1. Preserve or cite R8 oracle-sink provenance before task20, or cite the hardcoded
   `stride=1` call site plus the committed aggregate explicitly.
2. Remove plan/workflow markers and stale comments before final cleanup/merge.
3. Learned/distilled selector work remains out of scope unless explicitly approved
   under DEC-5.

## Mainline Gaps

1. **AC-4 task16/task17 remain active.**

   The lifted-budget path is still eager-only. That is acceptable as R14 research
   evidence, but AC-4 cannot close until either production hardening lands
   (alloc-free `out=`/scratch dequant, q-padding scratch, CUDA-graph replay
   allocation proof, perf impact) or task17 records an explicit deferred-with-evidence
   landing disposition with the DSA default untouched.

2. **AC-6 task19 remains active.**

   The final report still needs conc-1/conc-16 TTFT, decode TPS/req, GPU memory,
   graph replay status, admission behavior, radix/cache assumptions, and exact
   launch configs at the selected Loop-7 op-point.

3. **AC-2 task20 remains active.**

   The final decision record must supersede the Loop-6 Tier-2.A-primary gate with
   the complete M0/R7/R8/R14 evidence and the AC-4 disposition. `m0_decision.md`
   remains source evidence, not the plan's final closeout record.

## Goal Tracker Update Requests

I updated `goal-tracker.md` directly:

- bumped Plan Version to 19 for Round 14 Review;
- added a Round 14 Review plan-evolution row;
- moved task15 from Active to Completed and Verified;
- kept task16, task17, task19, and task20 active;
- left Blocking Side Issues empty and made no immutable-section changes.

No requested tracker change was rejected.

## Stagnation Check

No stagnation. Recent rounds have advanced sequential AC-4 work: R11 fail-closed
ABI gate, R12 compact remap/kernel proof, R13 served eager branch wiring, and R14
served recall plus TP/correctness evidence. The same remaining items are repeated
because they are the planned downstream gates, not because Claude is cycling on
unfixed defects.

PENDING
