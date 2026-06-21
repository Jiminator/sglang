Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop11b/plan.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop11b/plan.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-2-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 1 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
`ACs: 10/10 addressed | Forgotten items: 2 | Unjustified deferrals: 2`

I updated the mutable section of `goal-tracker.md` for this review: task8/task9/task11 remain active, the GLM DS summary wiring is blocking rather than follow-up, AC-8 close-out remains blocking, and the stale plan-completion state was rejected. I did not modify the immutable goal or acceptance criteria.

## Mainline Gaps

1. **AC-5 is not satisfied: the committed fail-closed no-op evidence still REFUSES every published DS SLO trial.**

Claude's summary claims "AC-5 no-op proven by direct evidence", but the plan and R1 contract explicitly require every published SLO trial to carry `dense_fallback_total == 0` plus sparse-selection proof, and missing fields are an input refusal. The committed validator enforces that contract in `development/loop11b/runs/20260616_mb/trial_evidence.py:12`, `development/loop11b/runs/20260616_mb/trial_evidence.py:106`, and `development/loop11b/runs/20260616_mb/trial_evidence.py:136`. The committed DS evidence files fail that validator: for example, `development/loop11b/runs/20260616_mb/results_v2/ds080/double_sparsity_gsp_isl4096_osl512_c64_t1.jsonl.evidence.json:3` has `dense_fallback_total: null`, `selected_tokens_mean: null`, `total_tokens_mean: null`, and `verdict: "REFUSE"` at `development/loop11b/runs/20260616_mb/results_v2/ds080/double_sparsity_gsp_isl4096_osl512_c64_t1.jsonl.evidence.json:18`. The same refusal pattern appears across all six DS verdict trials.

The gap is also admitted in `development/loop11b/runs/20260616_mb/ac5_no_op_evidence.md:37`: GLM DS requests emit null DS aggregate fields, because `Glm4MoeAttention`/`dsa_backend` never reaches the DeepSeekV2 publisher. The proposed "backend-side fix" at `development/loop11b/runs/20260616_mb/ac5_no_op_evidence.md:49` was deferred as out of scope, but that deferral directly violates AC-5. Log-level structural arguments (`top_k < ctx`, no selector errors, dense fallback greps) are useful diagnostics, but they are not a substitute for the required per-trial fields when the fail-closed validator says REFUSE.

Required implementation plan:

1. Add the missing GLM/dsa-backend DS summary publisher, not another report workaround. Mirror the existing DeepSeek side-channel shape from `python/sglang/srt/models/deepseek_v2.py:2065`, which writes `forward_batch.ds_per_request_summary["double_sparsity"]`.
2. Implement a helper in `python/sglang/srt/layers/attention/dsa_backend.py` near the DS decode path. It must return unless `self.enable_double_sparsity` is true, derive selected-token counts from the finalized DS page table (`page_table_1 >= 0`, the same validity condition already used at `python/sglang/srt/layers/attention/dsa_backend.py:2102`), use `forward_batch.seq_lens` as the total-token denominator, set `dense_fallback` to `0`, and populate `forward_batch.ds_per_request_summary["double_sparsity"]`.
3. Call that helper in `forward_decode` after `page_table_1` is finalized and before all DS decode backend returns. Do not affect native DSA or non-DS decode paths.
4. Rely on the existing transport path already present in `python/sglang/srt/model_executor/model_runner.py:3227` and the existing bench aggregation in `python/sglang/bench_serving.py:1847`; after the backend side-channel is populated, `bench_serving` should emit non-null `dense_fallback_total`, `selected_tokens_mean`, and `total_tokens_mean`.
5. Add a focused smoke/unit check or a short GLM DS `bench_serving --output-details` run proving the JSONL contains the DS fields and `trial_evidence.py` exits 0.
6. Rerun the DS verdict trials, regenerate `.evidence.json` files, and do not publish the verdict until every DS SLO trial evidence file has `verdict: "PASS"` with `dense_fallback_total == 0` and non-null sparse-selection aggregates.

2. **AC-8 close-out is incomplete: the ledger is stale, raw verdict artifacts are not committed, and push is still pending.**

The plan requires the reviewer to reproduce the verdict from committed artifacts alone. R1 commits sidecars and hashes, but the raw JSONLs/logs needed by the comparator are still ignored. The ignore rules are explicit in `development/loop11b/runs/20260616_mb/results_v2/.gitignore:3`, `development/loop11b/runs/20260616_mb/results_v2/.gitignore:4`, and the top-level `.gitignore:62` / `.gitignore:179`. `git ls-files` does not include representative raw verdict inputs such as DS trial JSONLs, serve logs, or tax JSONLs. `development/loop11b/results.md:96` says the bulky raw blobs are gitignored and reproducible from runners plus hashes, which is not the same as preserving stable raw evidence for review.

The queue ledger is also contradictory. `development/loop11b/queue.md:24` says M-B/M-C/closeout are complete, but `development/loop11b/queue.md:32` still lists the first R1 task in progress, and `development/loop11b/queue.md:100` through `development/loop11b/queue.md:111` still contains stale RUNNING/PENDING R1 tasks. Finally, `development/loop11b/results.md:100` records that the commits are local only and that push is pending owner direction. AC-8 says close-out includes push every round; if origin is unsafe because it is public upstream, that needs explicit owner direction or waiver before claiming AC-8 complete.

Required implementation plan:

1. Complete the AC-5 backend evidence fix first, because AC-8 cannot close over refused evidence.
2. Regenerate `development/loop11b/queue.md` and `development/loop11b/results.md` into one current state. Remove stale RUNNING/PENDING rows and make task8/task9/task11 reflect the actual final state.
3. Preserve raw evidence under stable names. Either force-add the exact verdict JSONLs and server logs, or commit lossless compressed raw payloads such as `.jsonl.zst`/`.log.zst` with documented decompression commands and hashes proving they reproduce the comparator inputs. Include per-trial JSONLs, per-boot server logs, server_info snapshots, tax probe JSONLs/logs, comparator inputs/outputs, run order, and command ledger.
4. Re-run the comparator from the committed or losslessly reconstructed artifacts only, then update `EVIDENCE_SHA256.txt`.
5. Push to an owner-approved remote/branch. If public `origin` must not receive this branch, get the owner-designated destination or an explicit written waiver and record it in `results.md`.

## Blocking Side Issues

- **GLM/dsa-backend DS summary publication is missing.** This is the direct cause of the AC-5 refusal and blocks publication. It must be fixed in the backend data path rather than papered over in docs.
- **AC-8 evidence transport is unresolved.** Hash-only ignored raw evidence and a pending push do not meet the plan's committed-artifact reproducibility requirement.

## Queued Side Issues

- **Plan terminology remains in implementation comments/help text.** This does not block the next round as strongly as AC-5/AC-8, but it is still drift from the plan's "implementation code and comments must not contain plan-workflow terminology" rule. Examples include `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:184`, `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:329`, `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:745`, and multiple AC/DEC references in `development/benchmark_compare.py`.

## Goal Alignment Check

- AC-0/1/6/7/UX: materially addressed in prior work and not regressed by R1 based on the reviewed diffs and ledger.
- AC-2/3: advanced. The R1 comparator artifacts are no longer refusals, both op-points are same-commit, and the DS conc-64 failure is an honest SLO failure rather than an admission-capped measurement. This progress cannot be published until AC-5 and AC-8 are fixed.
- AC-4: satisfied by the dedicated fixed-bs tax probe; the reviewed logs show DS/DSA TPOT ratios within the required bound.
- AC-5: incomplete and blocking. The required fail-closed per-trial evidence refuses all DS verdict trials.
- AC-8: incomplete and blocking. The current repo state does not preserve raw verdict evidence in committed artifacts, the queue ledger is stale, and push remains unresolved.
- AC-9: advanced by the accepted comparator outputs and reuse evidence, but still dependent on AC-5 for the required no-op proof embedded in each published SLO trial.

Do not mark Round 1 complete until the AC-5 DS summary fields are wired and all DS SLO trial evidence files pass `trial_evidence.py`, then AC-8 is redone from committed/replayable evidence and pushed or explicitly waived by the owner.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-2-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-2-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
