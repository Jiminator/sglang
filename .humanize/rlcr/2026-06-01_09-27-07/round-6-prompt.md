Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop7/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop7/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-6-contract.md

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
# Round 5 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the Round-5 contract: the DSA same-node NIAH reference exists, the DS-default / DS-hybrid / DSA matrix is reproducible from the committed inputs, and the R4 oracle analyzer artifact was corrected. The 16K DS-hybrid uplift claim is supported at N=20 by the stated directional materiality rule: 40% exceeds the DS-default baseline CI high of 24.9%.

Do not close Loop 7. Claude's "remaining items" are plan-derived acceptance work, not optional queued cleanup. The original plan still has active AC-1, AC-3, AC-4, AC-6, and final AC-2 closure work.

## Implementation Review

Accepted R5 work:

1. **The served-recall matrix is real and reproducible.**

   Evidence: `development/loop7/niah_recall_matrix.py` builds exact Clopper-Pearson CIs (`:24-30`) and marks 16K material only when the hybrid point exceeds the baseline CI high (`:93-101`). Re-running it against `niah_dsa_reference.json`, `ds_niah_baseline_mem07.json`, and `niah_ds_hybrid.json` reproduced `ds_vs_dsa_recall_matrix.json`. The matrix contains DSA 100% at 1024w/4K/16K/64K, DS-default 100/75/5/5, and DS-hybrid 100/85/40/0 (`development/loop7/ds_vs_dsa_recall_matrix.json:10-171`).

2. **The missing DSA-same-node reference is present.**

   Evidence: `development/loop7/niah_dsa_reference.json:10-56` records N=20 per length, 0 admission failures, and 20/20 hits at 1024w, 4K, 16K, and 64K. `development/serve_native_nsa.sh:7-10` confirms the launched baseline is intentionally non-DS native sparse attention.

3. **The R4 analyzer issue is fixed.**

   Evidence: `development/loop7/analyze_oracle.py:107-126` now emits separate `uplift_4096_minus_2048` and `uplift_8192_minus_2048` fields and a three-way `budget-limited` / `budget-partial` / `scorer-limited` verdict. Re-running the script with the same absolute sink path reproduced `development/loop7/oracle_budget_vs_scorer_r4.json`.

## Mainline Gaps

1. **AC-1/task4 and task6 remain incomplete: oracle-off graph allocation evidence and the dense/default stride reference are still missing.**

   Evidence: the tracker still has task4 active for CUDA allocation-detector evidence and task6 active for the dense/default stride reference (`goal-tracker.md:62-63`). R5's matrix records 1024w within-budget parity and calls it "dense-DS-equivalent" (`m2_recall_matrix_finding.md:41-45`), but there is no separate dense/default stride artifact or field in `niah_dsa_reference.json`, `niah_ds_hybrid.json`, or `ds_vs_dsa_recall_matrix.json`.

   Required implementation plan:
   - Add a durable artifact that explicitly records the dense/default stride reference required by the plan. Do not infer it from 1024w parity alone.
   - If this is the oracle sampling-stride reference, write a small harness run that records the default oracle stride and a stride=1 run side-by-side, with trial counts and any recall@K deltas.
   - If this is the dense-DS NIAH reference, run the DS dense/within-budget reference explicitly and label it as dense-DS, not just "dense-equivalent."
   - Complete task4 by running the CUDA graph replay allocation detector for oracle-off selection and recording byte-identical `selected_indices` / `valid_lengths` plus zero new allocations.

2. **AC-3/task8/task12 remain incomplete: the winning Tier-2.B scorer is still an eager research path without the required non-regression matrix.**

   Evidence: R5's hybrid run is explicitly eager (`m2_recall_matrix_finding.md:22-24`). The production validator still rejects non-default scorer configs under CUDA graph (`python/sglang/srt/layers/attention/double_sparsity/validator.py:97-114`), and `DeepseekV2AttentionMLA.forward` still routes non-default scorer configs to the eager selector (`python/sglang/srt/models/deepseek_v2.py:2242-2256`). MMLU re-anchor, N>=50 16K, graph-safe scorer support, and perf validation are still listed as active in the tracker (`goal-tracker.md:65-66`).

   Required implementation plan:
   - Port `scorer_norm={cosine,hybrid}`, `head_agg`, `scorer_norm_hybrid_threshold`, and supported anchor modes into the graph-safe `retrieve_topk_graph_safe` / Triton scorer path.
   - Add graph-state scratch for every new intermediate so replay stays allocation-free; do not route supported production variants through eager selection.
   - Add eager-vs-graph equality tests for raw, cosine, hybrid, mean/max head aggregation, and anchor modes over the TP=8 logical matrix.
   - Run the binding AC-3 matrix: DS-default, DS-hybrid graph-safe, and DSA at 1024w/4K/16K/64K; N>=50 at 16K; dense-DS/within-budget parity; MMLU re-anchored at the Loop-7 single-node mem0.7 op-point with <=1.0pp tolerance; and graph-vs-eager perf deltas.

3. **AC-4/task13-task17 are still unimplemented even though the oracle gate justifies bounded Tier-2.A work.**

   Evidence: repository search still finds no `enable_lifted_budget_decode` or `lifted_budget_top_k` implementation outside plan/review text, and no lifted-budget compact-remap decode path exists. The R4/R5 evidence says 4K is budget-limited and 16K budget-partial, so AC-4 cannot remain silently dangling.

   Required implementation plan:
   - Write the task13 design record first: explicit ABI `enable_lifted_budget_decode: bool` and `lifted_budget_top_k: int`, with `top_k > index_topk` rejected unless the opt-in lifted-budget backend path is selected.
   - Add the config fields and validation without reusing `max_top_k`, Twilight fields, or `SGLANG_DS_ALLOW_TOPK_MISMATCH`.
   - Implement the opt-in decode path using `flash_mla_sparse_fwd` plus `dequantize_k_cache_paged`: physical selected slot -> `page_table_1_flattened` -> request-local compact KV index.
   - Mask or safe-replace `-1` padding before any dequant/index operation; keep fixed `lifted_budget_top_k` shapes with padding; preserve the R23 deterministic tie-break.
   - Add reference sparse-attention tolerance tests, prefix-sharing remap tests, padding/duplicate/valid-length tests, TP=8 equality tests at 4096/8192, and graph replay allocation evidence.
   - Finish task17 with a landing/disposition record: production-ready landed path, or explicit hardening follow-on with recall evidence recorded and the DSA default untouched.

4. **AC-6/task19 and task20 final consolidation are still missing.**

   Evidence: no R5 artifact records conc-1/16 TTFT, decode TPS/req, GPU memory, graph replay success, admission, or Tier-1 spine non-regression. The tracker still has task19 and task20 pending (`goal-tracker.md:72-73`).

   Required implementation plan:
   - After task12 and task17 are complete, run the existing `development/benchmark.sh` / comparison tooling at the Loop-7 op-point for DS-default, DS-hybrid graph-safe, and DSA.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay status, admission, and any radix/cache assumptions.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final decision record that supersedes the Loop-6 strategic gate's Tier-2.A-primary ordering with the corrected M0/R5 evidence and the final Tier-2.A disposition.

5. **R5 evidence labels need cleanup before final decision-record consumption.**

   Evidence: `development/loop7/niah_dsa_reference.json:2` inherits the reused driver's `DS int8` op-point label even though the DSA reference is native NSA/no double sparsity; `development/loop7/niah_recall_matrix.py:9-12`, `:61-64`, and `m2_recall_matrix_finding.md:26-30` describe materiality as "outside baseline CI," while the implemented and plan-consistent rule is directional upward (`h > base_hi`, `niah_recall_matrix.py:87-101`). This does not invalidate the 16K uplift, but it is easy to misread in the final AC-2 record.

   Required implementation plan:
   - Give each input JSON an accurate config label: DSA/native-NSA/no-DS vs DS-default vs DS-hybrid eager/graph-safe.
   - Change the materiality prose and JSON `materiality_rule` to say "material uplift only when the variant point exceeds the DS-default baseline CI high." Keep below-CI/floor-noise reporting separate.

## Blocking Side Issues

None open for the R5 recall matrix itself. The matrix math, DSA reference, and analyzer regeneration are reproducible. The blockers are plan-derived unfinished acceptance tasks listed above.

## Queued Side Issues

1. Clean plan-specific workflow markers and stale comments before final merge.
2. Decide whether to archive raw oracle sink/trial-index artifacts or embed durable trial/failure-count metadata in the final decision record.
3. Keep learned/distilled selector work out of scope unless explicitly approved.
4. Clean the R5 evidence labels described above before task20; do not let this take over the next mainline implementation round.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 | Unjustified deferrals: 5

AC status:

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | PARTIAL | Oracle diagnostic and fail-closed records are implemented, but task4 CUDA graph allocation evidence and dense/default stride closure remain active. |
| AC-2 | PARTIAL/ADVANCED | R5 completed the DS-vs-DSA served-recall matrix and DSA same-node artifact. Final AC-2 closure still depends on non-regression evidence and task20 decision record. |
| AC-3 | PARTIAL | Eager hybrid materially improves 16K, but graph-safe scorer support, N>=50 16K, MMLU re-anchor, dense-DS proof, and perf remain missing. |
| AC-4 | NOT MET | No lifted-budget ABI, compact remap, implementation, tests, or disposition record. |
| AC-5 | MET | 64K servability at mem0.7 remains verified with served/admission separated. |
| AC-6 | PARTIAL | No final conc-1/16 perf guardrails or Tier-1 non-regression report. |

All original tasks are represented in Active, Completed, or Deferred; the Deferred table remains empty. The issue is not tracker forgetting, it is that Claude framed active acceptance work as queued follow-up.

## Goal Tracker Update Requests

I updated only the mutable tracker section:

- bumped Plan Version to 7 for Round 5 Review;
- recorded this review's acceptance of the R5 matrix and analyzer regeneration;
- cleared the stale task7 note that the R4 analyzer cleanup was still pending;
- added the R5 DSA op-point/materiality-wording cleanup to Queued Side Issues.

Do not mark any additional task complete yet. Keep task4, task6, task7, task8, task12, task13-task17, task19, and task20 active/pending until the evidence above exists.

Not complete: original-plan acceptance work remains active.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-6-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-6-summary.md

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
