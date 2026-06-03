# FULL GOAL ALIGNMENT CHECK - Round 4

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 4 Summary — Loop 7

## Mainline objective (round-4-contract.md)
Make the M0 recall oracle **fail-closed and binding** (AC-1/AC-2): config-borne
activation so it records on TP workers, strict span validation, explicit failure
artifacts, expected-record-count assertions in the sweep, and a re-run of
4K/16K/64K with no missing lengths — so task7's budget-vs-scorer attribution is
binding.

## Outcome: ACHIEVED — 64K oracle now MEASURED (not inferred), fail-closed, N=20.

## Work completed
1. **Config-borne activation (reaches TP workers).** Added `recall_oracle: bool`
   to `DoubleSparsityConfig` (`_ALLOWED_FIELDS` + dataclass + validation +
   `_coerce_bool` parse). Default off ⇒ byte-identical selection. The hook latches
   `oracle_artifact_sink.enable_via_config()` when the config flag is set, so the
   sink/trial paths resolve without env (env does not reach workers).
2. **Fail-closed hook** (`_maybe_record_recall_oracle`). No active trial,
   out-of-range needle span, or payload exception now emit explicit `failure`
   records keyed by `(request_id, trial_id, layer_id, decode_step)` instead of
   returning silently / swallowing. Out-of-range spans are **rejected, not
   filtered** (the old filter silently masked the absent 64K). Added a
   module-global sample counter so cross-process worker records don't all collide
   on `decode_step=0`.
3. **Rode the production long-context path.** Threaded `recall_oracle` into
   `retrieve_topk_graph_safe` (+ its line-1283 hook call + the deepseek_v2
   graph-safe call site). Reverted the initial `_force_eager_select |=
   recall_oracle` — the eager logical scorer does not scale to long-context int8
   tensors (error_containment silently dropped DS to dense, `ds`=None). Validator
   now requires `--disable-cuda-graph` when `recall_oracle` is on.
4. **Shared-FS cross-process files.** Default trial/sink under
   `./.sglang_ds_oracle/` (the repo bind-mount both driver and worker share);
   the old `/dev/shm` default is a per-sandbox tmpfs the worker can't see.
   Env-overridable via `SGLANG_DS_RECALL_ORACLE_DIR`. `os.makedirs` on write.
5. **Fail-closed sweep.** `niah_oracle_sweep.py` clears the sink, **forces decode
   steps** (`ignore_eos`, `decode_steps+1` tokens — DS selection is decode-only
   and NIAH prompts are immediate-EOS on raw `/generate`), then **asserts every
   issued trial produced records** and aborts on any missing length / hard
   failure. New `analyze_oracle.py` aggregates the sink into the budget-vs-scorer
   artifact with the uplift gate.
6. **serve script:** `RECALL_ORACLE=1` knob → `recall_oracle: true` in `DS_CONFIG`
   + auto-adds `--disable-cuda-graph`.

## Validation — binding GPU re-run (8×H200, DS int8 / mem 0.7 / TP=8, eager)
- **All three lengths recorded, fail-closed, N=20** (`[oracle-sweep] OK: all
  issued trials recorded; no hard failures`), ~4,880 (layer×decode-step) samples
  each. The previously silently-absent **64K is measured**.
- Budget-vs-scorer (`oracle_budget_vs_scorer_r4.json`, `m0_oracle_finding_r4.md`):
  - **4K budget-limited** — r@2048 44% → r@4096 86% → r@8192 100% (+56 pp).
  - **16K budget-partial** — 23% → 31% → 46% (+23 pp, caps ~46%): needs both a
    wider budget AND a better scorer.
  - **64K scorer-limited** — 15% → 20% → 24% (+9 pp): no feasible budget recovers.
- **342 DS unit tests pass** (fail-closed hook: no-trial/out-of-range/exception ⇒
  failure record; config-borne activation without env; validator recall_oracle
  guard; all prior scorer-variant/TP=8 tests).

## Decision impact
Confirms and sharpens the M0 A-vs-B decision: lead **Tier-2.B** (the only lever
for 64K and the binding lever for 16K); pursue **Tier-2.A** as a bounded win for
≤16K. Now binding at N=20 with 64K measured rather than inferred.

## Files changed
`config.py`, `oracle_artifact_sink.py`, `selection_kernel.py`, `selector.py`,
`validator.py`, `deepseek_v2.py`, `test_oracle_sink_and_force.py`,
`test_scorer_variants.py`, `niah_oracle_sweep.py`, `analyze_oracle.py` (new),
`serve_double_sparsity.sh`, `m0_oracle_finding{,_r4}.md`,
`oracle_budget_vs_scorer_r4.json`, `.gitignore`. Commit `bf2ce9b2b` (pushed).

## Remaining items (queued, justified)
- **AC-3 graph-safe Triton scorer port + full measurement matrix** (task #13):
  heavy kernel + GPU matrix (DSA same-node, N≥50 16K, MMLU at mem 0.7, dense-DS,
  within-budget parity, eager-vs-graph perf). Variants are correct + production-
  safe (R2/R3); the port + binding matrix is the next round's mainline.
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6** (task19–20):
  sequenced after AC-3 measurement.
- **Plan-marker code/comment cleanup**: pre-merge.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-ds-oracle-decode-only-and-shared-fs
- Notes: three stacked reasons the oracle recorded nothing despite config-borne +
  fail-closed (/dev/shm not shared across sandboxed processes; force-eager broke
  long-context DS via error_containment; prefill-only NIAH prompts do zero decode
  so decode-only DS selection never fires) — a genuinely reusable multi-round
  pitfall, captured with the verify-via-`double_sparsity`-meta method.

## Goal Tracker update request
- **Resolve Blocking Side Issue "oracle hook fail-open"** (flagged every review):
  fixed — fail-closed hook + config-borne activation + shared-FS + forced-decode
  sweep + expected-record assertions; 64K now measured.
- **task #12 (oracle fail-closed + 64K)** and **task #8 (oracle recording flaky)**
  → completed.
- **Keep Active**: task #13 (graph-safe Triton port + full AC-3 matrix) as the
  next round's mainline; AC-4 / M4 sequenced after.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
9ca1f5133 [Sparsity] Loop-7 recall R&D: plan + QA (active plan gitignored)
9914a3004 [Sparsity] Loop-7 M0: selection-recall oracle diagnostic math
8074cb1cf [Sparsity] Loop-7 M0: oracle sink + AC-1.1 force + flag-gated hook
c6ffcdea6 [Sparsity] Loop-7 M0: DS served-recall baseline at mem 0.7 (N=20)
78f6b5d17 [Sparsity] Loop-7 M0: oracle budget-vs-scorer evidence (A-vs-B decider)
a1e2c72dc [Sparsity] Loop-7 M0: A-vs-B decision (Codex-adjudicated)
599d7cc99 [Sparsity] Loop-7 M1: flag-gated cosine scorer (Tier-2.B candidate)
e2674f4f4 [Sparsity] Loop-7 M1: cosine scorer MEASURED — 16K recall 5%->40%
c5a829def [Sparsity] Loop-7: oracle trial-file read fresh; gitignore transient artifacts
273622705 [Sparsity] Loop-7 R1: length-conditional hybrid scorer (best of both regimes)
72c704edf [Sparsity] Loop-7 R2: scorer variants correct + production-safe
fc8871372 [Sparsity] Loop-7 R3: fix anchor over-budget + TP=8 logical-path matrix
bf2ce9b2b [Sparsity] Loop-7 R4: oracle fail-closed + config-borne + 64K binding re-run
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-3-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-3-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-2-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-2-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-1-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-1-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop7/refined_plan_v1.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Blocking Side Issues" or "Queued Side Issues" as appropriate
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If you reject a requested tracker change**: Include in your review why it was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New blocking issues: Add to "Blocking Side Issues"
- New queued issues: Add to "Queued Side Issues"
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop7/refined_plan_v1.md, we have completed **5 iterations** (Round 0 to Round 4).

The project's `.humanize/rlcr/2026-06-01_09-27-07/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-06-01_09-27-07/round-3-review-result.md` (previous round)
- `@.humanize/rlcr/2026-06-01_09-27-07/round-2-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-06-01_09-27-07/round-3-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-4-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-4-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
