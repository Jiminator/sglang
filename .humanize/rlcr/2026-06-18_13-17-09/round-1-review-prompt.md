# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop12/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Loop 12 — Round 1 Summary

## Mainline objective
Fix the Codex [P1] AC-8 gap: the perf wrapper ran the stock generated-shared-prefix default
(64 groups × 16 = 1024 requests across 64 prefix groups), NOT the loop-11b conc-64 workload
(1 prefix group, all prompts sharing the system prompt). Re-prove AC-8 on the correct workload.

## Outcome: AC-8 now validly PASS; branch corrected and re-pushed

### [P1] perf wrapper GSP grouping — FIXED + re-proven
- `benchmarks/bench_double_sparsity.py`: added `GSP_NUM_GROUPS = 1`; `build_bench_cmd` now passes
  `--gsp-num-groups 1 --gsp-prompts-per-group <num_prompts>` (over stock `bench_serving`). The
  verdict records `gsp_num_groups`, `gsp_prompts_per_group`, `expected_prompts`, `actual_completed`,
  `request_shape_ok`, and the wrapper **fails closed** if `actual_completed != num_prompts` so a
  future dataset-shape drift cannot pass silently.
- **AC-8 rerun on the live DS server (corrected workload)**: `--gsp-num-groups 1
  --gsp-prompts-per-group 256`, **actual_completed = 256** (`request_shape_ok = true`),
  **p50 decode TPS 35.05** (≥ 24.2), **P99 TTFT 22.90 s** (≤ 30.1), `parity: true`. M6 (DS active on
  long context) re-passed in the same boot. Evidence replaced in `development/loop12/perf_evidence/`.
  (The 1-group shape has maximal prefix reuse → cheaper prefill → more decode headroom than the prior
  wrong-shape run; native-DSA on the same base was 26.06 / 46.50 s.)

### [P3] plan-tracking markers in shipped comments — CLEANED
- Stripped `AC-`/`DEC-`/`Milestone`/`Loop-N`/`[R-N]` workflow markers from durable shipped code and
  test comments (`validator.py`, `selection_kernel.py`, `metrics.py`, `channel_mask.py`,
  `page_table_adapter.py`, `deepseek_v2.py`, `calibrate.py`, `test_double_sparsity_unit.py`,
  `test_ds_abort_path.py`, `bench_double_sparsity.py`), rewording them as technical statements.
- Deliberately preserved: `DeepSeek-R1`/`V3/R1` model names (not round markers); pre-existing base-code
  `Step N`/`Phase N` in `tokenizer_manager.py` / `server_args.py` / `model_runner.py` /
  `runner/decode_cuda_graph_runner.py` (not my changes — surgical principle); the
  `benchmarks/DOUBLE_SPARSITY.md` provenance doc's factual "loop-11b reference" baseline citation.

## Files changed (v2 clone, R1)
- `benchmarks/bench_double_sparsity.py` (grouping pin + shape guard + verdict fields)
- `benchmarks/DOUBLE_SPARSITY.md` (corrected perf numbers + command + completed count)
- 8 DS modules/tests (comment-marker strip; no logic change)
- v2 commits: `5dcd73ca6` (wrapper fix + marker strip) + `f05326636` (doc); branch re-pushed.

## Validation
- AC-8 rerun: 256/256 completed, 35.05 TPS / 22.90 s, parity true (1 group). Evidence:
  `development/loop12/perf_evidence/verdict.json`, `m6m8_eval_r1.out`.
- Final sweeps green: AC-1 42-file diff no dev scaffolding; AC-2 0 dropped-module refs + 0 plan
  markers in diff; AC-3 `import sglang` OK; 94 unit tests pass after the comment strip.
- Branch re-pushed to `Jiminator/sglang` (HEAD `f05326636`).

## Remaining Items
None. All 10 ACs pass (AC-8 now on the correct workload); both Codex findings resolved. PR can be
opened at `https://github.com/Jiminator/sglang/pull/new/double-sparsity-v2`.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260619-perf-parity-pin-request-shape
- Notes: Perf-parity evidence must pin the EXACT request shape of the reference workload, not just the
  per-request lengths — the stock generated-shared-prefix dataset is built from
  `--gsp-num-groups` (default 64) × `--gsp-prompts-per-group` (default 16) and ignores
  `--num-prompts`, so omitting those flags benchmarks a different request distribution. Pin
  `--gsp-num-groups 1 --gsp-prompts-per-group <n>` and assert `actual_completed == expected` in the
  perf gate so a silent dataset-shape drift fails closed.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
c6e3e943e [loop11b] R0: task10 DONE (UX pass); checkpoint — awaiting DSA matched re-run
d672d962f [loop11b] R0: matched-op-point verdict (task7/8/9 done) — DS meets SLO to conc32
425cdbcef [loop11b] R0: close-out — regenerate results.md (M-A+M-B+M-C complete)
65997cb4c [loop11b] R0: close-out evidence preflight — residual probe evidence + ignore .pt dumps
5df030348 [loop11b] R0: task11 close-out complete — all 11 tasks done; queue finalized
9af9d7835 [loop11b] R1: bench_serving emits prefix-reuse + DS no-op evidence (AC-5/AC-9)
8cde27faa [loop11b] R1: clean M-B re-run orchestrator (both op-points, tax probe, evidence)
73338e539 [loop11b] R1: fix mb_v2 tax_probe local-var bug; task10 serve-script de-plan
4ceba0ead [loop11b] R1: queue checkpoint — bench evidence + cleanup done, mb_v2 clean re-run running
86ddf6faf [loop11b] R1: fix stale a4be98c4 capacity claim (Codex gap 5) — note ld32 504640 reconfirm
1a29be00d [loop11b] R1: fix DS error-abort crash — check_finished was renamed upstream (#25725)
99ac584ac [loop11b] R1: document DS crash finding + selector reuse-edge; mb_v2 emits selector-error count
8fbe848ed [loop11b] R1: M-B verdict re-established clean — comparators ACCEPT both op-points
811c40420 [loop11b] R1: AC-5 no-op proof (dense_fallback=0 + structural sparsity) + GLM meta_info gap doc
9d2c4253d [loop11b] R1: headline M-B verdict + AC-4 dedicated per-step tax (both PASS)
f1b90c797 [loop11b] R1: AC-8 close-out — results.md + queue.md regenerated to the R1 publishable state
44310f230 [loop11b] R1: complete evidence package — DSA server_info + crash-probe txt + crash-log hashes
c16c0d202 [loop11b] R2: wire GLM/dsa-backend DS per-request summary (AC-5) — host-side, graph-robust
b5c4d72be [loop11b] R2: verdict re-established + AC-5 PASS + raw evidence committed (lossless)
8062039d8 [loop11b] R2: AC-8 ledgers regenerated to final state + push status; de-AC the new backend comment
df18a93d0 [loop11b] R3: fix total_tokens metric semantics (AC-5) — explicit field, not rate-inverse
96202e4c4 [loop11b] R3: corrected verdict evidence (results_r3) + supersede results_r2
2ce2adf4e [loop11b] R3: ledgers to one current state (AC-8) — results_r3, mask=regenerated, close-out ACTIVE-until-push
e0935e5a9 [loop11b] R3: AC-8 close-out COMPLETE — pushed to owner fork Jiminator/sglang
da12616a5 [loop11b] R3 review fix [P3]: build_corpus.py creates the output dir before writing
101926d76 [loop11b] R3 review fixes [P2 x2]: report verdict vs exit consistency + fail-closed partial DS evidence
9ab62e6ad [loop11b] R3 review fixes: DS abort test rename + comparator report verdict/labels + green test suite
bec9348c7 [loop12] add gen-plan output + draft; record pensieve doctor state
879532819 [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
6fee2b0ce [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-18_13-17-09/round-0-summary.md
- @.humanize/rlcr/2026-06-18_13-17-09/round-0-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: Required Finding Classification

You MUST classify your findings into these lanes:
- **Mainline Gaps**: plan-derived work or AC progress that is missing, incomplete, or regressing
- **Blocking Side Issues**: bugs or implementation issues that block the current mainline objective from succeeding safely
- **Queued Side Issues**: valid non-blocking follow-up issues that should be documented but must NOT take over the next round

Also include a one-line verdict:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
```

This verdict line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

If Claude mostly worked on queued side issues and failed to advance the mainline, say so explicitly.

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md yourself with the requested changes:
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

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
