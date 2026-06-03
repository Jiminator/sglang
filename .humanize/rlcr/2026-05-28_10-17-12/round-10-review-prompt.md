# Code Review - Round 10

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-10-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 10 Summary

## Work Completed
Ran the AC-11 directional comparator (task13) — a 3-trial radix-on DSA+DS sweep — and handled
#F honestly by surfacing effective-vs-nominal concurrency in the comparator/report. The
directional targets are MISSED (DS TTFT admission-bound), recorded as an AC-11 failure +
follow-up per DEC-7 (not a build-break).

- **AC-11 sweep on 8x H200.** DSA (radix-on default, mem 0.85) and DS (radix-on via the
  fixtures-passed artifact, no env override, mem 0.6), 3 trials × conc 16/32/64, 120s warmup /
  600s window, NUM_PROMPTS=64. 18 JSONLs, each `duration ≥ 602s`; sidecars carry the locked
  Option-B fields with matching `disable_radix_cache=false`, TP=8, fp8, page 64,
  `chunked_prefill_size=8192`.
- **Comparator verdict — directional MISS (DEC-7).** `benchmark_compare.py --ac11` exit 3.
  DS TPS: conc16 0.726 FAIL, conc32 0.900 FAIL, **conc64 1.146 pass** (DS faster). DS P99 TTFT
  FAIL at all conc (57.7/132.9/292.0s vs DSA 0.73/1.37/2.04s) — DS is admission/queue-bound.
- **#F handled (Codex option 3 — account, don't hide).** The comparator now:
  - treats `double_sparsity_radix_fixture_artifact` as a DS-only field;
  - treats `random_seed` (per-boot telemetry) and `mem_fraction_static` (DS reserves a
    TokenLabelTable → 0.6 vs DSA 0.85, an unavoidable asymmetry, NOT a locked Option-B field)
    as recorded-not-matched — while all locked Option-B fields stay strictly matched (a real
    mismatch still refuses, exit 2);
  - emits an **effective-vs-nominal concurrency** table (DS achieved 14.5/24.6/35.7 =
    91%/77%/56% of nominal; DSA ~100%) so the TTFT gap is shown to be partly admission-bound.
  Radix-on lifted DS effective concurrency far above the radix-off smoke's ~2.
- **Follow-up filed** (`ac11_analysis.md`): lift DS effective concurrency (TokenLabelTable
  footprint / KV budget) and re-sweep; profile DS admission at conc 64. Not blocking the
  recorded AC-11 artifact.

## Files Changed
- `development/benchmark_compare.py` — DS-only artifact key; `_AC11_IGNORED_SERVER_ARG_KEYS`
  (`random_seed`, `mem_fraction_static`); `RunMetrics.achieved_concurrency`; effective-vs-nominal
  table in the AC-11 report.
- `test/registered/unit/development/test_ac11_comparator.py` — removed `mem_fraction_static`
  from the refuse-cases; added a regression that the ignored fields don't refuse while real
  Option-B mismatches still do.
- `runs/20260528_dsv32_mvp/` — `mvp_compare_ac11.{md,json}`, `ac11_analysis.md`,
  `ac11_results/*.meta.json` (18 sidecars), `ac11_{dsa,ds}_server_info.json`.
- Commit `a24bc469c`. Pushed. (Raw `*.jsonl` gitignored; metrics preserved in sidecars +
  comparator JSON.)

## Validation
- `pytest test/registered/unit/development/test_ac11_comparator.py
  test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py
  test/registered/unit/development/test_option_b_scripts.py -q` → **359 passed** (28 subtests).
- AC-11 hardware: 18 JSONLs ≥602s, radix-on parity both sides; comparator ran to a directional
  verdict with the #F effective-concurrency table.

## Remaining Items
- **task14 AC-12:** NIAH 4K/16K/64K + MMLU 5-shot via `test_double_sparsity_v32.py`. The
  Round-9 long-context recall finding (top_k-bounded; `BL-...-longcontext-needle-recall-vs-topk`)
  is the key risk for NIAH 64K — distinguish a sparse-recall limit from a regression.
- **task15 evidence bundle:** assemble AC-Q/AC-10/AC-1b/AC-11/AC-12 artifacts, raw-JSONL
  locations, sidecars, server args, CUDA-graph status, mask provenance, comparator reports, and
  the AC-10 label-capture provenance note.
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-vs-dsa-memfraction-admission-asymmetry
- Notes: Added the lesson that DS and DSA cannot share `mem_fraction_static` (DS reserves a
  TokenLabelTable → 0.6 vs DSA 0.85), so an apples-to-apples serving comparator must treat
  mem-fraction (and per-boot random_seed) as recorded-not-matched while keeping the locked
  Option-B fields strict, and must surface achieved concurrency — because a DS TTFT miss at
  high nominal concurrency is admission-bound (mem-0.6 KV pool), not per-token latency (DS TPS
  was competitive-to-better). Records the comparator changes + the DEC-7 directional-miss
  handling.

## Goal Tracker Update Request

### Requested Changes:
- Confirm task13 / AC-11 as **EXECUTED with a recorded directional MISS** per DEC-7 (comparator
  ran, verdict + follow-up published, #F effective-concurrency surfaced). It is not a green
  pass — DS P99 TTFT misses the 1.10× target (admission-bound) and DS TPS misses at conc 16/32.
- Confirm #F as **accounted** (effective-vs-nominal in the AC-11 comparator/report), with the
  TokenLabelTable/KV-budget follow-up filed in `ac11_analysis.md`.

### Justification:
Per the immutable AC-11 (DEC-7), DS TPS-within-5% and P99-TTFT-≤1.10× are DIRECTIONAL targets;
a miss is recorded as an AC-11 failure + follow-up, not a build-break. The comparator ran to a
verdict and the report distinguishes effective from nominal concurrency, satisfying the "do not
hide queue-dominated admission" requirement Codex set for #F. The loop4-compatible MVP narrative
should state AC-11 as "comparator complete; directional TTFT target missed (admission-bound at
mem 0.6), follow-up filed." I did not alter the immutable AC or the directional thresholds.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
eb914678e [Sparsity] Loop-5: refined plan v1 + QA ledger
8979848ab [Sparsity] Loop-5: untrack active RLCR plan file
4f4c620df [Sparsity] Thread forward_batch into _write_token_labels (radix capture producer fix)
7cbbce088 [Sparsity] Calibration: native-FP8 sharded load + one-block dry-run mode
c99ed3644 [Sparsity] Calibration: load DeepSeek-V3.2 via deepseek_v3 remap + fail-closed dry-run
610f364c9 [Sparsity] Loop-5: V3.2 channel-mask calibration evidence (AC-4 complete)
df8d7c6c6 [Sparsity] Untrack .humanize/bitlesson.md (loop state, per .gitignore)
34b243b07 [Sparsity] Fix the DS serving path so DeepSeek-V3.2 boots on hardware
44a12d5d1 [Sparsity] Loop-5: round-2 DS boot evidence (AC-1 knobs + /generate probe)
610b65c15 [Sparsity] Loop-5: localize DS decode degeneration (DS-specific, selection over-count)
05a25f197 [Sparsity] Loop-5: refine decode diagnosis (eager scorer masks seq_len; instrument inputs in round 3)
2af5f4e65 [Sparsity] Fix DS decode selecting wrong domain: resolve req_to_token via ForwardContext
d9ad3066f [Sparsity] Loop-5: decode-degeneration is two bugs (req_to_token fixed; decode label-write open)
6429cf539 [Sparsity] Loop-5: complete bug #2 root cause (decode passes pre-projected k_nope, not latent)
8375b76a5 [Sparsity] Fix DS decode degeneration: label decode tokens (attn_mqa kv_b_proj + robust head_width)
b231942fa [Sparsity] Loop-5: DS genuine-sparse path OOB when seq_len>top_k (#18 finding)
da1ff651e [Sparsity] Loop-5: #18 deeper root cause — DS prefill selection bad req_pool_indices (long-prompt OOB)
802b51b84 [Sparsity] Loop-5: confirm #18 mechanism — DS selection uses decode batch shape, breaks on prefill per-token batch
ffe6c2b97 [Sparsity] Loop-5: critical review of loop4 DS scaffolding + pre-cutover loop5 fixes
eba4c640e [Sparsity] DS dense-prefill / sparse-decode: fix long-prompt OOB + unblock AC-1.1
590b0dc05 [Sparsity] Loop-5: extend code review to loops 1-3 foundational DS modules
3f9478128 [Sparsity] Loop-5: mark #18 resolved in review doc (dense-prefill fix)
8e9138af6 [Sparsity] Make radix fixture capture CUDA-graph-safe (no host copies during capture)
6f95a9711 [Sparsity] AC-0: radix-capture publish resolves req_to_token via backend/ForwardContext; dtype-safe SHA
bc534da7c [Sparsity] Fix /get_server_info crash (DS stashes tensors on server_args) + AC-0/AC-1 evidence
76eef9c80 [Sparsity] AC-1 negative test: invalid channel-mask path -> fail-closed validator rejection
6acdfb94f [Sparsity] Launcher parity: default MODEL_PATH to cluster weights; add DSA radix-off smoke knob
f2bc1eb6a [Sparsity] Make the TIER-1 smoke benchmark actually runnable on V3.2 FP8
2220a793f [Sparsity] TIER-1 smoke benchmark pair + comparator (AC-8/AC-9), radix-off both sides
99ac93691 [Sparsity] AC-Q quality smoke: single-node sequential capture/compare (#G)
d8fce372a [Sparsity] AC-Q evidence: single-node sequential quality smoke (3/4 gates; ROUGE-L miss analyzed)
bac3aaff6 [Sparsity] Quality smoke: generate via /v1/chat/completions (raw /generate is degenerate)
70bb52a15 [Sparsity] Diagnose AC-Q decode failure (#H): greedy degeneration, not a DS bug; harden ref validation (#I)
7861ca1d4 [Sparsity] AC-Q #H: reviewable DS-selection metadata proves no selection bug (greedy fragility)
85974608e [Sparsity] AC-Q: concise-answer measurement (user-approved) so the smoke tests answers, not greedy CoT
b0e43294c [Sparsity] AC-Q PASSES (all 4 gates) under user-approved concise measurement + first-8 prefix-overlap fix
d47dcbadb [Sparsity] Fix #J: first-8 overlap false-pass — alnum-subtoken normalization (not string prefix)
fa4473694 [Sparsity] AC-10 (DEC-5): no-env-override radix flip via a config-bound fixture state file
67422e698 [Sparsity] AC-10 MET on 8x H200: both radix fixtures pass; DS boots radix-on via artifact (no env)
0cb6b597b [Sparsity] gitignore development/results/ (benchmark + fixture runtime scratch outputs)
e7951a59d [Sparsity] Fix #K: update Option-B launcher-contract tests to the evolved radix contract; drop plan markers from new code
461119b46 [Sparsity] AC-1b chunked-prefill probe PASSES at the radix-on operating point
a24bc469c [Sparsity] AC-11 directional sweep (radix-on, 3-trial) + #F effective-concurrency accounting
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-9-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-9-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-8-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-8-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-7-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-7-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-10-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
