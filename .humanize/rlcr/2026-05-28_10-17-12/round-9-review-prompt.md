# FULL GOAL ALIGNMENT CHECK - Round 9

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 9 Summary

## Work Completed
Fixed the #K stale launcher-contract tests (+ plan-term cleanup) and ran/recorded the **AC-1b
chunked-prefill probe** — it passes, so the default chunked prefill is kept for the AC-11 sweep.

- **#K (blocking) — Option-B launcher tests updated to the evolved contract.**
  `test/registered/unit/development/test_option_b_scripts.py` encoded the pre-Round-4/Round-8
  contract and failed 2 tests. Rewrote them:
  - `test_dsa_server_radix_on_by_default_with_smoke_knob` — DSA default radix-on; `--disable-radix-cache`
    only inside the `DISABLE_RADIX_CACHE=1` guard.
  - `test_ds_server_radix_off_by_default` — `RADIX_ARGS=(--disable-radix-cache)` default.
  - `test_ds_server_artifact_driven_radix_on` — radix-on via `--double-sparsity-radix-fixture-artifact`;
    removed the obsolete fixed-marker assertion.
  Also reworded the Round-4/8 `AC-`/`DEC-`/`TIER` plan markers in production comments/help
  (serve_double_sparsity.sh, serve_native_nsa.sh, validator.py, server_args.py) to
  behavior-based language (plan Code Style); pre-plan markers left as-is.
- **AC-1b (mainline) — chunked-prefill probe PASSES at the radix-on operating point.** Booted
  DS radix-on via the fixtures-passed artifact (no env override) with `chunked_prefill_size=8192`.
  A 10565-token uncached prompt (radix flushed + unique prefix) prefilled in **genuine multiple
  chunks** (server log: `#new-token 8192` then `2432` for one seq) and was served without crash;
  the in-context needle "OSPREY-3141" was **recalled exactly**. `/get_server_info` confirms
  `chunked_prefill_size=8192`, `disable_radix_cache=false`, DS on, TP=8, page 64.
  Verdict: keep the default chunked prefill on BOTH DS and DSA (no disable needed; sidecars
  match for AC-11). A ~37k-token variant served (multi-chunk, no crash) but mangled the needle
  word — a DS sparse-decode (top_k=2048) long-context recall limit, not a chunked-prefill bug.

## Files Changed
- `test/registered/unit/development/test_option_b_scripts.py` — evolved launcher-contract tests.
- `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh` — plan-term reword.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`,
  `python/sglang/srt/server_args.py` — plan-term reword in the Round-8 additions.
- `runs/20260528_dsv32_mvp/` — `ac1b_probe.json` (verdict=PASS), `ac1b_server_info.json`.
- Commits: `e7951a59d` (#K + cleanup), `461119b46` (AC-1b probe). Pushed.

## Validation
- `pytest test/registered/unit/development/test_option_b_scripts.py
  test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q` → **301 passed**
  (option-B-scripts 23/23 + DS unit + sequential).
- AC-1b hardware: multi-chunk prefill (8192+2432) served radix-on without crash; needle
  recalled exactly; `chunked_prefill_size=8192` confirmed.

## Remaining Items
- **TIER-2:** task13 AC-11 (3-trial radix-on DSA+DS sweep at conc 16/32/64, 120s/600s; gated on
  #F), task14 AC-12 (NIAH 4K/16K/64K + MMLU 5-shot), task15 evidence bundle.
- **#F (queued, now front-of-line):** DS KV-pool/effective-concurrency at mem 0.6 — must be
  resolved or explicitly accounted for before the AC-11 TTFT comparison (next round's gate).
- AC-10 label-capture artifact provenance note (server_args null / stale commit SHA) — fold
  into task15. Stale `calibrate.py` operator recipe docstring — queued cleanup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-longcontext-needle-recall-vs-topk
- Notes: Added the lesson that DS long-context needle recall is bounded by `top_k` (a needle in
  a ~37k context isn't selected from top-2048 → garbled recall) and is a sparsity tradeoff, NOT
  a chunked-prefill/serving bug — recorded now because the upcoming AC-12 NIAH 64K gate will hit
  exactly this and must distinguish a sparse-recall limit from a regression. The #K launcher-test
  update is tracked as a tracker/plan-evolution entry rather than a separate lesson.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-8-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-8-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-7-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-7-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-6-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-6-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop5/refined_plan_v1.md) with the current goal-tracker:
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

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop5/refined_plan_v1.md, we have completed **10 iterations** (Round 0 to Round 9).

The project's `.humanize/rlcr/2026-05-28_10-17-12/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-28_10-17-12/round-8-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-28_10-17-12/round-7-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-28_10-17-12/round-8-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-9-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-9-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
