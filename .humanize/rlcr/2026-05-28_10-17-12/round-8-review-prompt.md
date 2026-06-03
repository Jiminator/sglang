# Code Review - Round 8

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-8-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 8 Summary

## Work Completed
Fixed the #J false-pass (AC-Q now cleanly passes) and implemented + verified **AC-10**
end-to-end on 8x H200 — DS boots radix-on via a no-env-override config-bound artifact. TIER-1
is clean; TIER-2 has begun.

- **#J (blocking) — first-8 false-pass fixed.** Replaced Round-7's string-prefix fallback in
  `first_n_tokens_match` with **alphanumeric-subtoken overlap**: each first-n token splits into
  alnum runs (`100°C`→{`100`,`C`}; `53,59,61`→{`53`,`59`,`61`}); overlap requires a shared alnum
  subtoken; the exact whitespace check still handles pure punctuation (`.`). Now `100`/`100°C`
  match (shared `100`), but `10`/`100` and `Paris.`/`London.` do NOT. +4 false-pass regressions.
  Recomputed the concise AC-Q gate offline from the saved deterministic outputs → `all_pass=true`
  under the corrected gate (prefix 0.95, rouge 0.944, niah 5/5, first8_div 0; the 100/100°C match
  now comes from the shared `100` subtoken, not a broad prefix).
- **AC-10 (mainline) — no-env-override radix flip (DEC-5).**
  - `write_radix_fixture_state(...)` writes a JSON state recording BOTH M3-B fixtures passed +
    a config fingerprint (model/tp/page/kv-dtype/channel-mask-SHA).
  - `--double-sparsity-radix-fixture-artifact` + `apply_radix_fixture_artifact` (called in
    `check_server_args` BEFORE `validate_double_sparsity`) verify the state matches THIS boot and
    record the flip — fail-closed on mismatch/missing/partial. 6 CPU regressions.
  - `serve_double_sparsity.sh`: `RADIX_FIXTURE_ARTIFACT` → pass artifact + drop
    `--disable-radix-cache`; the env override now only enables radix-on for dev/fixture runs.
  - **Hardware:** FP8 scale-stability fixture PASSED; label-capture fixture PASSED (cold==warm
    label SHAs, `cached_tokens>0`) after fixing its `/flush_cache` JSON-parse bug; wrote the state
    file; booted DS **radix-on authorized purely by the artifact (no `SGLANG_DS_RADIX_OVERRIDE`)** —
    validator logged the flip with the artifact SHA, `/get_server_info` shows
    `disable_radix_cache=false`, DS on, TP=8, fp8, page 64; `/generate`→"Paris".

## Files Changed
- `test/manual/_dsv32_quality_smoke_lib.py` — alnum-subtoken `first_n_tokens_match` (#J).
- `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` — +4 #J regressions.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — `write_radix_fixture_state`,
  `apply_radix_fixture_artifact`, `radix_fixture_config_fingerprint`, `RADIX_FIXTURE_STATE_SCHEMA`.
- `python/sglang/srt/server_args.py` — `double_sparsity_radix_fixture_artifact` field + CLI +
  `check_server_args` wiring.
- `development/serve_double_sparsity.sh` — `RADIX_FIXTURE_ARTIFACT` knob; env override = dev-only.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — +6 AC-10 regressions.
- `test/manual/test_dsv32_radix_label_capture_fixture.py` — `_flush_cache` plain-text fix.
- `runs/20260528_dsv32_mvp/` — `ac10_fp8_scale_stability.json`, `ac10_label_capture.json`,
  `ds_radix_fixture_state.json`, `ac10_radixon_server_info.json`, recomputed
  `dsv32_quality_smoke_concise.json`.
- Commits: `d47dcbadb` (#J), `fa4473694` (AC-10 mechanism), `67422e698` (AC-10 hardware). Pushed.

## Validation
- `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q` → **278 passed**
  (254 DS unit incl. 6 new AC-10 + 18 sequential incl. 4 new #J + first-8 tests).
- Hardware AC-10: both fixtures pass; radix-on boot via artifact (no env), `disable_radix_cache=false`,
  `/generate` coherent. AC-Q recomputed → `all_pass=true` under the precise gate.

## Remaining Items
- **TIER-2:** task12 AC-1b (chunked-prefill probe), task13 AC-11 (3-trial radix-on sweep; gated on
  #F), task14 AC-12 (NIAH 4K/16K/64K + MMLU 5-shot), task15 evidence bundle.
- **#F (queued):** DS KV-pool/effective-concurrency at mem 0.6 — resolve/account for before AC-11
  TTFT (now relevant since AC-11 runs radix-on next).
- Round-7 artifact-completeness note (graph-mode primes JSON / server-info in meta JSONs) — fold
  into the task15 bundle. Stale `calibrate.py` operator recipe docstring — queued cleanup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-radix-flip-config-bound-artifact
- Notes: Added the AC-10 no-env-override radix-flip pattern — a config-bound fixtures-passed state
  file (`write_radix_fixture_state` / `apply_radix_fixture_artifact`) verified against the boot's
  fingerprint before validation, with the env override demoted to dev/fixture-only; includes the
  chicken-and-egg resolution (run fixtures under the override → write state → final boot uses the
  artifact) and the `/flush_cache`-returns-plain-text fixture gotcha. The #J fix is recorded as a
  refinement of the AC-Q gate (alnum-subtoken overlap) in the goal tracker rather than a separate
  lesson.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-7-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-7-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-6-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-6-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-5-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-5-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-8-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
