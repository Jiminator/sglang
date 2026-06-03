# Code Review - Round 11

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-11-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 11 Summary

## Mainline objective (met)
Closed TIER 2 by **executing the AC-12 hard quality gate (task14) on real hardware** and
**assembling the task15 evidence bundle**. Both plan tasks ran. AC-12 is a **recorded HARD
FAILURE** — the honest, plan-anticipated outcome of DS's `top_k`-bounded long-context recall.

## Work completed

### AC-12 full quality gate (task14) — executed, HARD FAIL recorded
Ran the paired DS-vs-DSA gate on **two H200 nodes** (two TP=8 V3.2 servers can't co-reside on one
8-GPU node): DS radix-on via fixture artifact (mem 0.6, node 0), DSA radix-on (mem 0.85, node 1),
both at the locked Option B point. `/get_server_info` captured for both.

| Gate | DSA | DS | Δ (DSA−DS) | Threshold | Verdict |
|------|-----|-----|-----------|-----------|---------|
| MMLU 5-shot (200) | 89.00% | 89.00% | 0.00 pp | ≤ 1.0 pp | **PASS** |
| NIAH 4K (20) | 100% (20/20) | 75% (15/20) | 25.0 pp | ≤ 5 pp | **FAIL** |
| NIAH 16K (20) | 100% (20/20) | 5% (1/20) | 95.0 pp | ≤ 5 pp | **FAIL** |
| NIAH 64K (20) | served 20/20 | HTTP 400 (unservable) | — | ≤ 5 pp | **FAIL** |

Two real, non-bug mechanisms (`ac12_analysis.md`): (1) DS sparse decode is `top_k=2048`-bounded →
needle recall degrades monotonically with context (75% → 5%); (2) DS at mem 0.6 has
`max_total_num_tokens=53,056` < the 69,970-token 64K prompt → cannot admit 64K (DSA pool 910,784).
MMLU passes because short prompts (seq ≤ top_k) use effectively-dense selection → DS short-context
quality is identical to DSA. DSA (native long-context sparse attention) recalls 100% throughout,
validating the harness. **AC-12 is hard pass/fail (DEC-7 directional handling is AC-11-only) →
recorded as a hard failure, not reclassified.** Therefore the **loop4-compatible MVP is NOT
complete**: TIER-1 smoke complete, TIER-2 incomplete (AC-10/11/6/1b done, AC-12 MMLU passes,
AC-12 NIAH hard-fails).

### Enabling work
- **HOST knob (blocking #B1):** added a `HOST` env knob (default 127.0.0.1) to both Option-B
  launchers, passed through as `--host`, so the DSA baseline binds `0.0.0.0` for cross-node reach.
  Locked Option-B flags + default localhost behavior unchanged; +1 lock regression.
- **Node-1 sync:** node 1 was on a stale commit (`cb6004a36`, pre the NSA→DSA boot-chain fix and
  pre the locked Option-B launchers) — DSA crashed with `'…' object has no attribute 'use_nsa'`.
  Fast-forwarded node 1 to HEAD (`7478c27a0`); DSA then booted cleanly cross-node.
- **Harness transport fix:** `_generate` now uses `/v1/chat/completions` for NIAH (raw `/generate`
  returns an immediate-EOS empty string for these instruction prompts → would falsely pass the
  paired gate 0/0 on both servers) and KEEPS raw `/generate` for MMLU 5-shot (a few-shot
  *completion* benchmark the chat template breaks: verified raw 10/10 vs chat 0/10). Thresholds
  and prompt fixtures unchanged.

### Evidence bundle (task15)
`runs/20260528_dsv32_mvp/evidence_bundle.md` — AC-by-AC index with artifact paths, mask
provenance/SHA, server args/server_info, CUDA-graph + chunked-prefill status, radix fixture, and
an explicit **AC-10 label-capture provenance note** (resolving the Round-8 queued item). States
AC-11 as "executed; directional TTFT/TPS target missed; #F admission caveat + follow-up filed" and
AC-12 as a hard failure → loop4 MVP incomplete.

## Files changed
- `development/serve_native_nsa.sh`, `development/serve_double_sparsity.sh` — `HOST` knob.
- `test/registered/unit/development/test_option_b_scripts.py` — HOST-knob lock regression.
- `test/manual/test_double_sparsity_v32.py` — `_generate` task-specific transport (NIAH chat /
  MMLU raw); `_run_niah` uses chat.
- `runs/20260528_dsv32_mvp/` — `ac12_analysis.md`, `evidence_bundle.md`,
  `ac12_{ds,dsa}_server_info.json`, `ac12_results/` (MMLU+NIAH JSON, pytest summary, DS boot
  excerpt).
- Commits `7478c27a0` (HOST knob), `1a1293f01` (AC-12 + bundle). Both pushed.

## Validation
- **407 CPU tests pass** (`test_ac11_comparator` + `test_double_sparsity_unit` +
  `test_dsv32_quality_smoke_sequential` + `test_option_b_scripts` + `test_ac12_helpers`); up from
  359 (+47 ac12_helpers, +1 HOST-knob lock).
- Hardware: both servers booted at the locked Option B point (server_info captured); AC-12 gate
  ran to completion (`3 failed, 1 passed, 2 skipped`); all gate artifacts written and copied to
  `ac12_results/`. Both servers shut down afterward; both nodes' GPUs freed (the pre-existing
  port-30000 router was already down at round start and was not touched).

## Remaining Items
- **No mainline work remains** — all plan tasks (task1–task15) have been executed. AC-12 is a
  recorded hard failure; per the Ultimate Goal this makes the deliverable a TIER-1 smoke milestone
  with TIER-2 loop4 quality not met (recorded, not a build-break to fix).
- **Queued (out of scope, documented):** (a) comparator per-side `mem_fraction_static` validation
  hole — tighten when the comparator is next touched (AC-12 didn't touch it); (b) AC-11 directional
  performance follow-up (TokenLabelTable / KV-budget) — same lever now also bounds AC-12 64K
  admission; (c) stale `calibrate.py` `--tp 1` recipe docstring (doc-accuracy only; mask+SHA are
  authoritative).

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260529-ds-longcontext-needle-recall-vs-topk, BL-20260529-dsv32-quality-smoke-needs-chat-template
- Notes: Updated the long-context-recall lesson with the R11 AC-12 **quantified** result (DS NIAH
  recall 75%→5%→unservable at 4K/16K/64K vs DSA 100%) and a SECOND, independent limit it now
  documents: at mem 0.6 the DS KV pool is only ~53K tokens, so a >pool prompt is rejected (HTTP
  400) as an admission failure distinct from a recall miss — distinguish them from the server log.
  Updated the chat-template lesson with the R11 **counterpoint**: transport is task-specific — NIAH
  instruction prompts need chat completions, but MMLU few-shot *completion* needs RAW /generate
  (chat broke MMLU 0/10 vs raw 10/10). No new lesson added (both findings extend existing entries).

## Goal Tracker Update Request

### Requested Changes:
- Move **task14 / AC-12** to Completed and Verified as **EXECUTED with a recorded HARD FAILURE**
  (gate run on hardware, honest pass/fail recorded; MMLU PASS, NIAH 4K/16K/64K FAIL).
- Move **task15 / evidence bundle** to Completed and Verified (`evidence_bundle.md` assembled).
- Confirm the **AC-10 label-capture provenance** queued item RESOLVED (explicit provenance note
  added to `evidence_bundle.md`).
- Record the **comparator per-side `mem_fraction_static` hole** as a queued (non-blocking) item to
  fix when the comparator is next touched.

### Justification:
Both remaining plan tasks were executed this round. AC-12 is a hard pass/fail gate (DEC-7
directional handling is explicitly AC-11-only), so its failure is recorded as a hard failure, not
reclassified as directional — exactly per Codex's Round-10 directive ("If NIAH 64K fails … publish
that as a hard AC-12 failure with evidence. Do not reclassify AC-12 as directional. … If AC-12
fails, the bundle must say the Loop4-compatible MVP is not complete"). The Ultimate Goal itself
anticipates this terminal state: with AC-12 full quality not met, the deliverable is a TIER-1 smoke
milestone, not the loop4 MVP. The failure modes (top_k-bounded recall; mem-0.6 KV-budget admission
limit) are inherent DS design / operating-point tradeoffs, not bugs with a code fix that would turn
AC-12 green. No immutable AC or threshold was changed; the only harness change fixes the NIAH
measurement transport so the model actually answers (raw /generate returned empty), which makes the
gate meaningful rather than vacuous.
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
7478c27a0 [Sparsity] Add HOST knob to Option-B launchers for cross-node AC-12
1a1293f01 [Sparsity] AC-12 full quality gate executed: MMLU pass, NIAH hard-fail (task14+task15)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-10-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-10-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-9-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-9-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-8-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-8-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-11-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
