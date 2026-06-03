# Code Review - Round 5

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-5-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 5 Summary

## Work Completed
Built the single-node sequential AC-Q quality smoke (resolving #G) and ran it end-to-end
on 8x H200. AC-Q is **evidenced but NOT met** — 3/4 gates pass; the ROUGE-L gate misses on
benign temperature-0 decode drift (not a correctness regression). Reported honestly with a
Goal Tracker Update Request rather than altering the immutable threshold.

- **#G resolved — sequential harness.** Two TP=8 servers cannot co-reside on one 8-GPU
  node, but the smoke required both `DS_BASE_URL` and `DSA_BASE_URL` up at once and
  interleaved DSA/DS per prompt. Split into:
  - `test/manual/_dsv32_quality_smoke_lib.py` — shared prompt fixtures, generation,
    pure-Python ROUGE-L / first-n overlap, the load-bearing `compute_gates()`, plus
    `capture_reference_outputs()` and `evaluate_against_references()`.
  - `test/manual/test_dsv32_quality_smoke.py` — kept the legacy simultaneous unittest, added
    a `capture`/`compare` CLI (capture writes the 20+5 DSA refs with only DSA up; compare
    loads them with only DS up, scores the gates, exits non-zero on any miss).
  - `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` — 8-test CPU
    regression proving `compute_gates` verdicts + the capture→compare round-trip with no
    live servers (`generate` monkeypatched).
- **Generation switched to `/v1/chat/completions`.** The raw `/generate` path returned
  dataset/JSON scaffolding for the instruction prompts and **empty** outputs for the long
  NIAH prompts (base-model continuation with no chat template). Chat completions apply the
  template server-side, so the model actually answers (Hamlet→"William Shakespeare",
  NIAH→"ZEBRA-7"); both DS and DSA use the identical path.
- **Ran it on hardware (sequential):** booted DSA (radix-off, cluster path) → captured 20+5
  coherent references (NIAH 5/5) → shut DSA down → booted DS (radix-off, cluster path) →
  compared.

## Files Changed
- `test/manual/_dsv32_quality_smoke_lib.py` (new) — shared library; generation via chat
  completions.
- `test/manual/test_dsv32_quality_smoke.py` — refactored to use the lib; `capture`/`compare`
  CLI; legacy simultaneous unittest retained.
- `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` (new) — CPU regression.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — repointed
  `TestDSv32SmokeHelpers` at the shared lib (helpers moved there).
- `runs/20260528_dsv32_mvp/` — `dsa_quality_refs.json`, `dsv32_quality_smoke.json`,
  `ac_q_analysis.md` (logs gitignored).
- Commits: `99ac93691` (harness + regression), `d8fce372a` (AC-Q evidence + analysis).
  Pushed to remote.

## Validation
- `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q` → **262 passed**
  (254 DS unit + 8 new sequential regression).
- Hardware AC-Q gate verdict (`dsv32_quality_smoke.json`):
  - prefix_match_rate = 0.80 (≥ 0.80) — **PASS**
  - mean_rouge_l = 0.726 (≥ 0.85) — **FAIL**
  - niah_mini_recall = 5/5 (≥ 4/5) — **PASS**
  - first_8_tokens_divergence = 0 (== 0) — **PASS**
  - **AC-Q overall: FAIL** (hard gate).
- Analysis (`ac_q_analysis.md`): ROUGE-L median = 1.000; all short factual answers match
  DSA verbatim, NIAH recall perfect. The mean is dragged down by 7 open-ended explanatory
  prompts where DS and DSA agree on the answer + first tokens then diverge in wording/length
  under greedy decoding with different attention numerics — benign drift, not a regression.

## Remaining Items
- **AC-Q not met** as literally defined — see the Goal Tracker Update Request below.
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b chunked-prefill probe, task13 AC-11
  (gated on #F), task14 AC-12, task15 evidence bundle.
- **#F (queued):** DS KV-pool/effective-concurrency limit at mem 0.6 — resolve before the
  AC-11 TTFT comparison (does not affect AC-Q, which is sequential single-prompt).
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-dsv32-quality-smoke-needs-chat-template
- Notes: Added `BL-20260529-dsv32-quality-smoke-needs-chat-template` — instruction/QA eval
  prompts must go through `/v1/chat/completions` (chat template applied), not raw
  `/generate`, or the served model returns degenerate continuations / empty NIAH outputs;
  also records that ROUGE-L over long temp-0 free-form generations is a noisy DS-vs-DSA
  signal (greedy divergence after the shared prefix).

## Goal Tracker Update Request

### Requested Changes:
- Record AC-Q as **executed with evidence but the gate is NOT met** (mean_rouge_l 0.726 <
  0.85); keep task9's harness work (#G) as resolved/verified.
- Decide the reconciliation for the ROUGE-L gate. Options (I did NOT change the immutable
  AC or threshold myself):
  1. Treat the ROUGE-L gate like AC-11's directional targets (DEC-7) — a documented miss +
     follow-up rather than a hard TIER-1 blocker — given the evidence shows DS quality is
     substantively intact (median ROUGE-L 1.0, all short answers exact, NIAH 5/5, prefix
     0.80, zero first-8 divergence).
  2. Refine the AC-Q measurement to reduce benign long-generation noise (e.g. compare a
     bounded answer span, or lower `max_tokens` for the open-ended prompts) — a plan change
     requiring your approval.
  3. Treat it as a genuine DS gap and require investigation before TIER-1 is "complete".

### Justification:
The Ultimate Goal's TIER-1 narrative is "quality smoke passes on 20 paired prompts." The
run shows DS reproduces DSA's *answers* faithfully; the only gate miss is mean ROUGE-L over
free-form 256-token generations, which is inherently sensitive to temperature-0 greedy
divergence between two attention implementations (median ROUGE-L is a perfect 1.0). Lowering
the threshold unilaterally would be gaming the gate, so I'm surfacing the evidence + analysis
(`ac_q_analysis.md`) for you to reconcile against the immutable AC definition.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-4-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-4-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-3-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-3-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-2-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-2-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-5-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
