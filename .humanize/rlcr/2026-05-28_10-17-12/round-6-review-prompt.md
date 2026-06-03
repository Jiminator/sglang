# Code Review - Round 6

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-6-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 6 Summary

## Work Completed
Diagnosed the AC-Q failure (#H) to a definitive root cause on 8x H200 and resolved #I. The
mainline outcome is the contract's "not-a-DS-bug → measurement-change proposal" branch:
**AC-Q remains NOT MET**, but the failure is now proven to be temperature-0 greedy decode
degeneration, not a DS correctness or CUDA-graph bug, and a measurement-change proposal is
filed for approval (no threshold/prompt/decoding default changed unilaterally).

- **#H diagnosis (controls on hardware, `ac_q_diagnosis_round6.md`):**
  - **Eager == CUDA-graph.** DS booted with `--disable-cuda-graph` produces the *identical*
    `17 * 23` repetition loop (and the same dropped `17`). → not a CUDA-graph bug.
  - **DS knows the answers.** Asked concisely: `17 * 23`→`391`, primes→`53, 59, 61`, SI
    unit→ampere. → not a DS correctness/knowledge bug.
  - **DS escapes under sampling.** Temp 0.5 on the same prompt → reaches `391`. → the loop
    is a greedy (temperature-0) decode artifact.
  - **Trajectory divergence is early.** Offline first-N-token ROUGE: N=8 → 0.894, N=16 →
    0.815, N=32 → 0.790, full → 0.726. DS and DSA diverge in trajectory/verbosity within
    ~16 tokens on the 7 open-ended prompts, so a bounded-token comparison does not rescue
    the gate.
  - Root cause: DS decode attention differs numerically from DSA's, so temperature-0 greedy
    decoding follows different (both valid) trajectories on long-CoT prompts; on `17*23` DS
    falls into a greedy repetition loop. There is no DS code fix for temp-0 greedy
    degeneration. The ROUGE-L gate measures DS-vs-DSA lexical trajectory identity, which two
    different attention mechanisms cannot satisfy on open-ended generation; the other three
    gates (answer-agreement) pass.
- **#I resolved.** Hardened `_validate_reference_artifact` to enforce the exact 20 smoke
  prompts + 5 NIAH prompts/needles position-by-position; added 3 regressions
  (truncated / reordered / wrong-needle). A truncated/reordered reference can no longer pass
  a future compare on a subset.

## Files Changed
- `test/manual/_dsv32_quality_smoke_lib.py` — `_validate_reference_artifact` enforces the
  exact committed fixture (#I).
- `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` — +3 rejection
  regressions (now 11 tests).
- `runs/20260528_dsv32_mvp/` — `ac_q_diagnosis_round6.md`, `ds_diag_graph_chat_1723.json`,
  `ds_diag_eager_chat_1723.json`.
- Commit `70bb52a15` (diagnosis + #I). Pushed to remote.

## Validation
- `pytest test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py
  test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → **265 passed**
  (254 DS unit + 11 sequential regression).
- Hardware controls recorded as artifacts (eager-vs-graph, concise-answer, sampling-escape,
  offline first-N ROUGE table).

## Remaining Items
- **AC-Q not met** pending the measurement-change decision (Goal Tracker Update Request below).
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b, task13 AC-11 (gated on #F), task14
  AC-12, task15 evidence bundle.
- **#F (queued):** DS KV-pool/effective-concurrency at mem 0.6 — resolve before AC-11 TTFT.
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-greedy-decode-degeneration-vs-dsa
- Notes: Added the lesson that a DS-vs-DSA temp-0 quality miss must first be triaged with
  eager-vs-graph + concise-prompt + sampling controls; here they prove the `17*23` loop is
  greedy decode degeneration (DS answers are correct), not a DS bug, and that lexical ROUGE
  over open-ended temp-0 generations measures trajectory identity two different attention
  mechanisms cannot meet. (Complements `BL-20260529-dsv32-quality-smoke-needs-chat-template`.)

## Goal Tracker Update Request

### Requested Changes:
- Confirm #H as **diagnosed: temperature-0 greedy decode degeneration, not a DS correctness
  or CUDA-graph bug** (evidence: `ac_q_diagnosis_round6.md`, eager==graph, concise→correct,
  sampling→escape). Mark #I resolved.
- **Decide the AC-Q measurement** (I did NOT change the immutable AC, the threshold, the
  prompt fixture, or the decoding default). Candidate resolutions, in my recommended order:
  1. **Change the AC-Q decoding config** to one that is not greedy-degenerate, applied
     identically to DS and DSA with a fixed seed (e.g. a small repetition penalty, or a low
     fixed temperature) — keeps determinism, removes the greedy-loop confound, and tests DS
     quality on the same footing as DSA.
  2. **Make AC-Q an absolute DS-quality gate** on the known-answer prompts (DS output
     contains the expected answer / is non-degenerate) instead of DS-vs-DSA lexical ROUGE —
     but note `17*23` still fails absolutely under temp-0/256-token greedy, so this needs (1)
     too.
  3. **Constrain the open-ended prompts** to request concise answers (where DS == DSA).
  The harness already supports running these via the `capture`/`compare` CLI once a measure
  is chosen; the exact-fixture validator (#I) guards a legitimate future pass.
- If instead you judge this a DS decode regression that must be fixed in code, please say so
  — but the controls show DS produces correct answers and only greedy long-CoT degenerates,
  for which no DS-attention code fix exists (DS cannot be bit-identical to DSA).

### Justification:
The Ultimate Goal's TIER-1 narrative is "the quality smoke passes." The evidence shows DS's
*answers* are correct (concise arithmetic, primes, NIAH 5/5, prefix 0.80, zero first-8
divergence); the only failing gate, mean ROUGE-L, is confounded with temperature-0 greedy
trajectory chaos that DS is more susceptible to than DSA on long CoT. Lowering the threshold
or editing prompts unilaterally would game the gate, so I'm surfacing the controlled
evidence and concrete options for you to reconcile against the immutable AC. Once a
measurement is approved, the next round reruns the sequential `capture`→`compare` workflow to
a genuine all-four-pass.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-5-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-5-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-4-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-4-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-3-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-3-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-6-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
