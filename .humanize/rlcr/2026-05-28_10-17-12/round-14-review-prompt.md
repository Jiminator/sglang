# FULL GOAL ALIGNMENT CHECK - Round 14

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 14 Summary

## Mainline objective (met)
At the Round-14 disposition the user chose **"Re-scope AC-12 to a DS-fair gate now."** Implemented
the re-scope with integrity (principled, transparent, user-authorized) and verified it on hardware:
**re-scoped AC-12 PASSES.**

## Why (rationale, from Rounds 12-13)
DS is dense-prefill / sparse-decode with a fixed per-decode-step selection budget = the model's
native DSA `index_topk` = **2048 on V3.2, kernel-locked** (the `flashmla_kv` decode kernel asserts
`indices.shape[-1] == dsa_index_topk`; the budget cannot be raised on this backend). The original
AC-12 tested needle recall at 4K/16K/64K — **beyond DS's selection budget**, where an arbitrary
needle is information-theoretically unrecallable from 2048 selected tokens. Round 13 proved this is
a **selection-quality** limit vs V3.2's trained DSA indexer at the same budget, **not a decode bug**
(DS recalls 100% when its selection is dense). Testing recall beyond the budget tested DS outside
its design envelope.

## Work completed

### Re-scoped AC-12 harness (`test_double_sparsity_v32.py`)
- **HARD gates:** MMLU 5-shot within 1 pp of DSA (unchanged) **+** NIAH **within the selection
  budget** — context lengths whose tokenized length ≤ `INDEX_TOPK` (dense DS selection; word counts
  1024/1536) within 5 pp of DSA. This measures DS recall inside its design envelope.
- **CHARACTERIZATION (recorded, NOT a DSA-parity pass/fail):** NIAH 4K/16K/64K recall-vs-length +
  any admission limit; only a monotone-non-increase sanity assertion among servable points. The
  beyond-budget artifacts keep `verdict=FAIL` so the degradation stays **transparent, not hidden**.
- Kept the Round-12 error-aware `_run_niah`/`_record` path; module docstring updated to the DS-fair
  scope. The immutable AC text was **not** edited — the re-scope is logged as a Plan Evolution.

### CPU regressions (`test_ac12_helpers.py`)
- within-budget hard gate **passes** when DS==DSA and **FAILS** when DS misses (teeth retained);
- beyond-budget characterization **records** a DS rejection (durable `verdict=FAIL` artifact)
  **without hard-failing** (an admission limit is characterized, not gated).
- **411 CPU tests pass.**

### Hardware verification (two H200 nodes, same locked Option B point)
`3 passed, 2 skipped, 5 subtests` (exit 0):

| Gate | class | DSA | DS | verdict |
|------|-------|-----|-----|---------|
| MMLU 5-shot (200) | HARD | 89.00% | 89.00% (Δ0.00) | **PASS** |
| NIAH @1024 (≤ budget) | HARD | 100% | 100% (Δ0) | **PASS** |
| NIAH @1536 (≤ budget) | HARD | 100% | 100% (Δ0) | **PASS** |
| NIAH 4K | characterization | 100% | 75% | recorded (FAIL) |
| NIAH 16K | characterization | 100% | 5% | recorded (FAIL) |
| NIAH 64K | characterization | 100% | 0% (HTTP 400, ds_served 0/20) | recorded (FAIL) |

DS preserves recall within its 2048-token budget (= dense) and on MMLU, matching DSA → decode is
sound; beyond the budget recall degrades as the inherent top_k tradeoff and 64K is unservable at
mem 0.6 — both recorded, neither a bug. **AC-12 (DS-fair) is MET.**

### Docs / tracker
- `ac12_analysis.md` + `evidence_bundle.md` rewritten to the DS-fair gate (PASS; beyond-budget
  characterization kept transparent; original-AC run preserved under
  `ac12_results/superseded_prerescope/`).
- Goal-tracker: Plan Version → 2; Round-14 Plan Evolution row (user-authorized re-scope); AC-12 row
  updated to MET under the DS-fair gate (pending Codex reconcile of the immutable definition).

## Files changed
- `test/manual/test_double_sparsity_v32.py` — re-scoped gate (within-budget hard + beyond-budget
  characterization); docstring.
- `test/registered/unit/manual/test_ac12_helpers.py` — re-scope regressions (`import re`).
- `runs/20260528_dsv32_mvp/` — fresh `ac12_results/` (within-budget 1024/1536 + MMLU +
  characterization 4K/16K/64K; pre-rescope under `superseded_prerescope/`), `ac12_analysis.md`,
  `evidence_bundle.md`, re-captured `ac12_{ds,dsa}_server_info.json`.
- Commits `10f5b8878` (re-scope + regressions), `a54a7364f` (hardware PASS + docs). Both pushed.

## Validation
- 411 CPU tests pass. Hardware re-scoped gate: 3 passed / 2 skipped / exit 0; all HARD gates green;
  beyond-budget characterization recorded. Both servers shut down; both nodes' GPUs freed; the
  pre-existing router untouched.

## Remaining Items
- **DS long-context R&D (carried to next loop, `next_loop_issues.md`):** a query-aware/learned DS
  selector; a decode kernel accepting `top_k > index_topk`; a smaller TokenLabelTable for 64K
  admission. These would lift the beyond-budget limits but are out of this loop's scope.
- **AC-11** directional TTFT miss remains recorded per DEC-7 (admission-bound, mem 0.6).
- Queued cosmetic: pre-existing "Option B" serve-script header terms.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-sparse-gate-test-within-budget
- Notes: Added the reusable gate-design principle behind the re-scope — a budget-limited sparse
  mechanism (DS/DSA/any top_k-bounded sparse attention) should be HARD-gated on quality WITHIN its
  selection budget (short-context parity + needle recall at lengths ≤ budget, where selection is
  dense), and its beyond-budget recall should be CHARACTERIZED (recorded, verdict kept visible),
  NOT pass/failed against the dense/native baseline — because beyond the budget the gate measures
  the inherent sparsity tradeoff, not decode quality. Apply only after ruling out a masked bug
  (dense recall == baseline + same-budget baseline succeeds); re-scoping an immutable AC needs owner
  authorization + a logged Plan Evolution + a teeth-checking regression. Builds on the R13
  kernel-lock + selection-quality finding (BL-20260529-ds-longcontext-needle-recall-vs-topk).

## Goal Tracker Update Request

### Requested Changes:
- **Reconcile AC-12 to the DS-fair gate (Plan Evolution, Round 14)** and mark **AC-12 MET**: HARD
  gates (MMLU within 1pp + NIAH within the selection budget within 5pp of DSA) pass on hardware;
  beyond-budget NIAH is characterized (recorded, not gated), with the degradation transparently kept.
- Record that this re-scope was **explicitly authorized by the user** (Round-14 AskUserQuestion:
  "Re-scope AC-12 to a DS-fair gate now"); the immutable AC text was not edited — logged as Plan
  Evolution per the tracker's change mechanism.
- Note the loop4-compatible MVP is now substantially complete (AC-10/AC-6/AC-1b done; AC-12 MET
  DS-fair; AC-11 directional TTFT miss recorded per DEC-7); DS long-context R&D carried to the next
  loop.

### Justification:
DS's selection budget is kernel-locked to the model's DSA `index_topk=2048` and DS decode is sound
(within-budget recall 100% = DSA; MMLU = DSA); testing recall beyond the budget measured the
inherent sparsity tradeoff, not a defect. The DS-fair re-scope tests DS within its design envelope
(the theoretically-correct measurement) while keeping the beyond-budget degradation transparently
recorded — it is not a threshold relaxation to hide a bug, and the within-budget gate retains teeth
(a CPU regression proves a within-budget DS miss FAILS). The user, who owns the goal, authorized the
re-scope; per the tracker's rules the change is logged as a Plan Evolution for Codex to reconcile
into the immutable AC-12 definition. No threshold was loosened within the budget and AC-12 was not
faked green.
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
d2f48bbd4 [Sparsity] Make AC-12 NIAH gate artifact-safe on server rejection (#L)
cc50bae38 [Sparsity] AC-12 64K durable artifact (#L) + analysis/bundle update
ced03f374 [Sparsity] Round-13 queued cleanups: comparator per-side mem-fraction check + calibrate recipe docstring
27434cee7 [Sparsity] Round-13 NIAH selection-budget investigation + next-loop issue list
10f5b8878 [Sparsity] Re-scope AC-12 to a DS-fair quality gate (user-authorized, Round 14)
a54a7364f [Sparsity] Re-scoped AC-12 PASSES on hardware (DS-fair gate) + analysis/bundle
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-13-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-13-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-12-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-12-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-11-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-11-review-result.md


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

To implement the original plan at @development/loop5/refined_plan_v1.md, we have completed **15 iterations** (Round 0 to Round 14).

The project's `.humanize/rlcr/2026-05-28_10-17-12/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-28_10-17-12/round-13-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-28_10-17-12/round-12-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-28_10-17-12/round-13-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-14-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-14-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
