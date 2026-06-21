# Code Review - Round 5

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-5-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 5 Summary — DRIFT RECOVERY

Codex marked R3–R4 STALLED (2 consecutive). Recovered mainline: **begin AC-6 production-path
single-variable bisection on the REAL sparse workload.** Both of Codex's stated acceptable outcomes
were delivered (pruning-valid AC-2.3 AND a GSM8K-measured AC-6 arm), and the AC-6 arm produced a new,
verdict-refining result.

## Work Completed
- **AC-2.3 RESOLVED on real pruning rows** (retires AC-6 radix+width legs, suspicion-order 5–6).
  Rewrote `verify_ac2_3.py` to record the seq_len distribution + a `pruning_rows` count, evaluate
  radix==`torch.topk` on the **pruning subset** (seq_len > top_k), split the width check by the 5120
  boundary, and **fail (exit 2) if `pruning_rows==0`** — so it can never pass on smoke captures again
  (verified: it now exits 2 on the old seq_len=13 set). Captured the **SPARSE** regime (24-shot, eager
  production DS): on **4992** real rows (median seq_len **4280**, ~2048 of ~4280 pruned), the
  production blocked/radix algorithm == exact `select_topk_sequence_order` **4992/4992**, and
  selector-width [5120] == full **4992/4992**. The sparse capture run also re-confirmed production DS
  sparse = **0.000**.
- **AC-6 single-variable bisection arm (measured GSM8K).** New `serve.sh ref_cosine_noinc` mode =
  `ref_cosine` with the ONE variable flipped: `reference_include_current` true→false (production
  current-slot exclusion); cosine scorer, `head_agg=max`, exact fp32, TF32-off all held fixed
  (config-only; no selection/adapter fix lands). **Result: dense 0.940→0.625 (= production 0.620),
  sparse 0.940→0.313.** This completes the **scorer × current-slot 2×2**:

  | scorer \ current-slot | EXCLUDED (production) | INCLUDED (faithful) |
  |---|---|---|
  | raw-dot | production 0.620 / **0.000** | ref_faithful 0.950 / **0.013** |
  | cosine | ref_cosine_noinc 0.625 / **0.313** | ref_cosine 0.940 / **0.940** |

  **New finding:** sparse ≈0.94 needs **BOTH** the cosine scorer AND current-slot inclusion (neither
  alone: cosine+excl 0.313, rawdot+incl 0.013); the two regressions **interact**, and current-slot
  exclusion (H3) is a culprit in **both** regimes, not dense-only — refining the R1 reference-ceiling
  verdict. Corroborated by the `ds_anchor` arms (current-slot forced back on raw-dot stays 0.000/0.007).
- **Provenance single-source-of-truth (blocking).** `build_ledger.py` now patches `run_meta.json`'s
  generator blob + `git_sha_current` from the same `GEN_BLOB`/`GEN_HEAD` it stamps into per-arm JSONs,
  and asserts per-arm JSON == table header == run_meta blob (fails loud otherwise). The Codex-R4
  mismatch (run_meta `1391f0e` vs arms `f8771c7f2`) is closed; all three now agree.
- Downgraded the AC-2.3 "RESOLVED" over-claim, then re-resolved it from the pruning-valid artifact;
  `cheap_controls.json._status` no longer contradicts the stale 81/546 join summary. ROOT_CAUSE.md /
  findings.md updated with the 2×2 and the refined attribution.

## Files Changed (committed `c7b66f04b`)
- `development/loop13/verify_ac2_3.py` — pruning-aware, fail-closed on `pruning_rows==0`, width split.
- `development/loop13/serve.sh` — new `ref_cosine_noinc` single-variable bisection mode.
- `development/loop13/build_ledger.py` — run_meta provenance patch + consistency assertion; new arm;
  refined verdict line.
- `development/loop13/ROOT_CAUSE.md`, `evidence/findings.md` — scorer×current-slot 2×2, refined verdict.
- `evidence/ac2_3_radix_width_equivalence.json` (4992/4992 pruning-valid), `cheap_controls.json`,
  `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json` (+ new
  `ref_cosine_noinc.json`), `.gitignore` (new sparse capture dirs; raw .pt stay on disk per convention).

## Validation
- `verify_ac2_3.py` on sparse captures → **4992/4992** radix and width identical, exit 0; on the old
  seq_len=13 set → **exit 2** (`pruning_rows=0`).
- `build_ledger.py` → "provenance consistent" assertion passes; run_meta blob == per-arm == table.
- `test_reference_selectors.py` → **all 5 pass**.
- GSM8K `ref_cosine_noinc`: dense **0.625**, sparse **0.313** (batched, `--api completion`, temp 0).
- Discipline: one TP=8 server at a time — capture server torn down to 0 MiB before the measurement
  arm; both torn down at end (all 8 GPUs 0 MiB). No `.pt`/`.humanize` committed.

## Remaining Items (next mainline)
- **AC-6 production-NUMERIC legs** (fp8-absorbed vs exact fp32, bf16 vs fp32 reduce, head_agg cross-TP)
  — **untestable via config toggle** (the reference selector is exact fp32); a production-numerics
  cosine needs a production-path cosine **kernel** = code change, out of scope under "no fix". Documented
  as second-order (production raw-dot 0.000 ≈ exact raw-dot 0.013), not hand-waved.
- AC-2.4 recall-oracle@2048 corroboration for the arms; AC-2.1 `forced_all_assertions.json`; AC-4
  sample IDs/order + garbage counters; AC-3.1 captured-row materialized-K; AC-2.2 head-agg semantics.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-ds-control-must-exercise-pruning, BL-20260621-ds-bisection-interaction
- Notes: Added two lessons — (1) a top-k/selection equivalence control is vacuous unless the captured
  rows exercise the branch under test (seq_len > top_k); capture the risky regime and fail-closed on a
  zero pruning-row count. (2) A "faithful reference" that carries multiple non-production crutches can
  hide an interaction; peel one variable per arm and measure the full 2×2 rather than reading a
  single-variable cost off a multi-variable reference. Also UPDATED BL-20260621-ds-capture-step-alignment
  so its evidence cites the R5 pruning-valid 4992/4992 (the R4 624/624 was the invalid seq_len=13 set).

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-2.3 RESOLVED** (task4): radix==torch.topk 4992/4992 + width [5120]==full 4992/4992 on real
  sparse pruning rows (median seq_len 4280); verifier fails if `pruning_rows==0`.
  Evidence: `evidence/ac2_3_radix_width_equivalence.json`. Radix + selector-width suspects retired.
- Mark **AC-6 ADVANCED** (task11): first GSM8K-measured single-variable arm run (`ref_cosine_noinc`,
  0.625/0.313); scorer×current-slot 2×2 complete; verdict refined (sparse needs both fixes; H3 hurts
  both regimes). The remaining numeric legs require a production-path cosine kernel (code) and are out
  of scope under "no fix" — request they be reclassified from "blocking" to "documented out-of-scope".
- Close **Blocking: ledger provenance inconsistency** — build_ledger.py single-source-of-truth +
  consistency assertion; run_meta == per-arm == table.
- Close **Blocking: AC-2.3 does not exercise pruning** — resolved on real sparse rows.
- Plan Evolution Round-5 row already added (drift cause + recovery).

### Justification:
The recovery round delivered both Codex-stated acceptable outcomes (pruning-valid AC-2.3 and a
measured AC-6 arm) and produced a new, verdict-refining bisection result on the actual sparse workload
— directly countering the drift pattern (cheap CPU work while AC-6 GSM8K arms never ran). The two
remaining AC-6 numeric legs are genuinely blocked by the "no fix" constraint (they need a new
production-path cosine kernel), so they belong as documented out-of-scope, not as open blockers.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
3058bdc35 [loop12] add gen-plan output + draft; record pensieve doctor state
aaefdaf1e [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
8f88e1aef [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
4706b2138 [loop12] document double-sparsity v2 performance numbers
480fd70ba [loop12] add double-sparsity v2 run-and-evaluate runbook
2babc5afa [loop12] add gsm8k evidence + refresh perf verdict numbers
180f6dd6d [loop13] add diagnosis-loop plan, draft, and guarded harness
fc6ac20a7 [loop13] diagnostic DS selectors: fp32 raw-dot reference + forced-all dense control
29ec137bf [loop13] harness: ref/ds_capture/ds_forced_all serve modes + AC-1 baseline evidence
16caf4f5b [loop13] reference selector: gather-then-dequant (full-pool dequant was intractable)
5a1da871d [loop13] verdict: DS dense degradation = current-decode-slot exclusion (H3)
fea920c06 [loop13] reference selector: served cosine + faithful/leak-free ceiling
62ad64346 [loop13] Round 1 verdict FLIP: ceiling GOOD; sparse = raw-dot scorer_norm=off lock
ac479aeb3 [loop13] Round 2: per-arm evidence ledger, baseline consistency, captures, cleanup
29ed825fa [loop13] Round 3: ledger SHA provenance, capture row-identity, exact-join analyzer
393966c02 [loop13] Round 4: AC-2.3 RESOLVED on real captured rows; fail-closed analyzer; ledger provenance
c7b66f04b [loop13] Round 5 (drift recovery): pruning-valid AC-2.3 + AC-6 scorer×current-slot 2×2
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-4-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-4-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-3-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-3-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-2-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-2-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-5-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
