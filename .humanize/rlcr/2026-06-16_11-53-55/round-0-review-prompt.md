# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop11b/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 0 Summary — loop11b: finish loop 11's M4 verdict on a fresh 8×H200

**Outcome: the loop is COMPLETE — all 11 mainline tasks done, all milestones met, an honest measured
verdict delivered.** 32 commits on `dev/double-sparsity-standalone` (local; not pushed — see Remaining).

## What Was Implemented

**The headline verdict (the loop's reason to exist):** Table-free Double Sparsity on GLM-5.1-FP8 MEETS the
client SLO (decode-TPS p50 ≥ 30, P99 TTFT < 22 s) at concurrency 16 and 32, and FAILS at concurrency 64 (DS
decode-TPS 26.98 < 30 AND P99 TTFT 25.12 s > 22 s). Native DSA also fails at conc 64 (26.22 TPS, 33.32 s
TTFT) — the 30-TPS decode floor is the binding constraint for BOTH at high concurrency. At the matched
op-point DS is competitive-to-better than DSA: decode-TPS ratio 0.98–1.03, P99 TTFT ratio 0.46–0.75 (DS lower
at every conc), per-step TPOT 0.97–1.02. A documented FAIL at conc-64 is a complete result.
Report: `development/loop11b/runs/20260616_mb/DS_absolute_verdict.md`; ledger: `development/loop11b/results.md`.

**M-A op-point re-establishment (AC-0/5/6/7) — COMPLETE.**
- Regenerated the GLM-5.1 channel mask. CAUGHT + FIXED a recipe error: the plan/AC-0.1 recipe carried the
  DeepSeek-V3.2 values (`--dtype bfloat16 --label-dim 16`), which served −5.2pp recall at L4096; the
  GLM-native loop8 DEC-3 recipe is `--dtype fp8_e4m3 --label-dim 32`. Corrected mask `content_sha256=35155ac4…`
  serves; recall matches the frozen baseline (L4096 58.045% = baseline). FP8 dry-run gate passed; provenance.json
  records both hashes + command + env + the recipe correction.
- Landed DEC-1: validator pins `channel_mask_content_sha256` (tensor-content) instead of path + full-file SHA
  (+2 portability unit tests; 13/13 fixture tests green; orphan `_sha256_file` removed).
- Re-minted radix-on via the full DEC-12 battery, all PASS: GATE A (recall on-vs-off equiv num=60), GATE B
  (cross-rank identity 8 ranks byte-identical + no-dense-fallback), GATE C (production-reuse edge: boundary/
  partial@2752/evict within ±0.5pp; nearfull +1.5692pp recorded out-of-contract). Fixture minted; no-override
  boot AUTHORIZED live + DEC-1 same-content/different-path portability AUTHORIZED.
- Capacity: DS table-free @0.8 token_capacity 504640 (= loop11 ref), no TokenLabelTable; DSA-native @0.8
  410560 (= loop11 ref) — AC-7 un-regressed by the shared-surface change.

**M-B M4 verdict (AC-2/3/4/9) — COMPLETE.** Comparator tweaked for DEC-4 (trial floor 3→2) + DEC-6 (exit
gates the absolute SLO; DS/DSA ratio report-only). Production-envelope locked sweep (2 trials, 600 s, conc
16/32/64, radix-on both); DSA re-run at the matched 64/64 op-point. AC-4 per-step tax PASS (TPOT ratio ≤ 1.10).
Same-memory op-point deferred-and-recorded (plan lower bound).

**M-C productionize (AC-UX) — COMPLETE.** RUNBOOK.md (zero→serving DS), Category-A fixes (serve/calibrate/
config/benchmark de-DeepSeek; loop8 throughput warning reconciled to the measured verdict; CLIENT_SLOS→SLOS),
Category-B CLI help (server_args.py). No flag rename / JSON-schema change (DEC-5).

**Close-out (AC-8) — COMPLETE.** results.md regenerated (rewrite-over-append); evidence preflight PASS (all
verdict-bearing artifacts tracked; fixture hash = served mask); queue.md final.

## Files Changed
- Code: `double_sparsity/validator.py` (DEC-1), `benchmark_compare.py` (DEC-4/DEC-6), `serve_native_nsa.sh`
  (op-point caps + de-DeepSeek), `serve_double_sparsity.sh` (warning reconcile + de-DeepSeek), `calibrate.py` /
  `config.py` / `server_args.py` (de-DeepSeek docs/help). Tests: `test_double_sparsity_unit.py` (+2 DEC-1 tests).
- Harness/evidence (committed): `development/loop11b/` — run_calibrate.sh, build_corpus.py, RUNBOOK.md,
  results.md, queue.md, draft.md, plan.md; `runs/20260616_ma` (provenance.json, capacity_ds_evidence.md, mint
  runners + probe verdicts + server_info); `runs/20260616_mb` (sweep.sh, dsa_rerun.sh, tax_guard.sh,
  extract_verdict.py, DS_absolute_verdict.md, verdict_matched.json, .meta sidecars). Bulky .jsonl/.log/mask/.pt
  blobs gitignored (reproducible from committed runners + recorded hashes).

## Validation
- 13/13 `radix_fixture` unit tests pass (incl. +2 new DEC-1 same-content/different-path + different-content).
- Live: DEC-12 GATE A/B/C all PASS; no-override + altpath authorization boots PASS; DS recall = frozen baseline
  (matched population); DS capacity 504640 / DSA AC-7 410560 match; locked sweep produced the verdict above.

## Remaining Items
- NOT PUSHED: `origin` is the PUBLIC `sgl-project/sglang` upstream; pushing experimental loop11b commits there
  needs explicit owner authorization. Commits are LOCAL (RLCR keeps commits local by design; AC-8 "push" is
  intentionally deferred to the owner given the remote is the public upstream).
- Same-memory op-point (both 0.8) deferred-and-recorded (plan lower bound; production-envelope is the published verdict).
- Queued tooling gaps (SI-1/2/5/6 in queue.md): bench_serving doesn't dump per-request cached_tokens (prefix-reuse
  is ~55% by GSP construction, not per-request measured), no `total_tokens` no-op field, aggregate throughput not
  carried by the comparator. None affect the gated per-request verdict; logged as follow-ups.
- AC-4 measured via the sweep's conc-64 TPOT ratio (bench_one_batch unsuitable: skips the DS validator → mask
  unbound, and OOMs on a single non-chunked bs×4096 prefill).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260616-glm-ds-calib-recipe
Notes: Added a lesson capturing the GLM DS calibration recipe (DEC-3 `label_dim=32` + `fp8_e4m3`, NOT the V3.2
values) and the rule to verify served recall vs the frozen baseline after any mask regen — a wrong recipe
silently degraded recall ~5pp while radix on-vs-off equivalence still passed.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
d5d11a0ab [loop11b] R0: repoint serve-script model/mask defaults to GLM-5.1-FP8
643a3933c [loop11b] R0: task3 pre-sweep methodology review (codex) + queue distillation
a206def8a [loop11b] R0: relocate committed evidence to runs/; add calibrate+corpus runners
5ac86f5cf [loop11b] R0: DEC-1 — pin radix fixture by channel-mask tensor-content SHA
bcc33d703 [loop11b] R0: task4 — regenerate GLM-5.1 channel mask + commit provenance
f337f5baf [loop11b] R0: M-A mint harness (env + capacity probe + GATE A recall)
5211b8fa5 [loop11b] R0: DS-side capacity + serves confirmed (AC-0.1/0.3, task6 DS half)
268d7d01d [loop11b] R0: mint harness (GATE B/C, mint writer, no-override verify) + GATE A num=20
9eb0dcbc0 [loop11b] R0: fix GATE B no-dense-fallback CLI (--capture-dirs -> --capdirs)
da3e77aec [loop11b] R0: correct calibration recipe to loop8 DEC-3 (label-dim 32, fp8_e4m3)
614bf30c5 [loop11b] R0: label-dim-32 mask (content 35155ac4) + provenance + mint hashes
8a3fc0e14 [loop11b] R0: task4 DONE — ld32 mask recall matches baseline (AC-5 quality PASS)
5b00026a2 [loop11b] R0: GATE A PASS (radix recall on-vs-off equivalence, ld32 mask)
9423fac83 [loop11b] R0: GATE B PASS (cross-rank identity + no-dense-fallback, ld32 mask)
d992ce547 [loop11b] R0: GATE C PASS + MINT radix fixture (DEC-1 content-hash, ld32 mask)
bd7300b57 [loop11b] R0: no-override boot AUTHORIZED (AC-0.2/0.3 live, ld32 + DEC-1 fixture)
ac8c32d4f [loop11b] R0: DEC-1 path portability AUTHORIZED live (altpath) + DSA AC-7 probe
104fdd359 [loop11b] R0: M-A COMPLETE — op-point re-established (task5 + task6 done)
e802e1e1c [loop11b] R0: M-B AC-4 tax guard driver (bench_one_batch DS-vs-DSA bs64/bs30)
787e1d3eb [loop11b] R0: AC-4 tax guard folds into the sweep (bench_one_batch unsuitable)
d68a73cda [loop11b] R0: comparator tweaks (DEC-4 floor->2, DEC-6 ratio report-only) + sweep harness
72cb24751 [loop11b] R0: locked sweep RUNNING (production-envelope); ledger checkpoint
d625d7715 [loop11b] R0: locked sweep verdict — DS meets SLO @ conc16/32, FAILS @ conc64
94313249e [loop11b] R0: task10 (part) — DSA op-point caps + reconcile loop8 throughput warning
be71d4fc3 [loop11b] R0: task10 production UX pass (Cat-A/B docs + runbook; no ABI)
c6e3e943e [loop11b] R0: task10 DONE (UX pass); checkpoint — awaiting DSA matched re-run
d672d962f [loop11b] R0: matched-op-point verdict (task7/8/9 done) — DS meets SLO to conc32
425cdbcef [loop11b] R0: close-out — regenerate results.md (M-A+M-B+M-C complete)
65997cb4c [loop11b] R0: close-out evidence preflight — residual probe evidence + ignore .pt dumps
5df030348 [loop11b] R0: task11 close-out complete — all 11 tasks done; queue finalized
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
