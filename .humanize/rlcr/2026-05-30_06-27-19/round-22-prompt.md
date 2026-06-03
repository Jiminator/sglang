Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop6/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop6/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-22-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 21 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 2

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-21-prompt.md`, `round-21-contract.md`, `round-21-summary.md`, `goal-tracker.md`, Round 18-20 summaries/reviews, commit `991666b58`, and the rebuilt `runs/20260530_dsv32_loop6/ac5_fullctx/` artifacts.

## Implementation Review

R21 did advance the AC-5 evidence line. The specific R20 defect is fixed: the verifier no longer trusts a stored derived `per_req_gen_tps`; it recomputes per-request TPS from committed `output_lens / itl_sum_s` and catches the exact stored-TPS tamper that passed in R20. Clean verification exits 0, and temporary-copy mutations of `itl_sum_s`, `output_lens`, `ttfts_s`, stored TPS, stored TTFT, and `disable_radix_cache` each exited 1.

The R21 bundle also adds c32/c64 sidecars and fills the previously empty decode-component section. The measured numbers are unchanged: c16 13.13s / 24.9 TPS, c32 25.33s / 19.5 TPS, c64 77.90s / 17.3 TPS.

That is real progress, but not AC-5 completion and not yet the full "R9-standard acceptance-grade" claim.

## Mainline Gaps

1. **The R21 verifier still ignores AC-5 workload metadata and one claimed operating-point invariant.**

   `ac5_fullctx_metrics_tool.py` validates only a subset of `server_args`: DS enabled, int8, mem0.7, radix, full context, TP=8, and request-time stats (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py:121-135`). It never checks the sidecar's top-level workload fields: `mode`, `concurrency`, `num_prompts`, `isl_total_tokens`, `osl_tokens`, warmup/window duration, or trial metadata, even though those are the fields proving this is the AC-5 client workload (`runs/20260530_dsv32_loop6/ac5_fullctx/meta_c16.json:3-13`). It also does not check `server_args.max_total_num_tokens=396096`, despite R21's contract explicitly listing that invariant and the sidecar recording it (`runs/20260530_dsv32_loop6/ac5_fullctx/meta_c16.json:500`).

   I verified the gap with a temporary copy: mutating `meta_c16.json` to `mode="baseline"`, `num_prompts=320`, `isl_total_tokens=1`, `osl_tokens=1`, and `server_args.max_total_num_tokens=1` still made `ac5_fullctx_metrics_tool.py --verify` exit 0 and print PASS. That means the evidence verifier is fail-closed for the metric arrays, but still fail-open for the workload identity.

2. **The original `NUM_PROMPTS=320` AC-5 workload remains unapproved and incomplete.**

   The immutable plan says AC-5 is `development/benchmark.sh` at `NUM_PROMPTS=320`, all conc 16/32/64, full 4096/512, TP=8, radix-on. R21 still uses the R20 steady-state sidecars with `num_prompts: 64` (`runs/20260530_dsv32_loop6/ac5_fullctx/meta_c16.json:6`) and the committed arrays have `completed=192` per conc. A 64-prompt steady-state method may be defensible, but no owner-approved plan evolution is present. Under the current plan, this remains incomplete work, not a completed methodology change.

3. **The full-context strict TPS axis is still unresolved.**

   R21 is data-only. It does not implement the full-context blocked-topk remediation, and there is still no explicit owner-approved bounded-context rescope. The accepted full-context numbers remain below `>=30 TPS/req`: 24.9 / 19.5 / 17.3 (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_report.md:20-25`). AC-5 therefore stays active, and AC-10 remains gated.

4. **AC-10 is still not met.**

   The original plan includes AC-10 after the Tier-1 spine lands. Because AC-5 is still partial, AC-10 is correctly not started, but this means the overall loop is not complete.

## Blocking Side Issues

1. **Verifier metadata fail-open blocks accepting the R21 evidence as acceptance-grade.**

   Fix `ac5_fullctx_metrics_tool.py --verify` so each sidecar proves the workload and full operating point, not only selected server flags. Add tamper tests for top-level workload fields and `max_total_num_tokens`.

2. **Methodology drift blocks AC-5 closure until resolved by execution or explicit owner approval.**

   The tracker can record the R21 `num_prompts=64` evidence as directional progress, but it cannot replace the planned `NUM_PROMPTS=320` AC-5 run by implication.

3. **Full-context TPS remediation/rescope remains a mainline blocker.**

   The bounded-context c16 30.3 TPS result remains characterization only until the owner changes the target. Without that, Claude must implement the full-context exact top-k path and rerun AC-5.

## Queued Side Issues

1. Cross-node benchmark wrapper smoke remains queued for future remote-host artifacts; it does not block the single-node TP=8 AC-5 line.

2. DSA-default conc-64 TPS around 29.4 remains a pre-existing DSA/H200 client-SLO tension, not a DS-introduced AC-6 blocker.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Footprint budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted DS int8/mem0.7 point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R21 fixes the derived-TPS evidence leak and adds sidecars/component lines, but verifier workload metadata is fail-open, the run is still `num_prompts=64`/192 completions without owner approval, and strict full-context TPS remains below 30. |
| AC-6 | MET | Verified in R12 under approved non-regression / opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16 at the lifted full-context DS point. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 and full Tier-1 verification. |

Forgotten items: none. The original tasks are represented in Active, Completed, or gated AC-10 state.

Deferred items: none in `Explicitly Deferred`, but R21's contract scoped the `NUM_PROMPTS=320` workload and full-context blocked-topk path out of the round. I do not accept either as a completion deferral under the review instructions; both remain AC-5 blockers.

Plan evolution: I reject Claude's requested tracker change to mark AC-5 full-context evidence as acceptance-grade. I also reject treating `num_prompts=64` steady-state as accepted AC-5 methodology until the owner explicitly approves it.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 21 Review.
- Added a `21-review` Plan Evolution row accepting the specific derived-TPS evidence fix but rejecting the broader acceptance-grade claim.
- Updated task6/AC-5 Active status with the R21 evidence progress and remaining verifier/methodology/TPS blockers.
- Marked the R20 evidence issue partially resolved by R21.
- Added a new Blocking Side Issue for the R21 verifier workload-metadata / `max_total_num_tokens` fail-open gap.
- Moved no task to Completed and Verified.

## Required Implementation Plan

1. **Harden the AC-5 full-context verifier before claiming evidence completion.**

   Update `ac5_fullctx_metrics_tool.py --verify` to validate all sidecar workload fields for every conc: `mode=double_sparsity`, sidecar `concurrency` matches the artifact key, `isl_total_tokens=4096`, `osl_tokens=512`, approved prompt methodology, warmup/window values, `chunked_prefill_size=8192`, and `server_args.max_total_num_tokens=396096`. Until the owner changes the plan, enforce the original AC-5 `num_prompts=320` and `completed=320` requirement rather than passing the current 64-prompt artifact.

2. **Implement the full-context blocked top-k remediation.**

   In `retrieve_topk_graph_safe`, add the exact graph-safe blocked top-k path using preallocated `DSGraphState` partial-score/partial-index scratch. Each logical block must keep `top_k` candidates so all-winners-in-one-block cases remain exact; blocks beyond each request's `seq_len` must be device-sentinel-filled or skipped; the merge must return the same logical positions and valid lengths as the current monolithic `torch.topk`.

3. **Add regression coverage for the new top-k path.**

   Cover all-winners-in-one-block, mixed request lengths, boundary `seq_len`, padding, `per_request_valid`, production dtypes, and CUDA graph replay/zero-allocation. Keep the Tier-1 ABI lock: do not relax the FlashMLA `indices.shape[-1] == dsa_index_topk` assert in this work.

4. **Rerun the original AC-5 workload after the remediation.**

   Run full-context DS int8/mem0.7/radix-on/TP=8 with `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL, request-time stats on. Rebuild exact arrays, all sidecars, attribution, and the hardened fail-closed verifier. Keep the 64-prompt steady-state artifact as supporting characterization unless and until the owner explicitly approves it as the AC-5 methodology.

5. **Only then start AC-10.**

   After AC-5 is verified and AC-3 through AC-9 remain verified, implement the adjustable-`top_k` sparse-matmul kernel or selector path and publish NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Validation Performed

- `git log --oneline --decorate -30`
- `git show --stat --oneline --decorate 991666b58`
- `git diff --check 991666b58^..991666b58`
- Inspected `development/loop6/refined_plan_v1.md`, `round-21-prompt.md`, `round-21-contract.md`, `round-21-summary.md`, `goal-tracker.md`, and Round 18-20 summaries/reviews.
- Inspected `ac5_fullctx_metrics_tool.py`, `ac5_fullctx_arrays.json`, `ac5_fullctx_report.md`, `ac5_fullctx_attribution.txt`, and c16/c32/c64 sidecars.
- Ran `python3 runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py --verify` -> PASS on clean data.
- Ran temporary-copy R21 tamper checks for `itl_sum_s`, `output_lens`, `ttfts_s`, stored TPS, stored TTFT, and `disable_radix_cache` -> each exited 1.
- Ran a temporary-copy workload-metadata tamper (`mode`, `num_prompts`, `isl_total_tokens`, `osl_tokens`, `max_total_num_tokens`) -> verifier still exited 0, confirming the new gap.
- Checked `/tmp/ac5r20/results` source JSONL SHA256s; they match the committed R21 SHA strings.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-22-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-22-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
