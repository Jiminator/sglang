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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-13-contract.md

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
# Round 12 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 7/10 addressed, 6/10 met | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-12-prompt.md`, `round-12-contract.md`, `round-12-summary.md`, `goal-tracker.md`, Round 9-11 summaries/reviews, commits `d0cc9fdc9` and `f9bc51b13`, the AC-6 product-proof artifacts, benchmark scripts, `development/CLIENT_SLOS.md`, `development/loop6/runbook.md`, and `development/roadmap.md`.

## Implementation Review

Round 12 advanced the AC-6 cleanup. The benchmark wrappers now thread `--host "${HOST}"` to `sglang.bench_serving` in both scripts (`development/benchmark.sh:69-72`, `development/benchmark_baseline.sh:69-72`), so the load path no longer silently defaults away from the `/get_server_info` host. The AC-6 DSA-default artifact is now recomputable: `dsa_slo_metrics_tool.py --verify` recomputes P99 TTFT and per-request TPS from `dsa_slo_arrays.json`, exits 0 on clean data, and fails closed under temporary tampering of both a stored percentile and an array length (`dsa_slo_metrics_tool.py:60-86`).

The DS opt-in / DSA-default toggle evidence is also coherent: the captured server info has DS opt-in `enable_double_sparsity=True`, `signature_dtype=int8`, radix-on, and a fixture artifact, while DSA-default has `enable_double_sparsity=False`, `double_sparsity_config=None`, radix-on, and no table; the boot excerpts show all 8 DS ranks allocate the int8 table and the DSA boot has 0 table lines (`runs/20260530_dsv32_loop6/ac6_product_proof/get_server_info_keys.json`, `ds_table_boot_excerpt.txt`, `dsa_notable_boot_excerpt.txt`).

I accept AC-6 as verified only under the recorded R12 user decision: AC-6 is graded as a non-regression / opt-in product test, not as a literal DSA-default `>=30 TPS` gate. The artifact still honestly records the conc-64 DSA-default TPS miss: fresh DSA is 29.4 TPS and Loop-5 baseline is 29.5 TPS (`ac6_optin_dsa_default_product.md:46-61`, `:73-84`). That is not DS-introduced, but it remains a queued client-SLO-vs-DSA tension.

## Mainline Gaps

1. **The original Loop-6 work remains incomplete.**

   AC-5 is still directional-only, not strict/shippable: the tracked DS client-SLO artifact still misses conc-32/64 TTFT and all-conc TPS. AC-7, AC-8, and gated AC-10 are still pending. These are active plan tasks, so this round cannot end the loop or output `COMPLETE`.

2. **The Round 12 wrapper smoke was deferred despite being a success criterion.**

   The round contract required "`--host` added ...; cross-node smoke proves same-host targeting" (`round-12-contract.md:42-47`). The summary says no servers were booted and only static verification plus the earlier direct `bench_serving --host node1` banner were used (`round-12-summary.md:43-46`). The code fix is real, but the script-level wrapper smoke has not been run.

## Blocking Side Issues

1. **Script-level cross-node host-targeting smoke is still required before AC-7.**

   This does not invalidate AC-6's R11 direct-`--host` evidence, but it blocks safe publication of any new scripted cross-node AC-7 sweep. The next hardware round must first run a tiny wrapper smoke with `HOST=<remote>` and capture both the `bench_serving` readiness banner naming that host and the matching `/get_server_info` sidecar.

2. **Strict DS client SLO still blocks the Ultimate Goal.**

   AC-5 remains the mainline SLO blocker. Do not let AC-7/AC-8 hardening or the now-verified AC-6 product property hide the fact that DS is not shippable under the strict `P99 TTFT < 22 s` and `>=30 TPS/req` all-concurrency criterion.

## Queued Side Issues

1. **DSA-default conc-64 TPS is below the client threshold.**

   This is queued, not blocking AC-6 under the R12 user decision: DSA-default reproduces the pre-DS baseline and the miss is not introduced by DS. It should remain visible if the client later requires strict DSA-default `>=30 TPS/req` at conc 64.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, sidecar consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | Evidence and attribution are verified; strict DS SLO still fails. |
| AC-6 | MET | Verified in this review under the R12 user-approved non-regression/opt-in semantics. Literal DSA-default conc-64 `>=30 TPS` remains queued as a pre-existing DSA tension. |
| AC-7 | NOT MET | 3-trial DS+DSA lifted-point re-sweep pending; must start with the missing wrapper host smoke. |
| AC-8 | NOT MET | Lifted ~70K-token servability probe pending. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in Round 10 review. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original task is represented in Active, Completed, or the gated AC-10 path. Deferred items: one unjustified round-contract deferral, the cross-node wrapper smoke; I corrected the tracker so it remains blocking for AC-7 rather than silently resolved.

## Required Implementation Plan

1. Before any AC-7 artifact is trusted, boot the intended remote DSA or DS server and run a minimal wrapper smoke through `development/benchmark_baseline.sh` or `development/benchmark.sh` with `HOST=<remote>`, a single concurrency, one trial, and a very short measurement window. Capture the wrapper stdout readiness line proving `bench_serving` targeted `http://<remote>:<port>` and the generated `.meta.json` server-info sidecar from the same host. If either side disagrees, stop and fix the wrapper before sweeping.
2. Complete AC-7 exactly after the smoke passes: 3 DS+DSA trials at the lifted point, radix-on on both sides, per-side mem-fraction consistency recorded, refreshed `ac11_resweep.md` / `ac11_analysis.md`, and no hidden achieved-concurrency deficit.
3. Complete AC-8 next: run the lifted ~70K-token `/generate` probe and record either HTTP 200 with served `max_total_num_tokens` and no instability, or a characterized new ceiling with the server-side rejection reason.
4. Return to the AC-5 strict SLO blocker with the AC-7 data in hand. The next AC-5 remediation must target the measured throughput/scheduling bottleneck, not evidence churn: keep the client SLO and Option B constraints fixed, make the smallest scheduling/decode/operating-point change that can restore both `P99 TTFT < 22 s` and `>=30 TPS/req`, then publish exact recomputable benchmark evidence.
5. Start AC-10 only after AC-3 through AC-9 are verified. Then implement the adjustable-`top_k` kernel or learned selector path and record NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 12 Review.
- Added an R12-review plan-evolution row.
- Moved AC-6/task7 to Completed and Verified under the user-approved non-regression semantics.
- Kept AC-5, AC-7, AC-8, and gated AC-10 active.
- Changed the cross-node host issue from fully resolved to partial: code fixed, script-level wrapper smoke still blocking AC-7.
- Left the DSA-default conc-64 TPS miss queued, not blocking AC-6.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline d0cc9fdc9 f9bc51b13`
- Inspected `development/benchmark.sh` and `development/benchmark_baseline.sh`.
- Inspected `round-12-contract.md`, `round-12-summary.md`, `goal-tracker.md`, and Round 9-11 summaries/reviews.
- Inspected `ac6_optin_dsa_default_product.md`, `get_server_info_keys.json`, full server-info captures, boot excerpts, `dsa_default_matches_loop5_baseline.txt`, `dsa_default_slo_np64.txt`, and `dsa_slo_metrics_tool.py`.
- Ran `python3 runs/20260530_dsv32_loop6/ac6_product_proof/dsa_slo_metrics_tool.py --verify`.
- Temporary-copy tamper tests: modified stored P99 TTFT and dropped a TTFT element; both verifier runs exited 1.
- Compared Loop-5 DSA baseline server-info fields and t1 JSONL metrics against the fresh AC-6 DSA-default evidence.
- `git diff --check 0e1ce974d..HEAD`

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-13-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-13-summary.md

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
