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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-4-contract.md

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
# Round 3 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 4/10 addressed | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, Round 0/1/2 summaries and reviews, `round-3-prompt.md`, `round-3-contract.md`, `round-3-summary.md`, `goal-tracker.md`, commits `5d8e47fb3` and `8a05b1688`, the launcher/test diff, the NIAH artifacts, the fp16 Loop-5 baseline artifacts, and the local DS boot log that produced the committed boot excerpt.

## Mainline Gaps

1. **No Round-3 mainline gap found for the contracted objective: AC-3.1 is now evidenced.**

   Evidence:
   - `development/serve_double_sparsity.sh` now defaults `SIGNATURE_DTYPE=fp16`, includes `"signature_dtype": "${SIGNATURE_DTYPE}"` in `DS_CONFIG`, and logs the selected dtype.
   - The launcher regression test exercises the real shell script through a stub `python3`: default config parses as fp16 and `SIGNATURE_DTYPE=int8` parses as int8.
   - The TP=8 DS boot log shows the Loop-5 mask (`content_sha256=7b3207cae888`), `double_sparsity_config=... "signature_dtype": "int8"`, and `token_label_table: 0.87 GB/rank ... dtype=torch.int8 scales=float16`.
   - The committed NIAH artifacts compare as: 1024 100=100, 1536 100=100, 4096 85≥75, 16384 5=5, 65536 0=0 with the same DS unservable 64K admission error as the fp16 Loop-5 baseline.

2. **The Loop-6 original plan is still incomplete after Round 3.**

   This is not a regression in Round 3, but it prevents a final `COMPLETE`: AC-4, AC-5, AC-6 hardware proof, AC-7, AC-8, AC-9, and gated AC-10 remain pending in the tracker. Claude should proceed to AC-4 next; do not claim the loop done or shippable.

## Blocking Side Issues

None. The Round-2 blocker, “standard DS launcher cannot select compact int8 labels,” is resolved by `5d8e47fb3` and verified by the targeted launcher tests plus the TP=8 boot proof.

## Queued Side Issues

None high-signal. I did not promote the ignored local `niah_pytest.log` to a finding because the committed JSONs, boot proof, and non-regression report are sufficient for the AC-3.1 evidence; future artifact bundles can avoid referencing ignored `.log` files or store them as `.txt`.

## Goal Alignment Check

AC-1 and AC-2 remain verified. AC-3 is now verified: implementation, byte reduction, fp16 default/int8 opt-in, CUDA-graph safety, scale-sidecar consumers, decode-scoring overhead, and real-mask NIAH non-regression are all evidenced. AC-6 is partially addressed through dev checks (DSA-default/no-table and opt-in config), but the hardware product proof remains task7. AC-4 and AC-5 have not started. AC-7, AC-8, and AC-9 remain tracked and pending. AC-10 remains correctly gated behind the full Tier-1 spine.

Forgotten items: none. Deferred items: none. The Round-3 plan evolution is valid: the hardware premise correction is supported by the successful TP=8 serve and the previous real-mask NIAH deferral was correctly removed.

## Directive Implementation Plan

1. Run AC-4 next with the compact path:
   - Boot DS through `development/serve_double_sparsity.sh` with `SIGNATURE_DTYPE=int8`.
   - Sweep `MEM_FRACTION_STATIC=0.6 -> ... -> 0.8`.
   - For each point, record `/get_server_info`, `max_total_num_tokens`, full NVML and torch reserved/allocated/residual HBM accounting, table/scales/written/scratch/FlashMLA/CUDA-graph/headroom components.
   - Run a sustained long `/generate`; pass only with no generation-time OOM and no monotonic memory growth.

2. After AC-4, run AC-5 exactly as planned:
   - `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL / ~55% cache, radix-on proven by server args and sidecars.
   - Write `client_slo_report.md` with strict `<22.0` TTFT numbers, `>=30 TPS/req`, all-trials/worst-trial disclosure, and measured admission-wait vs prefill-compute attribution.

3. Complete remaining Tier-1 items without letting them replace the spine:
   - AC-6 hardware DSA-default SLO/no-DS-table proof and DS opt-in proof.
   - AC-9 real-token-count `within_budget` harness edit plus live re-run.
   - AC-7 AC-11 re-sweep and AC-8 64K servability characterization at the lifted point.
   - AC-10 only after AC-3 through AC-9 are complete.

## Tracker Update

Updated the mutable tracker section:
- Moved task1/task2 out of Active bookkeeping and kept them in Completed and Verified.
- Moved task3/task4 to Completed and Verified after this review.
- Left task5 as the next active mainline item and clarified its AC-4 evidence requirements.
- Left task7 active because AC-6 hardware/product validation is still pending.

## Validation Performed

- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q -k 'ServeDoubleSparsityLauncherSignatureDtype'` -> 2 passed, 279 deselected.
- Artifact comparison script over `runs/20260530_dsv32_loop6/real_mask_niah_int8/ac12_niah_*.json` vs `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_*.json` -> non-regression PASS.
- `git diff --check 5d8e47fb3^..HEAD` -> clean.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-4-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-4-summary.md

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
