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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-3-contract.md

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
# Round 2 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 4/10 addressed | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline, taste-review guidance, Round 0/1 summaries and reviews, `round-2-prompt.md`, `round-2-contract.md`, `round-2-summary.md`, `goal-tracker.md`, the R2 commit diff, changed production/test files, the decode microbench artifact, the launcher, and the NIAH harness.

## Mainline Gaps

1. **AC-3.1 / task4 is still incomplete: the required real-mask NIAH non-regression has no artifact and was moved to `Explicitly Deferred`.**

   Evidence:
   - The plan makes real-mask NIAH part of AC-3.1, not an optional AC-4 extra: `development/loop6/refined_plan_v1.md:47`, `:132`, `:183`, and DEC-8 at `:263`.
   - Task ordering is explicit: task5/AC-4 depends on task4 (`development/loop6/refined_plan_v1.md:183-184`).
   - The only Loop 6 R2 acceptance artifact under `runs/20260530_dsv32_loop6/` is `decode_scoring_microbench.md`; there is no int8-vs-fp16 real-mask NIAH artifact.
   - Claude's summary and tracker explicitly defer the item to the next cluster round.

   Impact: the compact int8 path still has no real-mask recall non-regression against the fp16 Loop-5 DS baseline. Synthetic top-k overlap and the decode microbench are useful, but they do not prove the core AC-3.1 quality gate. Per the review prompt, this deferral is incomplete work and cannot be accepted as AC-3 completion.

## Blocking Side Issues

1. **The standard DS launcher still cannot select the compact int8 table, so the next AC-4/NIAH run can silently validate fp16 labels.**

   Evidence:
   - `development/serve_double_sparsity.sh:53-54` builds `DS_CONFIG` with `top_k`, `page_size`, `channel_mask_path`, and `device_buffer_size` only. It never includes `signature_dtype`.
   - The launch log at `development/serve_double_sparsity.sh:75-85` also does not print the signature dtype.
   - The config default is fp16 (`python/sglang/srt/layers/attention/double_sparsity/config.py:43`), so running the documented script as-is validates the full-precision table, not the compact table.
   - The plan's AC-4 workflow tells Claude to sweep via `serve_double_sparsity.sh` (`development/loop6/refined_plan_v1.md:133`) and AC-4 is defined as validation **with the compact table** (`:58`).

   Impact: even if Claude gets the TP=8 cluster, the next hardware run is easy to mis-run: `MEM_FRACTION_STATIC=... bash development/serve_double_sparsity.sh` still boots fp16 DS. That would invalidate both the real-mask NIAH non-regression and the AC-4 mem-fraction/no-OOM evidence.

   Required fix:
   - Add `SIGNATURE_DTYPE="${SIGNATURE_DTYPE:-fp16}"` to `development/serve_double_sparsity.sh`.
   - Include `"signature_dtype": "${SIGNATURE_DTYPE}"` in `DS_CONFIG`.
   - Echo `signature_dtype` in the launcher log.
   - Run the next DS cluster server as `SIGNATURE_DTYPE=int8 ... bash development/serve_double_sparsity.sh`.

## Queued Side Issues

No separate queued side issues. The remaining findings block the AC-3 -> AC-4 handoff rather than being optional cleanup.

## Goal Alignment Check

AC-1 and AC-2 remain verified. AC-3 advanced: the scale-sidecar proof/sanity bug from Round 1 is fixed, and the decode-scoring microbench passed on H200. AC-3 is still not complete because the real-mask NIAH non-regression is missing. AC-6 advanced: the CPU DSA-default/no-table regression is present, but the hardware product proof remains task7 and is still pending. AC-4 through AC-10 remain pending; AC-10 is still properly gated.

Forgotten items: none. The tracker contains all original-plan tasks. The only rejected deferral is the real-mask NIAH item: the hardware dependency is real, but it still blocks AC-3/task4 and must not live in `Explicitly Deferred` as accepted completion.

## Directive Implementation Plan

1. Fix the launcher before any cluster run:
   - Patch `development/serve_double_sparsity.sh` with `SIGNATURE_DTYPE`, include it in `DS_CONFIG`, and log it.
   - Add a small unit/static test or shell check that `SIGNATURE_DTYPE=int8` produces a config containing `"signature_dtype": "int8"` and that the default remains `"fp16"`.

2. Run the real-mask NIAH non-regression on the TP=8 cluster before AC-4:
   - Boot DSA reference on port 30001 using the existing DSA launcher.
   - Boot DS on port 30000 with the Loop-5 mask and compact labels: `SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.6 TP_SIZE=8 CHANNEL_MASK_PATH=/models/dsv32-fp8-channel-mask.safetensors bash development/serve_double_sparsity.sh`.
   - Record `/get_server_info` and the DS server log excerpt proving `signature_dtype=int8`.
   - Run `DS_BASE_URL=http://localhost:30000 DSA_BASE_URL=http://localhost:30001 AC12_NIAH_NUM_PROMPTS=20 PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py -v -k niah`.
   - Copy the generated `development/results/ac12_niah_*.json` files into `runs/20260530_dsv32_loop6/real_mask_niah_int8/`.
   - Write `runs/20260530_dsv32_loop6/real_mask_niah_nonregression.md` comparing int8 DS recall against the fp16 Loop-5 DS baseline artifacts in `runs/20260528_dsv32_mvp/ac12_results/`. Pass only if each comparable length has `int8_ds_recall_pct >= fp16_loop5_ds_recall_pct` and no new DS unservable error where the fp16 baseline served.

3. Only after that artifact passes, proceed to AC-4:
   - Sweep `SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.6 -> ... -> 0.8`.
   - Log full NVML/torch residual HBM accounting, `/get_server_info`, and a sustained long `/generate` with no OOM or monotonic growth.

## Tracker Update

Updated the mutable section of `goal-tracker.md`:
- Rejected the Round 2 `Explicitly Deferred` classification for real-mask NIAH.
- Left task4 active/partial with real-mask NIAH pending.
- Added a blocking issue for the missing `SIGNATURE_DTYPE` launcher surface.
- Updated task5 notes so AC-4 cannot start without the compact launcher path and NIAH gate.

## Validation Performed

- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q -k 'CompactScaleSidecarConsumers or (CompactInt8Signatures and not cuda and not graph_safe and not decode_scoring_overhead)'` -> 17 passed, 262 deselected.
- `python -m pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -q` -> 13 passed.
- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q -k 'decode_scoring_overhead_within_tps_budget'` -> 1 passed, 278 deselected.
- `git diff --check 84d3410b9..e85cd2564` -> clean.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-3-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-3-summary.md

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
