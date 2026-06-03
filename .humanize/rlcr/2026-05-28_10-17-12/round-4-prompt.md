Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop5/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop5/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-4-contract.md

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

Goal Alignment Summary:
ACs: 5/11 addressed | Forgotten items: 0 | Unjustified deferrals: 8

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`, then the round-3 prompt/contract/summary, round 0/1/2 summaries and review results, `goal-tracker.md`, commits `6f95a9711`, `bc534da7c`, `76eef9c80`, the changed DS/scheduler code, serve/benchmark launchers, and Round-3 evidence under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 254 passed, 24 warnings in 11.79s
```

Artifact checks:

- `ac0_capture_positive.json`: `double_sparsity_radix_capture.prompt_len=11`, 11 `per_token_slot_sha` entries, 61 `per_layer_written_all_true` entries all true, no capture error.
- `ac1_server_info.json`: status ready, cluster `model_path`, TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `enable_double_sparsity=true`, `disable_radix_cache=true`, `attention_backend=dsa`; no `_`-prefixed runtime attrs in top-level or `internal_states[0]`.
- `ac1_generate.json`: non-empty coherent text and no capture key, covering the capture-disabled negative.
- `ac1_invalid_mask_rejection.md`: fail-closed `DoubleSparsityChannelMaskMissing` from `check_server_args` before model load.

## Mainline Gaps

1. **The original plan is still incomplete; Claude's "Remaining Items" are not acceptable deferrals.**

   Round 3 closed the two targeted AC-0/AC-1 gaps, but the original plan still requires task7/task8/task9 and task11-task15. That means the Smoke MVP is still missing the DS+DSA smoke benchmark pair, smoke comparator, and AC-Q quality smoke; the loop4-compatible tier is still missing AC-10 radix flip, AC-1b chunked-prefill probe, AC-11 sweep/comparator, AC-12 full quality, and the final evidence bundle.

   Required correction: keep these tasks active and complete them in plan order. Do not treat TIER-1 or TIER-2 as optional follow-up work.

2. **The next benchmark/comparator work is blocked by launcher parity gaps.**

   The plan says DEC-6 pins both serve scripts to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`, but both launchers still default to the HF id: `development/serve_double_sparsity.sh:29` and `development/serve_native_nsa.sh:28`. Round-3 AC-1 evidence is valid because the live endpoint shows the cluster path, but task7/task9 can silently drift if the scripts are run as documented without a manual override.

   The TIER-1 smoke also requires radix-off on both DS and DSA, and `benchmark_compare.py` refuses `disable_radix_cache` mismatches. DS passes `--disable-radix-cache`, but `development/serve_native_nsa.sh:21-24` explicitly leaves radix on and has no smoke knob to disable it. This must be fixed before publishing task7/task8 artifacts.

## Blocking Side Issues

1. **#D: serve launchers still default to the HF model id.**

   Evidence: `development/serve_double_sparsity.sh:29` and `development/serve_native_nsa.sh:28`.

   Required fix: change both launcher defaults to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` while preserving env override support for deliberate local runs. After the edit, capture fresh DS and DSA `/get_server_info` sidecars proving the cluster path before any benchmark or quality artifact is accepted.

2. **#E: DSA baseline launcher cannot produce the required radix-off TIER-1 smoke.**

   Evidence: `development/serve_native_nsa.sh:21-24` says radix cache is intentionally on, while AC-8/AC-9 require radix-off both sides for smoke; `development/benchmark_compare.py` includes `disable_radix_cache` in the required match set.

   Required fix: add one DSA launcher path for TIER-1 smoke that passes `--disable-radix-cache`, and verify both DS and DSA benchmark `.meta.json` sidecars report `disable_radix_cache=true`. Preserve the later AC-10/AC-11 path so the final sweep runs radix-on on both sides after the DS radix flip.

## Queued Side Issues

1. `calibrate.py` still has stale operator-facing recipe text after the native-FP8 sharded load redesign. This is non-blocking for task7/task8 because the validated mask and provenance already exist, but it should be cleaned up before the next calibration handoff.

## Verified Round-3 Work

No high-signal defects were found in the Round-3 AC-0/AC-1 fixes themselves.

- `_resolve_req_to_token_for_capture` fixes the production `ForwardBatch` shape gap, and the added regression covers the no-`req_to_token_pool` case.
- The dtype-safe SHA path works through byte views, and the full DS unit suite is green.
- Filtering `_`-prefixed runtime attrs from `get_internal_state` fixes the `/get_server_info` IPC crash without dropping public config fields required by the endpoint.

## Goal Alignment

Tracker updates applied during review:

- Moved task2/AC-0 and task5/AC-1 to Completed and Verified.
- Added #D and #E as Blocking Side Issues for task7/task8.
- Kept task7, task8, task9, task11, task12, task13, task14, and task15 active.
- Did not add any items to Explicitly Deferred; the remaining tasks are incomplete plan work, not accepted deferrals.

Plan evolution is valid for the Round-3 fixes: AC-0 and strict AC-1 advanced with hardware evidence. The summary's remaining TIER-1/TIER-2 items are not a plan change; they are still required work.

## Directive Implementation Plan

1. Patch `development/serve_double_sparsity.sh` and `development/serve_native_nsa.sh` so both default `MODEL_PATH` to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`.
2. Patch `development/serve_native_nsa.sh` to support the TIER-1 smoke baseline with radix cache disabled, then launch DSA smoke with `disable_radix_cache=true`. Keep the later AC-11 path able to run radix-on after AC-10.
3. Reboot DS and DSA from the scripts and save fresh `/get_server_info` artifacts proving cluster model path, TP=8, `fp8_e4m3`, page size 64, matching radix setting, and matching Option B knobs.
4. Run task7: DS and DSA smoke benchmarks at conc 16/32/64 with `TRIALS=1`, shortened `MEASUREMENT_WINDOW_S`, radix-off both sides, and valid `.meta.json` sidecars. Reject any JSONL whose observed duration is below the configured window.
5. Run task8: generate `mvp_compare.md` from those smoke JSONLs and require radix parity to pass.
6. Run task9: capture DSA references first, then DS, and require all four AC-Q gates.
7. Implement task11 AC-10 with no environment override: pass both radix fixtures, wire the pre-validation fixture-passed state through real launcher/server args or state-file plumbing, remove DS `--disable-radix-cache`, and boot radix-on.
8. Run task12 AC-1b before the sweep, then task13 AC-11, task14 AC-12, and task15 evidence bundle in that order.

Original plan work remains pending.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-4-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-4-summary.md

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
