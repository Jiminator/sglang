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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-2-contract.md

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
# Round 1 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 2/11 addressed | Forgotten items: 0 | Unjustified deferrals: 0

Review scope: `development/loop5/refined_plan_v1.md`, `round-1-prompt.md`,
`round-1-contract.md`, `round-0-summary.md`, `round-0-review-result.md`,
`goal-tracker.md`, commits `c99ed3644` / `610f364c9`, changed calibration code,
unit tests, and AC-4 artifacts under `runs/20260528_dsv32_mvp/`.

Verification runs:

```bash
python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 251 passed
```

```bash
python - <<'PY'
from sglang.srt.layers.attention.double_sparsity.channel_mask import load_channel_mask
m = load_channel_mask('/models/dsv32-fp8-channel-mask.safetensors')
print(m.dtype, m.page_size, m.label_dim, m.head_dim, tuple(m.channel_selection.shape), m.content_sha256)
PY
# fp8_e4m3 64 16 128 (61, 128, 16) 7b3207cae888c141173703384bfd7c8974b7adb64b1fddbdacac3ab26c7d6ac6
```

## Implementation Review

No high-signal implementation defects were found in the Round 1 AC-4 code path.

The loader now resolves `deepseek_v32` by reading the raw config and remapping it
to `deepseek_v3` before `AutoModelForCausalLM.from_pretrained(..., config=...,
torch_dtype="auto", device_map="auto")` in
[calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:281).
The dry-run guard logs a structured dtype/device report and fail-closes bad FP8
CUDA loads via `_enforce_dry_run_placement` at
[calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:235).
Regressions cover the remap, call shape, rejected off-GPU/single-GPU/upcast
loads, and the DeepGEMM skip path in
[test_double_sparsity_unit.py](/sgl-workspace/sglang/test/registered/unit/layers/attention/test_double_sparsity_unit.py:2393).

The hardware evidence matches AC-4: dry-run #5 logs FP8 presence
(`torch.float8_e4m3fn=604`), no CPU/disk/meta placement, all 8 GPUs in the
device histogram, and hooks fired on all 61 layers. The full calibration wrote
`/models/dsv32-fp8-channel-mask.safetensors`, and `load_channel_mask()` validates
metadata and index bounds in `mask_validation.txt`.

## Mainline Gaps

1. **The original plan is not complete: M2/M3 work remains active.**

   Round 1 completes AC-4 only. The root-blocker mask now exists, but the
   original plan still requires task2 and task5-task15: AC-0 hardware capture
   probe, DS boot smoke, genuine sparsity proof, DS/DSA smoke benchmarks, smoke
   comparator, paired quality smoke, CUDA-graph evidence, radix flip,
   chunked-prefill probe, AC-11 sweep, AC-12 quality gate, and the final evidence
   bundle. These are correctly active in the tracker and must not be treated as
   optional or complete.

2. **AC-0 remains only code/regression complete until the first DS `/generate` probe.**

   Task1 was verified in Round 0, but the AC-0 positive hardware test still
   requires `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` plus `/generate` returning
   non-empty `meta_info["double_sparsity_radix_capture"]` with
   `per_token_slot_sha` populated and `per_layer_written_all_true=True`. This is
   now unblocked by AC-4 and should run during the first DS boot.

## Blocking Side Issues

None open after this review.

Blocking Side Issues #1, #2, and #3 are resolved by the Round 1 implementation
and hardware artifacts:

- #1 HF cannot load `deepseek_v32`: resolved by the `deepseek_v3` remap.
- #2 dry-run only logged placement: resolved by fail-closed placement validation.
- #3 DeepGEMM hub-kernel failure: resolved for calibration by forcing the
  `finegrained-fp8` Triton fallback.

## Queued Side Issues

1. **Stale calibration script production recipe.**

   [calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:15)
   still documents the old DeepSeek-V3.2 production recipe with `--tp 1` and says
   `--dtype` is the model loading dtype. The proven Round 1 command records
   `--tp 8`, and `_load_calibration_model` now always uses `torch_dtype="auto"`
   with `device_map="auto"`. This does not block M2, because
   `calibration_provenance.md` records the exact command for the validated mask,
   but the script docstring should be corrected before the next calibration
   handoff.

## Goal Tracker Updates Applied

I updated only the mutable tracker section:

- Removed task3 and task4 from `Active Tasks`.
- Marked task3 and task4 as verified in `Completed and Verified`.
- Updated the Round 1 Plan Evolution row so AC-4 is complete rather than
  “in progress.”
- Added the stale calibration recipe as a queued side issue.

## Directive Implementation Plan

Execute the remaining original plan in this order. Do not start benchmark or
quality artifacts before the first DS boot is proven healthy.

1. Boot DS at TP=8 with `MODEL_PATH=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`,
   the validated mask, and `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`. Save
   `/get_server_info`, `/generate` text, and the
   `double_sparsity_radix_capture` payload. This completes task2 and task5.
2. Run the long-prompt genuine-sparsity check and require
   `0 < sparsity_rate < 1` plus `dense_fallback == 0` (task6).
3. Record regular CUDA-graph capture/replay status from that first DS boot,
   distinct from disabled piecewise CUDA graph (task10).
4. Run the Tier 1 smoke evidence: DS + DSA smoke benchmarks with radix off on
   both sides and explicit smoke labels (task7), smoke comparator (task8), and
   sequential paired quality smoke with all four gates (task9).
5. Implement the no-env-override radix flip and pass both fixtures (task11), then
   run the chunked-prefill probe (task12), AC-11 radix-on sweep/comparator
   (task13), and AC-12 full quality gate (task14).
6. Assemble the final evidence bundle only after task8, task9, task10, task13,
   and task14 artifacts exist (task15).

No COMPLETE: original plan work remains pending.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-2-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-2-summary.md

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
