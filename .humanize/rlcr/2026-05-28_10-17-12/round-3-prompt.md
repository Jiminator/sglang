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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-3-contract.md

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

Goal Alignment Summary:
ACs: 5/11 addressed | Forgotten items: 0 | Unjustified deferrals: 1

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`, then the round prompt/contract, round 0/1 summaries and review results, round 2 summary, `goal-tracker.md`, the round-2 commit diff, DS code hot spots, and the evidence under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 253 passed, 24 warnings in 11.59s
```

## Mainline Gaps

1. **Strict AC-1 is not complete: the required live server-info artifact is missing.**

   The plan requires `/get_server_info` to show DS enabled, TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, expected radix setting, and the cluster model path (`development/loop5/refined_plan_v1.md:38-40`; round-2 contract also requires saving it). The evidence has `ds_boot_knobs_AC1.json`, but that is a server-args scrape/log-derived artifact, not the required live endpoint response. The endpoint exists and delegates to `/server_info` (`python/sglang/srt/entrypoints/http_server.py:621-632`), so an empty/flaky `/get_server_info` response is an evidence gap to fix, not a reason to mark AC-1 fully verified.

   Required correction: run the DS server again, capture `/server_info` or `/get_server_info` with `curl --fail`, save the full JSON as an AC-1 artifact, and assert the required fields. Also capture the missing/invalid mask rejection log required by AC-1 if no existing artifact proves it verbatim. Until then, task5 stays **partial**, not completed.

2. **The original plan is still materially incomplete; do not stop at the smoke boot.**

   Round 2 advanced the DS-on path, but these original plan items remain active or blocked: task2/AC-0 hardware capture, task7-task8/AC-8+AC-9 smoke DS+DSA benchmarks and comparator, task9/AC-Q paired quality smoke, task11/AC-10 radix flip and fixtures, task12/AC-1b chunked-prefill probe, task13/AC-11 full sweep/comparator, task14/AC-12 quality gate, and task15 evidence bundle. Claude's final-state summary says remaining Tier-1 benchmark/comparator/quality and Tier-2 work are still pending; per the review instructions, those are incomplete work, not acceptable deferrals.

   Directive implementation order: first fix AC-0 hardware capture and finish strict AC-1 evidence; then run the Tier-1 radix-off smoke benchmark pair and comparator; then run AC-Q; then implement AC-10 without env overrides, run AC-1b, run AC-11, run AC-12, and assemble the evidence bundle.

## Blocking Side Issues

1. **AC-0 hardware radix capture cannot publish in the production ForwardBatch shape.**

   Evidence:
   - `runs/20260528_dsv32_mvp/ac0_capture_probe.json:1` has no `meta_info["double_sparsity_radix_capture"]`.
   - `_ds_radix_publish_extend_snapshot` reads `forward_batch.req_to_token_pool` and returns if that is absent (`python/sglang/srt/layers/attention/dsa_backend.py:344-353`).
   - Production `ForwardBatch` has `req_pool_indices` and `seq_lens`, but no `req_to_token_pool` field (`python/sglang/srt/model_executor/forward_batch_info.py:274-330`), and `ForwardBatch.init_new` constructs it without that field (`python/sglang/srt/model_executor/forward_batch_info.py:496-540`).
   - The DS selector already had to add a ForwardContext fallback for the same missing field (`python/sglang/srt/models/deepseek_v2.py:2153-2170`).
   - The current producer regression fakes `req_to_token_pool` into a `SimpleNamespace`, so it cannot catch the production mismatch (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:6949-6957`).

   Required fix: add a shared local resolver in `dsa_backend.py` that gets `req_to_token` from `forward_batch.req_to_token_pool` when present, otherwise from the active ForwardContext attention backend, unwrapping `TboAttnBackend.primary` just like the selector. Use that resolver in `_ds_radix_publish_extend_snapshot`. Add a production-shaped unit test where the forward batch has no `req_to_token_pool`, the ForwardContext backend supplies `req_to_token`, and `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` publishes `double_sparsity_radix_capture`. Then rerun the hardware probe until the response includes `per_token_slot_sha`, `per_layer_written_all_true=True`, and no error key.

## Queued Side Issues

1. `calibrate.py` still has stale operator-facing production recipe text after the native-FP8 sharded load redesign. This does not block the next mainline DS smoke work because `calibration_provenance.md` records the proven command, but it should be fixed before the next calibration handoff.

## Goal Alignment

The mutable tracker was corrected during this review:
- task6/AC-1.1 and task10/AC-6 moved to verified completed state.
- task5/AC-1 moved back to active partial because strict endpoint evidence is still missing.
- task2/AC-0 remains active blocked on the production radix-capture publish bug.
- The queued "radix-capture meta is Tier-2 only" classification was rejected because AC-0 explicitly requires the hardware `/generate` capture probe.
- #C historical OOB row was marked resolved by the dense-prefill/sparse-decode fix, removing the contradictory open status.

## Next Implementation Plan

1. Patch `_ds_radix_publish_extend_snapshot` to resolve `req_to_token` through ForwardContext when `forward_batch.req_to_token_pool` is absent, and add the production-shaped regression described above.
2. Reboot DS with `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, run `/generate`, and save the positive AC-0 capture artifact plus the capture-disabled and decode-only negative artifacts.
3. Reboot DS for AC-1 and save the live `/server_info` or `/get_server_info` JSON plus `/generate` output; also save the invalid/missing-mask validator rejection artifact.
4. Run task7 and task8 exactly as the plan specifies: DS and DSA `TRIALS=1` shortened-window smoke JSONLs, valid `.meta.json` sidecars, duration guard, and `mvp_compare.md`.
5. Run task9 AC-Q with DSA references captured first, then DS, and enforce all four gates.
6. Only after Tier 1 is complete, implement task11 AC-10 with no env override, run both radix fixtures, remove `--disable-radix-cache`, and capture a radix-on DS boot.
7. Run task12 AC-1b, task13 AC-11, task14 AC-12, and task15 evidence bundle in that order.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-3-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-3-summary.md

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
