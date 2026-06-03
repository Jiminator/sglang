# Code Review - Round 3

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-3-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 3 Summary

## Work Completed
Closed the two gaps Codex flagged in the round-2 "done" claims, making the DS-on smoke
boot fully evidence-complete on 8x H200.

- **AC-0 — hardware radix capture (task2), #B fixed.** The producer
  `_ds_radix_publish_extend_snapshot` returned before writing
  `double_sparsity_radix_capture` because production `ForwardBatch` carries no
  `req_to_token_pool` (same defect class as the selector's req_to_token bug).
  - Added `_resolve_req_to_token_for_capture(forward_batch, backend)` resolving in order:
    `forward_batch.req_to_token_pool.req_to_token` → `getattr(backend, "req_to_token", None)`
    → the active ForwardContext attention backend (unwrapping `TboAttnBackend.primary`).
  - Removed the broad `except Exception` wrapping the *required* context lookup — it was
    swallowing the missing-field error and silently dropping the publish. Best-effort
    catching now sits only around the snapshot read.
  - Made the capture SHA dtype-agnostic (`t.view(torch.uint8).numpy()`), fixing the
    `.numpy()` crash on fp16/bf16/fp8 tensors; kept the CUDA-graph-capture safety guard.
- **AC-1 — strict live server-info (task5).** Calling `/get_server_info` crashed the
  whole server every time after the DS bind. `get_internal_state` shipped
  `dict(vars(server_args))`, which now includes the DS bind's private CUDA tensors/pools
  (`_ds_channel_selection`, `_double_sparsity_token_label_table`, `_ds_token_to_kv_pool`,
  `_ds_req_to_token_pool`, `_double_sparsity_channel_mask`); pickling a torch tensor over
  the ZMQ pyobj IPC hit `torch.load`→`_legacy_load`→`UntypedStorage.dtype` AttributeError
  and killed the TokenizerManager recv loop. Fixed by excluding `_`-prefixed runtime
  attrs from the internal-state response (config fields are all public).
- Added a production-shaped CPU regression
  (`test_publishes_when_forward_batch_lacks_req_to_token_pool`): no `req_to_token_pool` on
  the batch, backend supplies `req_to_token`, `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` → asserts
  the capture publishes.

## Files Changed
- `python/sglang/srt/layers/attention/dsa_backend.py` — `_resolve_req_to_token_for_capture`;
  publish uses it; removed swallowing try/except around the context lookup.
- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py` —
  dtype-agnostic `_tensor_bytes_sha` (uint8 view).
- `python/sglang/srt/managers/scheduler.py` — `get_internal_state` excludes `_`-prefixed
  runtime attrs.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — production-shaped
  publish regression.
- `runs/20260528_dsv32_mvp/` — `ac0_capture_positive.json`, `ac1_server_info.json`,
  `ac1_generate.json`, `ac1_invalid_mask_rejection.md`.
- Commits: `6f95a9711` (AC-0 resolver + dtype-safe SHA), `bc534da7c` (get_internal_state
  fix + AC-0/AC-1 evidence), `76eef9c80` (AC-1 invalid-mask negative test).

## Validation
- DS unit suite green: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → **254 passed**.
- **AC-0 hardware (8x H200):** capture-enabled `/generate` returns non-empty
  `double_sparsity_radix_capture` — `prompt_len=11`, `per_token_slot_sha` count 11,
  `per_layer_written_all_true=True`, no `error` key. Capture-disabled negative: no key.
  (`runs/20260528_dsv32_mvp/ac0_capture_positive.json`)
- **AC-1 hardware (8x H200):** live `/get_server_info` returns and the server **stays
  alive** — `model_path=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`, `tp_size=8`,
  `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `enable_double_sparsity=True`,
  `disable_radix_cache=True`, `attention_backend=dsa`; `/generate` coherent
  (" Paris. The capital of the United States is Washington, D"). Invalid `CHANNEL_MASK_PATH`
  boot rejected fail-closed: `check_server_args` → `validate_double_sparsity` →
  `load_channel_mask` raises `DoubleSparsityChannelMaskMissing` before model load (no
  silent dense fallback). (`ac1_server_info.json`, `ac1_generate.json`,
  `ac1_invalid_mask_rejection.md`)
- GPUs freed after the probes (servers shut down).

## Remaining Items
- TIER-1 next round (gated on this fully-evidenced boot): AC-8/AC-9 smoke benchmarks
  (task7) + comparator (task8), AC-Q paired quality smoke (task9).
- TIER-2: AC-10 no-env-override radix flip + both fixtures (task11), AC-1b chunked-prefill
  probe (task12), AC-11 sweep (task13), AC-12 full quality (task14), evidence bundle (task15).
- Queued cleanup (non-blocking): stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260528-ds-private-server-args-attrs-crash-ipc
- Notes: Added `BL-20260528-ds-private-server-args-attrs-crash-ipc` — DS stashes private
  (`_`-prefixed) CUDA tensors on `server_args`; any handler that serializes
  `vars(server_args)` over the ZMQ pyobj IPC pickles a tensor and crashes the server on
  this torch build, so the introspection path must filter `_`-prefixed runtime attrs.
  I also extended the existing `BL-20260527-ds-metadata-via-forward-context` in
  `.humanize/bitlesson.md` (scope now covers the radix-capture producer and the
  `req_to_token` field; added the loop5-R3 `_resolve_req_to_token_for_capture` application
  and the rule that the required ForwardContext lookup must NOT sit inside a broad
  best-effort `except`). Both reinforce the same two-source-of-truth boundary: production
  `ForwardBatch` lacks DS fields (resolve via ForwardContext), and DS bind-time data lives
  on `server_args._*` (must be filtered before any IPC/pickle hop).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-2-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-2-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-1-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-1-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-0-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md and verify:

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

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-3-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
