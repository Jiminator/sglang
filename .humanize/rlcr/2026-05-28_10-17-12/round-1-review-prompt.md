# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 1 Summary — Complete AC-4: generate + validate the DeepSeek-V3.2 channel mask

## Round objective (from round-1-contract.md)
Complete AC-4: redesign the calibration loader so V3.2 actually loads, make the
one-block dry-run fail-closed, prove it on hardware, then run the full calibration
and validate the mask. This unblocks the root blocker
(`/models/dsv32-fp8-channel-mask.safetensors`) that gates every DS-on criterion.

## Outcome: AC-4 COMPLETE — the root-blocker mask exists and validates.

`/models/dsv32-fp8-channel-mask.safetensors` was generated and validated:
`content_sha256=7b3207cae888…`, `dtype=fp8_e4m3`, `page_size=64`, `label_dim=16`,
`head_dim=128`, `channel_selection` = `int32 (61, 128, 16)`, channel indices in `[0,128)`.

## What was implemented (commit `c99ed3644`)

### 1. V3.2 calibration loader redesign (task3 / Blocking Side Issue #1)
`_resolve_calibration_config(model_path)` reads the raw config via
`PretrainedConfig.get_config_dict`; for `model_type=="deepseek_v32"` it builds a
`deepseek_v3` config (`AutoConfig.for_model("deepseek_v3", …, architectures=
["DeepseekV3ForCausalLM"])`) and `_load_calibration_model` loads the FP8 weights
under the transformers V3 MLA modeling with `device_map="auto"`,
`torch_dtype="auto"`. V3.2 = V3 + the DSA indexer, which is irrelevant to
channel-importance calibration (only `kv_b_proj`/`q_b_proj` matter, identical to V3).
Falls back to `AutoConfig.from_pretrained` when the raw config can't be read.

### 2. Fail-closed dry-run validation (task5 / Blocking Side Issue #2)
`_summarize_param_placement` returns a structured dtype/device report;
`_enforce_dry_run_placement` rejects, on CUDA for an FP8 config: off-GPU
(cpu/disk/meta) placement, single-GPU placement, and a no-float8 (bf16 upcast)
load — BEFORE the full calibration runs. Logs the full histogram before raising.

### 3. DeepGEMM → Triton FP8 fallback (Blocking Side Issue #3, found this round)
transformers' deepseek_v3 FP8 forward fetches the `kernels-community/deep-gemm`
hub kernel (large cutlass JIT tree → HF Hub 429 storms with 230s backoffs) whose
cached metadata schema is rejected by `kernels` 0.14.1 with a `ValueError` that
escaped transformers' `except ImportError` and crashed the forward.
`_force_triton_fp8_for_calibration` makes `_load_deepgemm_kernel` report
`ImportError` immediately (no fetch), routing to transformers' own numerically
equivalent `finegrained-fp8` Triton kernel. Run online (not `HF_HUB_OFFLINE`) so
the Triton kernel's publisher-trust check passes.

### 4. Corpus (Pile-val queued issue, resolved)
`pip install zstandard` made Pile-val readable. Built a committed-by-reference
local corpus (`runs/20260528_dsv32_mvp/calib_corpus_pileval.txt`, 300 Pile-val docs
≥1500 chars) used via `--dataset`. See `calibration_provenance.md`.

## Hardware evidence (`runs/20260528_dsv32_mvp/`)
- `calibrate_dryrun5_*.log` — one-block dry-run PASSED: FP8 sharded across all 8
  GPUs (float8_e4m3fn=604 params, no upcast), validator passed, Method-1 Q/K hooks
  fired on all 61 layers (H=128, head_dim=128). (Dry-runs #2–#4 are the documented
  iterations that found the HF-load, deep-gemm, and offline-trust issues.)
- `calibrate_full_*.log` — full 256-block calibration wrote the mask (~8 min total).
- `mask_validation.txt` — `load_channel_mask` validation output.
- `calibration_provenance.md` — exact command, corpus build method + SHAs, mask SHA.
- `ROUND0_dryrun_finding.md` — round-0 discovery of the HF-load impossibility.

## Files modified / created
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — loader resolver,
  `_load_calibration_model`, structured placement report + `_enforce_dry_run_placement`,
  `_config_is_fp8`, `_force_triton_fp8_for_calibration`.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` —
  `TestCalibrationLoaderV32Remap` (9 tests: remap field/FP8 preservation, load-call
  args under CUDA, CPU device_map, validator rejects off-GPU/single-GPU/upcast,
  validator passes good sharded FP8, `_config_is_fp8`, deep-gemm→ImportError skip).
- `runs/20260528_dsv32_mvp/` evidence artifacts (above).

## Tests
- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
  → **251 passed** (was 242 at round 0; +9 loader regressions).

## Goal Tracker
Updated: task3 + task4 moved to Completed and Verified (pending Codex verification)
with evidence; Blocking Side Issues #1, #2, #3 and the Pile-val queued issue all
marked RESOLVED; Plan Evolution Log round-1 entry added. No immutable-section or AC
changes.

## Remaining items / next round
The mask exists → M2 is unblocked. Next round mainline (Codex directive step 6):
first DS boot (task5) — pin `MODEL_PATH=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
(DEC-6), boot `serve_double_sparsity.sh` at TP=8 with the new mask and
`SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, confirm `/get_server_info` knobs + `/generate`
text + the `double_sparsity_radix_capture` meta_info (satisfies the AC-0 hardware
probe / task2), then AC-1.1 genuine-sparsity (task6) and AC-6 CUDA-graph status
(task10). SGLang serving uses its own FP8 kernels, so the HF hub-kernel fragility
from this round does not apply there.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260528-dsv32-hf-calibration-load
- Notes: Added because the V3.2-calibration-load problem spanned rounds 0–1 and is
  now solved + validated end-to-end (dry-run #5 + mask generated/validated). The
  entry captures the three failure modes (unregistered `deepseek_v32` config;
  deep-gemm hub-kernel 429 + metadata-schema mismatch with `except ImportError` too
  narrow; offline breaking the Triton kernel's trust check) and the exact fixes
  (deepseek_v3 remap, force finegrained-fp8 Triton, run online, fail-closed dry-run,
  `zstandard` for Pile-val). `bitlesson-selector` for this round's tasks returned
  {`BL-20260527-mla-config-rope-dim-derivation`, `BL-20260527-reshape-before-slice-mla`},
  both respected (the remap preserves the MLA dims and the dry-run's all-61-layers
  hook-fire check directly exercises the rope-dim lesson; no projection slicing changed).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
