# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 0 Summary — Unblock M1 (AC-0 fix + calibration load change + dry-run)

## Round objective (from round-0-contract.md)
Unblock M1: land the AC-0 producer-bug fix + regression (task1), land the AC-4
calibration FP8-sharded load change + a `--dry-run-blocks` mode (task3), and run
the one-block dry-run on the real cluster weights as the round's hardware artifact.

## What was implemented

### AC-0 — radix-capture producer fix (task1) — COMPLETE (code + regression)
`dsa_backend._write_token_labels` referenced `forward_batch` without accepting
it, so the name lookup raised inside a swallowing `try/except` and the
radix-capture extend snapshot was never published (the Round-38 bug).

- Added `forward_batch: Optional[ForwardBatch] = None` to `_write_token_labels`.
- Threaded the live `forward_batch` from all four production call sites:
  - `forward_extend` (dsa_backend.py)
  - the second extend/decode write site (dsa_backend.py)
  - `_forward_trtllm` (dsa_backend.py)
  - the MHA_ONE_SHOT `_set_mla_kv_buffer` hook (forward_mha.py)
- Made the extend-snapshot publish gate explicit: publishes only when
  `forward_batch is not None and forward_mode.is_extend()`. Token-label writes
  stay first; the swallowing `try/except` that masked the original NameError is
  removed.
- Added a producer-side regression class `TestRadixCaptureExtendSnapshotProducer`
  (5 tests): publish-on-extend (asserts `per_token_slot_sha` populated +
  `per_layer_written_all_true`), no-key-when-capture-disabled,
  no-publish-on-decode, no-crash/no-publish when `forward_batch is None` (labels
  still written), and decode-does-not-overwrite an existing extend snapshot.
- Updated the two affected spy stubs (trtllm + MHA tests) to accept the new
  optional argument.

The AC-0 *hardware* `/generate` capture probe (task2) needs a booted DS server,
which needs the mask — so it is sequencing-gated by the first DS boot and runs in
a later round (logged in the goal tracker Plan Evolution Log). Code + producer
regression for AC-0 are done.

### AC-4 — calibration load change + dry-run (task3) — COMPLETE (code), load path BLOCKED on hardware
- `calibrate.py` model load changed from `torch_dtype=bfloat16` +
  `device_map={"":"cuda"}` to `torch_dtype="auto"` + `device_map="auto"` (native
  FP8, no upcast, sharded across the node's GPUs).
- Forward loop no longer assumes a single `model.device`: inputs route to the
  input-embedding device via a defensive resolver (handles dispatched real
  model, plain single-device model, and unit fakes).
- Added `--dry-run-blocks N`: loads the model, logs a parameter dtype +
  device-placement report (FP8-not-upcast evidence), runs N blocks to confirm
  the Method-1 Q/K hooks fire on every layer, then exits without writing a mask.
- Added `_log_param_dtype_device_report(model)` (dtype/device histogram +
  `hf_device_map` span).

## Hardware artifact + KEY FINDING (the round's main discovery)
Ran `calibrate.py --dry-run-blocks 1` against the real cluster weights on the 8×
H200 node. Artifacts under `runs/20260528_dsv32_mvp/`:
`calibrate_dryrun_20260528-103632.log` (gitignored by repo `*.log` policy; its
verbatim error is quoted in the committed finding), `ROUND0_dryrun_finding.md`,
`dryrun_prompts.txt`.

**Finding: the plan's HF-load premise for AC-4 is invalid.** The dry-run failed
at `AutoConfig.from_pretrained` — *before* dtype/device matter:
- transformers 5.8.1 `CONFIG_MAPPING` has `deepseek_v2/v3/v4` but **not**
  `deepseek_v32`.
- the DeepSeek-V3.2 checkpoint has **no `auto_map`** and ships **no remote
  modeling/config `.py`**, so `trust_remote_code=True` has nothing to load.
- SGLang serves V3.2 via its own `DeepseekV32ForCausalLM` and treats
  `deepseek_v32` as an *unregistered* HF type — it never registers a HF AutoModel
  modeling class.

So stock HF `AutoModelForCausalLM` cannot load V3.2 at all. This blocks
generating the root-blocker mask. The `device_map="auto"`/`torch_dtype="auto"`
and input-device-routing changes landed this round are still correct for once the
load works — they were not the blocker.

**Validated fix path (config-only probe PASSED):** remap the config
`model_type` → `deepseek_v3` (+ `architectures` → `DeepseekV3ForCausalLM`) and
load the FP8 weights under the transformers `deepseek_v3` modeling. The remapped
config builds a valid `DeepseekV3Config` preserving every field calibration needs
(L=61, qk_nope=128, qk_rope=64, v_head_dim=128, kv_lora_rank=512, FP8 quant
config). V3.2 = V3 + the DSA indexer, which is irrelevant to channel-importance
calibration (only the MLA `kv_b_proj`/`q_b_proj` projections matter, and they are
identical). Fallback: drive the forward via SGLang's own model loader.

## Files modified / created
- `python/sglang/srt/layers/attention/dsa_backend.py` (AC-0)
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` (AC-0)
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` (AC-4)
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` (AC-0 producer regression + spy updates + `mock` import)
- `runs/20260528_dsv32_mvp/ROUND0_dryrun_finding.md`, `dryrun_prompts.txt` (evidence)

Commits: `4f4c620df` (AC-0), `7cbbce088` (calibration + dry-run evidence).

## Tests
- New: `TestRadixCaptureExtendSnapshotProducer` — 5 producer regressions, all pass.
- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py` → **242 passed**.
- Calibrate-subset (`-k "Calibrate or calibrate"`) → 11 passed (after fixing the
  input-device resolver to tolerate unit fakes lacking `get_input_embeddings`).

## Remaining items / next round
- **Blocking Side Issue #1 (next round mainline):** redesign the AC-4 calibration
  load to remap `deepseek_v32` → `deepseek_v3` and load under transformers v3
  modeling; re-run `--dry-run-blocks 1` to confirm FP8-not-upcast + hooks fire on
  all 61 layers; then run the full calibration (task4) to produce + validate the
  mask via `load_channel_mask`.
- Queued: Pile-val (`mit-han-lab/pile-val-backup`) is not cached; decide
  download vs. local corpus for the full 256-block run.
- Then proceed to M2 (task5 boot smoke + AC-1.1) etc. per the plan.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: `bitlesson-selector` returned NONE for the AC-0 fix and
  {`BL-20260527-mla-config-rope-dim-derivation`, `BL-20260527-reshape-before-slice-mla`}
  for the calibration change; both were respected (my changes don't touch the
  rope-dim derivation or MLA projection slicing, and the dry-run's "hooks fired
  on all layers" guard directly exercises the rope-dim lesson). The new
  HF-cannot-load-`deepseek_v32` discovery is a strong lesson candidate, but its
  *solution* (the `deepseek_v3` remap load) is only validated at the config level,
  not end-to-end — so per the strict "specific problem + specific solution +
  validation evidence" rule I am NOT adding a lesson this round. I will add a
  precise entry once the remap load + dry-run forward is validated next round.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
eb914678e [Sparsity] Loop-5: refined plan v1 + QA ledger
8979848ab [Sparsity] Loop-5: untrack active RLCR plan file
4f4c620df [Sparsity] Thread forward_batch into _write_token_labels (radix capture producer fix)
7cbbce088 [Sparsity] Calibration: native-FP8 sharded load + one-block dry-run mode
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
