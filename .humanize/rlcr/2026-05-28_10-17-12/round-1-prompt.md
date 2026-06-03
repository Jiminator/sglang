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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-1-contract.md

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
# Round 0 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 2/11 addressed | Forgotten items: 0 | Unjustified deferrals: 0

Review scope: `development/loop5/refined_plan_v1.md`, `round-0-prompt.md`,
`goal-tracker.md`, commits `4f4c620df` and `7cbbce088`, changed code, and the
round artifacts in `runs/20260528_dsv32_mvp/`.

Verification run:

```bash
python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 242 passed
```

## Mainline Gaps

1. **AC-4 is not complete: the real dry-run failed before the required FP8/device/hook evidence.**

   The current load still enters `AutoConfig.from_pretrained()` at
   [calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:305), and the recorded hardware run fails there with
   `KeyError: 'deepseek_v32'` / `ValueError: model type deepseek_v32 not recognized` in
   [calibrate_dryrun_20260528-103632.log](/sgl-workspace/sglang/runs/20260528_dsv32_mvp/calibrate_dryrun_20260528-103632.log:3).
   That means the artifact is a useful blocking discovery, not the AC-4 positive
   dry-run. It does not show FP8-not-upcast placement, does not prove sharding,
   and does not prove hooks fired on all 61 layers. The root blocker mask is also
   still absent at `/models/dsv32-fp8-channel-mask.safetensors`.

   Required fix: implement the V3.2 calibration load redesign now, rerun a
   successful one-block dry-run, then run the full calibration and validate the
   mask with `load_channel_mask()`.

2. **AC-0 is only code/regression complete; the `/generate` capture probe is still pending.**

   Code inspection verifies the producer fix: `_write_token_labels` accepts
   `forward_batch` at
   [dsa_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:1501),
   token-label writes still happen before capture publish, and publish is gated on
   a present extend-mode `forward_batch` at
   [dsa_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:1589).
   The unit regression passes. The AC-0 hardware positive test remains incomplete:
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` plus `/generate` must return non-empty
   `meta_info["double_sparsity_radix_capture"]` after the first DS boot.

3. **All downstream plan tasks remain active.**

   Task5 through task15 are still pending and remain blocked primarily by AC-4.
   This is tracked, but it means Round 0 must not be treated as complete against
   the original plan. Smoke boot, sparsity proof, smoke benchmarks, smoke
   comparator, quality smoke, CUDA-graph evidence, radix flip, chunked-prefill
   probe, AC-11, AC-12, and the evidence bundle are not done.

## Blocking Side Issues

1. **HF AutoModel cannot load DeepSeek-V3.2 as written.**

   The checkpoint `config.json` uses `model_type: deepseek_v32`, while this
   Transformers install has `deepseek_v3` and `deepseek_v4` mappings but no
   `deepseek_v32`. The checkpoint also has no `auto_map` or remote code to make
   `trust_remote_code=True` useful. This blocks AC-4 and every DS-on criterion
   that depends on the mask.

2. **`--dry-run-blocks` logs dtype/device placement but does not enforce the guard.**

   `_log_param_dtype_device_report()` only logs histograms and `float8_present` at
   [calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:183).
   The dry-run success path at
   [calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:648)
   returns success based on hook coverage alone. A remapped load that silently
   upcasts, places parameters on CPU/disk/meta, or uses too few GPUs can still
   produce a successful dry-run log unless a human notices. That does not satisfy
   AC-4's "confirms no bf16 upcast before the full run" requirement.

## Queued Side Issues

1. **Pile-val is not cached.**

   This is correctly queued, not the next mainline blocker. Once the load works,
   choose one corpus path for the full 256-block calibration: either cache
   `mit-han-lab/pile-val-backup` on the node or use a committed/local
   newline-delimited corpus with the command recorded in the run directory.

## Goal Tracker Updates Applied

I updated the mutable tracker section directly:

- Moved task1 to `Completed and Verified` with the 242-pass unit evidence.
- Changed task3 from completed to `partially implemented; blocked`.
- Added Blocking Side Issue #2 for dry-run enforcement.
- Fixed task5's note so it is not circularly gated on task2; task2 should run
  during/after the first DS boot with capture enabled.

## Directive Implementation Plan

Execute the next work in this order. Do not start benchmark or quality tasks
until the mask exists and validates.

1. **Fix the calibration loader for V3.2.**

   In `calibrate.py`, add a helper that reads the raw config with
   `PretrainedConfig.get_config_dict(model_path)`. If `model_type` is
   `deepseek_v32`, copy the dict, set `model_type="deepseek_v3"` and
   `architectures=["DeepseekV3ForCausalLM"]`, then build the config with
   `AutoConfig.for_model("deepseek_v3", **config_without_model_type)`. For all
   other models, keep `AutoConfig.from_pretrained()`.

   Pass that config object into
   `AutoModelForCausalLM.from_pretrained(model_path, config=config,
   torch_dtype="auto", device_map="auto", trust_remote_code=True)`. Keep
   tokenizer loading unchanged.

2. **Make dry-run placement validation fail closed.**

   Change `_log_param_dtype_device_report()` into a structured report helper
   that returns dtype counts, parameter-device counts, and `hf_device_map`
   devices. In dry-run mode on CUDA for an FP8-quantized config:

   - require at least one `float8` parameter dtype,
   - reject `cpu`, `disk`, and `meta` placements,
   - require multi-GPU CUDA placement for the V3.2 H200 run,
   - still log the full histogram before raising.

   Keep CPU/unit fake behavior exempt so existing tests remain cheap.

3. **Add focused regressions.**

   Add tests that:

   - a temp `deepseek_v32` config is remapped to `DeepseekV3Config` while
     preserving `num_hidden_layers=61`, `qk_nope_head_dim=128`,
     `qk_rope_head_dim=64`, `v_head_dim=128`, `kv_lora_rank=512`, and the FP8
     `quantization_config`,
   - `from_pretrained()` receives the remapped config plus
     `torch_dtype="auto"` and `device_map="auto"` when CUDA is available,
   - dry-run validation raises if an FP8 config has no float8 parameters or has
     CPU/disk/meta placement,
   - dry-run validation passes for the existing CPU fake/unit path.

4. **Rerun the hardware one-block dry-run.**

   Use the same real model path and local dry-run corpus. The new artifact must
   show: remapped config, FP8 dtype present, no CPU/disk/meta placement, expected
   GPU sharding, and hooks fired on all 61 layers. If this does not pass, do not
   run full calibration.

5. **Run full calibration and validate the mask.**

   Produce `/models/dsv32-fp8-channel-mask.safetensors`, then validate it with
   `load_channel_mask()`. Record metadata values (`dtype=fp8_e4m3`,
   `page_size=64`, `label_dim=16`, `head_dim=128`), `channel_selection` shape
   and dtype, and the full content SHA in `runs/20260528_dsv32_mvp/`.

6. **Immediately run the first DS boot with the capture probe enabled.**

   Pin `MODEL_PATH=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`, boot DS
   TP=8 with the validated mask, set `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, and
   issue `/generate`. Save `/get_server_info`, the generated text, and the
   `double_sparsity_radix_capture` meta_info payload. This completes task2 and
   starts task5.

7. **Continue the original task order.**

   After task5 succeeds, run task6, task7, task8, task9, and task10 for the Tier
   1 smoke evidence. Then implement task11 before task12/task13/task14. Assemble
   task15 only after the required artifacts exist.

No COMPLETE: original plan work is still pending.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-1-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-1-summary.md

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
