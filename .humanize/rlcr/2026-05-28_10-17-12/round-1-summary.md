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
