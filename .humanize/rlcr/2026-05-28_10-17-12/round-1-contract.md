# Round 1 Contract

## Mainline Objective
**Complete AC-4: produce and validate the real DeepSeek-V3.2 channel mask.**
Redesign the calibration loader so V3.2 actually loads, make the one-block
dry-run fail-closed on dtype/device placement, prove it on hardware, then run the
full calibration and validate the mask. This unblocks the root blocker
(`/models/dsv32-fp8-channel-mask.safetensors`) that gates every DS-on criterion.

Concretely (Codex directive steps 1–5, all serving AC-4):
1. **Loader fix.** Add a helper that reads the raw config via
   `PretrainedConfig.get_config_dict(model_path)`; if `model_type == "deepseek_v32"`,
   build a `deepseek_v3` config (remap `model_type`→`deepseek_v3`,
   `architectures`→`["DeepseekV3ForCausalLM"]`) via `AutoConfig.for_model(...)`;
   otherwise keep `AutoConfig.from_pretrained`. Pass the resolved config into
   `AutoModelForCausalLM.from_pretrained(model_path, config=config,
   torch_dtype="auto", device_map="auto", trust_remote_code=True)`.
2. **Fail-closed dry-run validation.** Turn the parameter report into a structured
   return (dtype counts, param-device counts, `hf_device_map` devices). In dry-run
   on CUDA for an FP8-quantized config: require ≥1 float8 param dtype, reject
   `cpu`/`disk`/`meta` placements, require multi-GPU placement; log the full
   histogram before raising. CPU/unit-fake/non-FP8 paths stay exempt.
3. **Regressions** for the remap (field + FP8 quant preservation), the
   `from_pretrained` call shape under CUDA, and the fail-closed validator
   (raises on no-float8 / cpu-disk-meta; passes on the CPU fake path).
4. **Rerun the hardware one-block dry-run.** Artifact must show: remapped config,
   FP8 dtype present, no cpu/disk/meta, GPU sharding, hooks fired on all 61 layers.
   If it does not pass, STOP — do not run full calibration.
5. **Full calibration + mask validation.** Produce the mask, validate via
   `load_channel_mask()` (record `dtype=fp8_e4m3`, `page_size=64`, `label_dim=16`,
   `head_dim=128`, `channel_selection` int32 `[L,H,16]`, content SHA) into
   `runs/20260528_dsv32_mvp/`.

## Target ACs (≤ 2)
- **AC-4** — primary success condition for this round.
- (Continuation, not a gate) If AC-4 lands cleanly with time/feasibility left,
  begin **M2 / task5**: first DS boot at TP=8 with the validated mask and
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, which immediately satisfies the AC-0
  hardware capture probe (task2) and starts AC-1. This is reported as bonus
  progress, not a round-1 success requirement.

## Blocking Side Issues In Scope
- **#1: HF AutoModel cannot load `deepseek_v32`.** Resolved by step 1 (the
  `deepseek_v3` remap). This is the round's central blocker.
- **#2: dry-run does not enforce the placement/dtype guard.** Resolved by step 2
  (fail-closed validation).

## Queued Side Issues Out Of Scope (do not let these take over)
- Pile-val not cached: for the full calibration use a committed local corpus
  (`--dataset`) and record the exact command in the run dir; defer the
  download-vs-corpus policy decision. A local corpus is an acceptable production
  calibration source as long as the run is reproducible and recorded.
- task7–task15 (benchmarks, comparators, quality gates, radix flip, evidence
  bundle) — out of scope until the mask exists and a server boots.

## Round Success Criteria
1. `calibrate.py` resolves the V3.2 config via the `deepseek_v3` remap and loads
   the model successfully; regressions for the remap + validator pass; full DS
   unit suite still green.
2. The fail-closed dry-run raises on bad placement/dtype and passes on a good FP8
   sharded load.
3. A hardware one-block dry-run artifact under `runs/20260528_dsv32_mvp/` shows
   FP8-present, multi-GPU sharding, no cpu/disk/meta, hooks fired on all 61 layers.
4. The full calibration produces `/models/dsv32-fp8-channel-mask.safetensors`,
   validated by `load_channel_mask()` with metadata + content SHA recorded in the
   run dir.
5. Goal tracker updated; `round-1-summary.md` written with a BitLesson Delta.
