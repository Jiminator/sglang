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
