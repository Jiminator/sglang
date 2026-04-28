# CI Regression Bisection Report — `TestFlashinferTrtllmGenMoeBackendFP8` on B200

**Investigator:** Claude (sglang-bisect-ci-regression skill)
**Date:** 2026-04-28
**Repo:** `sgl-project/sglang`
**Reporting run:** https://github.com/sgl-project/sglang/actions/runs/24971499389/job/73115280727

---

## Failure Signature

- **Test:** `test/registered/backends/test_flashinfer_trtllm_gen_moe_backend.py::TestFlashinferTrtllmGenMoeBackendFP8` (`setUpClass`); `TestFlashinferTrtllmGenMoeBackendFP8Routed` and the BF16Routed sibling hit the same path.
- **Workflow / job:** `Nightly Test (Nvidia)` → job `nightly-test-perf-4-gpu-b200` (job id `73115280727`, run id `24971499389`); same code is reachable in `pr-test.yml::stage-b-test-4-gpu-b200`.
- **Model:** `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (snapshot `c5f5f263bdd5cc134092897864e8905d8fe7b928`, FP8-block checkpoint, `weight_block_size=[128,128]`).
- **Server command (verbatim):**

  ```bash
  sglang serve --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --attention-backend triton --moe-runner-backend flashinfer_trtllm \
    --tp-size 4 --ep-size 4 --mem-fraction-static 0.7 \
    --mamba-ssm-dtype bfloat16 --device cuda --host 127.0.0.1 --port 11000
  ```

- **Error:** `ValueError: Weight output_partition_size = 8 is not divisible by weight quantization block_n = 128.`
- **Failing symbol/path:** `Qwen3GatedDeltaNet.in_proj_ba = MergedColumnParallelLinear(...)` in `python/sglang/srt/models/qwen3_next.py:126` → `Linear.__init__` → `Fp8LinearMethod.create_weights` → `Fp8LinearMethod.validate_block_quant_shapes` (`python/sglang/srt/layers/quantization/fp8.py:351`).
- **Deterministic:** Yes. Reproduces on every B200 nightly run from `2026-04-23` onward (3 identical traces logged in run `24971499389`, including FP8, FP8Routed, and the BF16Routed/FP8 setup that retried offline→online).

---

## Boundary

| Status | Run ID | Job ID | Date (UTC) | Head SHA | Runner | Notes |
|---|---|---|---|---|---|---|
| **Last pass (setUpClass)** | `24754179764` | `72423730578` | 2026-04-22 07:32 | `1408d974080822788400c33cc3407994b98fdd2c` | `b200-inno-0123` (B200, drv 580.126.20) | `setUpClass` succeeded; `gsm8k_score=0.92` for FP8 and `0.945` for FP8Routed at 08:26–08:30 UTC. |
| **First fail (setUpClass)** | `24810524606` | `72614226614` | 2026-04-23 00:52 | `c689f774a4196563305a88e8faccecded3c6780c` | same `b200-inno-0123` | First `output_partition_size = 8` ValueError; setUpClass aborts before `test_gsm8k`. |
| Reproduction (the run linked in the question) | `24971499389` | `73115280727` | 2026-04-27 06:54 | `977830e91e4197628f581fd96cf257c6d9466f9d` | same `b200-inno-0123` | Same trace as 04-23, 04-25. |

Confirmed via `git merge-base --is-ancestor`:

- `4323fce82` is **not** an ancestor of `1408d974…` (last pass).
- `4323fce82` **is** an ancestor of `c689f774…` (first fail).

The same SHA has **not** been observed passing on a different runner — every B200-4GPU run since the boundary fails identically; older runs on the same runner passed. No other runner type is registered for this test (`register_cuda_ci(... suite="nightly-4-gpu-b200")`), so we cannot demonstrate "passes elsewhere", but the failure is in pure model-construction Python code reached before any kernel runs, so the B200 hardware is not the cause.

---

## Candidate Commits / PRs

| SHA | PR | Title | Files touched | Why suspicious |
|---|---|---|---|---|
| **`4323fce82a091fab154bf36baa5820659ec0fd16`** | **#23467** | **`fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert`** | `python/sglang/srt/layers/quantization/utils.py` | Adds `_FALLBACK_FUSED_SHARDS` containing `"in_proj_ba": ["in_proj_b","in_proj_a"]` and routes any layer whose `proj_name` is in that table through the fused-shard branch. **This is the root cause** (proven below). Authored 2026-04-22 22:16 +0800, exactly between last-pass and first-fail. |
| `175885676` | #23125 | `[CI] Fix mxfp8 TrtllmGenMoe test` | `test_flashinfer_trtllm_gen_moe_backend.py` | Only changed the **MXFP8** subclass; FP8 subclass and model name unchanged. Not the cause. |
| `ce0541404` | #20214 | `[FlashInfer v0.6.6][RL] ... flashinfer_trtllm_routed moe backend` | same test file (added `BF16Routed` 1 month earlier) | Pre-existing; passed for weeks before regression. |
| `e7e89349c` | #12543 | `Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell` | originally added `Qwen3-Next-80B-A3B-Instruct-FP8` (Nov 2025) | Long-standing; not the cause. |

No commit between the boundary touched `qwen3_next.py:Qwen3GatedDeltaNet`, `linear.py:MergedColumnParallelLinear`, or `fp8.py:validate_block_quant_shapes` in a way that would change `output_partition_size`. The diff of the suspect file across the boundary (`git diff 1408d97..977830e -- python/sglang/srt/layers/quantization/utils.py`) is exactly the patch in `4323fce82`.

---

## Root Cause Classification

**Code regression** introduced by PR #23467 (commit `4323fce82`).

---

## Root Cause Explanation

The `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` HF config lists `in_proj_ba` per concrete layer in `quantization_config.modules_to_not_convert`, e.g. `model.layers.0.linear_attn.in_proj_ba`, …, `model.layers.46.linear_attn.in_proj_ba`. The checkpoint stores it as a single weight, **not** split into `in_proj_b`/`in_proj_a`.

`Fp8Config.from_config` normalizes each entry into `{"layers.N.linear_attn.in_proj_ba","model.layers.N.linear_attn.in_proj_ba"}` (no `*` wildcards). For prefix `model.layers.0.linear_attn.in_proj_ba`, `is_layer_skipped(prefix, ignored_layers, fused_mapping)` decides whether to bypass FP8.

### Before `4323fce82` (`utils.py:46`, substring match)

`proj_name = "in_proj_ba"`. The model's `packed_modules_mapping = {"qkv_proj": [...], "gate_up_proj": [...]}` — `in_proj_ba` is not a key, so the `else` branch runs:

```python
is_skipped = any(ignored in prefix for ignored in ignored_layers)
```

`"model.layers.0.linear_attn.in_proj_ba" in "model.layers.0.linear_attn.in_proj_ba"` → **True** → `UnquantizedLinearMethod` → no `validate_block_quant_shapes` call → server boots; gsm8k passes (0.92 / 0.945 measured 2026-04-22).

### After `4323fce82` (`utils.py:60-66, 78-83, 91-95`)

```python
_FALLBACK_FUSED_SHARDS: Mapping[str, List[str]] = {
    "qkv_proj":     ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_ba":   ["in_proj_b", "in_proj_a"],     # WRONG: in_proj_ba is not fused
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],   # same problem
}
effective_fused = fused_mapping if proj_name in fused_mapping else _FALLBACK_FUSED_SHARDS
if proj_name in effective_fused:
    shard_prefixes = [prefix.replace(proj_name, s) for s in effective_fused[proj_name]]
    for shard_prefix in shard_prefixes:
        is_shard_skipped = any(_module_path_match(ignored, shard_prefix) ...)
        ...
```

- `proj_name = "in_proj_ba"` is now in `_FALLBACK_FUSED_SHARDS`, so the function takes the fused branch.
- It rewrites prefix into `["model.layers.0.linear_attn.in_proj_b", "model.layers.0.linear_attn.in_proj_a"]`.
- Neither shard appears in `ignored_layers` (the list contains `…in_proj_ba`, not `…in_proj_a`/`…in_proj_b`), so `_module_path_match` returns False for both → `is_skipped = False` → `Fp8LinearMethod` is selected.

I verified this by simulation:

```
=== OLD substring match (pre-4323fce82) ===
  any(ig in prefix): True
=== NEW dot-boundary match (post-4323fce82) ===
  shard 'model.layers.0.linear_attn.in_proj_b' matched any ignored: False
  shard 'model.layers.0.linear_attn.in_proj_a' matched any ignored: False
  -> is_skipped = False
```

Once `Fp8LinearMethod.create_weights` runs (`fp8.py:340-368`), it calls `validate_block_quant_shapes` with `output_partition_sizes = [linear_num_value_heads // tp, linear_num_value_heads // tp] = [32//4, 32//4] = [8, 8]`. The `MergedColumnParallelLinear` path forces the per-partition divisibility check (`fp8.py:328-338`), and `8 % 128 != 0` raises the observed `ValueError`. The failure is therefore independent of the MoE backend (`flashinfer_trtllm`/`flashinfer_trtllm_routed`) and of B200; it triggers during model construction before any forward pass.

---

## Evidence

Failing log (run `24971499389`, job `73115280727`):

```
2026-04-27T06:55:13Z [TP0 EP0] sglang is using nccl==2.27.7
2026-04-27T06:55:15Z [TP0 EP0] Detected fp8 checkpoint.
…
self.in_proj_ba = MergedColumnParallelLinear(
…
ValueError: Weight output_partition_size = 8 is not divisible by weight quantization block_n = 128.
ERROR: setUpClass (__main__.TestFlashinferTrtllmGenMoeBackendFP8)
Exception: Server process exited with code -9. Check server logs for errors.
```

Last passing log (run `24754179764`, 2026-04-22):

```
2026-04-22T08:26:11Z [METRIC] gsm8k_score=0.92  labels={"model":"Qwen/Qwen3-Next-80B-A3B-Instruct-FP8","eval":"gsm8k"}
2026-04-22T08:30:00Z [METRIC] gsm8k_score=0.945 labels={"model":"Qwen/Qwen3-Next-80B-A3B-Instruct-FP8","eval":"gsm8k"}
```

Code references (failing-SHA `977830e91…`):

- `python/sglang/srt/models/qwen3_next.py:126-134` — `self.in_proj_ba = MergedColumnParallelLinear(input_size=hidden_size=2048, output_sizes=[num_v_heads, num_v_heads]=[32,32], quant_config=quant_config, …)`.
- `python/sglang/srt/layers/linear.py:340-346` — `output_partition_sizes = [divide(s, tp_size) for s in output_sizes]` → `[8, 8]` with `tp_size=4`.
- `python/sglang/srt/layers/quantization/fp8.py:328-338` — per-partition `% block_n != 0` raises.
- `python/sglang/srt/layers/quantization/utils.py:46-110` (failing-SHA) — the new `_FALLBACK_FUSED_SHARDS` and `_module_path_match`; same file at `1408d974…` had only the substring branch.

HF config (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` `config.json`):

- `quantization_config.weight_block_size = [128, 128]`, `quant_method=fp8`, `activation_scheme=dynamic`, `fmt=e4m3`.
- `linear_num_value_heads = 32`, `linear_num_key_heads = 16`, `hidden_size = 2048`.
- `modules_to_not_convert` enumerates `model.layers.{0…46}.linear_attn.in_proj_ba` per concrete layer (no wildcards).

Runner / environment (constant across the boundary; rules out env cause):

- Runner: `b200-inno-0123`, machine `b48f74b00d89`, NVIDIA B200, driver `580.126.20`.
- `sglang==0.0.0.dev1+gb7113cadb`, `flashinfer-python==0.6.8.post1`, `flashinfer-cubin==0.6.8.post1`, `flashinfer-jit-cache==0.6.8.post1`, `nccl==2.27.7`.

---

## Recommended Fix

### Short-term (minimal patch in `python/sglang/srt/layers/quantization/utils.py`)

1. Drop the false fused entries from `_FALLBACK_FUSED_SHARDS` — `in_proj_ba`/`in_proj_qkvz` are *real* layers in Qwen3-Next FP8 checkpoints, not virtual fused names, so they should never trigger the unfusing branch:

   ```python
   _FALLBACK_FUSED_SHARDS: Mapping[str, List[str]] = {
       "qkv_proj":     ["q_proj", "k_proj", "v_proj"],
       "gate_up_proj": ["gate_proj", "up_proj"],
   }
   ```

   This restores `is_skipped=True` for `model.layers.N.linear_attn.in_proj_ba` because the `else` branch's `_module_path_match` will match the literal `…in_proj_ba` entry exactly (`ignored == prefix`), and the original Qwen3.6 / `mlp.gate` fix that PR #23467 was after still works.

2. **Or** keep PR #23467's intent but add a self-check before unfusing: only take the fused branch if **none** of the ignored entries equals the un-split prefix. (Belt-and-suspenders.)

### Longer-term

- Move fused-shard knowledge onto each model class. Models already declare `packed_modules_mapping` (Qwen3-Next does, on `qwen3_next.py:877`), and that mapping correctly excludes `in_proj_ba`. A global fallback that tries to guess shard names from substrings (`_FALLBACK_FUSED_SHARDS`) is fragile and was the proximate cause here. Either (a) require models to opt in via `packed_modules_mapping`, or (b) consult the checkpoint's safetensors index to decide whether a name is actually fused.
- Improve the error message in `validate_block_quant_shapes` to include the prefix/layer name and `output_partition_sizes`. Today the message is purely numeric, which is what made this look like a generic 4-GPU server crash.
- Add a regression test: instantiate the FP8 quant_config with the actual `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` `modules_to_not_convert`, build `Qwen3GatedDeltaNet` with `tp=4`, and assert that `in_proj_ba` ends up with `UnquantizedLinearMethod`.

---

## Next Files to Open

- `python/sglang/srt/layers/quantization/utils.py:46-110` — apply the patch above.
- `python/sglang/srt/layers/quantization/fp8.py:191-201` (`from_config` ignored-layer normalization) and `fp8.py:217-240` (`get_quant_method`) — confirm the call site after the patch.
- `python/sglang/srt/models/qwen3_next.py:877-898` — check whether `packed_modules_mapping` should also list `in_proj_ba`/`in_proj_qkvz` if you decide they really are fused (they are *layout* fused in the checkpoint via the custom `_make_packed_weight_loader`, but they are *named* as a single tensor, not split).
- `test/registered/backends/test_flashinfer_trtllm_gen_moe_backend.py` — no changes needed; once the quant logic is fixed, the existing test will run.

---

## Things Still Unclear

- Whether `in_proj_qkvz` would have hit the same bug on a different model. In Qwen3-Next FP8 the `modules_to_not_convert` does **not** list `in_proj_qkvz`, so it's quantized either way; for the present test it doesn't trip the validator because `tp=4` divides `[16*128, 16*128, 32*128, 32*128] = [2048,2048,4096,4096]` per shard cleanly. Worth checking once the PR is reverted that no other model's `modules_to_not_convert` was relying on the new fallback.
- Why the `2026-04-27` log shows the error trace under the BF16Routed launch line at `06:54:46` (the `server_args` dump above the error has `model_path='…-FP8'`). Cosmetic interleaving in the wrapper between BF16Routed teardown and FP8 setUpClass — the actual failure is unambiguously the FP8 `setUpClass` (the unittest banner `ERROR: setUpClass (__main__.TestFlashinferTrtllmGenMoeBackendFP8)` confirms it). Not load-bearing for the fix.
- Whether the Qwen3.6 model that motivated PR #23467 (`mlp.gate` collisions) has another path to the same fix that doesn't depend on `_FALLBACK_FUSED_SHARDS`. Worth coordinating the revert with @mickjagger19 / Qwen3.6 owner so the original `mlp.gate` vs `mlp.gate_up_proj` collision doesn't return.
