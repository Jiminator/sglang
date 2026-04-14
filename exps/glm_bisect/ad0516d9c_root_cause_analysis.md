# Root Cause Analysis: GLM-4.6-FP8 TP8+MTP Performance Regression

**Date**: 2026-04-14
**Root cause commit**: `ad0516d9c` -- [NPU] optimize glm4.7 (#19246)
**Status**: Confirmed via local A/B experiment

---

## 1. Confirmation

| Checkpoint | TP8+MTP bs=16 (tok/s) | TP8+MTP acc_len | State |
|------------|------------------------|-----------------|-------|
| `ad0516d9c~1` (parent) | **1011.69** | **1.98** | GOOD |
| `ad0516d9c` (this commit) | **557.50** | **1.10** | BAD |

The regression is introduced by exactly this one commit. TP8 throughput is unchanged (766.97 vs 767.27), confirming the issue is speculative-decoding-only.

---

## 2. What the Commit Changes

PR #19246 ([NPU] optimize glm4.7) modifies 4 files:

| File | Lines Changed | NVIDIA Impact |
|------|---------------|---------------|
| `python/sglang/srt/models/glm4_moe_nextn.py` | +16/-2 | **YES -- this is the root cause** |
| `python/sglang/srt/models/glm4_moe.py` | +35/-8 | No (all changes gated behind `_is_npu`) |
| `python/sglang/srt/hardware_backend/npu/utils.py` | +64/-0 | No (NPU-only utility functions) |
| `python/sglang/srt/layers/quantization/modelslim/modelslim.py` | +2/-2 | No (import relocation, NPU-only codepath) |

Only the `glm4_moe_nextn.py` change affects NVIDIA GPUs.

---

## 3. The Bug: `quant_config` Silently Set to `None` for GLM-4.6-FP8 Draft Model

### 3.1 The Code Change

In `Glm4MoeForCausalLMNextN.__init__()` (the GLM EAGLE draft model):

```python
# BEFORE (working):
self.quant_config = quant_config
self.model = Glm4MoeModelNextN(config, quant_config, ...)

# AFTER (broken for GLM-4.6-FP8):
self.needs_quant_draft = (
    get_global_server_args().speculative_draft_model_quantization
)
quant_config = quant_config if self.needs_quant_draft else None
self.model = Glm4MoeModelNextN(config, quant_config, ...)   # quant_config is now None!
```

### 3.2 Why `needs_quant_draft` Is Falsy for GLM-4.6-FP8

The value of `speculative_draft_model_quantization` is determined in `ServerArgs._handle_speculative_decoding_settings()`:

```python
# server_args.py line 1036-1039:
if self.speculative_draft_model_quantization is None:
    self.speculative_draft_model_quantization = self.quantization
elif self.speculative_draft_model_quantization == "unquant":
    self.speculative_draft_model_quantization = None
```

For GLM-4.6-FP8:
1. `--quantization` is NOT passed on the command line (the test uses `--tp=8 --trust-remote-code --speculative-algorithm=EAGLE ...`)
2. Therefore `self.quantization = None`
3. Line 1036-1037: `speculative_draft_model_quantization = self.quantization = None`
4. In the draft model: `self.needs_quant_draft = None` (falsy)
5. **`quant_config = None`** -- the original `CompressedTensorsConfig` is discarded

**Nightly CI logs confirm**: `quantization=None, speculative_draft_model_quantization=None`

### 3.3 How Quantization Actually Works for GLM-4.6-FP8

GLM-4.6-FP8 uses **compressed-tensors** quantization. This is NOT controlled by the `--quantization` server arg. Instead:

1. The model's HuggingFace `config.json` contains a `quantization_config` field specifying compressed-tensors
2. During model loading, `ModelConfig` reads this and creates a `CompressedTensorsConfig` object
3. This `CompressedTensorsConfig` is passed as `quant_config` to the model class `__init__`
4. Linear layers use `CompressedTensorsLinearMethod` to properly handle FP8 weight tensors with scales

The `server_args.quantization` field remains `None` because the user didn't explicitly pass `--quantization=fp8`. The model detects quantization from its own config file.

### 3.4 The Mismatch

**Before the commit**:
- Target model: `CompressedTensorsConfig` -> FP8 computation with proper scales
- Draft model: `CompressedTensorsConfig` -> FP8 computation with proper scales
- **Result**: Draft and target produce matching outputs. Acceptance length ~2.0.

**After the commit**:
- Target model: `CompressedTensorsConfig` -> FP8 computation with proper scales
- Draft model: `quant_config=None` -> `UnquantizedLinearMethod` -> weights loaded/computed in bfloat16
- **Result**: Draft produces numerically different outputs from target. Acceptance length collapses to ~1.0.

---

## 4. Mechanism of the Throughput Collapse

### 4.1 What Happens When `quant_config=None` for the Draft Model

With `quant_config=None`, the draft model's `Glm4MoeDecoderLayer` constructs all its subcomponents differently:

| Component | With `CompressedTensorsConfig` | With `quant_config=None` |
|-----------|-------------------------------|--------------------------|
| `QKVParallelLinear` | `CompressedTensorsLinearMethod` (FP8 matmul with scales) | `UnquantizedLinearMethod` (bfloat16 matmul) |
| `RowParallelLinear` (o_proj) | `CompressedTensorsLinearMethod` | `UnquantizedLinearMethod` |
| `FusedMoE` (experts) | `CompressedTensorsFusedMoEMethod` (FP8 MoE with scales) | Standard FusedMoE (bfloat16) |
| Shared experts | FP8 linear layers | bfloat16 linear layers |

The draft model's weights are loaded from the SAME checkpoint files as the target model (GLM-4.6-FP8 stores its NextN draft weights in the same safetensors files). But the weight loading process differs:

- **With compressed-tensors**: Loads FP8 weight tensors + scale tensors, applies proper dequant/requant during computation
- **Without compressed-tensors**: Loads weight tensors as bfloat16 parameters via `UnquantizedLinearMethod.create_weights()`, which allocates `params_dtype` (bfloat16) buffers. The weight loader must then convert FP8 checkpoint data to bfloat16.

The numerical difference between FP8-with-scales and dequant-to-bfloat16 is enough to cause the draft model's predictions to diverge from the target model at every token, reducing speculative acceptance from ~2.0 to ~1.0.

### 4.2 Why Acceptance Drops to ~1.0 (Not Zero)

An acceptance length of 1.0 means the system produces exactly 1 token per iteration -- the mandatory first token from the target model's verification pass. The ~3 draft tokens proposed by EAGLE are essentially all rejected because their probability distributions differ too much from the target's.

The acceptance isn't exactly 0 (it's ~1.1 at bs=8/16) because some high-probability tokens (e.g., common function words) may still match between FP8 and bfloat16 inference.

### 4.3 Why TP8 (Non-Speculative) Is Unaffected

The target model's `quant_config` is NOT modified by this commit. The `Glm4MoeForCausalLM` (base model) class is untouched for NVIDIA paths -- only the `Glm4MoeForCausalLMNextN` (draft model) class is affected. So non-speculative inference runs through the correctly-configured target model only.

---

## 5. The Intent of the Change

The commit was designed for NPU (Ascend) platforms where:
1. The draft model may need to run unquantized for compatibility or performance
2. `speculative_draft_model_quantization` can be explicitly set to control this

For DeepSeek models, a similar pattern exists in `deepseek_nextn.py` (lines 147-153), where environment variables control draft model quantization behavior at runtime. The GLM commit copies this pattern.

The bug is that the condition `if self.needs_quant_draft` is **wrong for models that use auto-detected quantization** (like compressed-tensors in GLM-4.6-FP8). The `server_args.speculative_draft_model_quantization` field is `None` because `server_args.quantization` is `None`, even though the model actually IS quantized via compressed-tensors.

---

## 6. Suggested Fix

### Option A: Don't override quant_config when server_args says None (minimal fix)

The condition should check whether the user explicitly requested an unquantized draft, not whether the server_args field happens to be None:

```python
# In Glm4MoeForCausalLMNextN.__init__:
server_args = get_global_server_args()
# Only strip quant_config if user explicitly requested "unquant"
# When speculative_draft_model_quantization is None, it means "use same as target"
# which is handled by _handle_speculative_decoding_settings setting it to self.quantization.
# But for auto-detected quant (compressed-tensors), self.quantization is None too,
# so we should preserve the quant_config that was passed to us from the model loader.
self.needs_quant_draft = (
    server_args.speculative_draft_model_quantization is not None
    or quant_config is not None  # Auto-detected quant from model config
)
if not self.needs_quant_draft:
    quant_config = None
```

### Option B: Preserve the original quant_config and only override when user says "unquant" explicitly

```python
# In Glm4MoeForCausalLMNextN.__init__:
server_args = get_global_server_args()
# Check if user explicitly asked for unquantized draft
explicit_unquant = (
    server_args.speculative_draft_model_quantization is None
    and server_args.quantization is None  # User didn't set --quantization
    # but the model HAS a quant_config from its HF config -> keep it
)
# Only discard quant_config when there's a clear signal that draft should be unquant
if quant_config is not None:
    # Model has auto-detected quantization; preserve it for draft
    self.needs_quant_draft = True
else:
    self.needs_quant_draft = bool(server_args.speculative_draft_model_quantization)
```

### Option C: Revert the glm4_moe_nextn.py changes entirely

```bash
git revert --no-commit ad0516d9c -- python/sglang/srt/models/glm4_moe_nextn.py
```

This preserves the NPU-specific optimizations in `glm4_moe.py` and `npu/utils.py` while reverting only the draft model quantization change. The `glm4_moe.py` changes are safely gated behind `_is_npu` and do not affect NVIDIA.

### Verification

Any fix should be verified by running the GLM-4.6-FP8 TP8+MTP benchmark and checking:
- TP8+MTP acceptance length returns to ~2.0
- TP8+MTP throughput at bs=16 returns to ~1000+ tok/s
- TP8 throughput remains unchanged

---

## 7. Impact Assessment

- **Affected models**: Any GLM MoE model using compressed-tensors quantization with EAGLE speculative decoding, where `--quantization` is not explicitly passed
- **Affected platforms**: All platforms (NVIDIA, AMD, etc.) -- the draft model quant_config override is not platform-gated
- **Severity**: High -- ~50% throughput regression for GLM-4.6-FP8 TP8+MTP
- **Introduced**: Apr 3, 2026 (merged to main)
- **First visible in nightly**: Apr 4, 2026
- **Still present**: Yes (as of Apr 14, 2026)

---

## 8. Full Bisection Evidence Trail

| SHA | Description | TP8+MTP bs=16 | acc_len | Source |
|-----|-------------|---------------|---------|--------|
| `afb32d76224e` | Mar 30 nightly | 1129.19 | ~2.0 | Nightly logs |
| `3650bfb19926` | Mar 31 nightly (after FP32 gate) | 1083.81 | ~2.0 | Nightly logs |
| `29d8e959d704` | Apr 3 nightly (last good) | 1022.93 | 1.96 | Nightly logs |
| `ad064c2f4~1` | Before FP32 gate | 1053.98 | ~2.0 | Local 10-trial |
| `ad064c2f4` | After FP32 gate, before MTP regression | 1010.72 | ~2.0 | Local 10-trial |
| **`ad0516d9c~1`** | **Parent of root cause** | **1011.69** | **1.98** | **Local 2-trial** |
| **`ad0516d9c`** | **ROOT CAUSE COMMIT** | **557.50** | **1.10** | **Local 2-trial** |
| `eb407b80f` (=84118acf5~1) | Before kernel bump | 559.44 | ~1.0 | Local 10-trial |
| `84118acf5` | After kernel bump | 559.54 | ~1.0 | Local 10-trial |
| `95cdbce34fa9` | Apr 4 nightly (first bad) | 576.19 | 1.01 | Nightly logs |
| `70658bfeb52a` | Apr 5 nightly | 576.30 | ~1.0 | Nightly logs |
