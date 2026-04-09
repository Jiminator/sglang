# Transformers Backend MMLU Regression Analysis
claude --resume 85f06a58-be0e-44bf-b3d3-15c1eea995ce   
**Commit**: `34ddf135fd2de6541ed577d63b8b875b1e6a72e1`
**PR**: https://github.com/sgl-project/sglang/pull/19163
**Test**: `TestTransformersFallbackEndpoint.test_mmlu` in `test/registered/models/test_transformers_models.py`
**Model**: `meta-llama/Llama-3.1-8B-Instruct`
**Symptom**: MMLU score dropped from consistently >= 0.65 to sometimes ~0.64, causing CI flakiness

---

## Root Cause: RMSNorm Kernel Replacement

The commit replaced Transformers' **Python `LlamaRMSNorm`** with SGLang's **fused CUDA kernel `sgl_kernel.rmsnorm`** in the transformers backend.

### Where the replacement happens

In the new `TransformersBase.recursive_replace()` method at `python/sglang/srt/models/transformers.py:780-784`:

```python
elif child_module.__class__.__name__.endswith("RMSNorm"):
    new_module = replace_rms_norm_class(
        child_module,
        self.text_config.hidden_size,
    )
```

The old `TransformersForCausalLM` class had **no RMSNorm replacement** at all. It used Transformers' native Python `LlamaRMSNorm` directly.

### Why the two implementations differ numerically

The two RMSNorm implementations differ in the **precision of the weight multiplication step**:

**Transformers Python `LlamaRMSNorm`** (old path):
```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype                              # fp16
    hidden_states = hidden_states.to(torch.float32)                # cast to fp32
    variance = hidden_states.pow(2).mean(-1, keepdim=True)         # fp32
    hidden_states = hidden_states * torch.rsqrt(variance + eps)    # fp32
    return self.weight * hidden_states.to(input_dtype)             # MULTIPLY IN FP16
```

**SGLang `sgl_kernel.rmsnorm`** (new path):
```
internally: normalize in fp32, multiply weight in fp32, THEN cast result to fp16
```

| Step | Transformers Python | SGLang CUDA kernel |
|---|---|---|
| Normalization | fp32 | fp32 |
| **Weight multiply** | **fp16** (`weight * x.to(fp16)`) | **fp32** (`(x * weight).to(fp16)`) |

### Empirical verification

Single-layer difference:
```
Kernel vs Transformers (fp16 mul):  Max diff = 0.00390625, Mean diff = 0.00011139
Kernel vs fp32-multiply reference:  Max diff = 0.00000000, Mean diff = 0.00000000
```

The kernel is an **exact match** for fp32 weight multiplication, confirming the precision difference.

After 32 transformer-like layers (RMSNorm + Linear + Residual):
```
Max abs diff:       0.0781
Mean abs diff:      0.010253
Cosine similarity:  0.99999833
```

Representations stay very close (cosine sim ~1.0), but individual element differences of ~0.01 are enough to flip borderline logit rankings.

### How this affects MMLU

MMLU evaluates by comparing logits for tokens A, B, C, D. When the correct answer's logit is close to the second-best answer's logit, a small numerical shift (~0.01) can flip the ranking. On a 64-example MMLU evaluation, flipping ~1 question causes a ~1.5% score drop (0.65 to 0.64).

### Other changes in the commit (ruled out)

| Change | Impact on MMLU? |
|---|---|
| `logit_scale` parameter added to `LogitsProcessor` | **No** -- `meta-llama/Llama-3.1-8B-Instruct` does not have a `logit_scale` config attribute; defaults to 1.0 which is a no-op |
| Linear layer replacement changes (new styles: `colwise_rep`, `rowwise_rep`) | **No** -- TP=1 in CI, no actual parallelism difference |
| Attention instances changed from `list` to `dict` | **No** -- same access pattern, same RadixAttention instances |
| Meta device initialization + AutoWeightsLoader | **No** -- weights are loaded correctly (catastrophic failure would be obvious) |
| Pipeline parallel support | **No** -- PP=1 in CI |
| `_init_on_device_without_buffers` (buffers stay on CPU) | **No** -- rotary embedding buffers (`inv_freq`) are handled by Transformers internally |

---

## Is This Fixable?

**Yes.** There are three options:

### Option A: Skip RMSNorm replacement for the transformers backend (Recommended)

The simplest fix. Remove the RMSNorm replacement branch in `recursive_replace()` so the transformers backend keeps using the original Transformers Python RMSNorm.

**File**: `python/sglang/srt/models/transformers.py`

**Current code** (~line 780):
```python
elif child_module.__class__.__name__.endswith("RMSNorm"):
    new_module = replace_rms_norm_class(
        child_module,
        self.text_config.hidden_size,
    )
```

**Fix**: Delete these 4 lines. RMSNorm modules will fall through to the `else` branch which recurses into children -- a no-op for leaf modules like RMSNorm.

After applying the fix, also restore the MMLU threshold in `test/registered/models/test_transformers_models.py`:
```python
cls.mmlu_lower_bound = 0.65  # restore from 0.64
```

**Trade-off**: Slight performance loss from using Python RMSNorm instead of the fused kernel. In practice this is negligible because the transformers backend's bottleneck is the HF model forward pass, not individual RMSNorm operations.

**Risk**: The `replace_rms_norm_class` function also wraps the norm in `HFCompatibleRMSNorm` to handle 3D inputs (batch dimension). The Transformers Python RMSNorm already handles arbitrary dimensions natively (it operates on the last dimension), so no wrapper is needed.

### Option B: Match Transformers' precision in the kernel path (More complex)

SGLang's `RMSNorm` already has a `cast_x_before_out_mul` flag in `forward_native` that matches Transformers' fp16-multiply behavior. However, `forward_cuda` ignores this flag and always uses the kernel.

**Changes needed**:
1. Modify `replace_rms_norm_class` to set `cast_x_before_out_mul=True`
2. Modify `RMSNorm.forward_cuda` to fall back to `forward_native` when `cast_x_before_out_mul=True`
3. OR modify the `sgl_kernel.rmsnorm` kernel to accept a flag for fp16 weight multiply

This is significantly more invasive and risky.

### Option C: Keep threshold at 0.64 (No code change)

**Justification**: The fp32 weight multiply in the kernel is actually *more* mathematically precise than fp16. The "regression" is a different numerical path, not a quality bug. The native SGLang backend uses the same kernel and scores well on MMLU. The 1% score difference on 64 MMLU examples (~0.6 questions) is within expected noise for such a small evaluation set.

---

## Investigation Methodology

This section documents how the precision difference between `LlamaRMSNorm` and `sgl_kernel.rmsnorm` was identified.

### Step 1: Reading Transformers' LlamaRMSNorm source

The Python source was retrieved at runtime via `inspect`:

```python
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import inspect
print(inspect.getsource(LlamaRMSNorm))
```

This revealed the key line — it casts `hidden_states` back to `input_dtype` (fp16) **before** multiplying with `weight`:

```python
return self.weight * hidden_states.to(input_dtype)  # both operands fp16 -> multiply in fp16
```

### Step 2: Reading SGLang's RMSNorm implementation

The SGLang `RMSNorm` class lives at `python/sglang/srt/layers/layernorm.py:148`. Its `forward_cuda` method (line 177) calls:

```python
out = rmsnorm(x, self.weight.data, self.variance_epsilon)
```

where `rmsnorm` is imported from `sgl_kernel`. The Python wrapper at `sgl-kernel/python/sgl_kernel/elementwise.py:76` provides a docstring but no implementation details — the actual computation is in compiled C++/CUDA code, so **the internal precision of the weight multiply could not be determined from source code alone**.

### Step 3: Empirical test to determine the kernel's internal behavior

Since the kernel source was opaque, a test was written to compare the kernel output against two reference implementations with known precision:

```python
import torch, sgl_kernel

torch.manual_seed(42)
x = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
w = torch.randn(4096, dtype=torch.float16, device='cuda')
eps = 1e-5

out_kernel = sgl_kernel.rmsnorm(x, w, eps)

# Normalize in fp32 (shared by both references)
x_fp32 = x.to(torch.float32)
variance = x_fp32.pow(2).mean(-1, keepdim=True)
x_normed = x_fp32 * torch.rsqrt(variance + eps)

# Reference A: multiply in fp16 (Transformers style)
out_fp16_mul = w * x_normed.to(torch.float16)

# Reference B: multiply in fp32 then cast (SGLang native style)
out_fp32_mul = (x_normed * w.to(torch.float32)).to(torch.float16)
```

Results:
```
Kernel vs Reference A (fp16 multiply):  Max diff = 0.00390625
Kernel vs Reference B (fp32 multiply):  Max diff = 0.00000000  <-- exact match
```

The **exact match with the fp32 path** conclusively shows that the kernel performs the weight multiplication in fp32 internally, then casts to fp16. This is the source of the numerical divergence from Transformers' fp16-multiply behavior.

### Step 4: Compound effect simulation

To verify that per-layer differences are large enough to affect MMLU, a simulation ran both paths through 32 transformer-like layers (RMSNorm + Linear + Residual):

```python
# For each of 32 layers:
#   Path A (Transformers): Python RMSNorm -> linear -> residual add
#   Path B (kernel):       sgl_kernel.rmsnorm -> linear -> residual add

# Result after 32 layers:
# Mean abs diff:      0.010253
# Cosine similarity:  0.99999833
```

The cosine similarity stays near 1.0, but individual element diffs of ~0.01 are sufficient to flip borderline logit rankings in MMLU's A/B/C/D token comparison.

---

## Recommendation

**Option A** is the best balance of simplicity, safety, and correctness:
- 4-line deletion
- Preserves exact numerical compatibility with the Transformers reference implementation
- Negligible performance impact
- Lets the MMLU threshold be restored to 0.65

If performance of the transformers backend's RMSNorm becomes a concern in the future, Option B can be revisited.
