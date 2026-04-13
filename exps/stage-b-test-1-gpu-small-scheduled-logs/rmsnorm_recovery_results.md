# RMSNorm Accuracy Recovery — Results

**Analysis doc**: [transformers_mmlu_regression_analysis.md](transformers_mmlu_regression_analysis.md)
**Test**: `TestTransformersFallbackTorchAO::test_mmlu`
**Model**: `meta-llama/Llama-3.1-8B-Instruct` + `int4wo-128`

---

## Executive Summary

Option B1-Native restored the MMLU score from **0.65625 → 0.703125** (full recovery) with a throughput of **782 tok/s** — approximately equal to Option A (784 tok/s), and ~10% below the unpatched baseline (872 tok/s).

The throughput cost is inherent to the Python `forward_native` path. A Triton kernel (Option B-Triton) was not implemented since B1-Native met the accuracy target and the throughput result aligned with Option A; the transformers backend's bottleneck is the HF model forward pass, not individual RMSNorm operations.

| Option | MMLU score | Throughput (tok/s) | Latency (s) | Test | Delta vs baseline |
|---|---|---|---|---|---|
| **Baseline (unpatched)** | 0.65625 | 872.5 | 23.35 | PASSED | — |
| **Option A** (remove kernel) | 0.703125 | 784.0 | 25.60 | PASSED | +7.1%, −10.2% thpt |
| **B1-Native (this fix)** | **0.703125** | **782.2** | **25.65** | **PASSED** | **+7.1%, −10.3% thpt** |

---

## Background

Commit `34ddf135fd` introduced `recursive_replace()` in `TransformersBase`, which replaced every `*RMSNorm` module with SGLang's `sgl_kernel.rmsnorm` kernel. The kernel normalizes in fp32 and multiplies the weight in fp32 before casting to fp16 — while HF's native Python `LlamaRMSNorm` casts the normalized output to fp16 *first*, then multiplies by the weight in fp16.

This fp16 vs fp32 weight-multiply difference compounded through 32+ transformer layers to produce mean element-wise diffs of ~0.01, sufficient to flip 3 borderline MMLU questions (0.703 → 0.656) under int4wo-128 quantization.

See the [regression analysis doc](transformers_mmlu_regression_analysis.md) for the empirical verification.

---

## Implementation: B1-Native

### Changes made

**`python/sglang/srt/layers/layernorm.py`** — `RMSNorm.forward_cuda`

Added an early dispatch before the `sgl_kernel.rmsnorm()` call:

```python
if self.cast_x_before_out_mul and residual is None:
    out = self.forward_native(x, None, None)
    if needs_reshape:
        out = out.reshape(original_shape)
    return out
```

`forward_native` already implements `cast_x_before_out_mul` correctly: when `True` it does `self.weight * x.to(fp16)` (fp16 weight multiply, matching HF semantics).

The `residual is None` guard is safe because HF model architectures never pass `residual` to their `*RMSNorm` modules — residuals are managed externally. This covers 100% of the transformers backend use case.

**`python/sglang/srt/models/transformers.py`** — `replace_rms_norm_class`

Added `cast_x_before_out_mul=True` in the non-Gemma `else` branch:

```python
kwargs["cast_x_before_out_mul"] = True  # match HF fp16-weight-multiply semantics
```

**Scope**: The change is bounded to `replace_rms_norm_class`, which is only called from the transformers backend. The native SGLang model loading path is unaffected. The Gemma path is excluded (Gemma uses `x * (1 + weight)` semantics, handled by a separate branch).

**`python/sglang/test/test_layernorm.py`** — new unit test

Added `TestRMSNormCastXBeforeOutMul.test_rmsnorm_cast_x_before_out_mul` that verifies the `cast_x_before_out_mul=True` path produces output with zero bit-level difference from HF's fp16-weight-multiply reference.

### Result

```
[METRIC] mmlu_score=0.703125
{'other': 0.75, 'stem': 0.545, 'humanities': 0.696, 'social_sciences': 0.786,
 'score': 0.703125, 'latency': 25.65, 'output_throughput': 782.2}
PASSED
```

---

## Final Ranking

| Rank | Option | Score | Throughput | Notes |
|---|---|---|---|---|
| 1 | **B1-Native (implemented)** | 0.703 | 782 | Simple, safe, full accuracy recovery |
| 2 | B-Triton (not implemented) | ~0.703 (expected) | ~850-872 (expected) | Would recover throughput; higher implementation risk |
| 3 | Option A (tested) | 0.703 | 784 | Identical outcome to B1-Native; no kernel in HFCompatibleRMSNorm |

Option B-Triton remains viable if throughput of the transformers backend becomes a bottleneck in the future.

---

## Appendix A — Code Diffs

### layernorm.py (forward_cuda)

```diff
+        if self.cast_x_before_out_mul and residual is None:
+            out = self.forward_native(x, None, None)
+            if needs_reshape:
+                out = out.reshape(original_shape)
+            return out
         if residual is not None:
```

### transformers.py (replace_rms_norm_class)

```diff
         kwargs["has_weight"] = getattr(rms_norm, "with_scale", True)
         if weight_meta is not None:
             kwargs["weight_dtype"] = weight_meta.dtype
         else:
             kwargs["has_weight"] = False
+        kwargs["cast_x_before_out_mul"] = True  # match HF fp16-weight-multiply semantics
         base_cls = RMSNorm
         norm = base_cls(**kwargs)
```

### test_layernorm.py (new test class, abbreviated)

```python
class TestRMSNormCastXBeforeOutMul(CustomTestCase):
    def test_rmsnorm_cast_x_before_out_mul(self):
        # Reference: HF LlamaRMSNorm semantics
        ref = w * x_normed.to(torch.float16)
        norm = RMSNorm(hidden_size, eps=eps, cast_x_before_out_mul=True, weight_dtype=torch.float16)
        norm.weight.data.copy_(w)
        out = norm(x)
        assert (out - ref).abs().max() == 0.0
```

---

## Appendix B — Category Breakdown

| Category | Baseline | B1-Native |
|---|---|---|
| other | — | 0.750 |
| stem | — | 0.545 |
| humanities | — | 0.696 |
| social_sciences | — | 0.786 |
| **overall** | **0.656** | **0.703** |

(Baseline per-category breakdown not collected; overall baseline score from `main.txt`.)
