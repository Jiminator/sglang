# RMSNorm Accuracy Recovery — Full Investigation & Final Results

**Analysis doc**: [transformers_mmlu_regression_analysis.md](transformers_mmlu_regression_analysis.md)
**Test**: `TestTransformersFallbackTorchAO::test_mmlu`
**Model**: `meta-llama/Llama-3.1-8B-Instruct` + `int4wo-128`

---

## Executive Summary

The CUDA C++ inline extension (B-CUDA) achieved **full recovery on both dimensions**:
- **Accuracy**: 0.703125 (fully recovered from regressed 0.65625)
- **Throughput**: 843.9 tok/s (above 840 tok/s target, within 3.2% of main.txt 872 tok/s baseline)
- **Test status**: PASSED

| Option | MMLU score | Throughput (tok/s) | Latency (s) | Test | Notes |
|---|---|---|---|---|---|
| **Baseline (unpatched)** | 0.65625 | 872.5 | 23.35 | PASSED | Regression introduced by `34ddf135fd` |
| **Option A** (remove kernel) | 0.703125 | 784.0 | 25.60 | PASSED | Accuracy fixed, throughput −10% |
| **B1-Native** | 0.703125 | 782.2 | 25.65 | PASSED | Same throughput outcome as A |
| **B-Triton** (custom op wrapper) | 0.703125 | 782.0 | ~25.65 | PASSED | Wrapper overhead ruled out as cause |
| **B-Triton** (direct dispatch) | 0.703125 | 782.0 | ~25.65 | PASSED | Python overhead ruled out as cause |
| **B-CUDA (final)** | **0.703125** | **843.9** | **23.76** | **PASSED** | Both targets met |

---

## Background

Commit `34ddf135fd` introduced `recursive_replace()` in `TransformersBase`, which replaced every `*RMSNorm` module with SGLang's `sgl_kernel.rmsnorm` kernel. The kernel normalizes in fp32 and multiplies the weight in fp32 before casting to the output dtype — while HF's native Python `LlamaRMSNorm` casts the normalized output to the activation dtype *first*, then multiplies by the weight.

This precision difference (fp32-weight-mul vs dtype-weight-mul) compounded through 32+ transformer layers to produce mean element-wise diffs of ~0.01, sufficient to flip 3 borderline MMLU questions (0.703 → 0.656) under int4wo-128 quantization.

The fix required:
1. **Accuracy**: route `cast_x_before_out_mul=True` calls through a path that preserves HF semantics
2. **Performance**: do so with a CUDA kernel, not Python, to avoid throughput regression

---

## Option A — Remove Kernel Replacement (Tested, Not Adopted)

**What it does**: Removes the `kwargs["weight_dtype"] = ...` / kernel replacement path in `replace_rms_norm_class`, so the transformers backend keeps HF's native Python `LlamaRMSNorm` instead of substituting `sgl_kernel.rmsnorm`.

**Result**: Accuracy fully recovered (0.703125). Throughput 784 tok/s — 10% below baseline because every RMSNorm call is now a Python forward pass (~5 tensor ops) instead of a single CUDA kernel launch.

**Why not adopted**: Throughput regression equivalent to all Python-path options. No advantage over B1-Native.

---

## B1-Native — Python forward_native Dispatch (Implemented, Superseded)

### What it does

Adds an early dispatch in `forward_cuda` that routes `cast_x_before_out_mul=True` calls to `forward_native` instead of `sgl_kernel.rmsnorm`. `forward_native` already implements the correct HF semantics when `cast_x_before_out_mul=True`:

```python
# forward_native (existing):
if self.cast_x_before_out_mul:
    x = self.weight * x.to(orig_dtype)   # cast first, multiply in dtype (HF semantics)
else:
    x = (x * self.weight).to(orig_dtype) # multiply in fp32, then cast
```

### Code change — `forward_cuda`

```diff
+        if self.cast_x_before_out_mul and residual is None:
+            out = self.forward_native(x, None, None)
+            if needs_reshape:
+                out = out.reshape(original_shape)
+            return out
         if residual is not None:
             fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
```

### Code change — `transformers.py`

```diff
+        kwargs["cast_x_before_out_mul"] = True  # match HF fp16-weight-multiply semantics
         base_cls = RMSNorm
         norm = base_cls(**kwargs)
```

### Result

```
mmlu_score=0.703125, output_throughput=782.2, latency=25.65
PASSED
```

Accuracy fully recovered. Throughput 782 tok/s — same 10% regression as Option A. The Python path runs ~5 tensor operations per call across 65 RMSNorm calls per Llama-3.1-8B forward pass. At the time, this was attributed to accumulated Python dispatch overhead.

---

## B-Triton — Triton Kernel with Custom Op Wrapper (Tested, Superseded)

### What it does

Writes a Triton kernel that implements HF semantics — normalizes in fp32, casts normalized x to fp16, multiplies weight in fp16 — and wraps it in `@register_custom_op` for CUDA graph compatibility.

### Triton kernel

```python
@triton.jit
def _rmsnorm_fp16_weight_kernel(y_ptr, x_ptr, w_ptr, DIM, EPS, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < DIM
    x_fp32 = tl.load(x_ptr + row * DIM + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x_fp32 * x_fp32, axis=0) / DIM
    rstd = tl.rsqrt(var + EPS)
    x_normed_fp16 = (x_fp32 * rstd).to(tl.float16)  # cast before weight multiply
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)   # fp16 weight
    tl.store(y_ptr + row * DIM + offs, x_normed_fp16 * w, mask=mask)

@register_custom_op(op_name="sglang_rmsnorm_fp16_weight", out_shape="x")
def _rmsnorm_fp16_weight(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)
    x_view = x.reshape(-1, shape[-1])
    y_view = y.reshape(-1, shape[-1])
    M, N = x_view.shape
    with torch.get_device_module().device(x.device):
        _rmsnorm_fp16_weight_kernel[(M,)](y_view, x_view, w, N, eps, BLOCK_N=triton.next_power_of_2(N))
    return y
```

### Dispatch in `forward_cuda`

```python
if self.cast_x_before_out_mul and residual is None:
    if x.dtype == torch.float16:
        out = _rmsnorm_fp16_weight(x, self.weight.data, self.variance_epsilon)
    else:
        out = self.forward_native(x, None, None)
```

### Result and diagnosis

**MMLU throughput: 782 tok/s** — identical to B1-Native. Suspicion: `@register_custom_op` wrapper overhead (2.4× in microbenchmark relative to sgl_kernel).

**Diagnostic test**: Replaced `@register_custom_op` with direct Triton kernel dispatch (no wrapper). Result: **still 782 tok/s**.

This ruled out Python dispatch overhead as the cause. The real cause was still unknown at this point.

### GPU kernel timing (CUDA Events, M=1, N=4096)

| Kernel | GPU time |
|---|---|
| `sgl_kernel.rmsnorm` (FlashInfer) | 5.56–6.42 μs |
| Triton fp16 kernel | 9.51–10.07 μs |

Triton is ~65% slower at the GPU level. Triton tuning (BLOCK_N=512/1024/2048/4096, num_warps=4/8/16) made no material difference — the gap is structural (Triton codegen vs FlashInfer hand-tuned CUDA).

---

## Root Cause Discovery

After ruling out Python dispatch overhead and Triton kernel quality, the focus shifted to **what dtype the model actually uses**.

**Finding**: The MMLU model (`Llama-3.1-8B-Instruct` + `int4wo-128`) loads with **bfloat16** activations:
```
KV cache dtype: torch.bfloat16
```

The fast-path guard in all previous implementations was:
```python
if x.dtype == torch.float16:   # ← always False for bf16 model
```

This evaluated to `False` for **every single RMSNorm call** in the entire MMLU benchmark. All implementations — B1-Native, B-Triton custom op, B-Triton direct — were silently falling through to `forward_native` (pure Python) on every call, regardless of implementation.

The 782 tok/s result was not a consequence of any specific implementation's quality; it was the result of **the fast path never being reached at all**.

The fix was:
1. Change the guard to `if x.dtype in (torch.float16, torch.bfloat16):`
2. Implement a bf16-capable CUDA kernel (Triton handles fp16 only via `tl.float16`)

---

## B-CUDA — CUDA C++ Inline Extension (Final Implementation)

### Design

512-thread warp-level reduction via `__shfl_xor_sync` + 32-slot shared memory. Separate fp16 and bf16 kernel variants. JIT-compiled via `torch.utils.cpp_extension.load_inline` at import time.

The bf16 kernel preserves HF's **double-rounding semantics**: cast `x * rstd` to bf16 first, then multiply with the bf16 weight. This matches `weight_bf16 * round_bf16(normalize_fp32(x))` exactly (max_diff = 0.0 vs HF reference).

### CUDA kernel source

```cuda
// Warp+block reduction helper
static __device__ __forceinline__ float _block_reduce_sum(float val, int threads) {
    for (int m = 16; m > 0; m >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, m);
    __shared__ float sm[32];
    int ln = threadIdx.x & 31, wp = threadIdx.x >> 5;
    if (ln == 0) sm[wp] = val;
    __syncthreads();
    float tot = 0.f;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (threads + 31) / 32) ? sm[threadIdx.x] : 0.f;
        for (int m = 16; m > 0; m >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, m);
        tot = v;
    }
    __shared__ float rstd_s;
    if (threadIdx.x == 0) rstd_s = tot;
    __syncthreads();
    return rstd_s;
}

// fp16: normalize fp32 → cast to fp16 → multiply weight in fp16
__global__ void _sglang_rmsnorm_fp16w_kernel(
    __half* y, const __half* x, const __half* w, int N, float eps
) {
    const __half* xr = x + blockIdx.x * N;
    __half* yr = y + blockIdx.x * N;
    float lsq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float xi = __half2float(xr[i]); lsq += xi * xi;
    }
    float rstd = rsqrtf(_block_reduce_sum(lsq, blockDim.x) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        __half xn = __float2half(__half2float(xr[i]) * rstd);
        yr[i] = __hmul(xn, w[i]);
    }
}

// bf16: same semantics with double-rounding to match HF exactly
__global__ void _sglang_rmsnorm_bf16w_kernel(
    __nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w, int N, float eps
) {
    const __nv_bfloat16* xr = x + blockIdx.x * N;
    __nv_bfloat16* yr = y + blockIdx.x * N;
    float lsq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]); lsq += xi * xi;
    }
    float rstd = rsqrtf(_block_reduce_sum(lsq, blockDim.x) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        // Cast normalized x to bf16 BEFORE multiplying (double-rounding = HF semantics)
        __nv_bfloat16 xn = __float2bfloat16(__bfloat162float(xr[i]) * rstd);
        yr[i] = __float2bfloat16(__bfloat162float(xn) * __bfloat162float(w[i]));
    }
}
```

### `forward_cuda` dispatch (final state)

```python
if self.cast_x_before_out_mul and residual is None:
    if x.dtype in (torch.float16, torch.bfloat16):   # both dtypes now covered
        x_c = x.contiguous()
        ext = _rmsnorm_fp16w_ext                      # direct global (no per-call overhead)
        if ext is not None:
            out = ext.sglang_rmsnorm_fp16w(x_c, self.weight.data, self.variance_epsilon)
        elif x_c.dtype == torch.float16:
            # Triton fallback (fp16 only, retained for environments without nvcc)
            y = torch.empty_like(x_c)
            M, N = x_c.shape
            _rmsnorm_fp16_weight_kernel[(M,)](
                y, x_c, self.weight.data, N, self.variance_epsilon,
                BLOCK_N=triton.next_power_of_2(N),
            )
            out = y
        else:
            out = self.forward_native(x_c, None, None)
    else:
        out = self.forward_native(x, None, None)
    if needs_reshape:
        out = out.reshape(original_shape)
    return out
```

**Key change from all prior implementations**: guard is `in (torch.float16, torch.bfloat16)` — both dtypes route to the CUDA kernel. The original `== torch.float16` was the silent bug.

### GPU Kernel Timing (CUDA Events, M=1, N=4096)

| Kernel | GPU time | vs sgl_kernel |
|---|---|---|
| `sgl_kernel.rmsnorm` (FlashInfer, baseline) | 5.56–6.42 μs | 1.00× |
| Triton fp16 kernel | 9.51–10.07 μs | ~1.65× slower |
| CUDA C++ ext (512 threads) | 4.91–4.98 μs | ~0.80× (**25% faster**) |

### End-to-End Microbenchmark (M=1–64, N=4096, 5000 iterations)

CUDA ext is 5–8% **faster** than sgl_kernel across all decode-relevant batch sizes. No gate failures at any M.

### CUDA Graph Compatibility

Capture and replay verified for both fp16 and bf16 paths via `exps/test_cuda_graph_rmsnorm.py`. Direct kernel launches (no custom op wrapper) are captured correctly by `torch.cuda.CUDAGraph`.

### Accuracy Verification

- fp16: max_diff = 0.0 vs HF LlamaRMSNorm reference (bit-exact)
- bf16: max_diff = 0.0 vs HF reference (double-rounding semantics preserved)

### MMLU Result

```
Total latency: 23.757 s
Score: 0.703
Output throughput: 843.931 token/s
[METRIC] mmlu_score=0.703125
{'other': 0.75, 'stem': 0.636, 'humanities': 0.652, 'social_sciences': 0.786,
 'score': 0.703125, 'latency': 23.757, 'output_throughput': 843.9}
PASSED
```

---

## Reproducing These Results

All commands are run from the **repo root** (`/sgl-workspace/sglang`). nvcc must be on PATH for the load_inline CUDA extension to compile (it is loaded at import time, ~22s cold, <1s from cache).

### 1. Install dependencies (PyPI sgl-kernel, no custom build)

```bash
CUSTOM_BUILD_SGL_KERNEL=false bash scripts/ci/cuda/ci_install_dependency.sh
```

### 2. Unit test — fp16 and bf16 accuracy

```bash
python -m pytest python/sglang/test/test_layernorm.py -xvs -k cast_x
```

Expected: 1 passed, max_diff = 0.0 for both end-to-end path and direct kernel call.

### 3. CUDA graph smoke test

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH python exps/test_cuda_graph_rmsnorm.py
```

Expected: All captures and replays OK, max_diff = 0.0.

### 4. Microbenchmark (end-to-end kernel throughput, fp16 and bf16)

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH python exps/bench_rmsnorm_fp16.py
```

Expected: B-CUDA ext ~5–8% faster than sgl_kernel baseline at all M. No gate failures.

### 5. MMLU integration test

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH python -m pytest \
  test/registered/models/test_transformers_models.py::TestTransformersFallbackTorchAO::test_mmlu \
  -xvs 2>&1 | tee output_v2.txt
```

Expected output:
```
Score: 0.703
Output throughput: 843.x token/s
[METRIC] mmlu_score=0.703125
PASSED
```

### Notes

- The load_inline CUDA extension compiles on first import. If `nvcc` is not in `PATH`, prepend `/usr/local/cuda-12.9/bin` as shown above.
- Compilation result is cached at `~/.cache/torch_extensions/py312_cu129/sglang_rmsnorm_fp16w/`. Delete this directory to force a recompile.
- The root cause of all prior 782 tok/s results was the dtype guard `if x.dtype == torch.float16:` — this always evaluated `False` for the bf16 MMLU model. The fix is `if x.dtype in (torch.float16, torch.bfloat16):`.

---

## Final Ranking

| Rank | Option | Score | Throughput | Notes |
|---|---|---|---|---|
| 1 | **B-CUDA** | 0.703 | 844 | Full recovery; bf16 + fp16 kernel |
| 2 | B-Triton direct | 0.703 | 782 | fp16-only guard missed bf16 model |
| 3 | B-Triton custom op | 0.703 | 782 | Same — wrapper overhead not the issue |
| 4 | B1-Native | 0.703 | 782 | Same — fast path never reached |
| 5 | Option A | 0.703 | 784 | Removes kernel; Python path always |

---

## Appendix A — Files Changed

| File | Change |
|---|---|
| `python/sglang/srt/layers/layernorm.py` | Added CUDA C++ ext (fp16+bf16 kernels); added Triton fp16 kernel; updated `forward_cuda` dispatch |
| `python/sglang/srt/models/transformers.py` | Added `cast_x_before_out_mul=True` in `replace_rms_norm_class` |
| `python/sglang/test/test_layernorm.py` | Added `TestRMSNormCastXBeforeOutMul` (fp16 end-to-end + direct kernel) |
| `exps/bench_rmsnorm_fp16.py` | Microbenchmark script |
| `exps/test_cuda_graph_rmsnorm.py` | CUDA graph smoke test |

---

## Appendix B — Category Breakdown

| Category | Baseline (unpatched) | B-CUDA (final) |
|---|---|---|
| other | — | 0.750 |
| stem | — | 0.636 |
| humanities | — | 0.652 |
| social_sciences | — | 0.786 |
| **overall** | **0.656** | **0.703** |

(Baseline per-category breakdown not collected; overall baseline score from `main.txt`.)
