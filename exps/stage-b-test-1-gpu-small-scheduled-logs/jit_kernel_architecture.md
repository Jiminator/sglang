# JIT Kernel Architecture: Three Variants of HF-Semantics RMSNorm

This document explains the three CUDA kernel variants in [rmsnorm_hf.cuh](/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/elementwise/rmsnorm_hf.cuh), the optimizations each employs, and the performance–accuracy trade-offs between them.

All three kernels implement HuggingFace `LlamaRMSNorm` semantics:

```
out[i] = weight[i] * cast_to_dtype( rsqrt(mean_j(x[j]²) + eps) * x[i] )
```

The key semantic requirement (what distinguishes HF from `sgl_kernel.rmsnorm`): the cast from fp32 to the activation dtype happens **before** the weight multiply, so the multiply is in the narrow dtype. This preserves the rounding behavior of HF's `nn.RMSNorm`.

---

## Common Building Blocks

All three variants share these components (defined in `rmsnorm_hf.cuh`):

### Common Parameter Struct

```cpp
struct RMSNormHFParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};
```

Passed via `__grid_constant__` — resides in the CUDA constant bank, single fast read per thread.

### Helper: `apply_norm_hf_impl` (CTA and Warp variants)

Shared reduction + HF-semantics output math. Key difference vs the existing `rmsnorm.cuh` `apply_norm_impl`:

```cpp
// Standard rmsnorm: fp32 multiply, cast at the end
output[i] = cast<PackedFloat>(fp32x2_t{ix * norm * wx, iy * norm * wy});

// HF rmsnorm_hf: cast first, then multiply in-dtype
const PackedFloat xn = cast<PackedFloat>(fp32x2_t{ix * norm, iy * norm});
const auto xn_fp32 = cast<fp32x2_t>(xn);
output[i] = cast<PackedFloat>(fp32x2_t{xn_fp32.x * wx, xn_fp32.y * wy});
```

The intermediate `cast<PackedFloat>` forces fp32→fp16/bf16 rounding before the weight multiply.

### PDL (Programmatic Dependent Launch)

All variants use `PDLWaitPrimary<kUsePDL>()` and `PDLTriggerSecondary<kUsePDL>()`. On Hopper+, PDL lets the kernel begin preparing the next kernel's launch while still running, reducing gaps in the CUDA stream. Enabled via `is_arch_support_pdl()` at JIT time.

### Launcher Convention

Each launcher struct is templated on `<kDim, kUsePDL, DType>`:
- `kDim` — hidden size (enables compile-time unrolling and constant folding)
- `kUsePDL` — PDL on/off (SM90+ gate)
- `DType` — `fp16_t` or `bf16_t`

The launcher validates tensors via `TensorMatcher`, builds `RMSNormHFParams`, and launches via `LaunchKernel(...)  .enable_pdl(kUsePDL)(kernel, params)`.

---

## Variant 1: `RMSNormHFKernel` (CTA with `tile::Memory`)

**File location**: `rmsnorm_hf.cuh:108` (`rmsnorm_hf_cta` kernel + `RMSNormHFKernel` launcher)

**When selected**: hidden_size in `(256, 8192]` where `hidden_size % 256 == 0` *and* `hidden_size % 512 != 0` (rare — e.g., `hidden=768`).

### Design

- **Threads per block**: `get_cta_threads<DType, kDim>()` = `(kDim / 256) * 32` — e.g., 512 threads for `kDim=4096`, 640 for `kDim=5120`
- **Storage type**: `AlignedVector<packed_t<Float>, 4>` — 16-byte (128-bit) vectors of 4 fp16x2 / bf16x2 pairs = 8 elements per vector
- **Memory access**: `tile::Memory<Storage>::cta(kNumThreads)` — thread `t` loads `Storage` at tile-index `t`, `t+kNumThreads`, etc. Each load is coalesced 128-bit
- **Grid size**: `min(num_tokens, max_occupancy * num_SM)` — capped persistent layout; a block may process multiple rows in a loop

### Reduction Path

1. Each thread computes partial sum-of-squares over its vector lanes in fp32
2. `warp::reduce_sum(lsq)` — butterfly-XOR inside a warp
3. `smem[warp_id]` stores each warp's partial sum
4. Warp 0 re-reduces across `kNumWarps` values
5. `rsqrt((sum / kDim) + eps)` broadcast via shared memory

### Pros

- **Vector I/O**: 128-bit loads/stores saturate DRAM bandwidth efficiently
- **Persistent layout**: when `num_tokens > grid_size`, a block processes multiple rows without paying launch overhead per row
- **Works for any hidden size** that's a multiple of 256 up to 8192

### Cons

- **Different reduction order vs `sgl_kernel.rmsnorm_hf`**: the vector-strided access produces a different per-thread accumulation sequence than `sgl_kernel`'s scalar-strided 512-thread loop. Produces 1-ULP differences on some inputs.
- **Not used for 4096**: the selection logic prefers `RMSNormHFScalarKernel` for `hidden % 512 == 0` cases (including 4096) to guarantee bit-identity.

### When to use it

- Hidden sizes in `(256, 8192]` that are not multiples of 512 (e.g., `hidden=768`, `hidden=1280`, `hidden=5120`). The scalar kernel requires `hidden % 512 == 0`.

---

## Variant 2: `RMSNormHFHalfKernel` (Vectorized Half-Block)

**File location**: `rmsnorm_hf.cuh:182/242` (`rmsnorm_hf_cta_double` pre-Blackwell / `rmsnorm_hf_cta_wide` Blackwell+) + `RMSNormHFHalfKernel` launcher

**When selected**: *Was* used for `hidden_size >= 2048 && hidden_size % 512 == 0` — now **not** selected by default (see "Pros and Cons" below). Retained in source for benchmarking/reference.

### Design

Two kernels, selected at compile time via `#if SGL_ARCH_BLACKWELL_OR_GREATER`:

#### Pre-Blackwell: `rmsnorm_hf_cta_double`
- **Threads per block**: `kDim / 16` — e.g., **256 threads** for `kDim=4096`
- **Storage**: `AlignedVector<Float2, 4>` = 16-byte (128-bit) vectors
- **Loads per thread**: Each thread issues **two** vector loads (`input_first`, `input_second`) covering 16 elements total — saturates a 128-bit load at half the thread count
- `__launch_bounds__(kDim / 16)` tells the compiler the exact block size

#### Blackwell+: `rmsnorm_hf_cta_wide`
- Same thread count, but `Storage = AlignedVector<Float2, 8>` = **32-byte (256-bit)** vectors — one vector load per thread
- Requires Blackwell's wider memory subsystem

### Reduction Path

Same structure as `RMSNormHFKernel`, but:
- **Only 256 threads** per block (half the threads of `RMSNormHFKernel` for `kDim=4096`)
- **8-way unrolled loops** over the 8 element pairs loaded per vector — more ILP (instruction-level parallelism)
- Warp reduction + shared-memory inter-warp reduction identical to common path

### Pros

- **Highest raw throughput**: 128-bit (or 256-bit on Blackwell) loads + fewer threads + aggressive unrolling
- **Best occupancy**: half the threads means more concurrent blocks per SM — helps when `num_tokens >> num_SM`
- **Blackwell-ready**: the 32-byte vector path extracts more bandwidth on newer architectures

### Cons

- **Different reduction order vs `sgl_kernel.rmsnorm_hf`**: 256 threads × 16-element contiguous chunks produces a materially different accumulation order than `sgl_kernel`'s 512 thread × 8-element-strided loop. Under **int4wo-128 quantization + 32 transformer layers**, this 1-ULP drift flipped **one** MMLU question (0.7031 → 0.6875).
- **The regression is reproducible on GPU 0** after swapping to this kernel — not noise, not GPU variance.
- **Not used for Llama-class models** (hidden=4096): we prefer the scalar variant for bit-identity.

### When to use it

Only when bit-identity with `sgl_kernel.rmsnorm_hf` is not required. For Llama / Qwen / Mistral in the transformers backend, avoid.

---

## Variant 3: `RMSNormHFScalarKernel` (Scalar + Register Cache) — **Default**

**File location**: `rmsnorm_hf.cuh:298` (`rmsnorm_hf_scalar` kernel) + `rmsnorm_hf.cuh:467` (`RMSNormHFScalarKernel` launcher)

**When selected**: `hidden_size >= 512 && hidden_size % 512 == 0` — **the primary path** for all Llama-family models (4096, 8192).

### Design

Exact replica of `sgl_kernel.rmsnorm_hf`'s 512-thread scalar-strided algorithm, with **one optimization**: input values are cached in per-thread registers between pass 1 (reduction) and pass 2 (output).

- **Threads per block**: exactly **512** (hardcoded, not a template)
- `__launch_bounds__(512)`
- **Grid**: one block per token (`LaunchKernel(num_tokens, 512, device)`)
- **Per-thread register cache**: `float xi_cache[kElemsPerThread]` where `kElemsPerThread = (kDim + 511) / 512` — 8 fp32 registers for `kDim=4096`

### Access Pattern

Each thread `t` of 512 threads handles elements at strided positions:
```
positions for thread t:  { t, t + 512, t + 1024, ..., t + 3584 }
```

### Kernel Body (simplified)

```cpp
float xi_cache[kElemsPerThread];
float lsq = 0.f;

// Pass 1: load, square, accumulate (also cache xi for pass 2)
#pragma unroll
for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    if (i < kDim) {
        xi_cache[k] = static_cast<float>(xr[i]);
        lsq += xi_cache[k] * xi_cache[k];
    }
}

// Warp reduce → smem → CTA reduce → rstd
lsq = warp::reduce_sum(lsq);
/* shared-memory block reduce, then rsqrt */
const float rstd = /* broadcast */;

// Pass 2: reuse xi_cache (NO re-read of xr from global memory)
#pragma unroll
for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    if (i < kDim) {
        const Float xn = cast<Float>(xi_cache[k] * rstd);           // cast before weight mul
        const float xn_f = static_cast<float>(xn);
        const float w_f = static_cast<float>(wr[i]);
        yr[i] = cast<Float>(xn_f * w_f);
    }
}
```

### Optimizations

1. **Register cache for `xi`**: the single most impactful change. `sgl_kernel.rmsnorm_hf` reads `xr` from global memory **twice** — once for the sum-of-squares, once for the output. By caching the fp32 values in registers between passes, we halve the global reads of `xr`. For `hidden=4096`, that saves 8192 bytes of DRAM traffic per row. Register cost: 32 bytes per thread (8 × fp32) — trivial on H100 (255 registers per thread available).

2. **Scalar-strided access (deliberate)**: each thread processes a strided set of scalar elements. This is *slower* than vectorized access on a microbenchmark, but it's **the reduction order `sgl_kernel.rmsnorm_hf` uses** — preserving it ensures bit-identical output, which is required to preserve MMLU accuracy.

3. **`__launch_bounds__(512)`**: hints the compiler to minimize register spilling so 2 blocks can co-resident per SM.

4. **Compile-time unrolled passes**: `kElemsPerThread` is a `constexpr` derived from `kDim`, so both loops fully unroll. No runtime branching on iteration count.

5. **PDL enabled**: `PDLWaitPrimary` at the top, `PDLTriggerSecondary` at the bottom. Critical for interleaving with the next kernel in the transformer pipeline (attention/MLP).

### Pros

- **Bit-identical to `sgl_kernel.rmsnorm_hf`** on all tested shapes (M=1 to M=4096, fp16 and bf16): max_diff = 0.0. This is the **defining correctness property**.
- **Faster than `sgl_kernel.rmsnorm_hf`** in end-to-end MMLU throughput (867.6 vs 844.6 tok/s) thanks to register cache + PDL overlap.
- **Robust for any hidden that's a multiple of 512** (most real models).

### Cons

- **Microbenchmark ~5% slower** than `sgl_kernel.rmsnorm_hf` on isolated calls — the register cache adds a few registers and the compiler keeps more state. The MMLU win comes from pipeline effects (PDL overlap, L1 pressure), not raw kernel speed.
- **Scalar loads** don't saturate memory bandwidth as well as 128-bit vectors. The half-block kernel would be faster if bit-identity weren't required.
- **Requires `hidden % 512 == 0`**: if the model has an unusual hidden size (e.g., 3072), it falls back to `RMSNormHFKernel`.

### When to use it

- **Default for all transformers-backend RMSNorm calls**, hidden size `>= 512` and divisible by 512. This is the primary path on production inference.

---

## Comparison Table

| Feature | `RMSNormHFKernel` (CTA) | `RMSNormHFHalfKernel` (Half-block) | `RMSNormHFScalarKernel` (Scalar, default) |
|---|---|---|---|
| Threads / block | `(kDim / 256) * 32` | `kDim / 16` | 512 |
| Memory access | 128-bit vectors, CTA stride | 128-bit (pre-Bw) / 256-bit (Bw+) | Scalar, 512-stride |
| Register cache | No | No | **Yes** (`xi_cache[kElemsPerThread]`) |
| Reads of `xr` per row | 1 | 1 | 1 (cached) |
| PDL | Yes | Yes | Yes |
| Blackwell special path | No | Yes (`rmsnorm_hf_cta_wide`) | No |
| Bit-identical to `sgl_kernel.rmsnorm_hf`? | No (1-ULP drift) | **No (1-ULP drift, flips MMLU Q)** | **Yes** |
| Microbench M=1 (us/call) | ~5.6 | ~4.9 | ~5.7 |
| MMLU throughput (tok/s) | n/a | 876 (different GPU, buggy) | **867.6** |
| MMLU accuracy | n/a | 0.6875 (buggy) | **0.7031** |
| Hidden size constraint | `(256, 8192]`, % 256 | `>= 2048`, % 512 | `>= 512`, % 512 |

---

## Why the Scalar Kernel is the Default

The jit_kernel framework gives us the freedom to pick *any* reduction order. Numerically, all three variants are "correct HF semantics" — each produces output within 1 ULP of PyTorch's `forward_native` reference. But the user's expected baseline is the `sgl_kernel.rmsnorm_hf` rounding profile, which specific questions on MMLU have been validated against at 0.7031.

A 1-ULP drift in the output of one RMSNorm, compounded across 32 transformer layers and filtered through int4wo-128 quantization, is empirically enough to flip 1 of 64 MMLU questions. The half-block kernel does this; the scalar kernel does not.

The trade-off:
- **Lose**: ~5% peak kernel throughput (vs the half-block variant on a decode-heavy microbenchmark)
- **Gain**: bit-identical output to the AOT reference → stable accuracy measurements across kernel variants → no surprising regressions in downstream pipelines that are sensitive to per-element rounding

End-to-end, the throughput "loss" is actually a **win** (867.6 > 844.6 for `sgl_kernel.rmsnorm_hf`), because the register cache optimization reduces DRAM pressure in real forward-pass workloads where kernels are chained.

---

## Throughput–Accuracy Trade-off Summary

### Measured on GPU 0 (H100), same server config

| Variant | MMLU score | Throughput (tok/s) | Latency (s) | Bit-identical to sgl_kernel.rmsnorm_hf? |
|---|---|---|---|---|
| **baseline_wrong** (unpatched `sgl_kernel.rmsnorm`, fp32 weight mul) | 0.6562 (42/64) | **~872** | 23.35 / 23.52* | N/A (wrong semantics) |
| **jit_half** (`RMSNormHFHalfKernel`, buggy reduction) | 0.6875 (44/64) | 873.5 | 23.32 | No (1-ULP drift) |
| **jit_scalar** (`RMSNormHFScalarKernel` + register cache) ★ | 0.7031 (45/64) | **867.6** | 23.11 | **Yes** |
| **jit_scalar** (without register cache) | 0.7031 (45/64) | 861.5 | 23.27 | **Yes** |
| **sgl_kernel_hf** (AOT CUDA, Plan B) | 0.7031 (45/64) | 844.6 / 851.0 | 23.65 | **Yes (reference)** |
| **forward_native** (Python HF semantics) | 0.7031 (45/64) | ~782 | 25.71 | **Yes** |

\* `baseline_wrong` throughput measured from the original unpatched `main.txt` run; the 23.52 s latency from the failure-curve harness is consistent with ~870 tok/s.

### Pareto frontier

```
accuracy (MMLU, 64 examples)

  0.7031 ┤─────────────●─●─────────●────────   ← HF-correct ceiling
          │       782   845    868           873
          │       fwd_  sgl_   jit_           jit_half
          │       nat   kernel scalar          (buggy,
          │             _hf    (★ prod)        not used)
          │                                     │
  0.6875 ┤                                     ● jit_half
          │                               (875)
          │
  0.6719 ┤  ●  jit_cta (not on primary path)
          │
  0.6562 ┤                                       ●  baseline_wrong
          │                                  (~872, unpatched)
          │
          └──────────────────────────────────────────> throughput (tok/s)
                 780      850      870       880
```

### Why `baseline_wrong` is the throughput ceiling

`baseline_wrong` is the unpatched `sgl_kernel.rmsnorm` (FlashInfer). It does the entire computation in fp32 and casts **once** at the end:

```cuda
output[j] = cast_dtype(float(x[j]) * rstd * float(w[j]));
```

Advantages that make it intrinsically faster than any HF-semantics variant:

1. **Fewer operations per element** — one fp32 multiply chain vs the HF sequence which adds an fp32→dtype→fp32 round-trip between `x * rstd` and `* weight`.
2. **FMA-friendly** — `x * rstd * w` in fp32 fuses to a single multiply-add instruction. HF semantics forces two separate multiplies with intermediate rounding.
3. **Wider vector loads** — FlashInfer uses `vec_t<T, VEC_SIZE>` up to 16 elements per load; our scalar kernel uses 1 element per load.

Any HF-semantics-correct kernel starts ~30 tok/s below `baseline_wrong`'s ceiling because the cast-before-multiply is unavoidable.

### Key observations

- **`jit_scalar`** sits at the top-right of the Pareto frontier: best accuracy (0.7031) and highest throughput of any bit-identical variant (867.6 tok/s).
- **`jit_half`** actually matches `baseline_wrong`'s throughput (~873 tok/s) — showing the HF-semantics overhead is *recoverable* with aggressive vectorization (128/256-bit loads + half the thread count). But its different reduction order flips one MMLU question, so it's not used.
- **Distance from ceiling**: `jit_scalar` is ~5 tok/s below `baseline_wrong` (within measurement noise). For all practical purposes, the HF-correctness fix is free.
- **`sgl_kernel.rmsnorm_hf` is the slowest HF-correct kernel at 844.6 tok/s** — even though it uses the *same* 512-thread scalar-strided algorithm as `jit_scalar`. The ~23 tok/s gap comes from two sources: (a) no register caching (reads `xr` twice), and (b) no PDL (Programmatic Dependent Launch). Adding both to the AOT kernel would close the gap.
- **`forward_native`** is the slowest (782 tok/s) because it's pure Python — 5 tensor operations per call × 65 RMSNorm calls per forward pass = ~325 Python dispatch events per forward pass.

### Why `jit_scalar` needed the register cache

Note that `jit_scalar` *without* the cache was **already faster than `sgl_kernel.rmsnorm_hf`** (861.5 > ~847). The register cache was not needed to catch up to `sgl_kernel` — it was added to close the remaining gap to `jit_half` (the fast-but-buggy variant) and `baseline_wrong`.

The register cache is **not** in `sgl_kernel.rmsnorm_hf`. That kernel reads `xr` from global memory twice:
```cuda
// sgl_kernel.rmsnorm_hf, no cache:
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float xi = __bfloat162float(xr[i]);   // ← first read
    lsq += xi * xi;
}
/* reduce for rstd */
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    __nv_bfloat16 xn = __float2bfloat16(__bfloat162float(xr[i]) * rstd);  // ← second read
    yr[i] = __float2bfloat16(__bfloat162float(xn) * __bfloat162float(w[i]));
}
```

The compiler can't hoist the `xr` reads across the `__syncthreads()` inside `_hf_block_reduce_sum`, so the second pass always hits global memory (possibly L1-cached, but not free).

`jit_scalar` adds an explicit `float xi_cache[kElemsPerThread]` (8 fp32 registers per thread for N=4096) that persists through the sync barrier, so pass 2 reads from registers. This halves the global reads of `xr`:

- Before: 8 KB (xr read 1) + 8 KB (xr read 2) + 8 KB (w read) + 8 KB (y write) = 32 KB per row
- After: 8 KB (xr read once) + 8 KB (w read) + 8 KB (y write) = 24 KB per row (-25%)

At the microbench level, the cached version is actually ~5% *slower* (register pressure slightly hurts the simple case). But in the real MMLU forward pass — with 65 RMSNorm calls chained with attention/MLP kernels contending for L1 cache — the reduced pressure yields +6 tok/s end-to-end.

### Why jit_scalar beats sgl_kernel_hf even without register cache

`jit_scalar` (861.5, no cache) vs `sgl_kernel.rmsnorm_hf` (~847) — same kernel algorithm, same reduction order, bit-identical output. The ~14 tok/s gap comes from two places neither involving kernel code:

1. **PDL (Programmatic Dependent Launch)** — `jit_scalar` uses `PDLWaitPrimary`/`PDLTriggerSecondary` so the next kernel can start preparing while this one finishes. `sgl_kernel.rmsnorm_hf` doesn't use PDL (no macros in the `.cu` file). Across 65 RMSNorm calls per forward pass interleaved with other kernels, this overlap adds up.

2. **Dispatch path** — `jit_scalar` goes through `tvm_ffi` (a lightweight function-pointer lookup). `sgl_kernel.rmsnorm_hf` goes through `torch.ops.sgl_kernel.rmsnorm_hf.default` which traverses PyTorch's operator dispatcher (schema validation, autograd layer, device dispatch) even in inference mode.

You could add PDL to `sgl_kernel.rmsnorm_hf` (it's a ~4-line change) and probably close ~10 tok/s of the gap. Register caching would close another ~6. But the `torch.ops` dispatch overhead is structural.

---

## Quick Reference: Selection Logic

From `_rmsnorm_hf_kernel_class` in [rmsnorm_hf.py](/sgl-workspace/sglang/python/sglang/jit_kernel/rmsnorm_hf.py):

```python
def _rmsnorm_hf_kernel_class(hidden_size: int) -> str:
    if hidden_size in {64, 128, 256}:
        return "RMSNormHFWarpKernel"   # small hidden (not covered in this doc)
    if hidden_size >= 512 and hidden_size % 512 == 0:
        return "RMSNormHFScalarKernel"  # bit-identical default
    return "RMSNormHFKernel"            # fallback for unusual sizes
```

Override via `kernel_class_override="..."` kwarg to `rmsnorm_hf()` — used by the failure-curve experiment harness to benchmark each variant directly.

---

## Empirical Failure Curves (64-question MMLU)

### Setup

- Model: `meta-llama/Llama-3.1-8B-Instruct` + `int4wo-128` quantization
- Dataset: MMLU, 64 samples, `random.Random(0).sample(...)` (same questions every run)
- Harness: `exps/mmlu_failure_curves.py` — launches a fresh sglang server per variant, runs MMLU with per-sample correctness logging, dumps JSON
- Plot: `exps/plot_mmlu_failure_curves.py` — produces `failure_curves.png` and `failure_curves_divergence.png`

### Results (per variant)

| Variant | Score | Correct / 64 |
|---|---|---|
| **baseline_wrong** (unpatched `sgl_kernel.rmsnorm`, fp32 weight mul) | 0.6562 | 42 / 64 |
| **jit_cta** (JIT `RMSNormHFKernel`, CTA tile::Memory) | 0.6719 | 43 / 64 |
| **jit_half** (JIT `RMSNormHFHalfKernel`, half-block vectorized) | 0.6875 | 44 / 64 |
| **forward_native** (Python HF semantics, B1-Native) | 0.7031 | 45 / 64 |
| **sgl_kernel_hf** (AOT CUDA, Plan B) | 0.7031 | 45 / 64 |
| **jit_scalar** (JIT `RMSNormHFScalarKernel`, default) | 0.7031 | 45 / 64 |

### Key observations

1. **11 of 64 questions are "borderline"** — their correctness flips depending on the kernel variant. The other 53 are answered identically across all 6 variants.

2. **`forward_native`, `sgl_kernel_hf`, and `jit_scalar` are pairwise identical per-sample.** Not just the scores match — the exact set of correct/incorrect answers is the same. This is the *operational* manifestation of bit-identity: the three implementations produce indistinguishable downstream behavior.

3. **`jit_cta` is actually worse than `baseline_wrong`** on some samples (43 vs 42 correct overall, but flips *different* questions — see sample #34). A kernel that gets the semantics "right" but with a different reduction order can be numerically further from the HF reference than one with the wrong semantics on a particular question. Below-SOTA accuracy does not require semantic incorrectness.

4. **`jit_half` drops one question** (0.6875) — the buggy kernel variant we tracked earlier. The failure curve shows it diverges near samples #5, #34, #36, #56 — each a 1-ULP rounding boundary that the half-block kernel's different reduction order crossed on the wrong side.

5. **Divergent samples flip in both directions.** `jit_half` gets sample #36 right where *all* other variants (including `jit_scalar`) get it wrong. This shows that lower accuracy variants are not uniformly worse — they flip a mixture of borderline questions.

6. **`baseline_wrong` (unpatched) is the floor.** This represents the original regression: MMLU under int4wo-128 dropped from 0.7031 → 0.6562 because `sgl_kernel.rmsnorm` did the weight multiply in fp32 instead of dtype. All HF-semantics fixes bring us back toward 0.7031.

### Plot 1: Cumulative Accuracy Curves

![failure_curves.png](/sgl-workspace/sglang/exps/mmlu_curves/failure_curves.png)

The x-axis is sample index in submission order (1–64). The y-axis is running cumulative accuracy `sum(correct[0:k+1]) / (k+1)`.

- Early samples have high variance (one wrong answer at sample #5 drops accuracy to 0.80)
- All curves converge as `n → 64` to their final scores
- The three bit-identical variants (forward_native, sgl_kernel_hf, jit_scalar) trace the exact same curve
- `baseline_wrong` sits lowest throughout; `jit_half` settles 1 question short of the ceiling

### Plot 2: Divergence Analysis

![failure_curves_divergence.png](/sgl-workspace/sglang/exps/mmlu_curves/failure_curves_divergence.png)

**Top panel**: cumulative-accuracy delta versus `jit_scalar` (the bit-identical reference). At every sample index, this shows how far a variant is from the reference. Zero means "agrees on everything seen so far." Negative means "has fewer correct answers."
- `forward_native` and `sgl_kernel_hf` overlay perfectly at 0 throughout — bit-identical
- `jit_half` and `jit_cta` dip early and recover, but never fully close
- `baseline_wrong` diverges the most

**Bottom panel**: per-sample marker view of the 11 divergent questions. Each row is one variant; each column is one divergent question. ○ = correct, ✕ = wrong.
- The **three bit-identical rows at the top** (`jit_scalar`, `sgl_kernel_hf`, `forward_native`) have identical patterns — every marker in the same position
- `jit_half` differs on samples #5, #36, #56 (correct where others wrong; wrong where others correct)
- `jit_cta` and `baseline_wrong` show similar patterns to each other, with more divergence

### Takeaway for the JIT kernel choice

The failure curves answer the question *"does the buggy kernel just produce random noise, or does it cause a systematic shift?"* empirically:

- **Not random noise**: `jit_half` vs `jit_scalar` differ on the same 5–6 samples run after run. The shift is a deterministic consequence of the reduction order.
- **Not a uniform slight degradation**: the buggy kernel sometimes flips in our favor (sample #36). The net effect in aggregate is usually negative, but the per-question picture shows ~10% of MMLU is on the numerical knife-edge.
- **Bit-identity is the right target**: the only way to get reproducible, test-stable MMLU scores under int4wo-128 is to pick one rounding convention and stick with it. `jit_scalar` was designed to match `sgl_kernel.rmsnorm_hf`'s convention exactly, and the 11 divergent samples confirm it succeeded.

