/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * RMSNorm with HuggingFace semantics:
 *   out = weight * cast_dtype(normalize_fp32(x))
 *
 * This differs from sgl_kernel.rmsnorm (FlashInfer) which computes:
 *   out = cast_dtype(weight_fp32 * normalize_fp32(x))
 *
 * The cast-before-multiply order matches HF LlamaRMSNorm and is required for
 * the transformers backend to produce numerically identical outputs to HF.
 * Supports fp16 and bf16 inputs/weights.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "utils.h"

// ---------------------------------------------------------------------------
// Warp + block reduction: returns sum(val) across all threads in the block.
// Uses shuffle-based intra-warp reduction + shared memory inter-warp reduction.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float _hf_block_reduce_sum(float val, int threads) {
    // Intra-warp reduction
    for (int m = 16; m > 0; m >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, m);
    __shared__ float sm[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = val;
    __syncthreads();
    float tot = 0.f;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (threads + 31) / 32) ? sm[threadIdx.x] : 0.f;
        for (int m = 16; m > 0; m >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, m);
        tot = v;
    }
    __shared__ float tot_s;
    if (threadIdx.x == 0) tot_s = tot;
    __syncthreads();
    return tot_s;
}

// ---------------------------------------------------------------------------
// fp16 kernel: normalize in fp32, cast to fp16 BEFORE multiplying weight (HF semantics)
// ---------------------------------------------------------------------------
__global__ void _sgl_rmsnorm_hf_fp16_kernel(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    int N,
    float eps
) {
    const __half* xr = x + blockIdx.x * N;
    __half* yr = y + blockIdx.x * N;
    float lsq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float xi = __half2float(xr[i]);
        lsq += xi * xi;
    }
    float rstd = rsqrtf(_hf_block_reduce_sum(lsq, blockDim.x) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        // Cast to fp16 first, then multiply weight in fp16 (HF LlamaRMSNorm semantics)
        __half xn = __float2half(__half2float(xr[i]) * rstd);
        yr[i] = __hmul(xn, w[i]);
    }
}

// ---------------------------------------------------------------------------
// bf16 kernel: normalize in fp32, cast to bf16 BEFORE multiplying weight.
// Uses double-rounding: __float2bfloat16(x*rstd) then bf16*bf16 product —
// matching HF's weight_bf16 * round_bf16(normalize_fp32(x)) exactly.
// ---------------------------------------------------------------------------
__global__ void _sgl_rmsnorm_hf_bf16_kernel(
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N,
    float eps
) {
    const __nv_bfloat16* xr = x + blockIdx.x * N;
    __nv_bfloat16* yr = y + blockIdx.x * N;
    float lsq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]);
        lsq += xi * xi;
    }
    float rstd = rsqrtf(_hf_block_reduce_sum(lsq, blockDim.x) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        // Cast normalized x to bf16 BEFORE multiplying (HF double-rounding semantics)
        __nv_bfloat16 xn = __float2bfloat16(__bfloat162float(xr[i]) * rstd);
        yr[i] = __float2bfloat16(__bfloat162float(xn) * __bfloat162float(w[i]));
    }
}

// ---------------------------------------------------------------------------
// Host wrapper: dispatches to fp16 or bf16 kernel based on input dtype
// ---------------------------------------------------------------------------
void sgl_rmsnorm_hf(
    at::Tensor& output,
    at::Tensor& input,
    at::Tensor& weight,
    double eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(2, input);
    CHECK_DIM(2, output);
    CHECK_DIM(1, weight);
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(input.size(1), output.size(1));
    CHECK_EQ(input.size(1), weight.size(0));
    TORCH_CHECK(
        input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
        "rmsnorm_hf: input must be fp16 or bf16, got ", input.scalar_type());
    TORCH_CHECK(input.scalar_type() == output.scalar_type(),
        "rmsnorm_hf: input and output must have the same dtype");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
        "rmsnorm_hf: input and weight must have the same dtype");

    int M = input.size(0), N = input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (input.scalar_type() == at::kHalf) {
        _sgl_rmsnorm_hf_fp16_kernel<<<M, 512, 0, stream>>>(
            (__half*)output.data_ptr<at::Half>(),
            (const __half*)input.data_ptr<at::Half>(),
            (const __half*)weight.data_ptr<at::Half>(),
            N, (float)eps);
    } else {
        _sgl_rmsnorm_hf_bf16_kernel<<<M, 512, 0, stream>>>(
            (__nv_bfloat16*)output.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)input.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)weight.data_ptr<at::BFloat16>(),
            N, (float)eps);
    }
}
