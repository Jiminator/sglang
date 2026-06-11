/* Copyright 2026 SGLang Team. All Rights Reserved.

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

// Deterministic sequence-aware top-K selection for Double Sparsity decode.
//
// Selects, per row, the top-K live finite scores under the exact
// (score DESCENDING, then logical position ASCENDING) contract and emits the
// selected logical positions in ascending order with -1 padding plus
// valid_lengths = min(num_selectable, K), where -inf and NaN are never
// selectable. Work is proportional to each row's
// live window (seq_lens bounds every scan), not the static score width.
//
// One thread block owns one row: four 8-bit radix rounds over order-
// preserving uint32 keys (zeros canonicalized so equal scores tie by
// position) with shared-memory histograms find the exact threshold key and
// tie quota; an ordered block-scan emission pass then admits boundary ties
// lowest-position-first and scatters each selected position directly into
// its ascending output slot. All ordering is block-sequential (the only
// atomics are shared-memory histogram counts, which are order-independent),
// so the result is bit-deterministic run-to-run and across TP ranks. No
// global scratch; a single kernel launch per call (CUDA-graph friendly).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kBins = 256;

__device__ __forceinline__ uint32_t sortable_key(float s) {
  s = s + 0.0f;  // canonicalize -0.0 -> +0.0 so equal scores tie by position
  uint32_t b = __float_as_uint(s);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ bool is_selectable(float s) {
  // -inf marks masked/unwritten slots and NaN is poison — neither is ever
  // selected. +inf (not producible by the scorer) stays selectable as the
  // maximal score, matching the torch reference selector and the Triton
  // suite so all implementations agree on every input.
  return !isnan(s) && s != -INFINITY;
}

// Exclusive block-wide prefix sum over one int per thread (Hillis-Steele in
// shared memory). Returns the exclusive prefix for this thread; *total gets
// the block-wide sum. Caller must __syncthreads() before reusing `buf`.
__device__ __forceinline__ int block_exscan(int v, int* buf, int* total) {
  int tid = threadIdx.x;
  buf[tid] = v;
  __syncthreads();
#pragma unroll
  for (int off = 1; off < kThreads; off <<= 1) {
    int x = (tid >= off) ? buf[tid - off] : 0;
    __syncthreads();
    buf[tid] += x;
    __syncthreads();
  }
  *total = buf[kThreads - 1];
  return buf[tid] - v;
}

__global__ void ds_topk_sequence_order_kernel(
    const float* __restrict__ scores,
    const int* __restrict__ seq_lens,
    int* __restrict__ out_indices,
    int* __restrict__ out_lengths,
    int64_t score_stride,
    int width,
    int max_top_k) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* row_scores = scores + row * score_stride;

  __shared__ unsigned int hist[kBins];
  __shared__ int scan_buf[kThreads];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_target;
  __shared__ int sh_k_target;
  __shared__ int sh_sel_base;
  __shared__ int sh_tie_seen;

  const int n_live = min(max(seq_lens[row], 0), width);

  // Initialize this row's output to the -1 padding in-kernel (true single
  // launch — no host-side fill). The emission phase below only overwrites the
  // selected slots; the first __syncthreads() of phase 1 orders this init
  // before any of it.
  for (int j = tid; j < max_top_k; j += kThreads) {
    out_indices[static_cast<int64_t>(row) * max_top_k + j] = -1;
  }

  // ---- Phase 1: four radix rounds narrow to the exact threshold key. ----
  if (tid == 0) {
    sh_prefix = 0u;
    sh_target = 0;
    sh_k_target = 0;
  }
  __syncthreads();

  for (int round = 0; round < 4; ++round) {
    const int shift = 24 - 8 * round;
    for (int b = tid; b < kBins; b += kThreads) {
      hist[b] = 0u;
    }
    __syncthreads();

    const uint32_t prefix = sh_prefix;
    for (int i = tid; i < n_live; i += kThreads) {
      const float s = row_scores[i];
      if (!is_selectable(s)) continue;
      const uint32_t k = sortable_key(s);
      if (round == 0 || (k >> (shift + 8)) == (prefix >> (shift + 8))) {
        atomicAdd(&hist[(k >> shift) & (kBins - 1)], 1u);
      }
    }
    __syncthreads();

    if (tid == 0) {
      int target = sh_target;
      if (round == 0) {
        int total = 0;
        for (int b = 0; b < kBins; ++b) total += static_cast<int>(hist[b]);
        target = min(total, max_top_k);
        sh_k_target = target;
        out_lengths[row] = target;
      }
      // Walk bins high -> low: the threshold digit is where the cumulative
      // count first reaches the target.
      int above = 0;
      int bstar = 0;
      int quota = 0;
      for (int b = kBins - 1; b >= 0; --b) {
        const int h = static_cast<int>(hist[b]);
        if (h > 0 && above < target && above + h >= target) {
          bstar = b;
          quota = target - above;
          break;
        }
        above += h;
      }
      sh_prefix = sh_prefix | (static_cast<uint32_t>(bstar) << shift);
      sh_target = quota;
    }
    __syncthreads();
  }

  const uint32_t thr = sh_prefix;
  const int tie_quota = sh_target;
  const int k_target = sh_k_target;
  if (k_target == 0) {
    return;  // the row's output is already all -1 from the in-kernel init
  }

  // ---- Phase 2: ordered emission (ascending positions, ties by lowest). ----
  if (tid == 0) {
    sh_sel_base = 0;
    sh_tie_seen = 0;
  }
  __syncthreads();

  for (int chunk = 0; chunk < n_live; chunk += kThreads) {
    const int i = chunk + tid;
    bool tie = false;
    bool above = false;
    if (i < n_live) {
      const float s = row_scores[i];
      if (is_selectable(s)) {
        const uint32_t k = sortable_key(s);
        above = k > thr;
        tie = k == thr;
      }
    }
    int tie_total;
    const int tie_prefix = block_exscan(tie ? 1 : 0, scan_buf, &tie_total);
    __syncthreads();
    const bool admitted = above || (tie && (sh_tie_seen + tie_prefix) < tie_quota);
    int sel_total;
    const int sel_prefix = block_exscan(admitted ? 1 : 0, scan_buf, &sel_total);
    if (admitted) {
      out_indices[static_cast<int64_t>(row) * max_top_k + sh_sel_base + sel_prefix] = i;
    }
    __syncthreads();
    if (tid == 0) {
      sh_sel_base += sel_total;
      sh_tie_seen += tie_total;
    }
    __syncthreads();
  }
}

}  // namespace

void ds_topk_sequence_order(
    at::Tensor score,
    at::Tensor seq_lens,
    at::Tensor out_indices,
    at::Tensor out_lengths) {
  TORCH_CHECK(score.is_cuda() && score.dim() == 2, "score must be CUDA [bs, width]");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
  TORCH_CHECK(score.stride(1) == 1, "score rows must be contiguous");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous());
  TORCH_CHECK(out_indices.scalar_type() == at::kInt && out_indices.is_contiguous());
  TORCH_CHECK(out_lengths.scalar_type() == at::kInt && out_lengths.is_contiguous());
  const int bs = static_cast<int>(score.size(0));
  TORCH_CHECK(seq_lens.size(0) >= bs && out_indices.size(0) >= bs && out_lengths.size(0) >= bs);
  const int width = static_cast<int>(score.size(1));
  const int max_top_k = static_cast<int>(out_indices.size(1));

  const at::cuda::OptionalCUDAGuard device_guard(score.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  ds_topk_sequence_order_kernel<<<bs, kThreads, 0, stream>>>(
      score.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      out_indices.data_ptr<int>(),
      out_lengths.data_ptr<int>(),
      score.stride(0),
      width,
      max_top_k);
}
