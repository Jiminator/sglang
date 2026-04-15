/**
 * RMSNorm with HuggingFace semantics:
 *   out = weight_dtype * cast_dtype(normalize_fp32(x))
 *
 * This differs from rmsnorm.cuh which computes:
 *   out = cast_dtype(normalize_fp32(x) * weight_fp32)
 *
 * The cast-before-weight-multiply order matches HF LlamaRMSNorm and is required
 * for the transformers backend to produce numerically identical outputs to HF.
 *
 * Fork of rmsnorm.cuh with the one-line semantic change applied to all kernel variants.
 */

#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil

#include <sgl_kernel/runtime.cuh>  // For get_blocks_per_sm, get_sm_count
#include <sgl_kernel/tile.cuh>     // For tile::Memory
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, SGL_DEVICE, type aliases, PDL
#include <sgl_kernel/vec.cuh>      // For AlignedVector

#include <sgl_kernel/impl/norm.cuh>  // For norm::StorageType, norm::kSmemBufferSize, norm::get_cta_threads

#include <tvm/ffi/container/tensor.h>

namespace {

struct RMSNormHFParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

// ---------------------------------------------------------------------------
// HF-semantics norm: cast normalized x to dtype BEFORE weight multiply.
// This helper replaces norm::apply_norm_impl for HF semantics.
// ---------------------------------------------------------------------------
namespace hf_norm {

template <int64_t kDim, bool kUseCTA, typename PackedFloat, std::size_t N>
SGL_DEVICE device::AlignedVector<PackedFloat, N> apply_norm_hf_impl(
    const device::AlignedVector<PackedFloat, N> input,
    const device::AlignedVector<PackedFloat, N> weight,
    const float eps,
    [[maybe_unused]] float* smem_buffer,
    [[maybe_unused]] uint32_t num_warps) {
  using namespace device;
  float sum_of_squares = 0.0f;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  float norm_factor;
  if constexpr (kUseCTA) {
    const auto warp_id = threadIdx.x / kWarpThreads;
    smem_buffer[warp_id] = sum_of_squares;
    __syncthreads();
    if (warp_id == 0) {
      const auto tx = threadIdx.x;
      const auto local_sum = tx < num_warps ? smem_buffer[tx] : 0.0f;
      sum_of_squares = warp::reduce_sum(local_sum);
      smem_buffer[32] = math::rsqrt(sum_of_squares / kDim + eps);
    }
    __syncthreads();
    norm_factor = smem_buffer[32];
  } else {
    norm_factor = math::rsqrt(sum_of_squares / kDim + eps);
  }

  AlignedVector<PackedFloat, N> output;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    // HF semantics: cast normalized x to dtype BEFORE multiplying by weight
    const PackedFloat xn = cast<PackedFloat>(fp32x2_t{
        fp32_input.x * norm_factor,
        fp32_input.y * norm_factor,
    });
    const auto xn_fp32 = cast<fp32x2_t>(xn);
    const auto fp32_weight = cast<fp32x2_t>(weight[i]);
    output[i] = cast<PackedFloat>(fp32x2_t{
        xn_fp32.x * fp32_weight.x,
        xn_fp32.y * fp32_weight.y,
    });
  }

  return output;
}

template <int64_t kDim, typename T>
SGL_DEVICE T apply_norm_hf_warp(const T& input, const T& weight, float eps) {
  return apply_norm_hf_impl<kDim, false>(input, weight, eps, nullptr, 0);
}

template <int64_t kDim, typename T>
SGL_DEVICE T apply_norm_hf_cta(
    const T& input, const T& weight, float eps, float* smem, uint32_t num_warps = blockDim.x / device::kWarpThreads) {
  return apply_norm_hf_impl<kDim, true>(input, weight, eps, smem, num_warps);
}

}  // namespace hf_norm

// ---------------------------------------------------------------------------
// Kernel variants: identical to rmsnorm.cuh but using hf_norm helpers
// ---------------------------------------------------------------------------

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_hf_cta(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kDim>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<Float, kDim>();
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<Float>(output, i * output_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    const auto output_vec = hf_norm::apply_norm_hf_cta<kDim>(input_vec, weight_vec, eps, smem, kNumWarps);
    gmem.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Pre-Blackwell: 16B vector, each thread loads/stores twice
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kDim / 16) void rmsnorm_hf_cta_double(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;

  constexpr auto kNumThreads = kDim / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[32];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = pointer::offset<Float>(input, blockIdx.x * input_stride);
  const auto output_ptr = pointer::offset<Float>(output, blockIdx.x * output_stride);

  const auto input_first = gmem.load(input_ptr, 0);
  const auto input_second = gmem.load(input_ptr, 1);
  const auto weight_first = gmem.load(weight_ptr, 0);
  const auto weight_second = gmem.load(weight_ptr, 1);

  float sum_of_squares = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_first[j]);
    sum_of_squares += x * x + y * y;
  }
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_second[j]);
    sum_of_squares += x * x + y * y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = sum_of_squares;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
    sum_of_squares = warp::reduce_sum(local_sum);
    smem[tx] = math::rsqrt(sum_of_squares / kDim + eps);
  }
  __syncthreads();
  const float norm_factor = smem[warp_id];

  // HF semantics: cast normalized x to dtype, then multiply weight in dtype
  Storage output_first, output_second;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_first[j]);
    Float2 xn = cast<Float2>(fp32x2_t{ix * norm_factor, iy * norm_factor});
    const auto [xn_x, xn_y] = cast<fp32x2_t>(xn);
    const auto [wx, wy] = cast<fp32x2_t>(weight_first[j]);
    output_first[j] = cast<Float2>(fp32x2_t{xn_x * wx, xn_y * wy});
  }
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_second[j]);
    Float2 xn = cast<Float2>(fp32x2_t{ix * norm_factor, iy * norm_factor});
    const auto [xn_x, xn_y] = cast<fp32x2_t>(xn);
    const auto [wx, wy] = cast<fp32x2_t>(weight_second[j]);
    output_second[j] = cast<Float2>(fp32x2_t{xn_x * wx, xn_y * wy});
  }

  gmem.store(output_ptr, output_first, 0);
  gmem.store(output_ptr, output_second, 1);

  PDLTriggerSecondary<kUsePDL>();
}

// Blackwell: 32B vector, each thread loads/stores once
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kDim / 16) void rmsnorm_hf_cta_wide(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 8>;

  constexpr auto kNumThreads = kDim / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[32];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = pointer::offset<Float>(input, blockIdx.x * input_stride);
  const auto output_ptr = pointer::offset<Float>(output, blockIdx.x * output_stride);

  const auto input_vec = gmem.load(input_ptr);
  const auto weight_vec = gmem.load(weight_ptr);

  float sum_of_squares = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 8u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_vec[j]);
    sum_of_squares += x * x + y * y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = sum_of_squares;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
    sum_of_squares = warp::reduce_sum(local_sum);
    smem[tx] = math::rsqrt(sum_of_squares / kDim + eps);
  }
  __syncthreads();
  const float norm_factor = smem[warp_id];

  // HF semantics: cast normalized x to dtype, then multiply weight in dtype
  Storage output_vec_out;
#pragma unroll
  for (auto j = 0u; j < 8u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_vec[j]);
    Float2 xn = cast<Float2>(fp32x2_t{ix * norm_factor, iy * norm_factor});
    const auto [xn_x, xn_y] = cast<fp32x2_t>(xn);
    const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j]);
    output_vec_out[j] = cast<Float2>(fp32x2_t{xn_x * wx, xn_y * wy});
  }

  gmem.store(output_ptr, output_vec_out);

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_hf_warp(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kDim>;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::warp();

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<Float>(output, i * output_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    const auto output_vec = hf_norm::apply_norm_hf_warp<kDim>(input_vec, weight_vec, eps);
    gmem.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Launcher structs: identical structure to rmsnorm.cuh
// ---------------------------------------------------------------------------

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormHFWarpKernel {
  static_assert(host::norm::is_config_supported<DType, kDim>(), "Unsupported norm configuration");
  static_assert(kDim <= 256, "Use RMSNormHFKernel for hidden sizes > 256");
  static constexpr auto kernel = rmsnorm_hf_warp<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormHFParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static constexpr uint32_t kNumThreads = device::kWarpThreads;
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kNumThreads);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormHFKernel {
  static_assert(host::norm::should_use_cta<DType, kDim>(), "Hidden size invalid for RMSNormHF");
  static constexpr auto kernel = rmsnorm_hf_cta<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormHFParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static constexpr auto kNumThreads = norm::get_cta_threads<DType, kDim>();
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kNumThreads);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormHFHalfKernel {
  static_assert(kDim % 512 == 0 && sizeof(DType) == 2);
#if SGL_ARCH_BLACKWELL_OR_GREATER
  static constexpr auto kernel = rmsnorm_hf_cta_wide<kDim, kUsePDL, DType>;
#else
  static constexpr auto kernel = rmsnorm_hf_cta_double<kDim, kUsePDL, DType>;
#endif
  static constexpr auto kBlockSize = static_cast<uint32_t>(kDim / 16);

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormHFParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
