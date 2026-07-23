/**
 * LayerNorm (mean-subtracted, with affine weight + bias):
 *   mean = mean_j(x[j])
 *   var  = mean_j(x[j]^2) - mean^2
 *   out[i] = cast_dtype( (x[i] - mean) * rsqrt(var + eps) * weight[i] + bias[i] )
 *
 * Matches torch F.layer_norm / aiter layernorm2d_fwd semantics: the reduction
 * and affine are computed in fp32, the result is rounded to the activation
 * dtype on store. Drop-in for the DSA indexer's k_norm (head_dim=128, bf16),
 * replacing aiter's ck_tile Layernorm2dFwd on gfx950.
 *
 * Two launch configs (mirrors rmsnorm_hf.cuh):
 *   - Warp kernel: 32 threads/row for small hidden sizes (indexer k_norm=128).
 *   - CTA kernel:  512-thread scalar-strided with register cache (token norms).
 */

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/math.cuh>     // For device::math::rsqrt
#include <sgl_kernel/runtime.cuh>  // For runtime::get_blocks_per_sm, get_sm_count
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, SGL_DEVICE, type aliases, PDL, cast
#include <sgl_kernel/warp.cuh>     // For warp::reduce_sum

#include <tvm/ffi/container/tensor.h>

namespace {

struct LayerNormParams {
  const void* input;
  const void* __restrict__ weight;
  const void* __restrict__ bias;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

// ---------------------------------------------------------------------------
// Warp kernel: one warp per row, for small hidden sizes (indexer k_norm at
// head_dim ∈ {32, 64, 96, 128, 256}). No shared memory, no block reduce —
// two warp reduces (Σx and Σx²) are sufficient. Grid-strided over rows.
// ---------------------------------------------------------------------------
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(32) void layernorm_warp_kernel(const LayerNormParams __grid_constant__ params) {
  using namespace device;
  constexpr int kElemsPerThread = kDim / kWarpThreads;

  const auto& [input, weight_ptr, bias_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto wr = static_cast<const Float*>(weight_ptr);
  const auto br = static_cast<const Float*>(bias_ptr);

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t row = blockIdx.x; row < num_tokens; row += gridDim.x) {
    const auto xr = static_cast<const Float*>(pointer::offset<Float>(input, row * input_stride));
    const auto yr = static_cast<Float*>(pointer::offset<Float>(output, row * output_stride));

    float xi_cache[kElemsPerThread];
    float lsum = 0.f;
    float lsq = 0.f;
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const int i = threadIdx.x + k * kWarpThreads;
      xi_cache[k] = static_cast<float>(xr[i]);
      lsum += xi_cache[k];
      lsq += xi_cache[k] * xi_cache[k];
    }
    lsum = warp::reduce_sum(lsum);
    lsq = warp::reduce_sum(lsq);
    const float mean = lsum / kDim;
    const float var = lsq / kDim - mean * mean;
    const float rstd = math::rsqrt(var + eps);

#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const int i = threadIdx.x + k * kWarpThreads;
      const float xn = (xi_cache[k] - mean) * rstd;
      yr[i] = cast<Float>(xn * static_cast<float>(wr[i]) + static_cast<float>(br[i]));
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Kernel: 512-thread scalar-strided LayerNorm + register cache.
//
// Pass 1: each thread loads its strided elements, caches them in registers,
//         accumulates fp32 Σx and Σx². Warp + block reduction yields
//         mean and rstd = rsqrt(var + eps).
// Pass 2: reuse cached fp32 values — no second global read of `x`.
// ---------------------------------------------------------------------------
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(512) void layernorm_scalar_kernel(const LayerNormParams __grid_constant__ params) {
  using namespace device;
  constexpr int kNumThreads = 512;
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kElemsPerThread = (kDim + kNumThreads - 1) / kNumThreads;

  const auto& [input, weight_ptr, bias_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto xr = static_cast<const Float*>(pointer::offset<Float>(input, blockIdx.x * input_stride));
  const auto yr = static_cast<Float*>(pointer::offset<Float>(output, blockIdx.x * output_stride));
  const auto wr = static_cast<const Float*>(weight_ptr);
  const auto br = static_cast<const Float*>(bias_ptr);

  PDLWaitPrimary<kUsePDL>();

  float xi_cache[kElemsPerThread];
  float lsum = 0.f;
  float lsq = 0.f;
#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    xi_cache[k] = static_cast<float>(xr[i]);
    lsum += xi_cache[k];
    lsq += xi_cache[k] * xi_cache[k];
  }

  lsum = warp::reduce_sum(lsum);
  lsq = warp::reduce_sum(lsq);

  __shared__ float smem_sum[32];
  __shared__ float smem_sq[32];
  const int warp_id = threadIdx.x / kWarpThreads;
  const int lane_id = threadIdx.x & (kWarpThreads - 1);
  if (lane_id == 0) {
    smem_sum[warp_id] = lsum;
    smem_sq[warp_id] = lsq;
  }
  __syncthreads();

  __shared__ float mean_s;
  __shared__ float rstd_s;
  if (threadIdx.x < kWarpThreads) {
    float vsum = (threadIdx.x < kNumWarps) ? smem_sum[threadIdx.x] : 0.f;
    float vsq = (threadIdx.x < kNumWarps) ? smem_sq[threadIdx.x] : 0.f;
    vsum = warp::reduce_sum(vsum);
    vsq = warp::reduce_sum(vsq);
    if (threadIdx.x == 0) {
      const float mean = vsum / kDim;
      mean_s = mean;
      rstd_s = math::rsqrt(vsq / kDim - mean * mean + eps);
    }
  }
  __syncthreads();
  const float mean = mean_s;
  const float rstd = rstd_s;

#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    const float xn = (xi_cache[k] - mean) * rstd;
    yr[i] = cast<Float>(xn * static_cast<float>(wr[i]) + static_cast<float>(br[i]));
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Warp launcher: occupancy-sized grid, 32 threads/block, one warp per row.
// Targets small hidden sizes (indexer k_norm). kDim must be a multiple of 32
// in [32, 512).
// ---------------------------------------------------------------------------
template <int64_t kDim, bool kUsePDL, typename DType>
struct LayerNormWarpKernel {
  static_assert(sizeof(DType) == 2, "layernorm: DType must be fp16_t or bf16_t");
  static_assert(
      kDim >= 32 && kDim < 512 && kDim % 32 == 0, "layernorm_warp: kDim must be a multiple of 32, in [32, 512)");
  static constexpr auto kernel = layernorm_warp_kernel<kDim, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = device::kWarpThreads;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device_).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device_).verify(weight);
    TensorMatcher({D}).with_dtype<DType>().with_device(device_).verify(bias);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device_).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "layernorm: num_tokens must be > 0");

    const auto params = LayerNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kBlockSize);
    static const uint32_t kNumSM = runtime::get_sm_count(device_.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

// ---------------------------------------------------------------------------
// CTA launcher: validates tensors, launches one block per row.
// ---------------------------------------------------------------------------
template <int64_t kDim, bool kUsePDL, typename DType>
struct LayerNormKernel {
  static_assert(sizeof(DType) == 2, "layernorm: DType must be fp16_t or bf16_t");
  static_assert(kDim >= 512 && kDim % 512 == 0, "layernorm: kDim must be a multiple of 512");
  static constexpr auto kernel = layernorm_scalar_kernel<kDim, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 512;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device_).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device_).verify(weight);
    TensorMatcher({D}).with_dtype<DType>().with_device(device_).verify(bias);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device_).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "layernorm: num_tokens must be > 0");

    const auto params = LayerNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
