#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tensor_access.cuh"

namespace {

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status__ = (expr);                                          \
    TORCH_CHECK(status__ == CUBLAS_STATUS_SUCCESS, "cuBLAS failure: ",         \
                static_cast<int>(status__));                                   \
  } while (0)

constexpr int kHmmWarpSize = 32;
constexpr int kHmmVecWidth = 4;
constexpr int kHmmFeatTile = kHmmWarpSize * kHmmVecWidth;

template <bool kUseMean>
__global__ void csr_agg_kernel(const int *row_ptr, const int *col_ind,
                               const float *x, float *out, int num_nodes,
                               int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int feat = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_nodes || feat >= feat_dim) {
    return;
  }

  float acc = 0.0f;
  const int row_start = row_ptr[row];
  const int row_end = row_ptr[row + 1];
  for (int edge = row_start; edge < row_end; ++edge) {
    const int col = col_ind[edge];
    acc += x[col * feat_dim + feat];
  }
  if constexpr (kUseMean) {
    const int deg = row_end - row_start;
    if (deg > 0) {
      acc /= static_cast<float>(deg);
    }
  }
  out[row * feat_dim + feat] = acc;
}

template <bool kUseMean, bool kVectorized>
__global__ void csr_agg_hmm_kernel(const int *row_ptr, const int *col_ind,
                                   const float *x, float *out, int num_nodes,
                                   int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (row >= num_nodes || lane >= kHmmWarpSize) {
    return;
  }

  const int feat_base =
      static_cast<int>(blockIdx.y) * kHmmFeatTile + lane * kHmmVecWidth;
  int row_start = 0;
  int row_end = 0;
  if (lane == 0) {
    row_start = row_ptr[row];
    row_end = row_ptr[row + 1];
  }
  row_start = __shfl_sync(0xffffffffu, row_start, 0);
  row_end = __shfl_sync(0xffffffffu, row_end, 0);

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int edge = row_start; edge < row_end; ++edge) {
    int col = 0;
    if (lane == 0) {
      col = col_ind[edge];
    }
    col = __shfl_sync(0xffffffffu, col, 0);
    const float *x_ptr = x + static_cast<int64_t>(col) * feat_dim + feat_base;
    if constexpr (kVectorized) {
      const float4 values = *reinterpret_cast<const float4 *>(x_ptr);
      acc0 += values.x;
      acc1 += values.y;
      acc2 += values.z;
      acc3 += values.w;
    } else {
      if (feat_base + 0 < feat_dim) {
        acc0 += x_ptr[0];
      }
      if (feat_base + 1 < feat_dim) {
        acc1 += x_ptr[1];
      }
      if (feat_base + 2 < feat_dim) {
        acc2 += x_ptr[2];
      }
      if (feat_base + 3 < feat_dim) {
        acc3 += x_ptr[3];
      }
    }
  }
  if constexpr (kUseMean) {
    const int deg = row_end - row_start;
    if (deg > 0) {
      const float inv_deg = 1.0f / static_cast<float>(deg);
      acc0 *= inv_deg;
      acc1 *= inv_deg;
      acc2 *= inv_deg;
      acc3 *= inv_deg;
    }
  }

  float *out_ptr = out + static_cast<int64_t>(row) * feat_dim + feat_base;
  if constexpr (kVectorized) {
    *reinterpret_cast<float4 *>(out_ptr) = make_float4(acc0, acc1, acc2, acc3);
  } else {
    if (feat_base + 0 < feat_dim) {
      out_ptr[0] = acc0;
    }
    if (feat_base + 1 < feat_dim) {
      out_ptr[1] = acc1;
    }
    if (feat_base + 2 < feat_dim) {
      out_ptr[2] = acc2;
    }
    if (feat_base + 3 < feat_dim) {
      out_ptr[3] = acc3;
    }
  }
}

__global__ void bias_relu_kernel(float *out, const float *bias, int rows,
                                 int cols, bool relu) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (idx >= count) {
    return;
  }
  float value = out[idx];
  if (bias != nullptr) {
    value += bias[idx % cols];
  }
  if (relu && value < 0.0f) {
    value = 0.0f;
  }
  out[idx] = value;
}

__global__ void tensor_add_inplace_kernel(float *dst, const float *src,
                                          float alpha, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  dst[idx] += alpha * src[idx];
}

__global__ void relu_inplace_kernel(float *tensor, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  if (tensor[idx] < 0.0f) {
    tensor[idx] = 0.0f;
  }
}

void check_inputs(const torch::Tensor &row_ptr, const torch::Tensor &col_ind,
                  const torch::Tensor &x, const torch::Tensor &out) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
  TORCH_CHECK(col_ind.is_contiguous(), "col_ind must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(row_ptr.scalar_type() == at::kInt, "row_ptr must be int32");
  TORCH_CHECK(col_ind.scalar_type() == at::kInt, "col_ind must be int32");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(x.sizes() == out.sizes(), "x and out must have the same shape");
  TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be rank-1");
  TORCH_CHECK(col_ind.dim() == 1, "col_ind must be rank-1");
  TORCH_CHECK(row_ptr.numel() == x.size(0) + 1,
              "row_ptr length must be num_nodes + 1");
  (void)get_device_accessible_const_ptr<int>(row_ptr);
  (void)get_device_accessible_const_ptr<int>(col_ind);
  (void)get_device_accessible_const_ptr<float>(x);
}

template <bool kUseMean>
void launch(const torch::Tensor &row_ptr, const torch::Tensor &col_ind,
            const torch::Tensor &x, const torch::Tensor &out) {
  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  const int threads = 128;
  dim3 grid(static_cast<unsigned int>(num_nodes),
            static_cast<unsigned int>((feat_dim + threads - 1) / threads));
  dim3 block(threads);
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  csr_agg_kernel<kUseMean><<<grid, block, 0, stream.stream()>>>(
      get_device_accessible_const_ptr<int>(row_ptr),
      get_device_accessible_const_ptr<int>(col_ind),
      get_device_accessible_const_ptr<float>(x),
      out.data_ptr<float>(), num_nodes, feat_dim);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <bool kUseMean>
void launch_hmm(const torch::Tensor &row_ptr, const torch::Tensor &col_ind,
                const torch::Tensor &x, const torch::Tensor &out) {
  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  dim3 grid(static_cast<unsigned int>(num_nodes),
            static_cast<unsigned int>((feat_dim + kHmmFeatTile - 1) / kHmmFeatTile));
  dim3 block(kHmmWarpSize);
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  if (feat_dim % kHmmVecWidth == 0) {
    csr_agg_hmm_kernel<kUseMean, true><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(x),
        out.data_ptr<float>(), num_nodes, feat_dim);
  } else {
    csr_agg_hmm_kernel<kUseMean, false><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(x),
        out.data_ptr<float>(), num_nodes, feat_dim);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void check_dense_inputs(const torch::Tensor &x, const torch::Tensor &weight,
                        const torch::Tensor &bias, const torch::Tensor &out) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(x.size(1) == weight.size(0), "Inner GEMM dimensions must match");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == weight.size(1),
              "Output shape must match x @ weight");
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias.dim() == 1, "bias must be rank-1");
    TORCH_CHECK(bias.numel() == out.size(1),
                "bias length must equal output feature dimension");
  }
  (void)get_device_accessible_const_ptr<float>(x);
  (void)get_device_accessible_const_ptr<float>(weight);
  if (bias.numel() != 0) {
    (void)get_device_accessible_const_ptr<float>(bias);
  }
}

void check_gemm_inputs(const torch::Tensor &x, const torch::Tensor &weight,
                       const torch::Tensor &out) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(x.size(1) == weight.size(0), "Inner GEMM dimensions must match");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == weight.size(1),
              "Output shape must match x @ weight");
  (void)get_device_accessible_const_ptr<float>(x);
  (void)get_device_accessible_const_ptr<float>(weight);
}

void check_bias_relu_inputs(const torch::Tensor &out, const torch::Tensor &bias) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias.dim() == 1, "bias must be rank-1");
    TORCH_CHECK(bias.numel() == out.size(1),
                "bias length must equal output feature dimension");
    (void)get_device_accessible_const_ptr<float>(bias);
  }
}

void run_bias_relu(torch::Tensor out, torch::Tensor bias, bool relu) {
  if (bias.numel() == 0 && !relu) {
    return;
  }
  const int64_t count = out.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  const float *bias_ptr =
      bias.numel() == 0 ? nullptr : get_device_accessible_const_ptr<float>(bias);
  bias_relu_kernel<<<blocks, threads, 0, stream.stream()>>>(
      out.data_ptr<float>(), bias_ptr, static_cast<int>(out.size(0)),
      static_cast<int>(out.size(1)), relu);
  C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void spmm_sum_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                            torch::Tensor x, torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, x, out);
  launch<false>(row_ptr, col_ind, x, out);
}

void spmm_mean_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                             torch::Tensor x, torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, x, out);
  launch<true>(row_ptr, col_ind, x, out);
}

void spmm_sum_hmm_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                                torch::Tensor x, torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, x, out);
  launch_hmm<false>(row_ptr, col_ind, x, out);
}

void spmm_mean_hmm_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                                 torch::Tensor x, torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, x, out);
  launch_hmm<true>(row_ptr, col_ind, x, out);
}

void linear_forward_cuda_(torch::Tensor x, torch::Tensor weight,
                          torch::Tensor bias, torch::Tensor out, bool relu) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_dense_inputs(x, weight, bias, out);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  CUBLAS_CHECK(cublasSetStream(handle, stream.stream()));

  const int64_t rows = x.size(0);
  const int64_t inner = x.size(1);
  const int64_t cols = weight.size(1);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(cols),
      static_cast<int>(rows), static_cast<int>(inner), &alpha,
      get_device_accessible_const_ptr<float>(weight), static_cast<int>(cols),
      get_device_accessible_const_ptr<float>(x), static_cast<int>(inner), &beta,
      out.data_ptr<float>(), static_cast<int>(cols)));
  CUBLAS_CHECK(cublasDestroy(handle));

  run_bias_relu(out, bias, relu);
}

void gemm_forward_cuda_(torch::Tensor x, torch::Tensor weight,
                        torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_gemm_inputs(x, weight, out);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  CUBLAS_CHECK(cublasSetStream(handle, stream.stream()));

  const int64_t rows = x.size(0);
  const int64_t inner = x.size(1);
  const int64_t cols = weight.size(1);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(cols),
      static_cast<int>(rows), static_cast<int>(inner), &alpha,
      get_device_accessible_const_ptr<float>(weight), static_cast<int>(cols),
      get_device_accessible_const_ptr<float>(x), static_cast<int>(inner), &beta,
      out.data_ptr<float>(), static_cast<int>(cols)));
  CUBLAS_CHECK(cublasDestroy(handle));
}

void bias_relu_forward_cuda_(torch::Tensor out, torch::Tensor bias, bool relu) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_bias_relu_inputs(out, bias);
  run_bias_relu(out, bias, relu);
}

void tensor_add_inplace_cuda_(torch::Tensor dst, torch::Tensor src,
                              double alpha) {
  at::cuda::CUDAGuard device_guard(dst.device());
  TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
  TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(dst.scalar_type() == at::kFloat, "dst must be float32");
  TORCH_CHECK(src.scalar_type() == at::kFloat, "src must be float32");
  TORCH_CHECK(dst.sizes() == src.sizes(), "dst and src must have the same shape");
  (void)get_device_accessible_const_ptr<float>(src);

  const int64_t count = dst.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  auto stream = at::cuda::getDefaultCUDAStream(dst.get_device());
  tensor_add_inplace_kernel<<<blocks, threads, 0, stream.stream()>>>(
      dst.data_ptr<float>(), get_device_accessible_const_ptr<float>(src),
      static_cast<float>(alpha), count);
  C10_CUDA_CHECK(cudaGetLastError());
}

void relu_inplace_cuda_(torch::Tensor tensor) {
  TORCH_CHECK(tensor.is_cuda(), "tensor must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, "tensor must be float32");
  at::cuda::CUDAGuard device_guard(tensor.device());

  const int64_t count = tensor.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  auto stream = at::cuda::getDefaultCUDAStream(tensor.get_device());
  relu_inplace_kernel<<<blocks, threads, 0, stream.stream()>>>(
      tensor.data_ptr<float>(), count);
  C10_CUDA_CHECK(cudaGetLastError());
}
