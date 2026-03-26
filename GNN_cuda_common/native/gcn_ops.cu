#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tensor_access.cuh"

namespace {

__global__ void gcn_spmm_kernel(const int *row_ptr, const int *col_ind,
                                const float *deg_inv_sqrt, const float *x,
                                float *out, int num_nodes, int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int feat = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_nodes || feat >= feat_dim) {
    return;
  }

  const float row_scale = deg_inv_sqrt[row];
  float acc = 0.0f;
  for (int edge = row_ptr[row]; edge < row_ptr[row + 1]; ++edge) {
    const int col = col_ind[edge];
    acc += row_scale * deg_inv_sqrt[col] * x[col * feat_dim + feat];
  }
  out[row * feat_dim + feat] = acc;
}

void check_inputs(const torch::Tensor &row_ptr, const torch::Tensor &col_ind,
                  const torch::Tensor &deg_inv_sqrt, const torch::Tensor &x,
                  const torch::Tensor &out) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
  TORCH_CHECK(col_ind.is_contiguous(), "col_ind must be contiguous");
  TORCH_CHECK(deg_inv_sqrt.is_contiguous(), "deg_inv_sqrt must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(row_ptr.scalar_type() == at::kInt, "row_ptr must be int32");
  TORCH_CHECK(col_ind.scalar_type() == at::kInt, "col_ind must be int32");
  TORCH_CHECK(deg_inv_sqrt.scalar_type() == at::kFloat,
              "deg_inv_sqrt must be float32");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(x.sizes() == out.sizes(), "x and out must have the same shape");
  TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be rank-1");
  TORCH_CHECK(col_ind.dim() == 1, "col_ind must be rank-1");
  TORCH_CHECK(deg_inv_sqrt.dim() == 1, "deg_inv_sqrt must be rank-1");
  TORCH_CHECK(row_ptr.numel() == x.size(0) + 1,
              "row_ptr length must be num_nodes + 1");
  TORCH_CHECK(deg_inv_sqrt.numel() == x.size(0),
              "deg_inv_sqrt length must equal num_nodes");
  (void)get_device_accessible_const_ptr<int>(row_ptr);
  (void)get_device_accessible_const_ptr<int>(col_ind);
  (void)get_device_accessible_const_ptr<float>(deg_inv_sqrt);
  (void)get_device_accessible_const_ptr<float>(x);
}

} // namespace

void spmm_gcn_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                            torch::Tensor deg_inv_sqrt, torch::Tensor x,
                            torch::Tensor out) {
  check_inputs(row_ptr, col_ind, deg_inv_sqrt, x, out);
  at::cuda::CUDAGuard device_guard(out.device());

  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  const int threads = 128;
  dim3 grid(static_cast<unsigned int>(num_nodes),
            static_cast<unsigned int>((feat_dim + threads - 1) / threads));
  dim3 block(threads);

  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  gcn_spmm_kernel<<<grid, block, 0, stream.stream()>>>(
      get_device_accessible_const_ptr<int>(row_ptr),
      get_device_accessible_const_ptr<int>(col_ind),
      get_device_accessible_const_ptr<float>(deg_inv_sqrt),
      get_device_accessible_const_ptr<float>(x),
      out.data_ptr<float>(), num_nodes, feat_dim);
  C10_CUDA_CHECK(cudaGetLastError());
}
