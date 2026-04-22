#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tensor_access.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kVecWidth = 4;
constexpr int kFeatTile = kWarpSize * kVecWidth;
constexpr int kOptimizedRowsPerCTA = 4;

__global__ void pyg_gcn_spmm_kernel(const int *row_ptr, const int *col_ind,
                                    const float *edge_weight, const float *x,
                                    float *out, int num_nodes, int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int feat = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_nodes || feat >= feat_dim) {
    return;
  }

  float acc = 0.0f;
  for (int edge = row_ptr[row]; edge < row_ptr[row + 1]; ++edge) {
    const int col = col_ind[edge];
    acc += edge_weight[edge] * x[static_cast<int64_t>(col) * feat_dim + feat];
  }
  out[static_cast<int64_t>(row) * feat_dim + feat] = acc;
}

template <bool kVectorized>
__global__ void pyg_gcn_spmm_plain_kernel(const int *row_ptr, const int *col_ind,
                                          const float *edge_weight,
                                          const float *x, float *out,
                                          int num_nodes, int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (row >= num_nodes || lane >= kWarpSize) {
    return;
  }

  const int feat_base =
      static_cast<int>(blockIdx.y) * kFeatTile + lane * kVecWidth;
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
    float weight = 0.0f;
    if (lane == 0) {
      col = col_ind[edge];
      weight = edge_weight[edge];
    }
    col = __shfl_sync(0xffffffffu, col, 0);
    weight = __shfl_sync(0xffffffffu, weight, 0);
    const float *x_ptr = x + static_cast<int64_t>(col) * feat_dim + feat_base;
    if constexpr (kVectorized) {
      const float4 values = *reinterpret_cast<const float4 *>(x_ptr);
      acc0 += weight * values.x;
      acc1 += weight * values.y;
      acc2 += weight * values.z;
      acc3 += weight * values.w;
    } else {
      if (feat_base + 0 < feat_dim) {
        acc0 += weight * x_ptr[0];
      }
      if (feat_base + 1 < feat_dim) {
        acc1 += weight * x_ptr[1];
      }
      if (feat_base + 2 < feat_dim) {
        acc2 += weight * x_ptr[2];
      }
      if (feat_base + 3 < feat_dim) {
        acc3 += weight * x_ptr[3];
      }
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

template <bool kVectorized>
__global__ void pyg_gcn_spmm_hmm_optimized_kernel(
    const int *row_ptr, const int *col_ind, const float *edge_weight,
    const float *x, const int64_t *row_schedule,
    const float *hot_feature_cache, int hot_node_cutoff, float *out,
    int num_nodes, int feat_dim) {
  const int row_slot =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) +
      static_cast<int>(threadIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (row_slot >= num_nodes || lane >= kWarpSize) {
    return;
  }
  const int row = static_cast<int>(row_schedule[row_slot]);

  const int feat_base =
      static_cast<int>(blockIdx.y) * kFeatTile + lane * kVecWidth;
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
    float weight = 0.0f;
    if (lane == 0) {
      col = col_ind[edge];
      weight = edge_weight[edge];
    }
    col = __shfl_sync(0xffffffffu, col, 0);
    weight = __shfl_sync(0xffffffffu, weight, 0);
    const float *x_ptr =
        col < hot_node_cutoff
            ? hot_feature_cache + static_cast<int64_t>(col) * feat_dim + feat_base
            : x + static_cast<int64_t>(col) * feat_dim + feat_base;
    if constexpr (kVectorized) {
      const float4 values = *reinterpret_cast<const float4 *>(x_ptr);
      acc0 += weight * values.x;
      acc1 += weight * values.y;
      acc2 += weight * values.z;
      acc3 += weight * values.w;
    } else {
      if (feat_base + 0 < feat_dim) {
        acc0 += weight * x_ptr[0];
      }
      if (feat_base + 1 < feat_dim) {
        acc1 += weight * x_ptr[1];
      }
      if (feat_base + 2 < feat_dim) {
        acc2 += weight * x_ptr[2];
      }
      if (feat_base + 3 < feat_dim) {
        acc3 += weight * x_ptr[3];
      }
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

void check_inputs(const torch::Tensor &row_ptr, const torch::Tensor &col_ind,
                  const torch::Tensor &edge_weight, const torch::Tensor &x,
                  const torch::Tensor &out) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
  TORCH_CHECK(col_ind.is_contiguous(), "col_ind must be contiguous");
  TORCH_CHECK(edge_weight.is_contiguous(), "edge_weight must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(row_ptr.scalar_type() == at::kInt, "row_ptr must be int32");
  TORCH_CHECK(col_ind.scalar_type() == at::kInt, "col_ind must be int32");
  TORCH_CHECK(edge_weight.scalar_type() == at::kFloat,
              "edge_weight must be float32");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(x.sizes() == out.sizes(), "x and out must have the same shape");
  TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be rank-1");
  TORCH_CHECK(col_ind.dim() == 1, "col_ind must be rank-1");
  TORCH_CHECK(edge_weight.dim() == 1, "edge_weight must be rank-1");
  TORCH_CHECK(row_ptr.numel() == x.size(0) + 1,
              "row_ptr length must be num_nodes + 1");
  TORCH_CHECK(col_ind.numel() == edge_weight.numel(),
              "col_ind and edge_weight lengths must match");
  (void)get_device_accessible_const_ptr<int>(row_ptr);
  (void)get_device_accessible_const_ptr<int>(col_ind);
  (void)get_device_accessible_const_ptr<float>(edge_weight);
  (void)get_device_accessible_const_ptr<float>(x);
}

void check_optimized_inputs(const torch::Tensor &row_schedule,
                            const torch::Tensor &hot_feature_cache,
                            const torch::Tensor &x,
                            int64_t hot_node_cutoff) {
  TORCH_CHECK(row_schedule.is_cuda(), "row_schedule must be CUDA");
  TORCH_CHECK(hot_feature_cache.is_cuda(), "hot_feature_cache must be CUDA");
  TORCH_CHECK(row_schedule.is_contiguous(), "row_schedule must be contiguous");
  TORCH_CHECK(hot_feature_cache.is_contiguous(),
              "hot_feature_cache must be contiguous");
  TORCH_CHECK(row_schedule.scalar_type() == at::kLong,
              "row_schedule must be int64");
  TORCH_CHECK(hot_feature_cache.scalar_type() == at::kFloat,
              "hot_feature_cache must be float32");
  TORCH_CHECK(row_schedule.dim() == 1, "row_schedule must be rank-1");
  TORCH_CHECK(hot_feature_cache.dim() == 2,
              "hot_feature_cache must be rank-2");
  TORCH_CHECK(row_schedule.numel() == x.size(0),
              "row_schedule length must equal num_nodes");
  TORCH_CHECK(hot_node_cutoff >= 1, "hot_node_cutoff must be >= 1");
  TORCH_CHECK(hot_node_cutoff <= x.size(0),
              "hot_node_cutoff must be <= num_nodes");
  TORCH_CHECK(hot_feature_cache.size(0) == hot_node_cutoff,
              "hot_feature_cache row count must equal hot_node_cutoff");
  TORCH_CHECK(hot_feature_cache.size(1) == x.size(1),
              "hot_feature_cache feature dimension must match x");
}

} // namespace

void spmm_pyg_gcn_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                                torch::Tensor edge_weight, torch::Tensor x,
                                torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, edge_weight, x, out);

  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  const int threads = 128;
  dim3 grid(static_cast<unsigned int>(num_nodes),
            static_cast<unsigned int>((feat_dim + threads - 1) / threads));
  dim3 block(threads);

  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  pyg_gcn_spmm_kernel<<<grid, block, 0, stream.stream()>>>(
      get_device_accessible_const_ptr<int>(row_ptr),
      get_device_accessible_const_ptr<int>(col_ind),
      get_device_accessible_const_ptr<float>(edge_weight),
      get_device_accessible_const_ptr<float>(x), out.data_ptr<float>(),
      num_nodes, feat_dim);
  C10_CUDA_CHECK(cudaGetLastError());
}

void spmm_pyg_gcn_plain_forward_cuda_(torch::Tensor row_ptr,
                                      torch::Tensor col_ind,
                                      torch::Tensor edge_weight,
                                      torch::Tensor x, torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, edge_weight, x, out);

  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  dim3 grid(static_cast<unsigned int>(num_nodes),
            static_cast<unsigned int>((feat_dim + kFeatTile - 1) / kFeatTile));
  dim3 block(kWarpSize);

  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  if (feat_dim % kVecWidth == 0) {
    pyg_gcn_spmm_plain_kernel<true><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(edge_weight),
        get_device_accessible_const_ptr<float>(x), out.data_ptr<float>(),
        num_nodes, feat_dim);
  } else {
    pyg_gcn_spmm_plain_kernel<false><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(edge_weight),
        get_device_accessible_const_ptr<float>(x), out.data_ptr<float>(),
        num_nodes, feat_dim);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void spmm_pyg_gcn_optimized_forward_cuda_impl_(
    torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor edge_weight,
    torch::Tensor x, torch::Tensor row_schedule,
    torch::Tensor hot_feature_cache, int64_t hot_node_cutoff,
    torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, edge_weight, x, out);
  check_optimized_inputs(row_schedule, hot_feature_cache, x, hot_node_cutoff);

  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  dim3 grid(
      static_cast<unsigned int>((num_nodes + kOptimizedRowsPerCTA - 1) /
                                kOptimizedRowsPerCTA),
      static_cast<unsigned int>((feat_dim + kFeatTile - 1) / kFeatTile));
  dim3 block(kWarpSize, kOptimizedRowsPerCTA);

  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  if (feat_dim % kVecWidth == 0) {
    pyg_gcn_spmm_hmm_optimized_kernel<true><<<grid, block, 0,
                                               stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(edge_weight),
        get_device_accessible_const_ptr<float>(x),
        row_schedule.data_ptr<int64_t>(),
        hot_feature_cache.data_ptr<float>(),
        static_cast<int>(hot_node_cutoff), out.data_ptr<float>(), num_nodes,
        feat_dim);
  } else {
    pyg_gcn_spmm_hmm_optimized_kernel<false><<<grid, block, 0,
                                                stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(edge_weight),
        get_device_accessible_const_ptr<float>(x),
        row_schedule.data_ptr<int64_t>(),
        hot_feature_cache.data_ptr<float>(),
        static_cast<int>(hot_node_cutoff), out.data_ptr<float>(), num_nodes,
        feat_dim);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void spmm_pyg_gcn_hmm_optimized_forward_cuda_(
    torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor edge_weight,
    torch::Tensor x, torch::Tensor row_schedule,
    torch::Tensor hot_feature_cache, int64_t hot_node_cutoff,
    torch::Tensor out) {
  spmm_pyg_gcn_optimized_forward_cuda_impl_(
      row_ptr, col_ind, edge_weight, x, row_schedule, hot_feature_cache,
      hot_node_cutoff, out);
}

void spmm_pyg_gcn_uvm_optimized_forward_cuda_(
    torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor edge_weight,
    torch::Tensor x, torch::Tensor row_schedule,
    torch::Tensor hot_feature_cache, int64_t hot_node_cutoff,
    torch::Tensor out) {
  spmm_pyg_gcn_optimized_forward_cuda_impl_(
      row_ptr, col_ind, edge_weight, x, row_schedule, hot_feature_cache,
      hot_node_cutoff, out);
}
