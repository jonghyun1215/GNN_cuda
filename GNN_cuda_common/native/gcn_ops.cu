#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tensor_access.cuh"

namespace {

constexpr int kHmmWarpSize = 32;
constexpr int kHmmVecWidth = 4;
constexpr int kHmmFeatTile = kHmmWarpSize * kHmmVecWidth;
constexpr int kOptimizedRowsPerCTA = 4;

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

template <bool kVectorized>
__global__ void gcn_spmm_hmm_kernel(const int *row_ptr, const int *col_ind,
                                    const float *deg_inv_sqrt, const float *x,
                                    float *out, int num_nodes, int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (row >= num_nodes || lane >= kHmmWarpSize) {
    return;
  }

  const int feat_base =
      static_cast<int>(blockIdx.y) * kHmmFeatTile + lane * kHmmVecWidth;
  int row_start = 0;
  int row_end = 0;
  float row_scale = 0.0f;
  if (lane == 0) {
    row_start = row_ptr[row];
    row_end = row_ptr[row + 1];
    row_scale = deg_inv_sqrt[row];
  }
  row_start = __shfl_sync(0xffffffffu, row_start, 0);
  row_end = __shfl_sync(0xffffffffu, row_end, 0);
  row_scale = __shfl_sync(0xffffffffu, row_scale, 0);

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
    const float scale = row_scale * deg_inv_sqrt[col];
    const float *x_ptr = x + static_cast<int64_t>(col) * feat_dim + feat_base;
    if constexpr (kVectorized) {
      const float4 values = *reinterpret_cast<const float4 *>(x_ptr);
      acc0 += scale * values.x;
      acc1 += scale * values.y;
      acc2 += scale * values.z;
      acc3 += scale * values.w;
    } else {
      if (feat_base + 0 < feat_dim) {
        acc0 += scale * x_ptr[0];
      }
      if (feat_base + 1 < feat_dim) {
        acc1 += scale * x_ptr[1];
      }
      if (feat_base + 2 < feat_dim) {
        acc2 += scale * x_ptr[2];
      }
      if (feat_base + 3 < feat_dim) {
        acc3 += scale * x_ptr[3];
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
__global__ void gcn_spmm_hmm_optimized_kernel(
    const int *row_ptr, const int *col_ind, const float *deg_inv_sqrt,
    const float *x, const int64_t *row_schedule,
    const float *hot_feature_cache, int hot_node_cutoff, float *out,
    int num_nodes, int feat_dim) {
  const int row_slot =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) +
      static_cast<int>(threadIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (row_slot >= num_nodes || lane >= kHmmWarpSize) {
    return;
  }
  const int row = static_cast<int>(row_schedule[row_slot]);

  const int feat_base =
      static_cast<int>(blockIdx.y) * kHmmFeatTile + lane * kHmmVecWidth;
  int row_start = 0;
  int row_end = 0;
  float row_scale = 0.0f;
  if (lane == 0) {
    row_start = row_ptr[row];
    row_end = row_ptr[row + 1];
    row_scale = deg_inv_sqrt[row];
  }
  row_start = __shfl_sync(0xffffffffu, row_start, 0);
  row_end = __shfl_sync(0xffffffffu, row_end, 0);
  row_scale = __shfl_sync(0xffffffffu, row_scale, 0);

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
    const float scale = row_scale * deg_inv_sqrt[col];
    const float *x_ptr =
        col < hot_node_cutoff
            ? hot_feature_cache + static_cast<int64_t>(col) * feat_dim + feat_base
            : x + static_cast<int64_t>(col) * feat_dim + feat_base;
    if constexpr (kVectorized) {
      const float4 values = *reinterpret_cast<const float4 *>(x_ptr);
      acc0 += scale * values.x;
      acc1 += scale * values.y;
      acc2 += scale * values.z;
      acc3 += scale * values.w;
    } else {
      if (feat_base + 0 < feat_dim) {
        acc0 += scale * x_ptr[0];
      }
      if (feat_base + 1 < feat_dim) {
        acc1 += scale * x_ptr[1];
      }
      if (feat_base + 2 < feat_dim) {
        acc2 += scale * x_ptr[2];
      }
      if (feat_base + 3 < feat_dim) {
        acc3 += scale * x_ptr[3];
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

__global__ void stage_feature_pages_kernel(const int64_t *page_ids,
                                           const float *x, float *out,
                                           int num_pages_to_stage,
                                           int rows_per_page, int feat_dim,
                                           int num_nodes) {
  const int cached_row = static_cast<int>(blockIdx.x);
  const int feat =
      static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
  const int total_cached_rows = num_pages_to_stage * rows_per_page;
  if (cached_row >= total_cached_rows || feat >= feat_dim) {
    return;
  }
  const int page_slot = cached_row / rows_per_page;
  const int row_in_page = cached_row - page_slot * rows_per_page;
  const int64_t page_id = page_ids[page_slot];
  const int64_t node = page_id * static_cast<int64_t>(rows_per_page) + row_in_page;
  const int64_t out_offset = static_cast<int64_t>(cached_row) * feat_dim + feat;
  out[out_offset] =
      node < num_nodes ? x[node * static_cast<int64_t>(feat_dim) + feat] : 0.0f;
}

__global__ void stage_feature_rows_kernel(const float *x, float *out,
                                          int num_rows, int feat_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int feat =
      static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
  if (row >= num_rows || feat >= feat_dim) {
    return;
  }
  out[static_cast<int64_t>(row) * feat_dim + feat] =
      x[static_cast<int64_t>(row) * feat_dim + feat];
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

void check_stage_inputs(const torch::Tensor &page_ids, const torch::Tensor &x,
                        const torch::Tensor &out, int64_t rows_per_page) {
  TORCH_CHECK(page_ids.is_cuda(), "page_ids must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(page_ids.is_contiguous(), "page_ids must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(page_ids.scalar_type() == at::kLong,
              "page_ids must be int64");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(page_ids.dim() == 1, "page_ids must be rank-1");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(rows_per_page >= 1, "rows_per_page must be >= 1");
  TORCH_CHECK(out.size(0) == page_ids.numel() * rows_per_page,
              "out rows must equal page_ids length * rows_per_page");
  TORCH_CHECK(out.size(1) == x.size(1),
              "out feature dimension must match x");
  (void)get_device_accessible_const_ptr<int64_t>(page_ids);
  (void)get_device_accessible_const_ptr<float>(x);
}

void check_stage_rows_inputs(const torch::Tensor &x, const torch::Tensor &out) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(out.size(0) <= x.size(0),
              "out rows must be <= x rows for staged prefix copies");
  TORCH_CHECK(out.size(1) == x.size(1),
              "out feature dimension must match x");
  (void)get_device_accessible_const_ptr<float>(x);
}

} // namespace

void spmm_gcn_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                            torch::Tensor deg_inv_sqrt, torch::Tensor x,
                            torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, deg_inv_sqrt, x, out);

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

void spmm_gcn_hmm_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                                torch::Tensor deg_inv_sqrt, torch::Tensor x,
                                torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, deg_inv_sqrt, x, out);

  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  dim3 grid(static_cast<unsigned int>(num_nodes),
            static_cast<unsigned int>((feat_dim + kHmmFeatTile - 1) / kHmmFeatTile));
  dim3 block(kHmmWarpSize);

  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  if (feat_dim % kHmmVecWidth == 0) {
    gcn_spmm_hmm_kernel<true><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(deg_inv_sqrt),
        get_device_accessible_const_ptr<float>(x),
        out.data_ptr<float>(), num_nodes, feat_dim);
  } else {
    gcn_spmm_hmm_kernel<false><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(deg_inv_sqrt),
        get_device_accessible_const_ptr<float>(x),
        out.data_ptr<float>(), num_nodes, feat_dim);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void spmm_gcn_hmm_optimized_forward_cuda_(torch::Tensor row_ptr,
                                          torch::Tensor col_ind,
                                          torch::Tensor deg_inv_sqrt,
                                          torch::Tensor x,
                                          torch::Tensor row_schedule,
                                          torch::Tensor hot_feature_cache,
                                          int64_t hot_node_cutoff,
                                          torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_inputs(row_ptr, col_ind, deg_inv_sqrt, x, out);
  check_optimized_inputs(row_schedule, hot_feature_cache, x, hot_node_cutoff);

  const int num_nodes = static_cast<int>(x.size(0));
  const int feat_dim = static_cast<int>(x.size(1));
  dim3 grid(
      static_cast<unsigned int>((num_nodes + kOptimizedRowsPerCTA - 1) /
                                kOptimizedRowsPerCTA),
            static_cast<unsigned int>((feat_dim + kHmmFeatTile - 1) / kHmmFeatTile));
  dim3 block(kHmmWarpSize, kOptimizedRowsPerCTA);

  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  if (feat_dim % kHmmVecWidth == 0) {
    gcn_spmm_hmm_optimized_kernel<true><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(deg_inv_sqrt),
        get_device_accessible_const_ptr<float>(x),
        row_schedule.data_ptr<int64_t>(),
        hot_feature_cache.data_ptr<float>(),
        static_cast<int>(hot_node_cutoff),
        out.data_ptr<float>(), num_nodes, feat_dim);
  } else {
    gcn_spmm_hmm_optimized_kernel<false><<<grid, block, 0, stream.stream()>>>(
        get_device_accessible_const_ptr<int>(row_ptr),
        get_device_accessible_const_ptr<int>(col_ind),
        get_device_accessible_const_ptr<float>(deg_inv_sqrt),
        get_device_accessible_const_ptr<float>(x),
        row_schedule.data_ptr<int64_t>(),
        hot_feature_cache.data_ptr<float>(),
        static_cast<int>(hot_node_cutoff),
        out.data_ptr<float>(), num_nodes, feat_dim);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void stage_feature_pages_cuda_(torch::Tensor page_ids, torch::Tensor x,
                               torch::Tensor out, int64_t rows_per_page) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_stage_inputs(page_ids, x, out, rows_per_page);

  const int total_cached_rows = static_cast<int>(out.size(0));
  const int feat_dim = static_cast<int>(out.size(1));
  const int threads = 128;
  dim3 grid(static_cast<unsigned int>(total_cached_rows),
            static_cast<unsigned int>((feat_dim + threads - 1) / threads));
  dim3 block(threads);
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  stage_feature_pages_kernel<<<grid, block, 0, stream.stream()>>>(
      page_ids.data_ptr<int64_t>(),
      get_device_accessible_const_ptr<float>(x),
      out.data_ptr<float>(),
      static_cast<int>(page_ids.numel()),
      static_cast<int>(rows_per_page),
      feat_dim,
      static_cast<int>(x.size(0)));
  C10_CUDA_CHECK(cudaGetLastError());
}

void stage_feature_rows_cuda_(torch::Tensor x, torch::Tensor out) {
  at::cuda::CUDAGuard device_guard(out.device());
  check_stage_rows_inputs(x, out);

  const int num_rows = static_cast<int>(out.size(0));
  const int feat_dim = static_cast<int>(out.size(1));
  const int threads = 128;
  dim3 grid(static_cast<unsigned int>(num_rows),
            static_cast<unsigned int>((feat_dim + threads - 1) / threads));
  dim3 block(threads);
  auto stream = at::cuda::getDefaultCUDAStream(out.get_device());
  stage_feature_rows_kernel<<<grid, block, 0, stream.stream()>>>(
      get_device_accessible_const_ptr<float>(x), out.data_ptr<float>(),
      num_rows, feat_dim);
  C10_CUDA_CHECK(cudaGetLastError());
}
