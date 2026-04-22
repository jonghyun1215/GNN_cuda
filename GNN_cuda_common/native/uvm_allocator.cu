#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <sstream>
#include <string>
#include <vector>

namespace {

at::ScalarType decode_dtype(int dtype_code) {
  switch (dtype_code) {
  case 0:
    return at::kFloat;
  case 1:
    return at::kDouble;
  case 2:
    return at::kHalf;
  case 3:
    return at::kInt;
  case 4:
    return at::kLong;
  default:
    TORCH_CHECK(false, "Unsupported dtype code: ", dtype_code);
  }
}

int64_t multiply_sizes(const std::vector<int64_t> &sizes) {
  int64_t numel = 1;
  for (const int64_t dim : sizes) {
    TORCH_CHECK(dim >= 0, "Negative shape dimension: ", dim);
    numel *= dim;
  }
  return numel;
}

void check_supported_tensor(const torch::Tensor &tensor) {
  TORCH_CHECK(tensor.is_cuda() || tensor.device().is_cpu(),
              "Expected a CPU or CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "Expected a contiguous tensor");
}

bool device_supports_pageable_memory_access(int device_index) {
  int pageable = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess,
                             device_index));
  return pageable != 0;
}

void ensure_hmm_supported(int device_index) {
  TORCH_CHECK(device_supports_pageable_memory_access(device_index),
              "The selected CUDA device does not support Linux HMM pageable "
              "memory access");
}

bool current_device_supports_pageable_memory_access() {
  int device_index = 0;
  cudaError_t status = cudaGetDevice(&device_index);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  int pageable = 0;
  status = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess,
                                  device_index);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return pageable != 0;
}

std::string pointer_type_name(cudaMemoryType type) {
  switch (type) {
  case cudaMemoryTypeHost:
    return "host";
  case cudaMemoryTypeDevice:
    return "device";
  case cudaMemoryTypeManaged:
    return "managed";
  case cudaMemoryTypeUnregistered:
    return "unregistered";
  default:
    return "unknown";
  }
}

} // namespace

torch::Tensor managed_empty_cuda(std::vector<int64_t> sizes, int dtype_code,
                                 int device_index) {
  TORCH_CHECK(device_index >= 0, "device_index must be >= 0");
  const auto dtype = decode_dtype(dtype_code);
  const int64_t numel = multiply_sizes(sizes);
  const size_t bytes = static_cast<size_t>(numel) * c10::elementSize(dtype);

  at::cuda::CUDAGuard device_guard(device_index);
  void *ptr = nullptr;
  if (bytes > 0) {
    C10_CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
  }
  auto options =
      torch::TensorOptions().device(torch::kCUDA, device_index).dtype(dtype);
  auto deleter = [device_index](void *memory) {
    if (memory == nullptr) {
      return;
    }
    at::cuda::CUDAGuard device_guard(device_index);
    cudaFree(memory);
  };
  auto tensor = torch::from_blob(ptr, sizes, deleter, options);
  TORCH_CHECK(tensor.is_cuda(),
              "Managed allocation did not materialize as a CUDA tensor");
  return tensor;
}

torch::Tensor hmm_empty_cpu(std::vector<int64_t> sizes, int dtype_code,
                            int device_index) {
  TORCH_CHECK(device_index >= 0, "device_index must be >= 0");
  ensure_hmm_supported(device_index);
  const auto dtype = decode_dtype(dtype_code);
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(dtype);
  return torch::empty(sizes, options);
}

void prefetch_cuda_(torch::Tensor tensor, int location_code, int device_index) {
  check_supported_tensor(tensor);
  TORCH_CHECK(device_index >= 0, "device_index must be >= 0");
  if (tensor.device().is_cpu()) {
    ensure_hmm_supported(device_index);
  }
  at::cuda::CUDAGuard device_guard(device_index);
  auto stream = at::cuda::getDefaultCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemPrefetchAsync(tensor.data_ptr(), tensor.nbytes(),
                                      location_code, stream.stream()));
}

void advise_preferred_location_cuda_(torch::Tensor tensor, int location_code,
                                     int device_index) {
  check_supported_tensor(tensor);
  if (tensor.device().is_cpu()) {
    ensure_hmm_supported(device_index);
  }
  at::cuda::CUDAGuard device_guard(device_index);
  C10_CUDA_CHECK(cudaMemAdvise(tensor.data_ptr(), tensor.nbytes(),
                               cudaMemAdviseSetPreferredLocation,
                               location_code == 0 ? device_index
                                                  : location_code));
}

void advise_accessed_by_cuda_(torch::Tensor tensor, int location_code,
                              int device_index) {
  check_supported_tensor(tensor);
  if (tensor.device().is_cpu()) {
    ensure_hmm_supported(device_index);
  }
  at::cuda::CUDAGuard device_guard(device_index);
  C10_CUDA_CHECK(cudaMemAdvise(tensor.data_ptr(), tensor.nbytes(),
                               cudaMemAdviseSetAccessedBy,
                               location_code == 0 ? device_index
                                                  : location_code));
}

void advise_read_mostly_cuda_(torch::Tensor tensor, bool enabled) {
  check_supported_tensor(tensor);
  int device_index = tensor.is_cuda() ? tensor.get_device() : 0;
  if (tensor.device().is_cpu()) {
    int current_device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    device_index = current_device;
    ensure_hmm_supported(device_index);
  }
  at::cuda::CUDAGuard device_guard(device_index);
  const auto advise =
      enabled ? cudaMemAdviseSetReadMostly : cudaMemAdviseUnsetReadMostly;
  C10_CUDA_CHECK(
      cudaMemAdvise(tensor.data_ptr(), tensor.nbytes(), advise, device_index));
}

pybind11::dict pointer_info_cuda(torch::Tensor tensor) {
  pybind11::dict out;
  out["device_type"] = std::string(tensor.device().type() == c10::kCUDA ? "cuda"
                                                                        : "cpu");
  out["device_index"] = tensor.device().has_index() ? tensor.device().index()
                                                    : -1;
  out["is_managed"] = false;
  out["is_host_mapped"] = false;
  out["is_hmm"] = false;
  out["pointer_type"] = tensor.device().is_cpu() ? std::string("cpu")
                                                 : std::string("unknown");

  if (tensor.numel() == 0) {
    return out;
  }

  cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, tensor.data_ptr());
  if (status != cudaSuccess) {
    cudaGetLastError();
    if (tensor.device().is_cpu() && current_device_supports_pageable_memory_access()) {
      out["is_hmm"] = true;
      out["pointer_type"] = std::string("hmm");
    }
    return out;
  }
  out["is_managed"] = attrs.type == cudaMemoryTypeManaged;
  const bool is_host_mapped =
      attrs.type == cudaMemoryTypeHost && attrs.devicePointer != nullptr;
  out["is_host_mapped"] = is_host_mapped;
  const bool is_hmm = tensor.device().is_cpu() && !is_host_mapped &&
                      current_device_supports_pageable_memory_access();
  out["is_hmm"] = is_hmm;
  out["pointer_type"] = is_host_mapped
                            ? std::string("host_mapped")
                            : (is_hmm ? std::string("hmm")
                                      : pointer_type_name(attrs.type));
  return out;
}
