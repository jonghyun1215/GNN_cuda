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

void check_cuda_tensor(const torch::Tensor &tensor) {
  TORCH_CHECK(tensor.is_cuda(), "Expected a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "Expected a contiguous tensor");
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

torch::Tensor host_mapped_empty_cpu(std::vector<int64_t> sizes, int dtype_code,
                                    int device_index) {
  TORCH_CHECK(device_index >= 0, "device_index must be >= 0");
  const auto dtype = decode_dtype(dtype_code);
  const int64_t numel = multiply_sizes(sizes);
  const size_t bytes = static_cast<size_t>(numel) * c10::elementSize(dtype);

  at::cuda::CUDAGuard device_guard(device_index);
  void *ptr = nullptr;
  if (bytes > 0) {
    C10_CUDA_CHECK(
        cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    void *device_ptr = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr, ptr, 0));
    TORCH_CHECK(device_ptr != nullptr,
                "Mapped host allocation did not expose a device pointer");
  }
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(dtype);
  auto deleter = [](void *memory) {
    if (memory == nullptr) {
      return;
    }
    cudaFreeHost(memory);
  };
  return torch::from_blob(ptr, sizes, deleter, options);
}

void prefetch_cuda_(torch::Tensor tensor, int location_code) {
  check_cuda_tensor(tensor);
  auto device_index = tensor.get_device();
  at::cuda::CUDAGuard device_guard(device_index);
  auto stream = at::cuda::getDefaultCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemPrefetchAsync(tensor.data_ptr(), tensor.nbytes(),
                                      location_code, stream.stream()));
}

void advise_preferred_location_cuda_(torch::Tensor tensor, int location_code,
                                     int device_index) {
  check_cuda_tensor(tensor);
  at::cuda::CUDAGuard device_guard(device_index);
  C10_CUDA_CHECK(cudaMemAdvise(tensor.data_ptr(), tensor.nbytes(),
                               cudaMemAdviseSetPreferredLocation,
                               location_code == 0 ? device_index
                                                  : location_code));
}

void advise_accessed_by_cuda_(torch::Tensor tensor, int location_code,
                              int device_index) {
  check_cuda_tensor(tensor);
  at::cuda::CUDAGuard device_guard(device_index);
  C10_CUDA_CHECK(cudaMemAdvise(tensor.data_ptr(), tensor.nbytes(),
                               cudaMemAdviseSetAccessedBy,
                               location_code == 0 ? device_index
                                                  : location_code));
}

void advise_read_mostly_cuda_(torch::Tensor tensor, bool enabled) {
  check_cuda_tensor(tensor);
  auto device_index = tensor.get_device();
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
  out["pointer_type"] = tensor.device().is_cpu() ? std::string("cpu")
                                                 : std::string("unknown");

  if (tensor.numel() == 0) {
    return out;
  }

  cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, tensor.data_ptr());
  if (status != cudaSuccess) {
    cudaGetLastError();
    return out;
  }
  out["is_managed"] = attrs.type == cudaMemoryTypeManaged;
  const bool is_host_mapped =
      attrs.type == cudaMemoryTypeHost && attrs.devicePointer != nullptr;
  out["is_host_mapped"] = is_host_mapped;
  out["pointer_type"] =
      is_host_mapped ? std::string("host_mapped") : pointer_type_name(attrs.type);
  return out;
}
