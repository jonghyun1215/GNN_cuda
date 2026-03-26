#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

inline void *get_device_accessible_void_ptr(const torch::Tensor &tensor) {
  if (tensor.numel() == 0) {
    return nullptr;
  }
  if (tensor.is_cuda()) {
    return tensor.data_ptr();
  }
  TORCH_CHECK(tensor.device().is_cpu(),
              "Expected CPU or CUDA tensor for device access");

  cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, tensor.data_ptr());
  if (status != cudaSuccess) {
    cudaGetLastError();
    TORCH_CHECK(false,
                "CPU tensor is not backed by CUDA-mapped host memory");
  }

  void *device_ptr = attrs.devicePointer;
  if (device_ptr == nullptr) {
    status = cudaHostGetDevicePointer(&device_ptr, tensor.data_ptr(), 0);
    if (status != cudaSuccess) {
      cudaGetLastError();
      TORCH_CHECK(false,
                  "Failed to obtain a device alias for CPU mapped memory");
    }
  }
  TORCH_CHECK(device_ptr != nullptr,
              "CPU tensor does not expose a device-accessible pointer");
  return device_ptr;
}

template <typename T> inline T *get_device_accessible_ptr(const torch::Tensor &tensor) {
  return static_cast<T *>(get_device_accessible_void_ptr(tensor));
}

template <typename T>
inline const T *get_device_accessible_const_ptr(const torch::Tensor &tensor) {
  return static_cast<const T *>(get_device_accessible_void_ptr(tensor));
}

