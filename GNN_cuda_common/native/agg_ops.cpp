#include <torch/extension.h>

void spmm_sum_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                            torch::Tensor x, torch::Tensor out);
void spmm_mean_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                             torch::Tensor x, torch::Tensor out);
void linear_forward_cuda_(torch::Tensor x, torch::Tensor weight,
                          torch::Tensor bias, torch::Tensor out, bool relu);
void tensor_add_inplace_cuda_(torch::Tensor dst, torch::Tensor src, double alpha);
void relu_inplace_cuda_(torch::Tensor tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_sum_forward_", &spmm_sum_forward_cuda_,
        "CSR sum aggregation into a preallocated output tensor");
  m.def("spmm_mean_forward_", &spmm_mean_forward_cuda_,
        "CSR mean aggregation into a preallocated output tensor");
  m.def("linear_forward_", &linear_forward_cuda_,
        "Dense linear update into a preallocated output tensor");
  m.def("tensor_add_inplace_", &tensor_add_inplace_cuda_,
        "In-place dst += alpha * src with CPU-mapped/CUDA source support");
  m.def("relu_inplace_", &relu_inplace_cuda_, "In-place ReLU");
}
