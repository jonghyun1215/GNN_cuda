#include <torch/extension.h>

void spmm_gcn_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                            torch::Tensor deg_inv_sqrt, torch::Tensor x,
                            torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_gcn_forward_", &spmm_gcn_forward_cuda_,
        "GCN normalized CSR aggregation into a preallocated output tensor");
}

