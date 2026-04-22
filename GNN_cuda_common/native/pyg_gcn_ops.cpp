#include <torch/extension.h>

void spmm_pyg_gcn_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                                torch::Tensor edge_weight, torch::Tensor x,
                                torch::Tensor out);
void spmm_pyg_gcn_plain_forward_cuda_(torch::Tensor row_ptr,
                                      torch::Tensor col_ind,
                                      torch::Tensor edge_weight,
                                      torch::Tensor x, torch::Tensor out);
void spmm_pyg_gcn_hmm_optimized_forward_cuda_(
    torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor edge_weight,
    torch::Tensor x, torch::Tensor row_schedule,
    torch::Tensor hot_feature_cache, int64_t hot_node_cutoff,
    torch::Tensor out);
void spmm_pyg_gcn_uvm_optimized_forward_cuda_(
    torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor edge_weight,
    torch::Tensor x, torch::Tensor row_schedule,
    torch::Tensor hot_feature_cache, int64_t hot_node_cutoff,
    torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_pyg_gcn_forward_", &spmm_pyg_gcn_forward_cuda_,
        "PyG-style weighted CSR GCN aggregation into a preallocated output tensor");
  m.def("spmm_pyg_gcn_plain_forward_", &spmm_pyg_gcn_plain_forward_cuda_,
        "PyG-style weighted CSR GCN aggregation with shared vectorized loads");
  m.def("spmm_pyg_gcn_hmm_optimized_forward_",
        &spmm_pyg_gcn_hmm_optimized_forward_cuda_,
        "PyG-style weighted CSR GCN aggregation with row-scheduled hot-row caching");
  m.def("spmm_pyg_gcn_uvm_optimized_forward_",
        &spmm_pyg_gcn_uvm_optimized_forward_cuda_,
        "PyG-style weighted CSR GCN aggregation with row-scheduled hot-row caching on UVM inputs");
}
