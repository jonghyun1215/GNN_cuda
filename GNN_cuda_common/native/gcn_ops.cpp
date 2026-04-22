#include <torch/extension.h>

void spmm_gcn_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                            torch::Tensor deg_inv_sqrt, torch::Tensor x,
                            torch::Tensor out);
void spmm_gcn_hmm_forward_cuda_(torch::Tensor row_ptr, torch::Tensor col_ind,
                                torch::Tensor deg_inv_sqrt, torch::Tensor x,
                                torch::Tensor out);
void spmm_gcn_hmm_optimized_forward_cuda_(torch::Tensor row_ptr,
                                          torch::Tensor col_ind,
                                          torch::Tensor deg_inv_sqrt,
                                          torch::Tensor x,
                                          torch::Tensor row_schedule,
                                          torch::Tensor hot_feature_cache,
                                          int64_t hot_node_cutoff,
                                          torch::Tensor out);
void stage_feature_pages_cuda_(torch::Tensor page_ids, torch::Tensor x,
                               torch::Tensor out, int64_t rows_per_page);
void stage_feature_rows_cuda_(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_gcn_forward_", &spmm_gcn_forward_cuda_,
        "GCN normalized CSR aggregation into a preallocated output tensor");
  m.def("spmm_gcn_hmm_forward_", &spmm_gcn_hmm_forward_cuda_,
        "HMM-optimized GCN normalized CSR aggregation into a preallocated output tensor");
  m.def("spmm_gcn_hmm_optimized_forward_", &spmm_gcn_hmm_optimized_forward_cuda_,
        "Hot-node-partitioned HMM GCN aggregation with scheduled rows and a staged device feature slab");
  m.def("stage_feature_pages_", &stage_feature_pages_cuda_,
        "Stage selected feature pages into a device-side scratch buffer");
  m.def("stage_feature_rows_", &stage_feature_rows_cuda_,
        "Stage a contiguous prefix of feature rows into a device-side scratch buffer");
}
