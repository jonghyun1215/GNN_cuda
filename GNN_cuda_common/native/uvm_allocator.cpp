#include <torch/extension.h>

#include <vector>

torch::Tensor managed_empty_cuda(std::vector<int64_t> sizes, int dtype_code,
                                 int device_index);
torch::Tensor hmm_empty_cpu(std::vector<int64_t> sizes, int dtype_code,
                            int device_index);
void copy_cpu_to_managed_cuda_(torch::Tensor dst, torch::Tensor src);
void prefetch_cuda_(torch::Tensor tensor, int location_code, int device_index);
void prefetch_range_cuda_(torch::Tensor tensor, int64_t offset_bytes,
                          int64_t byte_count, int location_code,
                          int device_index);
void advise_preferred_location_cuda_(torch::Tensor tensor, int location_code,
                                     int device_index);
void advise_accessed_by_cuda_(torch::Tensor tensor, int location_code,
                              int device_index);
void advise_read_mostly_cuda_(torch::Tensor tensor, bool enabled);
pybind11::dict pointer_info_cuda(torch::Tensor tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("managed_empty", &managed_empty_cuda,
        "Allocate a CUDA tensor backed by cudaMallocManaged");
  m.def("hmm_empty", &hmm_empty_cpu,
        "Allocate a CPU tensor backed by ordinary system memory for HMM");
  m.def("copy_cpu_to_managed_", &copy_cpu_to_managed_cuda_,
        "Populate a managed tensor from CPU using host-side memcpy");
  m.def("prefetch_", &prefetch_cuda_, "Prefetch a managed tensor");
  m.def("prefetch_range_", &prefetch_range_cuda_,
        "Prefetch a byte range of a managed or HMM tensor");
  m.def("advise_preferred_location_", &advise_preferred_location_cuda_,
        "Set cudaMemAdviseSetPreferredLocation");
  m.def("advise_accessed_by_", &advise_accessed_by_cuda_,
        "Set cudaMemAdviseSetAccessedBy");
  m.def("advise_read_mostly_", &advise_read_mostly_cuda_,
        "Toggle cudaMemAdviseSetReadMostly");
  m.def("pointer_info", &pointer_info_cuda, "Inspect pointer attributes");
}
