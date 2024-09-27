#include "grkan_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(grkan, m) {
  m.def("grkan_forward_cuda(Tensor x, Tensor n, Tensor d, int group) -> Tensor");
  m.def("grkan_backward_cuda(Tensor grad_output, Tensor x, Tensor n, Tensor d, int group) -> Tensor[]");
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(grkan, CUDA, m) {
  m.impl("grkan_forward_cuda", &grkan::grkan_forward_cuda);
  m.impl("grkan_backward_cuda", &grkan::grkan_backward_cuda);
}

