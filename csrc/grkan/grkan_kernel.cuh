#include <torch/extension.h>

namespace grkan {

    at::Tensor grkan_forward_cuda(at::Tensor x, at::Tensor n, at::Tensor d, int64_t group);
    
    std::vector<at::Tensor> grkan_backward_cuda(at::Tensor grad_output, at::Tensor x, at::Tensor n, at::Tensor d, int64_t group);

}
