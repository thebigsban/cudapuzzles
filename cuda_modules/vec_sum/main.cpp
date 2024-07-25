#include <torch/extension.h>
torch::Tensor vec_sum(torch::Tensor a, torch::Tensor b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("vec_sum", torch::wrap_pybind_function(vec_sum), "vec_sum");
}