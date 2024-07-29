#include <torch/extension.h>
torch::Tensor vec_dot_prod(torch::Tensor a, torch::Tensor b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("vec_dot_prod", torch::wrap_pybind_function(vec_dot_prod), "vec_dot_prod");
}