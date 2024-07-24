#include <torch/extension.h>
torch::Tensor add_ten(torch::Tensor vec)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("add_ten", torch::wrap_pybind_function(add_ten), "add_ten");
}