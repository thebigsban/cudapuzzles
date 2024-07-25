#include <torch/extension.h>
torch::Tensor add_ten_mat(torch::Tensor input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("add_ten_mat", torch::wrap_pybind_function(add_ten_mat), "add_ten_mat");
}