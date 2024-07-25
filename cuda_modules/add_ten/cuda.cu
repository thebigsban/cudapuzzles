#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_ten_kernel(const float* vec, float* result, int maxlen) {
    int local_idx = threadIdx.x;
    if (local_idx < maxlen) {
        result[local_idx] = vec[local_idx] + 10;
    }
}

torch::Tensor add_ten(torch::Tensor vec) {
    const auto maxlen = vec.size(0);

    auto result = torch::empty_like(vec);

    add_ten_kernel<<<1, maxlen>>>(vec.data_ptr<float>(), result.data_ptr<float>(), maxlen);

    return result;
}
