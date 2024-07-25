__global__ void vec_sum_kernel(const float* a, const float* b, float* result, int len) {
    int local_idx = threadIdx.x;

    if (local_idx < len){
        result[local_idx] = a[local_idx] + b[local_idx];
    }
}

torch::Tensor vec_sum(torch::Tensor a, torch::Tensor b){
    int len = a.size(0);
    auto result = torch::empty_like(a);
    vec_sum_kernel<<<1, len>>>(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), len);
    return result;
}