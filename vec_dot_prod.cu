#define THREADS_PER_BLOCK 256
__global__ void vec_dot_prod_kernel(const float* a, const float* b, float* result, int len){
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gidx < len){
        result[gidx] = a[gidx] * b[gidx];
    }

    __syncthreads();
    
    if (threadIdx.x == 0){
        for (int i = 1; i < len; i++){
            result[0] += result[i];
        }
    }
}

torch::Tensor vec_dot_prod(torch::Tensor a, torch::Tensor b){
    int len = a.size(0);
    auto result = torch::empty_like(a);
    int num_blocks = (len + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
    vec_dot_prod_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(a.data_ptr<float>(),b.data_ptr<float>(), result.data_ptr<float>(), len);
    
    return result;
}