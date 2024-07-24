import torch
from torch.utils.cpp_extension import load_inline
#from profile import time_cuda_function

cuda_source = '''
__global__ void add_ten_kernel(const float* vec, float* result, const int* maxlen) {
    local_idx = threadIdx.x;
    if (local_idx < maxlen){
        result[local_idx] = input[local_idx] + 10;
    }
}

torch::Tensor add_ten(torch::Tensor vec){
    const auto maxlen = vec.size();

    auto result = torch::empty_like(vec);

    add_ten_kernel<<<1, maxlen>>>(vec.data_ptr<float>(), result.data_ptr<float>(), maxlen);

    return result;
}
'''

cpp_source = "torch::Tensor add_ten(torch::Tensor vec)"

add_ten_extension = load_inline(
    name = 'add_ten_extension',
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['add_ten'],
    with_cuda = True,
    extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
    build_directory = "./load_inline_cuda",
)

a = torch.tensor([1,2,3,4,5,6,67,7,8,9], device ='cuda')
print(add_ten_extension.add_ten(a))