import torch
from torch.utils.cpp_extension import load_inline
#from profile import time_cuda_function
import pdb 

cuda_source = '''
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
'''

cpp_source = "torch::Tensor add_ten(torch::Tensor vec);"

add_ten_extension = load_inline(
    name = 'add_ten_extension',
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['add_ten'],
    with_cuda = True,
    extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
    build_directory = "./load_inline_cuda",
)

a = torch.tensor([1,2,3,4,5,6,67,7,8,9], device ='cuda').type(torch.float)
b = add_ten_extension.add_ten(a)
print(b)

assert torch.allclose(a+10, b), "first result is not correct"

# tried 10k but max number of threads per block is 1024
c = torch.randn(1000, dtype = torch.float).cuda()
d = add_ten_extension.add_ten(c)
#pdb.set_trace()
print(a.shape, b.shape, c.shape, d.shape)
assert torch.allclose(c+10, d), "second result is not correct"


#a = torch.randn(10000, dtype = torch.float).cuda()



#def pyt_add_ten(a):
#    return a + 10




#with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
#    add_ten_extension.add_ten(a)
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit = 10))
#a.cpu()

# a = torch.randn(10000, dtype = torch.float).cuda()
# with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
#     pyt_add_ten(a)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit = 10))
# 
