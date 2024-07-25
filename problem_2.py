import torch
from torch.utils.cpp_extension import load_inline
#from profile import time_cuda_function
import pdb 

with open('./vector_sum.cu') as f:
    cuda_source = f.readlines()

with open('./vector_sum.cpp') as f:
    cpp_source = f.readlines()

vec_sum_extension = load_inline(
    name = 'vec_sum_extension',
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['vec_sum'],
    with_cuda = True,
    extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
    build_directory = "./cuda_modules/vec_sum/",
)

a = torch.tensor([1,2,3,4,5], device ='cuda').type(torch.float)
b = torch.tensor([5,6,7,8,9], device ='cuda').type(torch.float)

c = vec_sum_extension.vec_sum(a,b)

assert torch.allclose(c, a+b), "result is not correct"