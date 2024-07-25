import torch
from torch.utils.cpp_extension import load_inline
#from profile import time_cuda_function
import pdb 

with open('./sum_last_three.cu') as f:
    cuda_source = f.readlines()

with open('./sum_last_three.cpp') as f:
    cpp_source = f.readlines()

sum_last_three_extension = load_inline(
    name = 'sum_last_three_extension',
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['sum_last_three'],
    with_cuda = True,
    extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
    build_directory = "./cuda_modules/sum_last_three/",
)

a = torch.tensor([1,2,3,4,5,6,7,8,9,0], device ='cuda').type(torch.float)

c = sum_last_three_extension.sum_last_three(a)
pdb.set_trace()
ans = torch.tensor([1.,  3.,  6.,  9., 12., 15., 18., 21., 24., 17.])
assert torch.allclose(c, ans), "result is not correct"