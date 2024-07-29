import torch
from torch.utils.cpp_extension import load_inline
#from profile import time_cuda_function
import pdb 

with open('./vec_dot_prod.cu') as f:
    cuda_source = f.readlines()

with open('./vec_dot_prod.cpp') as f:
    cpp_source = f.readlines()

vec_dot_prod_extension = load_inline(
    name = 'vec_dot_prod_extension',
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['vec_dot_prod'],
    with_cuda = True,
    extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
    build_directory = "./cuda_modules/vec_dot_prod/",
)

a = torch.tensor([1,2,3,4,5,6,7,8,9,0], device ='cuda').type(torch.float)

c = vec_dot_prod_extension.vec_dot_prod(a,a)
pdb.set_trace()
assert torch.allclose(c, ans), "result is not correct"