import torch
from torch.utils.cpp_extension import load_inline
#from profile import time_cuda_function
import pdb 

with open('./add_ten_mat.cu') as f:
    cuda_source = f.readlines()

with open('./add_ten_mat.cpp') as f:
    cpp_source = f.readlines()

add_ten_mat_extension = load_inline(
    name = 'add_ten_mat_extension',
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['add_ten_mat'],
    with_cuda = True,
    extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
    build_directory = "./cuda_modules/add_ten_mat/",
)

a = torch.tensor([[1,2,3,4,5],[6,7,8,9,0]], device ='cuda').type(torch.float)

c = add_ten_mat_extension.add_ten_mat(a)

assert torch.allclose(c, a+10), "result is not correct"