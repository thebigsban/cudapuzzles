#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void add_ten_mat_kernel(const float* input, float* result, int height, int width){

    // i know it's not necessary but i will try to use blocks here



    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;



    if (row_idx < height && col_idx < width){

        int ndx = row_idx * width + col_idx;

        result[ndx] = input[ndx] + 10; 

    }







}



torch::Tensor add_ten_mat(torch::Tensor input){

    int height = input.size(0);

    int width = input.size(1);



    auto result = torch::empty_like(input);

    dim3 threads_per_block(16,16);

    

    dim3 num_blocks((width + threads_per_block.x - 1)/threads_per_block.x, (height + threads_per_block.y - 1)/threads_per_block.y);



    add_ten_mat_kernel<<<num_blocks, threads_per_block>>>(input.data_ptr<float>(), result.data_ptr<float>(), height, width);



    return result;



}