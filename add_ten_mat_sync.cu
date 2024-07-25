#define BLOCKSIZE 16

__global__ void add_ten_mat_sync_kernel(const float* input, float* result, int height, int width){
    // implemented with shared memory and thread synchronization as an exercise

    __shared__ float temp [BLOCKSIZE][BLOCKSIZE];


    int grow_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gcol_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lrow_idx = threadIdx.x;
    int lcol_idx = threadIdx.y;

    int ndx = grow_idx * width + gcol_idx;
    if (grow_idx < height && gcol_idx < width){
        temp[lrow_idx][lcol_idx] = input[ndx];
        // synchronize
        __syncthreads();   
    }

    if (grow_idx < height && gcol_idx < width){
        result[ndx] = temp[lrow_idx][lcol_idx] + 10;
    }
}

torch::Tensor add_ten_mat_sync(torch::Tensor input){

    int height = input.size(0);
    int width = input.size(1);

    auto result = torch::empty_like(input);
    dim3 threads_per_block(BLOCKSIZE,BLOCKSIZE);
    
    dim3 num_blocks((width + threads_per_block.x - 1)/threads_per_block.x, (height + threads_per_block.y - 1)/threads_per_block.y);

    add_ten_mat_sync_kernel<<<num_blocks, threads_per_block>>>(input.data_ptr<float>(), result.data_ptr<float>(), height, width);

    return result;

}