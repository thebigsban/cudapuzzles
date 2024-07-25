#define SUMOVER 3
#define THREADS_PER_BLOCK 256
void __global__ sum_last_three_kernel(const float* input, float* result) {

    // initialize temporary shared array with all zeroes
    __shared__ float temp[THREADS_PER_BLOCK + SUMOVER - 1];

    // calculate global and local indices
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x;

    // the index of the temp array is longer than the number of outputs by SUMOVER-1
    int tindex = lindex + SUMOVER - 1;
    
    // read input into shared memory
    temp[tindex] = input[gindex];
    
    // synchronize threads
    __syncthreads();

    // number to keep track of sum
    float out = 0;

    for (int i = 0; i <= SUMOVER-1; i++){
        out += temp[lindex + i];
    }
    result[gindex] = out;
}


torch::Tensor sum_last_three(torch::Tensor input){

    int totallen = input.size(0);
    //int threads_per_block = 256;
    dim3 numblocks((totallen + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);

    auto result = torch::empty_like(input);

    sum_last_three_kernel<<<numblocks, THREADS_PER_BLOCK>>>(input.data_ptr<float>(), result.data_ptr<float>());

    return result;

}