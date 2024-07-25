# Cuda Puzzles

Working through puzzles from [Sasha Rush's github repo](https://github.com/srush/GPU-Puzzles/tree/main). Instead of using his interface though, I'll write it in raw CUDA and load them through the PyTorch's `load_inline`. 

Also maybe doing some profiling vs. raw Python code :3. 

## Environment Setup

```bash
yes | conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
yes | conda install -c conda-forge jupyter matplotlib 
yes | pip3 install einops
yes |pip3 install tqdm
```

## Problems

### Problem 1 - DONE
Implement a "kernel" (GPU function) that adds 10 to each position of vector a and stores it in vector out. You have 1 thread per position.

### Problem 2 - DONE
Implement a kernel that adds together each position of a and b and stores it in out. You have 1 thread per position.

### Problem 3 - DONE (Problem 1)
Implement a kernel that adds 10 to each position of a and stores it in out. You have more threads than positions.

### Problem 4 - DONE
Implement a kernel that adds 10 to each position of a and stores it in out. Input a is 2D and square. You have more threads than positions.

### Problem 5 - DONE
Implement a kernel that adds a and b and stores it in out. Inputs a and b are vectors. You have more threads than positions.

### Problem 6 - DONE
Implement a kernel that adds 10 to each position of a and stores it in out. You have fewer threads per block than the size of a.

### Problem 7 - DONE
Implement the same kernel in 2D. You have fewer threads per block than the size of a in both directions.

### Problem 8 - DONE
Implement a kernel that adds 10 to each position of a and stores it in out. You have fewer threads per block than the size of a.

Warning: Each block can only have a constant amount of shared memory that threads in that block can read and write to. This needs to be a literal python constant not a variable. After writing to shared memory you need to call cuda.syncthreads to ensure that threads do not cross.

(This example does not really need shared memory or syncthreads, but it is a demo.)

### Problem 9
Implement a kernel that sums together the last 3 position of a and stores it in out. You have 1 thread per position. You only need 1 global read and 1 global write per thread.

### Problem 10
Implement a kernel that computes the dot-product of a and b and stores it in out. You have 1 thread per position. You only need 2 global reads and 1 global write per thread.

### Problem 11
Implement a kernel that computes a 1D convolution between a and b and stores it in out. You need to handle the general case. You only need 2 global reads and 1 global write per thread.

### Problem 12
Implement a kernel that computes a sum over a and stores it in out. If the size of a is greater than the block size, only store the sum of each block.

We will do this using the parallel prefix sum algorithm in shared memory. That is, each step of the algorithm should sum together half the remaining numbers. 

### Problem 13
Implement a kernel that computes a sum over each column of a and stores it in out.

### Problem 14
Implement a kernel that multiplies square matrices a and b and stores the result in out.

Tip: The most efficient algorithm here will copy a block into shared memory before computing each of the individual row-column dot products. This is easy to do if the matrix fits in shared memory. Do that case first. Then update your code to compute a partial dot-product and iteratively move the part you copied into shared memory. You should be able to do the hard case in 6 global reads.
