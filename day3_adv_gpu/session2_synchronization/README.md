# Warp-Level Synchronization

The function `__syncthreads();` provides synchronization across an entire block, however, this can be a costly operation as all the warps within the block must wait until any can proceed past this point.
However, because GPUs operate in a SIMT fashion there is inherent synchronization at the warp level.
In this activity, we will use warp-level synchronization to obtain considerable performance improvement.

## Setup

Copy your `matVecMult` code (that uses shared memory to do a tree based reduction) from the previous session into this directory.
Compile and run the code, and measure the time taken to perform the multiplication. 

## Synchronizing Between Threads in a Warp 

The number of columns in our matrix is 1024, which happens to be an exact multiple of 32.
If we launch blocks of 1024 threads we will have 32 warps in each block (each warp contains 32 threads).
Rather than using shared memory for storing the intermediate results, we can just use registers and synchronize within the warps. 

The optimized reduction may be broken into four stages:
+ Have each thread calculate the multiplication of a element from the matrix and an element of the `v1` and store the result in a register (just a local variable).
+ Within each warp, use  `__shfl_down_sync` to sum the result into a single value.
Store this value into shared memory (so each block needs an array of 32 floats). 
+ Similarly, have the first warp within the block reduce the result down to a single value.
+ Finally, use 1 thread in the block to store the result to the global memory location. 
Compare the performance of this approach with the previous one. 




