# Reduce

Implementing a "map" operation on a GPU is straightforward: just use threads to work in parallel on different parts of the data.
However, implementing a parallel "reduce" operation requires more effort to optimize for performance. 

## Profiling Sequential Performance

The file `matVecMult.cu` contains a simple sequential implementation of matrix-vector multiplication.
As the code does not take advantage of multiple CUDA threads, you might expect it to run slower on the GPU than an equivalent sequential implementation for the CPU.
Read though the code to understand how it is working.
Pay particular attention to the way the matrix is stored.
Compile and run the program, and use `nvprof` to measure how long the kernel takes to execute.
Make a note of this so you can compare performance as we make improvements to the approach (either in a text file or just on paper).  


## Implementing a Parallel Reduction

Use grid-block striding to divide the work of calculating the dot product between the vector and a row of the matrix to produce a single element of the result vector.
Play around with the number of block and threads per block to see if you can improve performance.

What choice of dimensions gives you the best results?
How does this compare to the original implementation?

Note: your code should also work with 1 block and 1 thread per block (check and time this also - it should be basically the same as the original approach).

## Optimized Reduction Using Shared Memory

The vector `v1` is read multiple times within a block,  as all the rows use it within its calculation.
As such this can be loaded into shared memory and read from shared memory rather than global memory.
Note you can either assume the vector size is fixed and constant (1024 elements), or better still, you can use the 3rd parameter in "<<<>>>" to specify the number of elements to allocate for the shared memory array.

Use a team of threads to copy this data over to shared memory; you will need to `__syncthreads();` before any threads of the block read this shared memory to ensure it has all been properly loaded.

What performance improvements, if any, do you gain?

Remember to record the time the kernel takes.

## Comparing with Outer-Product-Based Implementation

Restructure your kernel such that each block is responsible for one row and each thread within this block does a single multiplication between an element of the matrix and an element of "v1".  Store this multiplication into an array in shared memory (Is there any point loading "v1" into shared memory now?).

Use a tree based reduction approach to sum this result in shared memory.u
Again, you will need to use `__syncthreads();` at each step to ensure the previous levels results are completed before the next level uses their values for its calculation.
Once the sum is calculated one of the threads in the block should copy the result into the correct location in `v2`, which is in global memory.

What are the performance results and how do they compare with previous approaches?

The current implementation only works with NxN matrices.   If you modified the implementation such that it worked on MxN matrices then how would the approach taken in Part 3 compare to the approach in Part 4 if M was much larger than N?   How about if N was much larger than M?

