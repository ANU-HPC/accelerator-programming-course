# Knapsack

In this activity you will gain some more practice in converting some existing code into a parallel GPU implementation using CUDA.   

The Knapsack problem is an optimization problem that finds an optimal subset of objects to place in your "knapsack".
Each item has a weight and a value.
Your "knapsack" has a weight limit that you can't go over.
Within this weight limit, you need to find a subset that returns the maximum total value. 


You will need to gain an understanding of the serial implementation before you move to porting it to the GPU.
In particular you need to understand the data structures along with the data dependencies.
A dynamic programming approach is used, however, this is still slow given the size of the problem it is expected to run on.
The serial implementation works by using a 2D array called `m`.   `m[k][j]` is the maximum value that can be stored in the knapsack keeping the weight less than or equal to `j` and using a subset of items which have index less than `k`.

## Understanding the Sequential Implementation

The directory contains:
+ `knapsackCPU.c` - This holds a CPU implementation that executes on a small data set.   It is mainly just to help you understand the basic algorithm.
+ `knapsackCPUbig.c` - This is the CPU implementation that has the large test set and is the base version that you should used to compare your GPU implementation to in terms of correctness and performance. 
+ `knapsackCUDAv1.cu` - This is the file you need to modify to create your solution.
  The provided version is essentially a copy of `knapsackCPUbig.c` with added header includes for CUDA and the error macro.
  However you will still need to do most of the work.
  You may wish to explore a number of versions; just copy the file and rename it with the next version number.
+ Makefile - This is set up to compile and test your code.

Compile, run and time the CPU implementation.

Have a look at the code and think about the follow questions:
+ How is the data set out?
+ What loops are there and how big are they?
+ What are the data dependencies?

## Parallelizing Knapsack

Decide on how you are going to parallelize your code and get the basics of this working.  This will involve:
+ setting up device memory,
+ transferring the data, and
+ creating and launching the kernel.

With this basic GPU implementation what performance improvements did you gain?

## Optimization

Explore different approaches for improving performance.


