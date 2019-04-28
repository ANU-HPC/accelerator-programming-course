# GPU Memory

Significant performance gains can often be made with strategic use of the available memory on the GPU.
An initial implementation of a program might place all data in either registers (which are just variables) or global memory.
Once this is done, and your implementation is working correctly, then you should consider if any advantages could be gained by moving data into shared and constant memory -- and this is exactly what we will be doing in this lab. 

## Analyzing Memory Use

Copy your GPU implementation of `listprimes.cu` from Session 1 (along with the `makefile`) into this directory.
Check it is still working correctly.
Then read through the code of the kernel and consider what memory is being read/written and were this memory is located.
Also use the NVIDIA Visual Profiler `nvvp` to provide an analysis of the kernel.
What is the bottleneck?

## Using Shared Memory Within a Thread Block

You may have noticed that every thread repeatedly reads the global memory location that is used to flag when a factor has been found.
This is not actually required, as if a thread checks extra factors after the number has already found to be not prime, it will not change the correctness of the implementation.
So rather than have every thread read this flag, just have one thread in the block read the value and one thread write to it if needed.
Then maintain a copy of this data in shared memory for that group of threads.  

You will need to use `__shared__` to mark some memory as shared, and also `__syncthreads` to synchronize the threads within a block.

What performance improvement have you gained?

## Using Both Shared and Constant Memory

Perform a similar optimization for the `keywordfinder` program from Session 2.
For this program, some of the data will benefit greatly by being placed in `__constant__` memory, other data would be better placed in `__shared__` memory.

What performance improvement have you gained? 
