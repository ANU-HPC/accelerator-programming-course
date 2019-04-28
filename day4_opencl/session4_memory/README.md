# Using Private and Local Memory

In this exercise, you will explore the use of different memory types within the OpenCL memory hierarchy to improve the performance of data access within an OpenCL kernel.

## Private Memory

Starting with your naive matrix multiplication code from the previous lab, modify the kernel so that each work-item copies its own row of A into private memory (i.e. a local variable within the kernel).
Verify that your kernel continues to produce the correct result.

Measure the runtime and GFLOP/s on both CPU and GPU, and compare to the previous version.
How did the use of private memory affect performance for each device?

## Local Memory

Now modify the kernel so that each work-group collaborates to copy its own column of B into local memory.
You will need to modify both the kernel and the host program to set up a local memory argument.
Verify that your kernel continues to produce the correct result.

Again, measure the runtime and GFLOP/s on both CPU and GPU, and compare to the private-memory version.
How did the use of local memory affect performance for each device?

Of all the implementations you've created so far, which performed best on each device?
Can this be explained by your understanding of the differences in memory hierarchies between CPU and GPU?

## Extra Time?

Optimize the matrix multiplication further to implement *blocking*, so that the kernel operates on tiles of matrices A, B and C that just fit in the fastest memory type available.
