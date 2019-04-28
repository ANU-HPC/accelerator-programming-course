# Matrix Multiplication with OpenCL

In this lab, you will write your first OpenCL kernel from scratch to perform multiplication of matrices.

## Implementing a Kernel

The provided file `matmul.cpp` implements an OpenCL host program which calls a kernel `mmul`.
The kernel is currently implemented as a stub in the file `kernel.cl`.
Starting with the sequential matrix multiplication code provided in the file `matrix_lib.c`, implement the `mmul` kernel.

Verify that your kernel produces the correct results.
How many work items are created?

Measure and compare the runtime and GFLOP/s between CPU and GPU.
You can choose which device is used by passing the device number as a command line argument as follows:
```
./matmul-c++ --device 1
```

Now try reducing the number of work items by assigning a whole row of A to a single work item.
How does this affect the performance on each device?

## Extra Time?

On the GPU, what local work group size is used?
How does this affect the scheduling of work items corresponding to individual elements of the matrix?

