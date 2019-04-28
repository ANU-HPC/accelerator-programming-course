
# Synchronization in OpenCL

In this lab you will use synchronization between work-items to implement a parallel reduction in OpenCL.

## Numerical Integration of Pi

The file `pi.cpp` contains a simple sequential implementation of rectangular integration of a quarter circle to compute an approximation to pi.
Using this code as a base, write an OpenCL host program `pi_ocl.cpp` and kernel `pi_ocl.cl` to compute the integration using OpenCL.

You will need to implement a reduction of the partial results from each OpenCL work-item.
Note: each work-item should compute multiple iterations of the loop, i.e. don't create one work item per iteration.
(To do so would make the reduction so costly that performance would be terrible.)
An initial implementation could simply compute an array of partial results, transfer it back to the host and sum the results on the host.

A better implementation might perform the reduction as a kernel on the device.

## Extra Time?

Implement a tree-based reduction as you did with CUDA in the previous lab.
What are the challenges for implementing a tree-based reduction in OpenCL?

The time to perform the reduction is not a major component of the overall runtime of the numerical integration.
Create another host program which measures the time required to perform a large number of reductions.
How does the performance of your tree-based reduction compare to the simple implementation where one work-item sums all partial values for the work group?

