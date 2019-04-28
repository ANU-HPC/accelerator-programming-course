
# Single-Instruction Multiple-Data (SIMD) Vectorization in OpenCL

In this lab you will learn how to migrate the mandelbrot generator from the last session into OpenCL and will ultimately show the hetereogenous platform layer can be used to write cleaner and portable SIMD code.

The [OpenCL Extended Instruction Set](https://www.khronos.org/registry/spir-v/specs/1.0/OpenCL.ExtendedInstructionSet.100.pdf) will help in identifying kernel level instructions and their supported data-types.

## From Intrinsics to OpenCL

The OpenCL program is divided into two parts, the host-side and the device-side.
For this exercise we will be focusing entirely on the device-side -- known as the `kernel`.

To build the program and run the program:

```
make
./mandelbrot 1 0
```
**Note** the `1 0` corresponds to the Intel platform and the CPU device, OpenCL allows you to run the same code on any accelerator, so the user must make this decision -- this will be discussed in greater detail in future workshops.
You can verify that platform 1 and device 0 corresponds to the Intel CPU on your systems by rerunning the `opencl_device_query` application -- from session 1.

The same tools from the prior lab `convert_mandelbrot_csv2png.py` and `plot_times.py` can again be used to view your solution and measure the performance.

This exercise will require you to edit the `mandelbrot_vectorization_kernel.cl` file.
Can you locate and fix the two lines of the
```
__kernel void mandelbrot_vectorized(__global int* map)
```
kernel function? **Hint:** use the provided
```
__kernel void mandelbrot(__global int* map)
```
function for the sample functionality.

Once you've fixed the program, examine the performance of the vectorized code relative to the base-line implementation -- is this what you'd expect? Why?


## Extra Time?

* How does the OpenCL vectorization compare to intrinsics -- in terms of performance, programmability and portability?
