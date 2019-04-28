
# Metaprogramming

In this lab you will learn how to change the meta-characteristics of an OpenCL code: the parameters used to compile the kernel when it is built for a particular device.

Metaprogramming can be used to dynamically change the precision of a kernel e.g. by designing the kernel to operate on a parametrized type REAL (rather than a specified type float/double), then specifying REAL at runtime using OpenCL build options: –DREAL=type.

It can also be used to make runtime decisions that change the functionality of the kernel, or make implementation decisions to improve performance portability.
These can include:

* Switching between scalar and vector types
* Changing whether data is stored in buffers or images
* Toggling use of local memory

This approach requires that the OpenCL kernel be compiled at runtime (it will not work if we are precompiling kernels or using SPIR).

## Kernel Compilation

The example is a simple bilateral filter -- an edge-preserving smoothing/noise reduction filter.
Each pixel of the output image is some function of its neighbouring pixels from the input image and uses `sqrt`, `exp` and `distance` builtins.

A fully working implementation of this code is provided as a starting point.

```
make
./bilateral
```


Experiment with some OpenCL compiler options to improve performance (in line 93 of bilateral.cpp).

Try embedding some simulation parameters into the kernel as compile-time constants using OpenCL build options.
This might not help for every parameter or on every device – try it with a few!
Tip: If verification is too slow, use --noverify flag or set verify = false

## Extra Time?

Our results from 2 different versions were:

```
Using OpenCL device: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz

Processing image of size 1920x1080

Running OpenCL...
OpenCL took 768.5ms (24.0ms / frame)

Running reference...
Reference took 2536.4ms

Verification passed.

```

and

```
Using OpenCL device: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz

Processing image of size 1920x1080

Running OpenCL...
OpenCL took 623.5ms (19.5ms / frame)

Running reference...
Reference took 2552.2ms

Verification passed.

```

Can you do better?

Try the code on both the GPU and CPU.
Get the compiler to generate the assembly code and look through this, correlating it to your source code.
