# Chaining OpenCL Kernels

In this lab you will manipulate kernel invocations and buffers using the OpenCL C++ API.

## Vector Addition

The file `vadd.cpp` contains an OpenCL host program implemented using the OpenCL C++ API, which computes the sum of two vectors C = A + B.

The program accepts a single optional integer argument which is the platform ID, and runs on the first device found on that platform.
For example, to run on platform 0 (on your lab computer, the GPU):
```
./vadd-c++ 0
```
or to run on platform 1 (on your lab computer, the CPU):
```
./vadd-c++ 1
```

Run the provided code on both platforms and observe the time taken.

Read the code and consider: where might memory be copied between host and device?
Which operations are synchronous or asynchronous?

## Chaining Additions

Use the existing kernel function to create a chain of vector additions as follows:
- C = A + B
- D = C + E
- F = D + G

As per the existing vectors A and B, the new vectors E and G should be initialized with random values.
You will need to create additional buffers on the host and device side with the appropriate memory access (see the existing code for examples of how to do this).
Modify the test code so that it correctly checks the result of the chain of additions.

## Extra Time?

Compare the time to perform a single vector addition on the CPU to the time previously measured with OpenMP (you may need to modify the timing code to make a truly fair comparison).
