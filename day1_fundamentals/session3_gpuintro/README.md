# Introduction to GPU programming - developing a running a simple kernel. 

In any new programming development environment it is important to check that you can compile and run a simple program.  So in the first part of the lab we do this with a provided simple "hello world" program.   Given we wish to check that the parallel aspects are working we use different threads to construct the different characters within the "hello world" string.  This is a little contrived and will be slower and considerably more complex than a simple serial implementation.  However, it does highlight key parts of the programming models,  moreover, it gives us a starting point for developing code in which GPUs may be effectively used.

## Part 1

To find out a little about the GPU you are using you can run "nvidia-smi".  What card do you have and how much memory is available?  Is the GPU currently in use?

## Part 2

Download, compile and run the following programs:
+ helloworldOpenCL.c - OpenCL version of the hello world program.
+ helloworldCUDAv1.cu - CUDA version using pinned memory with explicit host/device memory transfers.  
+ helloworldCUDAv2.cu - CUDA version that uses unified memory.

So to compile the OpenCL version:
```
gcc -o helloworldOpenCL helloworldOpenCL.c -lOpenCL
```

And the CUDA versions:
```
nvcc -o helloworldCUDAv1 helloworldCUDAv1.cu
nvcc -o helloworldCUDAv2 helloworldCUDAv2.cu
```

There is also a `Makefile` provided. 


Read through the code and answer the following questions:
+ Which is the code that executes on the GPU?
+ The kernel's execution needs to finished before the result can be copied from the device to the host.  How is this achieved in the 3 different approaches?

## Part 3

Nvidia have some handy profiling tools, namely "nvprof" and "nvvp",  run them on the CUDA versions of the implementation.   

To get the "nvvp" working you may need to give the full path of the CUDA program and create and set a working directory (this can be done in /tmp and is set in the dialog when the program is run). e.g.

```
mkdir /tmp/nvvp
nvvp $PWD/helloworldCUDAv1
```


Also run "nvprof" with the "--print-gpu-trace" on the two different CUDA versions.  What is the difference between the unified memory approach and the explicitly coping the memory from the device approach?

## Part 4


Modify the OpenCL and one of the CUDA programs such that it:
+ outputs "Hello <your name>" and  
+ makes 1 thread responsible for 2 consecutive characters in the result (so you are using less threads - getting each one to do more work). 







