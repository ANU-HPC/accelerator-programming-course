# Streaming 

In this lab, you will use CUDA streams to overlap data transfers with computation.

In the initial GPU implementation of the `findkeywords` program, there is considerable time spent in transferring data which is not overlapped by computation.
With data parallel tasks, both computation and transfers can be partitioned, allowing them to be overlapped to improve performance.
This can have the added benefit of reducing the memory requirements on the GPU, as you can just allocate a fixed number of buffers and recycle them.
This approach is described as *streaming* and is facilitated by CUDA streams.

## Analyzing Data Transfer 

Copy your current implementation of `findkeywords` into this directory, and check that it compiles and runs.
Now use `nvvp` to see how much time is spent transferring data compared to the time it takes to execute the kernel.

Using a streaming approach, what is the maximum speed up you could hope to obtain?
When would it not be worth attempting to use a streaming approach?

## Partitioning the Kernel

To start with just break up the kernel execution into sections (so leave the copying of the text data in one large transfer).  This will enable you to first check your kernel is working correctly.  Also note the performance cost is associated with just dividing the kernel and executing it sections (basically the additional overhead of launching more kernels - also if the sections are too small you will not have enough work for the GPU which will adversely affect performance).

## Partitioning Data Transfer

To apply streaming you also need to break up the transfers into sections which correspond to the sections of the kernel .
For now, keep the full allocation of memory for the text data on the GPU.
Use a stream per kernel execution to enforce completion of the required transfers prior to execution of the kernel.
Note: you will need to transfer the first two sections of data prior to executing the first kernel, as the kernel needs to read past the section of data it is responsible for when matching (this assumes that all keywords are shorter than the section length).  

Once this is working correctly, use `nvvp` to verify that you are overlapping the transfer with computation.
What performance improvement do you observe?

## Extra Time? 

Use a fixed number of buffers on the GPU for storing the text data.
Once the processing on a buffer is completed, you can reuse it for following sections.
Note: not only do you need to ensure that the data is available prior to executing the kernel, you also need to ensure that later transfers only reuse a buffer after the kernel is finished. 

Some hints:
+ use an array of buffers with a circular index (take the modulus of this index by the number of buffers every time you increment it).
+ use a fixed number of streams, also in an array with a circular index. 
