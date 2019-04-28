# Observing PTX and SASS code, Debugging and Profiling

## CUDA Backend Targets

PTX (Parallel Thread Execution) is a low-level intermediate representation used by CUDA.   Below PTX is SASS (Stream Assembler) which is tied to a particular architecture.
The `cuobjdump` tool allows us to view these lower-level representations to better understand how a kernel maps to the underlying architecture.

To generate the PTX for a compiled program named `helloworldCUDAv1`, run the following command:

```
cuobjdump -ptx helloworldCUDAv1
```

To generate the SASS, run:
```
cuobjdump -sass helloworldCUDAv1
```

+ What do you notice about how the string literal "Hello World" is stored within the kernel?

+ What is the difference in terms of registers between the two ISAs?

## Profiling with `nvprof`

Take one of your application codes and try some more detailed profiling using nvprof.
Test the following options:

```
nvprof --print-gpu-trace <path-to-executable>
nvprof --metrics all <path-to-executable>
nvprof --events all <path-to-executable> 
nvprof --dependency-analysis <path-to-executable> 
```

Note that collecting events/metrics can adversely affect performance.
Once you have identified a metric of interest, you can choose to collect only that metric.

You can also try some of the guided kernel evaluations in `nvvp`.

## Debugging a CUDA Program

The file `bugs.cu` contains a broken CUDA program.

Find and fix the bugs using any of the tools at your disposal, including:
- error checking on CUDA calls-
- `printf` statements in the kernel code
- debugging with `cuda-gdb`
- code analysis provided within `nsight`

Note: you could simply use careful code inspection to fix these errors, however, the objective of this exercise is to familiarize you with the tools available to diagnose problems with CUDA code.

The `cuda-gdb` debugger is very similar to `gdb`.
To use it, compile your code with the options `g -G`, and then run `cuda-gdb` with the path to your application as the first argument. For example:
```
cuda-gdb <path-to-executable>
```

or
```
cuda-gdb --args <path-to-executable> arg1 arg2
```

Once the debugger has started, you can set a breakpoint with:
```
break kernelname
```

then `run` and it will run until the kernel.
You can then `step` through the execution, `print` variable values, or change thread focus e.g. `cuda thread 3`  

For some more info see:
http://developer.download.nvidia.com/GTC/PDF/1062_Satoor.pdf

