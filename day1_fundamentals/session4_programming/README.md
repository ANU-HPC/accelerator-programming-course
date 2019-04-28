# Implementing a simple data-parallel program on a GPU - finding primes!


In this lab you are given a simple serial implementation of a program that checks if a number is prime.
The objective is to turn this into a GPU implementation that is functionally equivalent, yet executes on the GPU and provides a performance improvement.
The program simply works by checking if there are any factors up to the square root of the number,  if there are no factors then it is prime otherwise the number is not prime. 

## Estimating Performance

Before we get into coding, read the sequential code in `listprimes.c` and estimate how long the program should take to run on a CPU, and what kinds of performance improvements you would get out of a GPU.
Most composite numbers will be identified as such relatively quickly, but confirming that a number is prime may take a long time.
Note: we are looking for primes between 9223372036854775707 and 9223372036854775806; as it turns out, there is just one, namely 9223372036854775783.
So it is this prime that takes most of the effort to check.
Estimate how long your CPU would take to complete this calculation (count the number of times the main loop goes around,  guess the number of clock cycles it would take for the calculations within the loop, multiply them to work out the total number of cycles,  and divide by the clock frequency).
How long would you expect this to take on the GPU?


## Measuring and Analyzing Performance

Compile and run the following program:
+ listprimes.c - does a prime test on a sequence of large numbers

Time how long it takes to run using the Unix `time` command.
How near were you to your estimate from part 1? 

Read through the code and answer the following questions:
+ Which loop(s) could be parallelized?
+ What data dependencies are there and how can they be addressed? 

Now you could focus on improving the implementation by reducing the number of modulo operations (or possibly changing the algorithm),   however, in this activity the focus is on using the GPU to gain a performance improvement while keeping the calculations basically the same. 


## Dividing Work on the GPU

Copy the code to "listprimesCUDA.cu" and attempt to convert it to execute on the GPU.  Use the CUDA code from the previous lab as a template for the GPU parts of your implementation.

Time the performance of your implementation and see how much faster you can get it running.   How did this compare to what you estimated in Part 1. 

Hints:
+ Use gid/block striding to determine which numbers each thread is responsible for. 
+ Use one device global memory location to "reduce" the results of the different threads.  Start it off with the value 1 meaning "prime" and then any thread can set it to 0 if the thread finds a factor.  This simple solution works without creating a race,  however,  in such situations great care must be taken as often one would require atomics or synchronisation between threads.  

## Extra Time?

Use "nvvp" to profile your application and understand what is slowing it down.

Reflect back on Part 1 and work out the approximate number of clock cycles for the calculations within the loop.  


