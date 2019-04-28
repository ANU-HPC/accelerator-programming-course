
# Single-Instruction Multiple-Data (SIMD) Vectorization on x86

This lab will show how vectorization is achieved on x86 CPU architectures.
This approach will be generalized to all platforms in a following lab.

The [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/) will help in identifying vector instructions and the data types supported for each architecture.

We will start with some simple C++ code which generates the Mandelbrot set and explore how packing data together can achieve speedup.
This code is based on the [Intel 64 and IA-32 Architectures Optimization Reference manual](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf).

## Mandelbrot Set

The Mandelbrot-set map evaluation is useful to illustrate a situation with more complex control flows in nested loops.
The Mandelbrot-set is a set of height values mapped to a 2-D grid.
The height value is the number of Mandelbrot iterations (defined over the complex number space as I(n) = I(n-1^2) + I(0) needed to get |In| > 2.
It is common to limit the map generation by setting some maximum threshold value of the height, all other points are assigned with a height equal to the threshold. 

## Intrinsics

The file `mandelbrot.cpp` contains code to generate the Mandlebrot set.
Compile the code and run it, and note the time taken to perform the computation.

```
make
./mandelbrot
```

The output of this program is two matrices (`mandelbrot_set.csv` and `mandelbrot_set_vectorized.csv`).You'll note that the mandelbrot_set_vectorized.csv is incomplete -- it's largely empty.
Using the

```
void mandelbrot(int *map, int DIMX, int DIMY, float X_STEP, float Y_STEP)
```

function as a guide, can you make the

```
void mandelbrot_128(int *map, int DIMX, int DIMY, float X_STEP, float Y_STEP)
```

function generate the same result?

Also in this directory is two Python programs: `convert_mandelbrot_csv2png.py` can be used to convert the matrix files into images (`mandelbrot_set.png` and `mandelbrot_set_vectorized.png`), which can be used to visually compare your solution; and `plot_times.py` is used to generate `runtimes.png`, this uses the timing data stored from your run in `lsb.mandelbrot.r0` -- **Note** you will want to remove this file between runs to generated a new runtime figure.

LibSciBench is used as a high resolution timing library to measure execution times and store the results in a usable way.
It allows us to examine the granularity of codes by collecting execution time information of multiple regions at once.
We can also use it to measure energy usage and hardware events -- ask us if you'd like to know more.
We just present the runtimes of 100 runs of each of the mandelbrot functions, but do take a look at `lsb.mandelbrot.r0` to see the other regions we time -- this may be useful in the next session where you can look at the overhead of OpenCL host-side setup.

Generate and view `runtimes.png`, is the speed-up of our vectorized intrinsics as good as you would expect?

## Extra Time?

* Can you identify the lines of code that hinder performance? **hint** bit-packing is only usually required because of availability of data-types for certain intrinsics
* Can you figure out how to do the same thing for longer vector widths?
* How easy is this process?
* Fancy achieving the same with (ARM Neon Intrinsics)[https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics]?

