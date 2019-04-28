
# OpenMP Parallel Loops

In this lab you will explore the use of OpenMP work sharing directives to implement data parallelism.

The [OpenMP Reference Guide](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf) may be of use in remembering the syntax for OpenMP directives and functions.

## Data Parallel Loops

The file `vector_add.cpp` contains code to add corresponding elements of two large vectors and store the results in a third vector.
Compile the code and run it, and note the time taken to perform the computation.
```
make
./vector_add
```

Now parallelize the code using an OpenMP parallel loop directive.
Do you observe a speedup?

The file `dot_product.cpp` contains code to compute the dot product of two vectors, i.e. the sum of the products of each pair of corresponding elements.
Parallelize this code using an OpenMP parallel loop directive.

## Examining CPU scaling

Docker can be used to restrict the cores available to the system for our experiment.
For example, to restrict Docker to use only the first two cores, add a `cpuset` option as in the following command:
```
docker run -it --mount src=`pwd`,target=/workspace,type=bind --cpuset-cpus="0,1" workspace /bin/bash
```

## Extra Time?

Set the environment variable `OMP_DISPLAY_ENV`, run one of your OpenMP programs and view the output, e.g.
```
OMP_DISPLAY_ENV=true ./dot_product
```

You should see something like this:
```
OPENMP DISPLAY ENVIRONMENT BEGIN
  _OPENMP = '201511'
  OMP_DYNAMIC = 'FALSE'
  OMP_NESTED = 'FALSE'
  OMP_NUM_THREADS = '4'
  OMP_SCHEDULE = 'DYNAMIC'
  OMP_PROC_BIND = 'FALSE'
  OMP_PLACES = ''
  OMP_STACKSIZE = '0'
  OMP_WAIT_POLICY = 'PASSIVE'
  OMP_THREAD_LIMIT = '4294967295'
  OMP_MAX_ACTIVE_LEVELS = '2147483647'
  OMP_CANCELLATION = 'FALSE'
  OMP_DEFAULT_DEVICE = '0'
  OMP_MAX_TASK_PRIORITY = '0'
OPENMP DISPLAY ENVIRONMENT END
```
The first line shows the version of the OpenMP specification that is implemented by your compiler, e.g. '201511' is OpenMP 4.5.
Each of the remaining lines shows a value that can be set as an environment variable to affect how work is scheduled.

* What is the effect of setting `OMP_PLACES` to 'sockets', 'cores', or 'threads'?
* What is the effect of setting `OMP_PROC_BIND` to 'close', 'spread', or 'master'?
