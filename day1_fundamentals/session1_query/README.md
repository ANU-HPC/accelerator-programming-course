
# Querying OpenCL Device Properties

The purpose of this lab is to get comfortable with the environment used throughout the rest of the workshop.
Docker is our tool of choice when it comes to OS level virtualization, and we've used it to set-up a usable Linux environment with all required development tools required in our course.
We'll be using Docker in this workshop to verify that both CPU and GPU devices on your laptops are available.
In our image we install NVIDIA and Intel drivers along with a simple test program -- written in OpenCL -- to list the compute platforms on your system.

## Docker Environment

To build the Docker image run the following command on the root directory of the git repository:

    docker build -t workspace .

This will produce an image named workspace.
Next to create an instance of this image execute:

    docker run --runtime=nvidia -it --mount src=`pwd`,target=/workspace,type=bind  workspace

The `--runtime=nvidia` stipulates that an NVIDIA GPU driver should be used -- which is critical to all of the GPU specific workshops.

`-it` denotes an interactive session with a pseudo-TTY.

`--mount src=/...,target=/...,type=bind` is used to mount the top level of this directory as `/workspace`. This allows any changes you make in the mounted directory to be maintained after the Docker session is terminated.

`workspace` is the name of the image to use for the instance.

`/bin/bash` or any other command can be appended to the end of the `run` command to execute it in this new instance.

You should now be in a new bash session with the name student, you should also have root access to this virtualized OS.

### Sharing instances

If you wish to have multiple sessions into the same instance, it can be shared by determining the name of the instance and executing a new command in it.

To get the shorthand name, run:

     docker ps

Copy the desired name, for example naughty_davinci, and use it in the following:

    docker exec -it <name> /bin/bash

You can run as many sessions of a shared instance as you wish.
Sessions will be shared until the terminal pane -- which executed `docker run` -- is closed/terminated.
Once this occurs all shared sessions will also terminate.

## Accessing OpenCL Devices

There is a program located in this directory called opencl_query.c.
From within your running docker instance go to this directory `day1_fundamentals/session1_query/` and run `make`.

Now you've Built a simple OpenCL program called `opencl_device_query`, this performs as advertised -- it queries the OpenCL platforms and devices available on your system.
Run the following to run the query:

    ./opencl_device_query

The different vendors offer separate *platforms* in the OpenCL setting, *devices* for each *platform* are also listed.

## The OpenCL (heterogeneous platform) view of hardware

* Can you find how many CPU cores are available?
* How many GPU cores? *Note* these are the number of compute units -- hyper-threaded cores on Intel CPUs, or streaming multiprocessors (SMM) on NVIDIA GPUs. In the NVIDIA setting, the number of CUDA cores of this accelerator will be the number of SMMs * 128.
* Check the online specs of your devices to verify -- maybe [CPU](https://ark.intel.com/content/www/us/en/ark/products/134899/intel-core-i7-8850h-processor-9m-cache-up-to-4-30-ghz.html) or [GPU](https://www.notebookcheck.net/NVIDIA-Quadro-P2000-Max-Q-GPU-Benchmarks-and-Specs.355016.0.html)?
* Is there a significant difference between the clock frequency of these two accelerators?
* What is the size of the respective global and local memory sizes of these systems?
* What do you think the global and local refer to on CPU and GPU devices?


## Extra time?

Take a look at the source code for the `opencl_device_query` program.

Note how platforms then devices are queried, how a command queue is created, the way to create memory objects to run on different devices and how to populate them, how kernels are compiled from source code and how they're executed.
Finally, see how computed results are collected from the device back into main memory.

This is the composition of most OpenCL programs -- the boilerplate -- and will be examined in detail in day two's workshops.

