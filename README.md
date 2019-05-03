Accelerator Programming for High Performance Computing: Lab Exercises
---------------------------------------------------------------------------------

# Acknowledgement

Portions of this course are taken from the 'Advanced HandsOnOpenCL' course developed by the High Performance Computing Group at the University of Bristol. Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.

# Installation

All the lab exercises for this course will be run within a Docker container, for easier management of software dependencies.

Run the install script in this directory:

    ./install.sh

to install the necessary Ubuntu packages required for CUDA development within Docker:

* NVidia driver for Quadro P2000 GPU (418)
* Docker CE -- available [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* nvidia-docker2, install instructions found [here](https://github.com/NVIDIA/nvidia-docker)

# Build

To generate a docker image named workspace, run:

    docker build -t workspace .

# Run

To test the docker image:

    docker run --runtime=nvidia -it --mount src=`pwd`,target=/workspace,type=bind  workspace /opencl_device_query/opencl_device_query


To start the docker image run:

    docker run --runtime=nvidia -it --mount src=`pwd`,target=/workspace,type=bind workspace

