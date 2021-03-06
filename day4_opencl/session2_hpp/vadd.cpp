//------------------------------------------------------------------------------
//
// Name:       vadd_cpp.cpp
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <util.hpp>

#include "err_code.h"

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//------------------------------------------------------------------------------

#define TOL (0.001)       // tolerance used in floating point comparisons
#define LENGTH (2 << 24)  // length of vectors a, b, and c

int main(int argc, char* argv[]) {
  std::vector<float> h_a(LENGTH);  // a vector
  std::vector<float> h_b(LENGTH);  // b vector
  std::vector<float> h_c(LENGTH,
                         (float)0xdeadbeef);  // c = a + b, from compute device

  cl::Buffer d_a;  // device memory used for the input  a vector
  cl::Buffer d_b;  // device memory used for the input  b vector
  cl::Buffer d_c;  // device memory used for the output c vector

  // Fill vectors a and b with random float values
  int count = LENGTH;
  for (int i = 0; i < count; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  try {
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    size_t platform_id = 0;
    if (argc > 1) platform_id = atoi(argv[1]);
    cl::Platform platform = platforms[platform_id];

    cl::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
    cl::Device device = devices[0];
    std::cout << std::endl
              << "Using OpenCL device: " << device.getInfo<CL_DEVICE_NAME>()
              << std::endl;

    cl::Context context(device);

    // Load in kernel source, creating and building a program object for the
    // context
    cl::Program program(context, util::loadProgram("vadd.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functor

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> vadd(program, "vadd");

    d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);

    d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

    util::Timer timer;

    vadd(cl::EnqueueArgs(queue, cl::NDRange(count)), d_a, d_b, d_c);

    queue.finish();

    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    printf("\nThe kernels ran in %lf seconds\n", rtime);

    cl::copy(queue, d_c, h_c.begin(), h_c.end());

    // Test the results
    int correct = 0;
    float tmp;
    for (int i = 0; i < count; i++) {
      tmp = h_a[i] + h_b[i];        // expected value for d_c[i]
      tmp -= h_c[i];                // compute errors
      if (tmp * tmp < TOL * TOL) {  // correct if square deviation is less
        correct++;                  //  than tolerance squared
      } else {
        printf(" tmp %f h_a %f h_b %f  h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
      }
    }

    // summarize results
    printf("vector add to find C = A+B:  %d out of %d results were correct.\n",
           correct, count);
  } catch (cl::BuildError error) {
    std::string log = error.getBuildLog()[0].second;
    std::cerr << std::endl << "Build failed:" << std::endl << log << std::endl;
  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
              << std::endl;
  }

#if defined(_WIN32) && !defined(__MINGW32__)
  system("pause");
#endif
}
