//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multiplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "device_picker.hpp"
#include "matrix_lib.hpp"
#include <util.hpp>

int main(int argc, char *argv[]) {

  int N;    // A[N][N], B[N][N], C[N][N]
  int size; // Number of elements in each matrix

  double start_time; // Starting time
  double run_time;   // Timing
  util::Timer timer; // Timing

  N = ORDER;
  size = N * N;

  std::vector<float> h_A(size); // Host memory for Matrix A
  std::vector<float> h_B(size); // Host memory for Matrix B
  std::vector<float> h_C(size); // Host memory for Matrix C

  cl::Buffer d_a, d_b, d_c; // Matrices in device memory

  //--------------------------------------------------------------------------------
  // Create a context and queue
  //--------------------------------------------------------------------------------

  try {

    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex);

    // Get list of devices
    std::vector<cl::Device> devices;
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices) {
      std::cout << "Invalid device index (try '--list')\n";
      return EXIT_FAILURE;
    }

    cl::Device device = devices[deviceIndex];

    std::string name = getDeviceName(device);
    std::cout << "\nUsing OpenCL device: " << name << "\n";

    std::vector<cl::Device> chosen_device;
    chosen_device.push_back(device);
    cl::Context context(chosen_device);
    cl::CommandQueue queue(context, device);

    //--------------------------------------------------------------------------------
    // Run sequential matmul
    //--------------------------------------------------------------------------------

    initmat(N, h_A, h_B, h_C);

    timer.reset();

    printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n", N);
    for (int i = 0; i < COUNT; i++) {
      zero_mat(N, h_C);

      start_time = static_cast<double>(timer.getTimeNanoseconds()) / 1000000000.0;

      seq_mat_mul_sdot(N, h_A, h_B, h_C);

      run_time = (static_cast<double>(timer.getTimeNanoseconds()) / 1000000000.0) - start_time;
      results(N, h_C, run_time);
    }

    //--------------------------------------------------------------------------------
    // Setup the buffers, initialize matrices, and write them into global memory
    //--------------------------------------------------------------------------------

    //  Reset A, B and C matrices (just to play it safe)
    initmat(N, h_A, h_B, h_C);

    d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);

    d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);

    d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    //--------------------------------------------------------------------------------
    // OpenCL matrix multiplication ... Naive
    //--------------------------------------------------------------------------------

    timer.reset();

    // Create the compute program from the source buffer
    cl::Program program(context, util::loadProgram("kernel.cl"), true);

    // Create the compute kernel from the program
    cl::KernelFunctor<int, cl::Buffer, cl::Buffer, cl::Buffer> naive_mmul(program, "mmul");

    printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n", N);

    // Do the multiplication COUNT times
    for (int i = 0; i < COUNT; i++) {
      zero_mat(N, h_C);

      start_time = static_cast<double>(timer.getTimeNanoseconds()) / 1000000000.0;

      // Execute the kernel over the entire range of C matrix elements ... computing
      // a dot product for each element of the product matrix.  The local work
      // group size is set to NULL ... so I'm telling the OpenCL runtime to
      // figure out a local work group size for me.
      cl::NDRange global(N, N);
      naive_mmul(cl::EnqueueArgs(queue, global),
                 N, d_a, d_b, d_c);

      queue.finish();

      run_time = (static_cast<double>(timer.getTimeNanoseconds()) / 1000000000.0) - start_time;

      cl::copy(queue, d_c, h_C.begin(), h_C.end());

      results(N, h_C, run_time);

    } // end for loop

  } catch (cl::BuildError error) {
    std::string log = error.getBuildLog()[0].second;
    std::cerr << std::endl
              << "Build failed:" << std::endl
              << log << std::endl;
  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: "
              << err.what()
              << "("
              << err_code(err.err())
              << ")"
              << std::endl;
  }

#if defined(_WIN32) && !defined(__MINGW32__)
  system("pause");
#endif

  return EXIT_SUCCESS;
}
