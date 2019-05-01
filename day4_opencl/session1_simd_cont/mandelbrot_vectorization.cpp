/*
 * Copyright 2019 Australian National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either or express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <liblsb.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/opencl.h>
#endif

const float EPSILON = 0.00001f;

inline void except(bool condition, const std::string &error_message = "") {
  if (!condition)
    throw std::runtime_error(error_message);
}

inline void print_payload_as_integer(int *x, unsigned int size) {
  for (int i = 0; i < size; i++) {
    std::cout << x[i] << ' ';
  }
  std::cout << std::endl;
}

inline void write_matrix_to_file(int *mat, unsigned int x, unsigned int y, std::string filename) {
  std::ofstream myfile;
  myfile.open(filename);
  for (int i = 0; i < y; i++) {       // rows
    for (int j = 0; j < x - 1; j++) { // cols
      myfile << mat[i * x + j] << ',';
    }
    myfile << mat[i * x + x] << '\n';
  }
  myfile.close();
}

inline void zero_payload_as_integer(int *x, unsigned int size) {
  for (int i = 0; i < size; i++) {
    x[i] = 0;
  }
}

int main(int argc, char **argv) {
  except(argc == 3, "./mandelbrot <platform id> <device id>");
  const char *synthetic_kernel_path = "./mandelbrot_vectorization_kernel.cl";
  int platform_id = atoi(argv[1]);
  int device_id = atoi(argv[2]);

  LSB_Init("mandelbrot", 0);
  LSB_Set_Rparam_string("kernel", "none_yet");

  LSB_Set_Rparam_string("region", "host_side_setup");
  LSB_Res();
  // read synthetic kernel file
  std::ifstream sk_handle(synthetic_kernel_path, std::ios::in);
  except(sk_handle.is_open(), "synthetic kernel doesn\'t exist");
  std::cout << "Attempting kernel: " << synthetic_kernel_path << " with contents:\n"
            << sk_handle.rdbuf() << std::endl;
  std::filebuf *sk_buf = sk_handle.rdbuf();
  int sk_size = sk_buf->pubseekoff(0, sk_handle.end, sk_handle.in);
  sk_buf->pubseekpos(0, sk_handle.in);
  char *sk_source = new char[sk_size];
  sk_buf->sgetn(sk_source, sk_size);

  //set-up open compute language
  int sbd_err;
  cl_uint num_platforms = 0;
  cl_uint num_devices = 0;

  cl_device_id *sbd_devices;
  cl_platform_id *sbd_platforms;

  cl_context sbd_context;
  cl_command_queue sbd_queue;

  sbd_err = clGetPlatformIDs(0, NULL, &num_platforms);
  except(sbd_err == CL_SUCCESS, "can't get platform counts");
  sbd_platforms = new cl_platform_id[num_platforms];
  sbd_err = clGetPlatformIDs(num_platforms, sbd_platforms, NULL);
  except(sbd_err == CL_SUCCESS, "can't get platform info");
  except(num_platforms, "no OpenCL platforms found");
  except(platform_id >= 0 && platform_id < num_platforms, "invalid platform selection");

  sbd_err = clGetDeviceIDs(sbd_platforms[platform_id], CL_DEVICE_TYPE_ALL, 0, 0, &num_devices);
  except(sbd_err == CL_SUCCESS, "can't get device counts");
  except(num_devices, "no OpenCL devices found");
  sbd_devices = new cl_device_id[num_devices];
  sbd_err = clGetDeviceIDs(sbd_platforms[platform_id], CL_DEVICE_TYPE_ALL, num_devices, sbd_devices, NULL);
  except(sbd_err == CL_SUCCESS, "can't get device info");
  except(device_id >= 0 && device_id < num_devices, "invalid device selection");

  sbd_context = clCreateContext(NULL, 1, &sbd_devices[device_id], NULL, NULL, &sbd_err);
  except(sbd_err == CL_SUCCESS, "can't create context");
  sbd_queue = clCreateCommandQueue(sbd_context, sbd_devices[device_id], 0, &sbd_err);
  except(sbd_err == CL_SUCCESS, "can't create command queue");

  //set-up memory for payload/problem size
  size_t KiB = 7900;

  unsigned int c_bytes = (KiB * 1024);
  cl_int c_elements = static_cast<cl_int>(c_bytes / sizeof(cl_int));
  //MxN matrix (but actually square matrix)
  int w = 32;
  int M = floor(sqrt(c_elements));
  M = floor(M / w) * w; //but rounded down so it's a multiple of 32 -- 32x32 divisible blocks

  unsigned int map_bytes = M * M * sizeof(cl_int);
  w = 1;

  std::cout << "M = " << M << " total KiB = " << map_bytes / 1024 << std::endl;

  LSB_Rec(0);

  std::cout << "Operating on a " << M << "x" << M << " map with a tile size " << w << "..." << std::endl;

  LSB_Set_Rparam_string("region", "kernel_creation");
  LSB_Res();
  //compile kernels
  std::string compiler_flags = "-DDIMX=" + std::to_string(M) + " -DDIMY=" + std::to_string(M) + " -DX_STEP=(0.5f/DIMX) -DY_STEP=(0.4f/(DIMY/2))  ";
  cl_program sbd_program = clCreateProgramWithSource(sbd_context, 1, (const char **)&sk_source, NULL, &sbd_err);
  except(sbd_err == CL_SUCCESS, "can't build kernel");
  sbd_err = clBuildProgram(sbd_program, 1, &sbd_devices[device_id], compiler_flags.c_str(), NULL, NULL);
  if (sbd_err != CL_SUCCESS) { //print error during kernel compilation
    size_t log_size;
    clGetProgramBuildInfo(sbd_program, sbd_devices[device_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *build_log = new char[log_size];
    clGetProgramBuildInfo(sbd_program, sbd_devices[device_id], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << build_log << std::endl;
    delete[] build_log;
  }
  except(sbd_err == CL_SUCCESS, "can't build program");
  cl_kernel mandelbrot_kernel = clCreateKernel(sbd_program, "mandelbrot", &sbd_err);
  except(sbd_err == CL_SUCCESS, "can't create kernel");
  cl_kernel mandelbrot_vectorized_kernel = clCreateKernel(sbd_program, "mandelbrot_vectorized", &sbd_err);
  except(sbd_err == CL_SUCCESS, "can't create kernel");
  LSB_Rec(0);

  //memory setup
  LSB_Set_Rparam_string("region", "device_side_buffer_setup");
  LSB_Res();
  cl_mem sbd_map = clCreateBuffer(sbd_context, CL_MEM_READ_WRITE, map_bytes, NULL, &sbd_err);
  except(sbd_err == CL_SUCCESS, "can't create device memory map");

  int *map = new int[M * M];
  LSB_Rec(0);

  int sample_size = 100;

  //mandelbrot case
  LSB_Set_Rparam_string("kernel", "mandelbrot");
  for (int i = 0; i < sample_size; i++) {
    LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
    zero_payload_as_integer(map, M * M);
    LSB_Rec(i);

    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    sbd_err = clEnqueueWriteBuffer(sbd_queue, sbd_map, CL_TRUE, 0, map_bytes, map, 0, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "can't write to device memory!");
    LSB_Rec(i);

    //run the kernel
    //size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
    //size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)};

    // just one thread
    size_t global_work[2] = {static_cast<size_t>(1), static_cast<size_t>(1)};
    size_t local_work[2] = {static_cast<size_t>(1), static_cast<size_t>(1)};

    LSB_Set_Rparam_string("region", "mandelbrot_kernel");
    LSB_Res();
    sbd_err = clSetKernelArg(mandelbrot_kernel, 0, sizeof(cl_mem), &sbd_map);
    except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

    sbd_err = clEnqueueNDRangeKernel(sbd_queue, mandelbrot_kernel, 1, NULL, global_work, local_work, 0, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "failed to execute kernel (with error " + std::to_string(sbd_err) + ")");
    clFinish(sbd_queue);

    LSB_Rec(i);

    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
    sbd_err = clEnqueueReadBuffer(sbd_queue, sbd_map, CL_TRUE, 0, map_bytes, map, 0, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "can't read from device memory");
    LSB_Rec(i);
  }
  std::cout << "Mandelbrot:" << std::endl;
  write_matrix_to_file(map, M, M, "mandelbrot_set.csv");
  //print_payload_as_integer(map,M*M);
  //mandelbrot vectorized case
  LSB_Set_Rparam_string("kernel", "mandelbrot_vectorized");
  for (int i = 0; i < sample_size; i++) {
    LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
    zero_payload_as_integer(map, M * M);
    LSB_Rec(i);

    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    sbd_err = clEnqueueWriteBuffer(sbd_queue, sbd_map, CL_TRUE, 0, map_bytes, map, 0, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "can't write to device memory!");
    LSB_Rec(i);

    //run the kernel
    // just one thread
    size_t global_work[2] = {static_cast<size_t>(1), static_cast<size_t>(1)};
    size_t local_work[2] = {static_cast<size_t>(1), static_cast<size_t>(1)};
    //size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
    //size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)};

    LSB_Set_Rparam_string("region", "mandelbrot_vectorized_kernel");
    LSB_Res();
    sbd_err = clSetKernelArg(mandelbrot_vectorized_kernel, 0, sizeof(cl_mem), &sbd_map);
    except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

    sbd_err = clEnqueueNDRangeKernel(sbd_queue, mandelbrot_vectorized_kernel, 1, NULL, global_work, local_work, 0, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "failed to execute kernel (with error " + std::to_string(sbd_err) + ")");
    clFinish(sbd_queue);

    LSB_Rec(i);

    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
    sbd_err = clEnqueueReadBuffer(sbd_queue, sbd_map, CL_TRUE, 0, map_bytes, map, 0, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "can't read from device memory");
    LSB_Rec(i);
  }
  std::cout << "\nMandelbrot vectorized:" << std::endl;
  write_matrix_to_file(map, M, M, "mandelbrot_set_vectorized.csv");
  //print_payload_as_integer(map,M*M);
  delete map;

  clReleaseMemObject(sbd_map);
  clReleaseKernel(mandelbrot_kernel);
  clReleaseKernel(mandelbrot_vectorized_kernel);
  clReleaseProgram(sbd_program);
  clReleaseCommandQueue(sbd_queue);
  clReleaseContext(sbd_context);

  delete sk_source;
  delete sbd_devices;
  delete sbd_platforms;
  LSB_Finalize();
}

