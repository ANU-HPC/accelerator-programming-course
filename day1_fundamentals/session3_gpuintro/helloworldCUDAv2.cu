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

#include <cuda.h>
#include <stdio.h>

// HelloworldCUDAv2 - uses the unified memory approach rather than explicitly
// copying the memory between the device and the host.

// this macro checks for errors in cuda calls
#define Err(ans) \
  { gpucheck((ans), __FILE__, __LINE__); }
inline void gpucheck(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Err: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    exit(code);
  }
}

__global__ void hello(char *res) {
  char cstr[] = "Hello World!";
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 13)
    res[idx] = cstr[idx];
}

int main(void) {
  char *str;
  const int size = 13;
  const int blocks = 1;
  Err(cudaMallocManaged(&str, size));

  hello<<<blocks, size>>>(str);

  Err(cudaDeviceSynchronize());

  printf("Result : %s\n", str);
  Err(cudaFree(str));
}

