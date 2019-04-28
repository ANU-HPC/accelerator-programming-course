#include <stdio.h>
#include <cuda.h>
 
// this macro checks for errors in cuda calls
#define Err(ans) { gpucheck((ans), __FILE__, __LINE__); }
inline void gpucheck(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU Err: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}
 
 
 
__global__ void hello(char *res, int size) {
    char str[] = "Hello World!";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) res[idx] = str[idx];
}

int main(void) {
    char *str_h, *str_d;
    const int size = 13;
    Err(cudaMallocHost(&str_h,size));  // note we could just use a normal malloc although this gives us pinned memory 
    Err(cudaMalloc(&str_d, size));  
 
    hello<<<1,13>>>(str_d, size);
    Err(cudaMemcpy(str_h, str_d, size, cudaMemcpyDeviceToHost));

    printf("Result : %s\n", str_h);
    
    cudaFree(str_h); 
    cudaFree(str_d);
}

