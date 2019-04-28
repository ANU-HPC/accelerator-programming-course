#include <stdio.h>
#include <cuda.h>
 

// HelloworldCUDAv2 - uses the unified memory approach rather than explicitly copying the memory between 
//                    the device and the host. 

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
 

__global__ void hello(char *res) {
    char cstr[] = "Hello World!";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 13) res[idx] = cstr[idx];
}

int main(void) {
    char *str;
    const int size = 13;
    const int blocks = 1;
    Err(cudaMallocManaged( &str, size));  
 
    hello<<<blocks,size>>>(str);
 
    Err(cudaDeviceSynchronize());
    
    printf("Result : %s\n", str); 
    Err(cudaFree(str));
}

