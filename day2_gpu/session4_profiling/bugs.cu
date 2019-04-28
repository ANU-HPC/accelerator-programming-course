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
 
// add one to an array of n integers - indexed from 0
__global__ void addone(int *data, int n) {
    int idx = blockIdx.x;
    if (idx <= n) data[idx] = data[idx] + 1;
}

// add 2 to elements of the array and then double it 
__global__ void doublesubten(int *data, int n) {
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
          data[idx] = 2*data[idx]; 
          __syncthreads();
    }
    if (idx < n) {
        data[idx] = data[idx] - 10;
    }
}

// set to zero all the elements that are over 100
__global__ void  zerooutoverhundred(int *data, int n) {
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
         int val = data[idx];
         int count = 0;
         while (val != 0) {
             val--;
             count++;
         } 
         if (count > 100) data[idx] = 0;
    }
}

// sum all the elements in the array form 0 to i-1  and place the result in data[i].
__global__ void prefixsum(int *data, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
            int sum = 0;
            for (int i = 0;i<idx;i++) sum += data[i];
            data[idx] = sum;
    }
}

int main(void) {
    int *data_h, *data_d;
    const int size = 100;
    cudaMallocHost(&data_h,size); 
    for (int i =0;i<size;i++) data_h[i] = i; 
    cudaMemcpy(data_d, data_h, size, cudaMemcpyDeviceToHost); // copy the array to the GPU

    addone<<<1,100>>>(data_d,size);
    doublesubten<<<1,100>>>(data_d,size);
    zerooutoverhundred<<<1,100>>>(data_d,size);
    prefixsum<<<1,100>>>(data_d,size);
    
    cudaMemcpy(data_h, data_d, size, cudaMemcpyDeviceToHost);

    for (int i = 0;i<size;i++) printf("%d, ", data_d);
    
    cudaFree(data_h); 
    cudaFree(data_d);
}

