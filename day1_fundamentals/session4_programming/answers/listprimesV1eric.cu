
// listPrimes - shows the prime numbers between a fixed range.
//      this is a CUDA version that uses 1 thread in 1 block just using 
//      a simple serial approach
// Eric McCreath 2019 - GPL 

// based on https://en.wikipedia.org/wiki/Integer_square_root
// assumes a positive number


#include<stdio.h>
#include<cuda.h>

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

// based on https://en.wikipedia.org/wiki/Integer_square_root
// assumes a positive number
long intsquroot(long n) {
	long shift = 2;
	long nShifted = n >> shift;

	while (nShifted != 0 && nShifted != n) {
		shift += 2;
		nShifted = n >> shift;
	}
	shift -= 2;
	long result = 0;
    while (shift >= 0) {
	    result = result << 1;
	    long candidateResult = result + 1;
	    if (candidateResult*candidateResult <= n >> shift) {
	        result = candidateResult;
	    }
	    shift = shift - 2;
	}
	return result;
}

__global__  void divides(long v, long sqrt, int *isprime) {
   long i;
   for (i = 2 + blockIdx.x * blockDim.x + threadIdx.x;
        (i <= sqrt) && (*isprime == 1);
         i +=  blockDim.x* gridDim.x) {
         if (v % i == 0) *isprime = 0;    
   }
}

int isPrime(long v) {

	long srt = intsquroot(v);
	int isprime = 1;
	int *isprime_d;
	Err(cudaMalloc( &isprime_d, sizeof(int)));
	
	Err(cudaMemcpy(isprime_d, &isprime, sizeof(int), cudaMemcpyHostToDevice));

        divides<<<100000,1024>>>(v,srt,isprime_d);

	Err(cudaMemcpy(&isprime,isprime_d,sizeof(int),cudaMemcpyDeviceToHost));
	
	Err(cudaFree(isprime_d));

	return isprime;
}

void listPrimes(long start, long end) {
	    for (long num = start; num < end ; num++) {
		    printf ("%ld : %s\n", num, (isPrime(num)? "yes" : "no"));
	    }
}

int main() {
    long largestlong = 0x7FFFFFFFFFFFFFFFL;
	listPrimes(largestlong-100L,largestlong);
    return 0;
}
