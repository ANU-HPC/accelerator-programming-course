#include <stdio.h>

// listPrimes - shows the prime numbers between a fixed range.
// Eric McCreath 2019 - GPL

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
    if (candidateResult * candidateResult <= n >> shift) {
      result = candidateResult;
    }
    shift = shift - 2;
  }
  return result;
}

__device__ int prime[1];

__global__ void checkFactors(long v, long srt) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long totalThreads = blockDim.x * gridDim.x;
  int check = 0;
  for (long i = idx + 2; i <= srt; i += totalThreads) {
    if (v % i == 0) {
      prime[0] = 0;
    }
    if (prime[0] == 0)
      break;
  }
}

// isPrime - just a simple loop to check if the number is divisible by
//            any number from 2 to the square root of the number
int isPrime(long v) {
  long srt = intsquroot(v);
  int h_prime;
  h_prime = 1;
  cudaMemcpyToSymbol(prime, &h_prime, sizeof(int), 0);
  checkFactors<<<64, 128>>>(v, srt);
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&h_prime, prime, sizeof(int), 0);
  return h_prime;
}

// listPrimes - check each number in the range
void listPrimes(long start, long end) {
  for (long num = start; num < end; num++) {
    printf("%ld : %s\n", num, (isPrime(num) ? "yes" : "no"));
  }
}

/*
 * The only prime in the range below is 9223372036854775783
 * Noting on my Intel i7-4790K it takes 23s to run.
 */
int main() {
  long largestlong = 0x7FFFFFFFFFFFFFFFL;
  listPrimes(largestlong - 100L, largestlong);
  return 0;
}
