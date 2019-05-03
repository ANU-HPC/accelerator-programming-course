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

// isPrime - just a simple loop to check if the number is divisible by
//            any number from 2 to the square root of the number
int isPrime(long v) {
  long srt = intsquroot(v);
  for (long i = 2; i <= srt; i++) {
    if (v % i == 0)
      return 0;
  }
  return 1;
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
