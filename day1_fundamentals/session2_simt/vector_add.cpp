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

#include <cstdio>
#include "benchmark.h"

/**
 * Compute the element-wise sum of two vectors:
 * c = a + b
 */
void vectorAdd(double *a, double *b, double *c, int N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

int main(void) {
  const size_t N = 2 << 24;

  printf("Adding vectors of length %zu...\n", N);

  double *x = new double[N];
  double *y = new double[N];
  double *z = new double[N];
  for (size_t i = 0; i < N; i++) {
    x[i] = i;
    y[i] = i;
  }

  measureFunctionTime([&] { vectorAdd(x, y, z, N); }, "vector addition");

  for (size_t i=0; i<N; i++) {
    checkWithinTolerance(2.0 * i, z[i]);
  }
}
