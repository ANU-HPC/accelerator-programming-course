#include <cstdio>
#include "benchmark.h"

/**
 * Compute the element-wise sum of two vectors:
 * c = a + b
 */
void vectorAdd(double *a, double *b, double *c, int N) {
#pragma omp parallel for
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
