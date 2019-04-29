#include <cstdio>
#include "benchmark.h"

/**
 * Compute the dot product of two vectors:
 * d = sum(a[i] * b[i])
 */
double dotProduct(double *a, double *b, int N) {
  double dot = 0.0;
  for (size_t i = 0; i < N; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

int main(void) {
  const size_t N = 2 << 24;

  printf("Computing dot product for vectors of length %zu\n", N);

  double *x = new double[N];
  double *y = new double[N];
  for (size_t i = 0; i < N; i++) {
    x[i] = i;
    y[i] = i;
  }

  double dot;

  measureFunctionTime([&] { dot = dotProduct(x, y, N); }, "dot product");

  double expected = (N - 1.0) * N * (2.0 * N - 1.0) / 6.0;
  checkWithinTolerance(expected, dot);
}
