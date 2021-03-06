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
#include <stdlib.h>

// 0-1 Knapsack problem - Eric McCreath 2019

// see https://en.wikipedia.org/wiki/Knapsack_problem
// for problem and solution I used in this code.

int n = 2500;   // number of items to select from
int *w;         // this is weight of the items (these are positive)
int *v;         // value gained by placing the item into the knapsack
int W = 100000; // the maximum weight the knapsack can take

int maxvalue = 1000; // the maximum value any item can be worth

// Objective - find a set of items to placing into the knapsack that maximizes the value while keeping the total weight less than or
// equal to W.

/* A dynamic programming approach can be used to solve the problem in psueo-polynomial time.  This works by using a 2D array, called "m",  where m[k][j] is the maximum value that can be stored in the knapsack keeping the weight less than or equal to "j" and using a subset of items which have index less than "k".  */

#define max(A, B) ((A) > (B) ? (A) : (B))

#define m(K, J) (m_array[(K) + (J) * (n + 1)])

int main() {
  int *m_array;
  int j, k;

  // obtain the memory for the 2D array
  m_array = malloc(sizeof(int) * (W + 1) * (n + 1));
  if (!m_array) {
    printf("malloc error");
    exit(1);
  }

  // set up and initialize the weights with values
  srand(0);
  w = malloc(sizeof(int) * n);
  v = malloc(sizeof(int) * n);
  for (k = 0; k < n; k++) {
    w[k] = (rand() % (W - 1)) + 1;
    v[k] = (rand() % (maxvalue - 1)) + 1;
  }

  // initailize the first column to 0
  for (j = 0; j <= W; j++)
    m(0, j) = 0;

  for (k = 0; k < n; k++) {
    for (j = 0; j <= W; j++) { // the next column is set based on the previous
      if (w[k] > j) {
        // If the items weight is greater than the total weight we are considering it
        // can not be added, so use the best value from the previous column for this weight
        m((k + 1), j) = m(k, j);
      } else {
        // If item "k" not add then best value us just taken
        // from the previous column (same row).
        // If item "k" is added then best value is the sum of the items value and the
        // best value from the previous column and row offset by the items weight.
        // The maximum of these two options is taken.
        m(k + 1, j) = max(m(k, j), v[k] + m(k, j - w[k]));
      }
    }
  }

  // Work out a sub set that gives the maximum value.
  int cw;
  cw = W;
  printf("best value is %d using : ", m(n, cw));
  for (k = n - 1; k >= 0; k--) {
    if (m(k + 1, cw) > m(k, cw)) {
      if (W != cw)
        printf(",");
      printf("%d", k);
      cw -= w[k];
    }
  }
  printf("\n");
}
