#include<stdio.h>
#include<stdlib.h>


// see https://en.wikipedia.org/wiki/Knapsack_problem
// for problem and solution used below


// 0-1 Knapsack problem - Eric McCreath 2019

int n = 4;             // number of items to select from
int w[] = {2,1,4,4};   // this is weight of the items to place in the (these are non-negative) 
int v[] = {4,2,5,3};   // value gained by placing the item into the knapsack
int W = 5;             // the maximum weight the knapsack can take

// Objective - find a set of items to placing into the knapsack that maximizes the value while keeping the total weight less than or
// equal to W.

/* A dynamic programming approach can be used to solve the problem in psueo-polynomial time.  This works by using a 2D array, called "m",  where m[k][j] is the maximum value that can be stored in the knapsack keeping the weight less the weight less than or equal to "j" and using a subset of items which have index less than "k".   */

#define max(A,B) ((A)>(B)?(A):(B))

#define m(K,J) (m_array[(K)*(W+1) + J])

int main() {

  int *m_array;
  m_array = malloc(sizeof(int) * (W+1) * (n+1));

  int j,k;
  // initailize the first column to 0
  for (j = 0; j <= W; j++) m(0,j) = 0;

 
  for (k = 0; k < n; k++) {
     for (j = 0; j <= W; j++) {  // the next column is set based on the previous
        if (w[k] > j) {
            // If the items weight is greater than the total weight we are considering it
            // can not be added, so use the best value from the previous column for this weight
            m((k+1),j) = m(k,j);   
        } else {
            // If item "k" not add then best value us just taken 
            // from the previous column (same row). 
            // If item "k" is added then best value is the sum of the items value and the 
            // best value from the previous column and row offset by the items weight.
            // The maximum of these two options is taken. 
            m(k+1,j) = max (m(k,j), v[k] + m(k, j-w[k]));
        }
     }
  }

  // Show the table
  printf("Weight :     ");
  for (k = 0; k < n;k++) {
    printf(" %2d ", k);
  } 
  printf("\n");
  printf("-------:------");
  for (k = 0; k < n;k++) {
    printf("----");
  } 
  printf("\n");
  for (j = 0; j <= W; j++) {
     printf(" %2d    : ",j);
     for (k = 0; k <= n;k++) {
       printf(" %2d ", m(k,j));
     }
     printf("\n");
  }

  // Work out a sub set that gives the maximum value. 
  int cw;
  cw = W;
  printf("best value is %d using : ", m(n,cw));
  for (k = n-1; k>= 0; k--) {
      if (m(k+1,cw) > m(k,cw)) {
         if (W != cw) printf(","); 
         printf ("%d",k);
         cw -= w[k];
      } 
  }   
  printf("\n");
}
