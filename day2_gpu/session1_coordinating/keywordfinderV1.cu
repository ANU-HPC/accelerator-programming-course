/*
 * keywordfinder.c - finds key words in a large file
 *
 *  Created on: 19 Feb. 2019
 *      Author: Eric McCreath
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// this macro checks for errors in cuda calls
#define Err(ans) \
  { gpucheck((ans), __FILE__, __LINE__); }
inline void gpucheck(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Err: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

void err(const char *str) {
  printf("error : %s\n", str);
  exit(1);
}

// get the size of a file, allocate pinned memory, and load the file into memory
int loadfile(char *str, char **data) {
  struct stat filestat;
  if (stat(str, &filestat) == -1)
    err("problem stating file");
  FILE *file;
  if ((file = fopen(str, "r")) == NULL)
    err("problem opening file");

  Err(cudaMallocHost(data, filestat.st_size));
  fread(*data, filestat.st_size, 1, file);
  return filestat.st_size;
}

// count - used for counting the number of "marker"s in some text
int count(char marker, char *data, int size) {
  int i;
  int sum = 0;
  for (i = 0; i < size; i++) {
    if (data[i] == marker)
      sum++;
  }
  return sum;
}

struct find {
  int pos;
  int word;
};

// check - determine if "word" matches within "data" starting at "pos".
//         the "word" is assumed to be null terminated
__device__ int check(char *data, int datasize, int pos, char *word) {
  int i = 0;
  while (word[i] != 0 && data[pos + i] == word[i] && pos + i < datasize) {
    i++;
  }
  if (word[i] == 0)
    return 1;
  return 0;
}

// findkeywords - search for the keywords within the "data"
__global__ void findkeywords(char *data, int datasize, char *keywords,
                             int *wordsindex, int numwords,
                             struct find *finds, int maxfinds,
                             int *findcount) {
  int pos;
  int i;
  char *word;

  for (pos = 0; pos < datasize; pos++) {
    for (i = 0; i < numwords; i++) {
      word = keywords + wordsindex[i];
      if (check(data, datasize, pos, word)) {
        if (*findcount < maxfinds) {
          finds[*findcount].pos = pos;
          finds[*findcount].word = i;
        }

        (*findcount)++;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3)
    err("usage: keywordfinder textfile keywords");

  // load the text files into memory
  char *data_h, *data_d;
  int datasize;
  datasize = loadfile(argv[1], &data_h);
  Err(cudaMalloc(&data_d, datasize));

  char *keywords_h, *keywords_d;
  int keywordssize;
  keywordssize = loadfile(argv[2], &keywords_h);
  Err(cudaMalloc(&keywords_d, keywordssize));

  // obtain an index into the keywords.  So "wordsindex[i]" is
  // the position within "keywords" that keyword "i"  starts.
  int numwords = count('\n', keywords_h, keywordssize);
  int *wordsindex_h, *wordsindex_d;
  Err(cudaMallocHost(&wordsindex_h, sizeof(int) * numwords));
  Err(cudaMalloc(&wordsindex_d, sizeof(int) * numwords));

  int i;
  int pos = 0;
  wordsindex_h[pos++] = 0;
  for (i = 0; i < keywordssize; i++) {
    if (keywords_h[i] == '\n') {
      keywords_h[i] = 0;
      if (pos < numwords)
        wordsindex_h[pos++] = i + 1;
    }
  }

  // set aside some memory for the finds (we fix a maximum number of finds)
  // A "struct find" is used to store a find,
  // basically just a mapping between the key word index and the position.
  int maxfinds = 2000;
  struct find *finds_h, *finds_d;
  Err(cudaMallocHost(&finds_h, maxfinds * sizeof(struct find)));
  Err(cudaMalloc(&finds_d, maxfinds * sizeof(struct find)));

  // copy the data across to the device
  Err(cudaMemcpy(data_d, data_h, datasize, cudaMemcpyHostToDevice));
  Err(cudaMemcpy(keywords_d, keywords_h, keywordssize,
                 cudaMemcpyHostToDevice));
  Err(cudaMemcpy(wordsindex_d, wordsindex_h, sizeof(int) * numwords,
                 cudaMemcpyHostToDevice));

  int *findcount_d;
  int findcount = 0;
  Err(cudaMalloc(&findcount_d, sizeof(int)));
  Err(cudaMemcpy(findcount_d, &findcount, sizeof(int),
                 cudaMemcpyHostToDevice));

  // find the keywords
  findkeywords<<<1, 1>>>(data_d, datasize, keywords_d, wordsindex_d, numwords,
                         finds_d, maxfinds, findcount_d);

  // copy the results back
  Err(cudaMemcpy(&findcount, findcount_d, sizeof(int),
                 cudaMemcpyDeviceToHost));
  Err(cudaMemcpy(finds_h, finds_d, sizeof(struct find) * findcount,
                 cudaMemcpyDeviceToHost));

  // display the result
  for (int k = 0; k < findcount; k++) {
    printf("%s : %d\n", &keywords_h[wordsindex_h[finds_h[k].word]],
           finds_h[k].pos);
  }

  return 0;
}
