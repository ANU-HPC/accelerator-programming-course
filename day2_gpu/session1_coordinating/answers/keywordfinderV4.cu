/*
 * keywordfinder.c - finds key words in a large file
 *
 *  Created on: 19 Feb. 2019
 *      Author: Eric McCreath
 */

#include<stdio.h>
#include<stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/time.h>
#include<cuda.h>

// this macro checks for errors in cuda calls
#define Err(ans) { gpucheck((ans), __FILE__, __LINE__); }
inline void gpucheck(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU Err: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		exit(code);
	}
}

struct timeval tv1, tv2;

void timestart() {
	gettimeofday(&tv1, NULL);
}

void timestop() {
	gettimeofday(&tv2, NULL);
}

void timereport() {
	printf("%f",
			(double) (tv2.tv_usec - tv1.tv_usec) / 1000000
					+ (double) (tv2.tv_sec - tv1.tv_sec));
}

void err(const char *str) {
	printf("error : %s\n", str);
	exit(1);
}

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

__device__ int check(char *sdata, int datasize, int pos, int tidx, char *word) {
	int i = 0;
	while (word[i] != 0 && sdata[tidx + i] == word[i] && pos + i < datasize) {
		i++;
	}
	if (word[i] == 0)
		return 1;
	return 0;
}

#define MAXKEYWORDSIZE 1024
__constant__ char keywords[MAXKEYWORDSIZE];
#define MAXKEYWORDS 100
__constant__ int wordsindex[MAXKEYWORDS];

#define MAXKEYLEN 20

__global__ void findkeywords(char *data, int datasize, int numwords,
		struct find *finds, int maxfinds, unsigned int *findcount) {
	int i;
	char *word;
	int resultspot;

	extern __shared__ char datashared[];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	datashared[threadIdx.x] = data[pos];
	if (threadIdx.x < MAXKEYLEN && pos+blockDim.x < datasize) datashared[threadIdx.x + blockDim.x] = data[pos+blockDim.x];
		__syncthreads();

//	printf("pos : %d", pos);
		for (i = 0; i < numwords; i++) {
			word = keywords + wordsindex[i];
			if (check(datashared, datasize, pos, threadIdx.x, word)) {

				resultspot = atomicInc(findcount, datasize);
				if (resultspot < maxfinds) {
					finds[resultspot].pos = pos;
					finds[resultspot].word = i;
				}

			}
		}
		//	printf("fc : %d\n", *findcount);

//	printf("efc : %d\n", *findcount);
}

int main(int argc, char *argv[]) {

	if (argc != 3)
		err("usage: keywordfinder textfile keywords");

	// load the text files into memory
	char *data_h, *data_d;
	int datasize;
	datasize = loadfile(argv[1], &data_h);
	Err(cudaMalloc(&data_d, datasize));

	char *keywords_h;
	int keywordssize;
	keywordssize = loadfile(argv[2], &keywords_h);
//Err(cudaMalloc(&keywords_d, keywordssize));

// obtain an index into the keywords.  So "wordsindex[i]" is
// the position within "keywords" that keyword "i"  starts.
	int numwords = count('\n', keywords_h, keywordssize);
	int *wordsindex_h;
	Err(cudaMallocHost(&wordsindex_h, sizeof(int) * numwords));
	//Err(cudaMalloc(&wordsindex_d, sizeof(int) * numwords));
	if (numwords > MAXKEYWORDS || keywordssize > MAXKEYWORDSIZE)
		err("problem too many keywords for constant memory");

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

	// display the key words
	//for (int j=0;j<numwords;j++)
	//printf("word : %d %s\n", wordsindex[j], &keywords[wordsindex[j]]);

	// set aside some memory for the finds (we fix a maximum number of finds)
	// A "struct find" is used to store a find,  basically just a mapping between the key word index and the position.
	int maxfinds = 2000;
	struct find *finds_h, *finds_d;
	Err(cudaMallocHost(&finds_h, maxfinds * sizeof(struct find)));
	Err(cudaMalloc(&finds_d, maxfinds * sizeof(struct find)));

	// find the keywords
	timestart();

	Err(cudaMemcpy(data_d, data_h, datasize, cudaMemcpyHostToDevice));
	Err(cudaMemcpyToSymbol(keywords, keywords_h, keywordssize));
	Err(cudaMemcpyToSymbol(wordsindex, wordsindex_h, sizeof(int) * numwords));

	unsigned int *findcount_d;
	unsigned int findcount = 0;
	Err(cudaMalloc(&findcount_d, sizeof(unsigned int)));
	Err(
			cudaMemcpy(findcount_d, &findcount, sizeof(unsigned int),
					cudaMemcpyHostToDevice));
int t = 256;
	findkeywords<<<(datasize-1)/t +1, t, t+MAXKEYLEN>>>(data_d, datasize, numwords, finds_d, maxfinds,
			findcount_d);

	Err(
			cudaMemcpy(&findcount, findcount_d, sizeof(unsigned int),
					cudaMemcpyDeviceToHost));
	Err(
			cudaMemcpy(finds_h, finds_d,
					sizeof(struct find) * min(findcount, maxfinds),
					cudaMemcpyDeviceToHost));

	timestop();

	// display the result
	for (int k = 0; k < min(findcount, maxfinds); k++) {
		printf("%s : %d\n", &keywords_h[wordsindex_h[finds_h[k].word]],
				finds_h[k].pos);
	}

	//  printf("time : ");
	//  timereport();
	//  printf("(s)\n");

	return 0;
}
