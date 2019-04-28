/*
 * keywordfinder.c - finds key words in a large file
 *
 *  Created on: 19 Feb. 2019
 *      Author: Eric McCreath
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define min(A, B) ((A) < (B) ? (A) : (B))

void err(char *str) {
  printf("error : %s\n", str);
  exit(1);
}

// get the size of a file, allocate memory, and load the file into memory
int loadfile(char *str, char **data) {
  struct stat filestat;
  if (stat(str, &filestat) == -1)
    err("problem stating file");
  FILE *file;
  if ((file = fopen(str, "r")) == NULL)
    err("problem opening file");

  *data = malloc(filestat.st_size);
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
int check(char *data, int datasize, int pos, char *word) {
  int i = 0;
  while (word[i] != 0 && data[pos + i] == word[i] && pos + i < datasize) {
    i++;
  }
  if (word[i] == 0)
    return 1;
  return 0;
}

// findkeywords - search for the keywords within the "data"
int findkeywords(char *data, int datasize, char *keywords, int *wordsindex,
                 int numwords, struct find *finds, int maxfinds) {
  int pos;
  int i;
  char *word;
  int findcount = 0;
  for (pos = 0; pos < datasize; pos++) {
    for (i = 0; i < numwords; i++) {
      word = keywords + wordsindex[i];
      if (check(data, datasize, pos, word)) {
        if (findcount < maxfinds) {
          finds[findcount].pos = pos;
          finds[findcount].word = i;
        }
        findcount++;
      }
    }
  }
  return findcount;
}

int main(int argc, char *argv[]) {
  if (argc != 3)
    err("usage: keywordfinder textfile keywords");

  // load the text files into memory
  char *data;
  int datasize;
  datasize = loadfile(argv[1], &data);
  char *keywords;
  int keywordssize;
  keywordssize = loadfile(argv[2], &keywords);

  // obtain an index into the keywords.  So "wordsindex[i]" is
  // the position within "keywords" that keyword "i" starts.
  int numwords = count('\n', keywords, keywordssize);
  int *wordsindex;
  wordsindex = malloc(sizeof(int) * numwords);
  int i;
  int pos = 0;
  wordsindex[pos++] = 0;
  for (i = 0; i < keywordssize; i++) {
    if (keywords[i] == '\n') {
      keywords[i] = 0;
      if (pos < numwords)
        wordsindex[pos++] = i + 1;
    }
  }

  // display the key words
  // for (int j=0;j<numwords;j++)
  //   printf("word : %d %s\n", wordsindex[j], &keywords[wordsindex[j]]);

  // set aside some memory for the finds (we fix a maximum number of finds)
  // A "struct find" is used to store a find,
  // basically just a mapping between the key word index and the position.
  int maxfinds = 2000;
  struct find *finds;
  finds = malloc(maxfinds * sizeof(struct find));

  // find the keywords

  int findcount = findkeywords(data, datasize, keywords, wordsindex, numwords,
                               finds, maxfinds);

  // display the result
  for (int k = 0; k < min(maxfinds, findcount); k++) {
    printf("%s : %d\n", &keywords[wordsindex[finds[k].word]], finds[k].pos);
  }

  return 0;
}
