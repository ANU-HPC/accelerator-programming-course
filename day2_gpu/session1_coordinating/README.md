# Coordinating Between Threads on the GPU

In this lab you will explore ways of coordinating dependencies between multiple threads, including the writing of results.
The example program finds keyword matches in a large text file; it is basically a simple version of `grep`.
The text data need to be transferred to the GPU, the matches computed, and the results transferred back.
Once again, the serial implementation is simple and there has not been any attempt at algorithmic optimization, however, that is not the aim.
The objective is to keep the algorithm basically the same, and use the GPU to accelerate the computation.

The file `keywordfinder.c` contains the serial CPU implementation, and `keywordfinderV1.cu` contains a simple CUDA implementation that only uses a single thread.
This will save you some leg work in setting up memory on the device and transferring data to and from the device.

## Finding Keywords on the Way to the Moon

First compile both versions of the program:
```
make
```

The `keywordfinder` program accepts two parameters: the name of a text file to search, and the name of a file containing the keywords to match on, one per line.
It prints the keyword matching to standard output, in no particular order.

For example, if the file `small.txt` contains:
```
This is a short file to test finding keywords.
```
and `smallkeywords.txt` contains:
```
short
keywords
```

We expect the following output when run:
```
$ ./keywordfinder small.txt smallkeywords.txt 
short : 10
keywords : 37
```

The CPU version is implemented using a main loop that checks for each character in the text file whether a match starts at that character.
This checking is done by looping over the possible keywords, where each keyword is checked by looping over the characters of the word and checking if it matches with the corresponding character in the file. 

Have a look through the CPU version and consider the following questions:
+ What loops could you parallelize?  Which is the best one and why?  What data dependencies need addressing?
+ What is the data structure used for storing the matches? 

The file `small.txt` is, as the name suggests, small. 
To see real performance improvements we need to deal with a much larger data sets (to amortise setup costs).  
The provided file "verne.txt" contains the text of "From the Earth to the Moon" by Jules Verne.
Even that is a bit small for our purposes, so the `make` command also generates the file "verne128.txt" which is "verne.txt" concatenated 128 times.
Run the following command to see how long it takes to execute on your machine:
```
time ./keywordfinder verne128.txt keywords.txt

```

## Grid Block Striding

"keywordfinderV1.cu" holds the first step in implementing a GPU version.
The `findkeywords` function has been turned into a kernel.
As only one thread is launched, this kernel includes the main loop that does all the work.

Modify this implementation to launch many threads, each working on different parts of the text file concurrently.
Use grid block striding.
At this point, leave the `findcount` global variable as is.

+ What performance improvement did you observe?
+ How does it change with the changing of the number of blocks and blocks per thread?
+ Is the solution correct?

One way of checking for correctness is storing the output of the CPU version into a file and then "diff"ing the result of your new version with this file.
```
./keywordfinder verne128.txt keywords.txt | sort > resultvern128cpu.txt
./keywordfinderV1 verne128.txt keywords.txt | sort | diff - resultvern128cpu.txt
```
Note: the `sort` command is required as the order of the matches in the output is non-deterministic.

## Global Memory Atomics

Use the `atomicInc` function to increment the `findcount` variable.
The result of this function can be used to determine the location of the match data.
This should produce a result that is fast, and more importantly correct!

If you run into problems, add a few "printf" statements within your kernel to help with your debugging.
Remember, first test on the smaller data set and then progress to the larger one.
If you don't have any errors and it works first time, you may still wish to add a few "printf" statements within your kernel just to see this working. 


## Extra Time?

Use "nvvp" to gain an understanding of the performance and limitations of this implementation. 















