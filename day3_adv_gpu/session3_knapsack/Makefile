
targets = knapsackCPU knapsackCPUbig  expectedresult.txt knapsackCUDAv1

all : $(targets)

% : %.c Makefile
	gcc -O3 -o $@ $<

% : %.cu Makefile
	nvcc -o $@ $<

expectedresult.txt : knapsackCPUbig Makefile
	./knapsackCPUbig > expectedresult.txt

test : knapsackCUDAv1  expectedresult.txt Makefile
	./knapsackCUDAv1 | diff -s - expectedresult.txt

clean :
	rm  $(targets)
