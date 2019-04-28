
targets = knapsackCPU knapsackCPUbig  expectedresult.txt knapsackCUDAv1

all : $(targets)

% : %.c Makefile
	gcc -O3 -o $@ $<

% : %.cu Makefile
	nvcc -o $@ $<

expectedresult.txt : knapsackCPUbig makefile
	./knapsackCPUbig > expectedresult.txt

test : knapsackCUDAv1  expectedresult.txt makefile
	./knapsackCUDAv1 | diff -s - expectedresult.txt

clean :
	rm  $(targets)
