all : listprimes
	
% : %.c Makefile
	gcc -o $@ $< -lOpenCL

% : %.cu Makefile
	nvcc -o $@ $< 


