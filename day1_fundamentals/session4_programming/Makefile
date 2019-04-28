all : listprimes
	
% : %.c makefile
	gcc -o $@ $< -lOpenCL

% : %.cu makefile
	nvcc -o $@ $< 


