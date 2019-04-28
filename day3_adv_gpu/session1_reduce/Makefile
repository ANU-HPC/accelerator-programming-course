all : matVecMult
	
% : %.c Makefile
	gcc -o $@ $< -lOpenCL

% : %.cu Makefile
	nvcc -o $@ $< 


