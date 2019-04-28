all : helloworldOpenCL helloworldCUDAv1 helloworldCUDAv2
	
% : %.c Makefile
	gcc -o $@ $< -lOpenCL

% : %.cu Makefile
	nvcc -o $@ $< 


