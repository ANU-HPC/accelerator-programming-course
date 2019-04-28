all : helloworldOpenCL helloworldCUDAv1 helloworldCUDAv2
	
% : %.c makefile
	gcc -o $@ $< -lOpenCL

% : %.cu makefile
	nvcc -o $@ $< 


