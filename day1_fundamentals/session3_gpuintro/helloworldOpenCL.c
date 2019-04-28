#include<stdio.h>
#include<CL/cl.h>

// Hello World using OpenCL - Eric McCreath 2019
// a very basic hello world implementation using opencl. 


void err(char *str) {
    printf("error: %s\n",str);
    exit(-1);
}

int main() {

// get hold of the platform
    unsigned int numPlatforms;
    cl_platform_id platforms[10];
    if (clGetPlatformIDs(10,platforms,&numPlatforms)) err("Getting Platform");
    printf("clGetPlatformsIDs numPlatforms: %d\n",numPlatforms);


// get hold of the first GPU device on the platform
    unsigned int numDev;
    cl_device_id devIDs[10];
    if (clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,10,devIDs,NULL)) err("Getting Devices");
    cl_device_id device;
    device = devIDs[0];


// get some basic info about this device
    char info[100];
    long unsigned int size;
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, 100, info, &size)) err("Getting Info");
    printf("NAME: %s\n",info);
    if (clGetDeviceInfo(device, CL_DEVICE_VERSION, 100, info, &size)) err("Getting Info");
    printf("VERSION: %s\n",info);
  
// create a context for launching kernel on the device
    cl_int retcode;
    cl_context context;
    if (!(context = clCreateContext(NULL,1,&device,NULL,NULL,&retcode ))) {printf("retcode : %d\n",retcode);err("context");}
  

// make a commend queue so we can give tasks for the device to do
    cl_command_queue commandQueue;
    if (!(commandQueue = clCreateCommandQueueWithProperties(context, devIDs[0], NULL, &retcode))) err("command queue");
    // if (!(commandQueue = clCreateCommandQueue(context, devIDs[0], 0, &retcode))) err("command queue");




// compile up our kernel
    const char* source =  "__constant char str[] = \"Hello World!\";\n"
                          "__kernel \n"
                          "void hello(__global char *out) {\n"
                          "     int idx = get_global_id(0);\n"
                          "     if (idx < 12) out[idx] = str[idx];\n"
                          "}\n";

    cl_program program;
    if (!(program = clCreateProgramWithSource(context,
                                              1,
                                              &source,
                                              NULL,
                                              &retcode))) err("problem compiling code");

    char *message;
    message = malloc(1000);
    if (clBuildProgram(program,
                       1,
                       &device,
                       NULL,
                       NULL,
                       NULL)) {
        if (clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_LOG,
                                  1000,
                                  message,
                                  NULL)) err("build info");
             printf("log: %s\n",message);
             err("problem building");
    }

    cl_kernel kernel;
    if (!(kernel = clCreateKernel(program,
                                  "hello",
                                  &retcode))) err("problem creating kernel");

// make a buffer for the result to go
    cl_mem mem;
    if (!(mem = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               12,
                               NULL,
                               &retcode)))  err("problem creating buffer");


// start the kernel
    clSetKernelArg(kernel, 0 , sizeof(cl_mem), &mem);
    size_t workSize[1];
    workSize[0] = 12;
    if (clEnqueueNDRangeKernel(commandQueue,
                               kernel,
                               1,
                               NULL,
                               workSize,
                               NULL,
                               0,
                               NULL,
                               NULL)) err("enqueue kernel");

// transfer the result back to the CPU
    char *resBuf;
    resBuf = malloc(12);
    clEnqueueReadBuffer(commandQueue, mem, CL_TRUE, 0, 12, resBuf,0,NULL,NULL);

    printf("Result : %s\n", resBuf);
    return 0;
}
