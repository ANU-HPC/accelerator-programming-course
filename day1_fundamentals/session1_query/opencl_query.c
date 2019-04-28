#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

//OpenCL header files
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
//if Linux
#include <CL/cl.h>
#endif

int main(int argc, char** argv)
{
	cl_uint number_of_platforms = 0;
	clGetPlatformIDs((cl_uint)0,        //num_entries
			NULL,                 //platforms
			&number_of_platforms);//num_platforms
	if(number_of_platforms==0){
		printf("Error: No platforms found!\n");
		printf("\tIs an OpenCL driver installed?");
		return EXIT_FAILURE;
	}

	//get all those platforms
	cl_platform_id* my_platforms =
		(cl_platform_id*)malloc(sizeof(cl_platform_id)*
				number_of_platforms);
	clGetPlatformIDs(number_of_platforms,	//num_entries
			 my_platforms,       	//platform
			 NULL);              	//num_platforms

	printf("\n");
	printf("Your system platform id(s) are:\n");
	//print some details about each platform
	for(size_t i = 0; i < number_of_platforms; i++){
		printf("\tplatform no. %zu\n",i);

		//print it's name
		size_t total_buffer_length = 1024;
		size_t length_of_buffer_used = 0;
		char my_platform_name[total_buffer_length];
		clGetPlatformInfo(my_platforms[i],      //platform
				CL_PLATFORM_NAME,       	//param_name
				total_buffer_length,    	//param_value_size
				&my_platform_name,      	//param_value
				&length_of_buffer_used);	//param_value_size_ret
		printf("\t\tname:\t\t%*.*s\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_platform_name);

		//print the vendor
		char my_platform_vendor[total_buffer_length];
		length_of_buffer_used = 0;
		clGetPlatformInfo(my_platforms[i],      //platform
				CL_PLATFORM_VENDOR,     	//param_name
				total_buffer_length,    	//param_value_size
				&my_platform_vendor,    	//param_value
				&length_of_buffer_used);	//param_value_size_ret
		printf("\t\tvendor:\t\t%*.*s\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_platform_vendor);

		//print the profile
		char my_platform_profile[total_buffer_length];
		length_of_buffer_used = 0;
		clGetPlatformInfo(my_platforms[i],      //platform
				CL_PLATFORM_PROFILE,    	//param_name
				total_buffer_length,    	//param_value_size
				&my_platform_profile,   	//param_value
				&length_of_buffer_used);	//param_value_size_ret
		printf("\t\tprofile:\t%*.*s\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_platform_profile);

		//print the version
		char my_platform_version[total_buffer_length];
		length_of_buffer_used = 0;
		clGetPlatformInfo(my_platforms[i],      //platform
				CL_PLATFORM_VERSION,    	//param_name
				total_buffer_length,    	//param_value_size
				&my_platform_profile,   	//param_value
				&length_of_buffer_used);	//param_value_size_ret
		printf("\t\tversion:\t%*.*s\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_platform_profile);

		//print the extensions
		char my_platform_extensions[total_buffer_length];
		length_of_buffer_used = 0;
		clGetPlatformInfo(my_platforms[i],      //platform
				CL_PLATFORM_EXTENSIONS, 	//param_name
				total_buffer_length,    	//param_value_size
				&my_platform_extensions,	//param_value
				&length_of_buffer_used);	//param_value_size_ret
		printf("\t\textensions:\t%*.*s\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_platform_extensions);

		//given this platform, how many devices are available?
		//start by getting the count of available devices
		cl_uint number_of_devices = 0;
		clGetDeviceIDs(my_platforms[i],    	//platform_id
				CL_DEVICE_TYPE_ALL, 		//device_type
				(cl_uint)0,      		//num_entries
				NULL,               		//devices
				&number_of_devices);		//num_devices
		if(number_of_devices==0){
			printf("Error: No devices found for this platform!\n");
			return EXIT_FAILURE;
		}
		printf("\n\t\twith device id(s):\n");

		//get all those platforms
		cl_device_id* my_devices =
			(cl_device_id*)malloc(sizeof(cl_device_id)*number_of_devices);
		clGetDeviceIDs(my_platforms[i],    	//platform_id
				CL_DEVICE_TYPE_ALL, 		//device_type
				number_of_devices,  		//num_entries
				my_devices,         		//devices
				NULL);              		//num_devices
		//for each device print some of its details:
		for(size_t j = 0; j < number_of_devices; j++){
			printf("\t\tdevice no. %zu\n",j);

			//print the name
			char my_device_name[total_buffer_length];
			length_of_buffer_used = 0;
			clGetDeviceInfo(my_devices[j],      //device
					CL_DEVICE_NAME,         	//param_name
					total_buffer_length,    	//param_value_size
					&my_device_name,        	//param_value
					&length_of_buffer_used);	//param_value_size_ret
			printf("\t\t\tname:\t\t%*.*s\n",(int)length_of_buffer_used,
					(int)length_of_buffer_used,my_device_name);

			//print the vendor
			char my_device_vendor[total_buffer_length];
			length_of_buffer_used = 0;
			clGetDeviceInfo(my_devices[j],      //device
					CL_DEVICE_VENDOR,       	//param_name
					total_buffer_length,    	//param_value_size
					&my_device_vendor,      	//param_value
					&length_of_buffer_used);	//param_value_size_ret
			printf("\t\t\tvendor:\t\t%*.*s\n\n",(int)length_of_buffer_used,
					(int)length_of_buffer_used,my_device_vendor);

			//device type
			//CL_DEVICE_TYPE
			cl_device_type my_device_type;
			clGetDeviceInfo(my_devices[j],	//device
					CL_DEVICE_TYPE, //param_name
					sizeof(cl_device_type),	        //param_value_size
					&my_device_type,//param_value
					NULL);	        //param_value_size_ret

			switch(my_device_type){
				case CL_DEVICE_TYPE_CPU:
					printf("\t\t\tdevice type:\t\tCPU\n");
					break;
				case CL_DEVICE_TYPE_GPU:
					printf("\t\t\tdevice type:\t\tGPU\n");
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					printf("\t\t\tdevice type:\t\tACCELERATOR\n");
					break;
				case CL_DEVICE_TYPE_DEFAULT:
					printf("\t\t\tdevice type:\t\tDEFAULT\n");
					break;
				default:
					printf("\t\t\tno device type!\n");
			}
			
			//no. cores
			cl_uint my_number_of_cores;
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_MAX_COMPUTE_UNITS, 	//param_name
					sizeof(cl_uint),	     	//param_value_size
					&my_number_of_cores,	     	//param_value
					NULL);	        		//param_value_size_ret

			printf("\t\t\tcore count:\t\t%u\n",my_number_of_cores);

			//core clock frequency
			cl_uint my_clock_frequency;
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_MAX_CLOCK_FREQUENCY, 	//param_name
					sizeof(cl_uint),	     	//param_value_size
					&my_clock_frequency,	     	//param_value
					NULL);	        		//param_value_size_ret

			printf("\t\t\tmax clock frequency:\t%u MHz\n",my_clock_frequency);
			
			//workgroups
			//max total workgroup size 
			size_t my_max_work_group_size;
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_MAX_WORK_GROUP_SIZE, 	//param_name
					sizeof(size_t),		     	//param_value_size
					&my_max_work_group_size,  	//param_value
					NULL);	        		//param_value_size_ret

			printf("\t\t\tmax total work size:\t%zu\n",my_max_work_group_size);
			//no. work dimensions		
			cl_uint my_max_work_item_dimensions;
			clGetDeviceInfo(my_devices[j],				//device
					CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 	//param_name
					sizeof(cl_uint),	     		//param_value_size
					&my_max_work_item_dimensions,  		//param_value
					NULL);	        			//param_value_size_ret

			printf("\t\t\tmax work dimensions:\t%u\n",my_max_work_item_dimensions);
			//CL_DEVICE_MAX_WORK_ITEM_SIZES size_t[]
			//max total workgroup size 
			size_t my_max_work_item_sizes[my_max_work_item_dimensions];
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_MAX_WORK_ITEM_SIZES, 	//param_name
					sizeof(size_t)*
					my_max_work_item_dimensions,   	//param_value_size
					&my_max_work_item_sizes,  	//param_value
					NULL);	        		//param_value_size_ret
			printf("\t\t\tmax work item sizes:\t");
			for(size_t k = 0; k < my_max_work_item_dimensions; k++){
				printf("%zu\t",my_max_work_item_sizes[k]);
			}
			printf("\n");
			
			//memory	
			//(total global)
			cl_ulong my_global_mem_size;	
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_GLOBAL_MEM_SIZE, 	//param_name
					sizeof(cl_ulong),	     	//param_value_size
					&my_global_mem_size,	     	//param_value
					NULL);	        		//param_value_size_ret

			printf("\t\t\tglobal memory size:\t%lu MiBs\n",(my_global_mem_size>>20));
			
			//(global cache size)
			cl_ulong my_global_cache_size;	
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,//param_name
					sizeof(cl_ulong),	     	//param_value_size
					&my_global_cache_size,		//param_value
					NULL);	        		//param_value_size_ret

			printf("\t\t\tglobal cache size:\t%lu KiBs\n",(my_global_cache_size>>10));

			//(global cache line size)
			cl_uint my_global_cacheline_size;	
			clGetDeviceInfo(my_devices[j],				//device
					CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,	//param_name
					sizeof(cl_uint),	     		//param_value_size
					&my_global_cacheline_size,		//param_value
					NULL);	        			//param_value_size_ret

			printf("\t\t\tglobal cacheline size:\t%u bytes\n",(my_global_cacheline_size));

			cl_ulong my_local_mem_size;	
			clGetDeviceInfo(my_devices[j],			//device
					CL_DEVICE_LOCAL_MEM_SIZE, 	//param_name
					sizeof(cl_ulong),	     	//param_value_size
					&my_local_mem_size,	     	//param_value
					NULL);	        		//param_value_size_ret

			printf("\t\t\tlocal memory size:\t%lu KiBs\n",(my_local_mem_size>>10));

		}
		printf("\n");
		free(my_devices);
	} 

	free(my_platforms);
	return EXIT_SUCCESS;
}

