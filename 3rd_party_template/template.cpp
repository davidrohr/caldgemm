#include <CL/opencl.h>
#include <stdio.h>

#define STD_OUT stdout

#ifdef __WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define ERRRET(...) {fprintf(STD_OUT, __VA_ARGS__);fprintf(STD_OUT, "\n");return(1);}
#define CHKRET(result, ...) \
	if (result != CL_SUCCESS) \
	{ \
		fprintf(STD_OUT, __VA_ARGS__); \
		fprintf(STD_OUT, ":\n"); \
		fprintf(STD_OUT, "OpenCL Error %d: (%s: %d) %s\n", result, __FILE__, __LINE__, opencl_error_string(result)); \
		return(0); \
	}

//We must export two functions, kernelLibCreate to return the kernel object, kernelLibQuerySettings to return some parameters
extern "C" DLL_EXPORT cl_kernel kernelLibCreate(cl_context* context, int nDevices, cl_device_id* devices, int kernelType, int k, int betazero); 
extern "C" DLL_EXPORT void kernelLibQuerySettings(int* tiling_x, int* tiling_y, bool* transposeA, bool* transposeB, bool* texture_buffers, int* group_size_x, int* group_size_y, int* min_tile_size, int* min_k);
extern "C" DLL_EXPORT void kernelLibTerminate();

//The kernels can be subject to some optimizations, depending on the parameters:
//betazero indicates that beta can be assumed zero, regardless of other parameters
//kernelType:
//0 - no further optimizations
//1 - can assume alpha = 1
//2 - can assume alpha = 1 and k is fixed to the parameter passed as k
//3 - not used
//4 - can assume alpha = -1 and k is fixed to the parameter passed as k

//kernelLibQuerySettings must return
//The tiling size in x and y (defines how many work-items are started
//transposeA and transposeB define whether the kernel expects A or B input matrices in transposed form or not
//texture_buffers = 1 means input is read from images, 0 stands for standard buffers
//group_size_x/y defines the work-group-size

cl_program ocl_program, ocl_programx;

const char* kernel_str =
#include "kernel.cl"
;

int program_initialized = 0;

const char* opencl_error_string(int errorcode)
{
    switch (errorcode)
	{
		case CL_SUCCESS:                            return "Success!";
		case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
		case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
		case CL_OUT_OF_RESOURCES:                   return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
		case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
		case CL_MAP_FAILURE:                        return "Map failure";
		case CL_INVALID_VALUE:                      return "Invalid value";
		case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
		case CL_INVALID_PLATFORM:                   return "Invalid platform";
		case CL_INVALID_DEVICE:                     return "Invalid device";
		case CL_INVALID_CONTEXT:                    return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
		case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
		case CL_INVALID_SAMPLER:                    return "Invalid sampler";
		case CL_INVALID_BINARY:                     return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
		case CL_INVALID_PROGRAM:                    return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
		case CL_INVALID_KERNEL:                     return "Invalid kernel";
		case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
		case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
		case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
		case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
		case CL_INVALID_EVENT:                      return "Invalid event";
		case CL_INVALID_OPERATION:                  return "Invalid operation";
		case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
		default: return "Unknown Errorcode";
	}
}

cl_kernel kernelLibCreate(cl_context* context, int nDevices, cl_device_id* devices, int kernelType, int k, int betazero)
{
	cl_int ocl_error;
	if (program_initialized == 0)
	{
		ocl_program = clCreateProgramWithSource(*context, 1, &kernel_str, NULL, &ocl_error);
		CHKRET(ocl_error, "Error creating program object");
		ocl_error = clBuildProgram(ocl_program, nDevices, devices, 0, NULL, NULL);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "OpenCL Error while building program: %d\n", ocl_error);
			fprintf(STD_OUT, "OpenCL Kernel:\n\n%s\n\n", kernel_str);
			char build_log[16384];
			for (int i = 0;i < nDevices;i++)
			{
				clGetProgramBuildInfo(ocl_program, devices[i], CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL);
				fprintf(STD_OUT, "Build Log (device %d):\n\n%s\n\n", i, build_log);
			}
			return(0);
		}
		program_initialized = 1;
	}
	cl_kernel tmp = clCreateKernel(ocl_program, "oclkernel", &ocl_error);
	CHKRET(ocl_error, "Error creating kernel");
	
	return(tmp);
}

void kernelLibTerminate()
{
	if (program_initialized)
	{
		clReleaseProgram(ocl_program);
		program_initialized = 0;
	}
}

void kernelLibQuerySettings(int* tiling_x, int* tiling_y, bool* transposeA, bool* transposeB, bool* texture_buffers, int* group_size_x, int* group_size_y, int* min_tile_size, int* min_k)
{
	*group_size_x = *group_size_y = 8; //We start a grid with work-group-size 8x8 and in total m/tilingx x n/tiling_y work items
	*tiling_x = *tiling_y = 4;
	*texture_buffers = false;
	*transposeA = true;
	*transposeB = false;
	*min_tile_size = 32;
	*min_k = 4;
}
