/*
 * CPU side of CALDGEMM implementation.
 *
 * Copyright 2010:
 *  - David Rohr (drohr@jwdt.org)
 *  - Matthias Bach (bach@compeng.uni-frankfurt.de)
 *  - Matthias Kretz (kretz@compeng.uni-frankfurt.de)
 *
 * This file is part of CALDGEMM.
 *
 * CALDGEMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CALDGEMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CALDGEMM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "caldgemm_opencl.h"
#include <CL/cl_ext.h>

#define OCL_KERNEL_PRE \
"#ifdef cl_amd_fp64\n" \
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n" \
"#else\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"#endif\n" \
"//const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n"


#define OCLKernelName OCLKernel
#include "caldgemm.cl"
#undef OCLKernelName
#define OCLKernelName OCLKernelALPHA1
#define CALDGEMM_ALPHA1
#include "caldgemm.cl"
#undef OCLKernelName
#define OCLKernelName OCLKernelLinpack
#define CALDGEMM_LINPACK_KERNEL
#include "caldgemm.cl"
#undef CALDGEMM_LINPACK_KERNEL
#undef CALDGEMM_ALPHA1
#undef OCLKernelName

const char* caldgemm_opencl::OCLConvertKernel =
OCL_KERNEL_PRE
"__kernel void oclconkernel(__global const uint4* iBuffer, __write_only image2d_t oBuffer, int width, int height, int transpose)\n"
"{\n"
"	int i, j;\n"
"	for (i = get_global_id(1);i < height / 2;i+=get_global_size(1))\n"
"	{\n"
"		for (j = get_global_id(0);j < width / 2;j+=get_global_size(0))\n"
"		{\n"
"			uint4 tmp, tmp2;\n"
"			tmp = iBuffer[(2 * i) * (width / 2) + j];\n"
"			tmp2 = iBuffer[(2 * i + 1) * (width / 2) + j];\n"
"			int2 coord, coord2;\n"
"			uint4 val, val2;\n"
"			if (transpose)\n"
"			{\n"
"				val.x = tmp.x;val.y = tmp.y;\n"
"				val.z = tmp2.x;val.w = tmp2.y;\n"
"				val2.x = tmp.z;val2.y = tmp.w;\n"
"				val2.z = tmp2.z;val2.w = tmp2.w;\n"
"				coord.x = i;\n"
"				coord.y = 2 * j;\n"
"				coord2.x = i;\n"
"				coord2.y = 2 * j + 1;\n"
"			}\n"
"			else\n"
"			{\n"
"				coord.x = j;\n"
"				coord.y = 2 * i;\n"
"				coord2.x = j;\n"
"				coord2.y = 2 * i + 1;\n"
"				val = tmp;\n"
"				val2 = tmp2;\n"
"			}\n"
"			write_imageui(oBuffer, coord, val);\n"
"			write_imageui(oBuffer, coord2, val2);\n"
"		}\n"
"	}\n"
"}\n"
;

static const char* opencl_error_string(int errorcode)
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

#define ERRRET(...) {fprintf(STD_OUT, __VA_ARGS__);fprintf(STD_OUT, "\n");return(1);}
#define CHKRET(result, ...) \
	if (result != CL_SUCCESS) \
	{ \
		fprintf(STD_OUT, __VA_ARGS__); \
		fprintf(STD_OUT, ":\n"); \
		fprintf(STD_OUT, "OpenCL Error %d: (%s: %d) %s\n", result, __FILE__, __LINE__, opencl_error_string(result)); \
		return(1); \
	}

caldgemm_opencl::caldgemm_opencl() : caldgemm()
{
	C_matrix_base = NULL;
}

caldgemm_opencl::~caldgemm_opencl()
{
}

int caldgemm_opencl::WaitForEventAndRelease(cl_event* pEvent)
{
	cl_int ocl_error;
	if ((ocl_error = clWaitForEvents(1, pEvent)) != CL_SUCCESS)
	{
		fprintf(STD_OUT, "Error while waiting for event (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
		return(1);
	}
	if ((ocl_error = clReleaseEvent(*pEvent)) != CL_SUCCESS)
	{
		fprintf(STD_OUT, "Error releasing event (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
		return(1);
	}
	return(0);
}

int caldgemm_opencl::WaitForEvent(int a, int b, int)
{
	if (Config->Debug) fprintf(STD_OUT, "\tWaiting for event from device %d obuffer %d...\n", b, a);
	return(WaitForEventAndRelease(&ocl_events[b][a]));
}

int caldgemm_opencl::Initialize(bool nocalinit)
{
	int deviceNum = Config->DeviceNum;
	if (!Config->Quiet) fprintf(STD_OUT, "Initializing CALDGEMM (OpenCL Runtime)\n");
	if (Config->Debug) fprintf(STD_OUT, "OPENCL Initialice\n");
	cl_int ocl_error;

	cl_device_id* devices = new cl_device_id[nDevices];
	if (devices == NULL) ERRRET("Memory allocation error\n");
	CHKRET(clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, nDevices, devices, NULL), "Error getting OpenCL devices\n");

	for (int i = 0;i < nDevices;i++)
	{
		char device_vendor[64], device_name[64];
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, device_name, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, device_vendor, NULL);
		if (Config->Debug) fprintf(STD_OUT, "Device %d: %s %s\n", i, device_vendor, device_name);
	}

	if (nDevices > (signed) max_devices) nDevices = max_devices;
	if (nDevices > Config->NumDevices) nDevices = Config->NumDevices;

	if (deviceNum >= nDevices) ERRRET("OpenCL Device %d not available\n", deviceNum);
	if (deviceNum >= 0) nDevices = 1;
	gpu_available = (nDevices > 0);

	for (int i = 0;i < nDevices;i++)
	{
		if (deviceNum >= 0) ocl_devices[i] = devices[deviceNum];
		else ocl_devices[i] = devices[Config->DeviceNums[i]];
	}
	
	delete[] devices;

	for (int i = 0;i < nDevices;i++)
	{
		ocl_contexts[i] = clCreateContext(NULL, 1, &ocl_devices[i], NULL, NULL, &ocl_error);
		CHKRET(ocl_error, "Error creating OpenCL context");
	}

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < obuffercount;j++)
		{
			ocl_command_queues[i][j] = clCreateCommandQueue(ocl_contexts[i], ocl_devices[i], 0, &ocl_error);
			CHKRET(ocl_error, "Error creating OpenCL command queue");
		}
	}

	return(0);
}

int caldgemm_opencl::ValidateRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ValidateRuntime\n");

	if (Config->GPU_C == -1) Config->GPU_C = 1;

	cl_uint num_platforms;
	CHKRET(clGetPlatformIDs(0, NULL, &num_platforms), "Error getting OpenCL Platform Count");
	if (num_platforms == 0) ERRRET("No OpenCL Platform found\n");
	if (Config->Debug) fprintf(STD_OUT, "%d OpenCL Platforms found\n", num_platforms);

	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	if (platforms == NULL) ERRRET("Memory allocation error");
	CHKRET(clGetPlatformIDs(num_platforms, platforms, NULL), "Error getting OpenCL Platforms");

	for (unsigned int i = 0;i < num_platforms;i++)
	{
		char platform_profile[64], platform_version[64], platform_name[64], platform_vendor[64];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 64, platform_profile, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 64, platform_version, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 64, platform_name, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 64, platform_vendor, NULL);
		if (Config->Debug) fprintf(STD_OUT, "Platform %d: (%s %s) %s %s\n", i, platform_profile, platform_version, platform_vendor, platform_name);
	}

	if (Config->OpenCLPlatform >= (signed) num_platforms) ERRRET("OpenCL Platform %d not available\n", Config->OpenCLPlatform);
	ocl_platform = platforms[Config->OpenCLPlatform];
	delete[] platforms;

	cl_uint num_devices;
	clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	nDevices = num_devices;
	if (nDevices == 0) ERRRET("No OpenCL device for this platform found\n");
	if (Config->Debug) fprintf(STD_OUT, "%d OpenCL devices found for this platform\n", nDevices);

	return(0);
}

int caldgemm_opencl::CheckDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL CheckDevices\n");
	return(0);
}

int caldgemm_opencl::InitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL InitDevices\n");

	cl_int ocl_error;

	int num_bbuffers;
	if (Config->DstMemory == 'g') num_bbuffers =  max_bbuffers_g;
	else num_bbuffers = max_bbuffers;

	BufferHeight = Config->Height;
	BufferWidth = Config->Width;

	cl_image_format ocl_image_format;
	ocl_image_format.image_channel_order = CL_RGBA;
	ocl_image_format.image_channel_data_type = CL_UNSIGNED_INT32;

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < ibuffercount;j++)
		{
			cl_image_desc image_desc;
			memset(&image_desc, 0, sizeof(image_desc));
			image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
#ifdef CALDGEMM_TRANSPOSED_B
			image_desc.image_width = BufferWidth / 2;
			image_desc.image_height = BufferHeight;
			ocl_abuffers[i][j] = clCreateImage(ocl_contexts[i], CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
#elif defined(CALDGEMM_TRANSPOSED_A)
			image_desc.image_width = BufferHeight / 2;
			image_desc.image_height = BufferWidth;
			ocl_abuffers[i][j] = clCreateImage(ocl_contexts[i], CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
#endif
			CHKRET(ocl_error, "Error allocating device memory (A)");
		}

		for (int j = 0;j < obuffercount;j++)
		{
			if (Config->GPU_C == 0 || Config->DstMemory == 'g')
			{
				ocl_cbuffers[i][j] = clCreateBuffer(ocl_contexts[i], CL_MEM_READ_WRITE, BufferHeight * BufferHeight * sizeof(double), NULL, &ocl_error);
				CHKRET(ocl_error, "Error allocating device memory (C)");
			}
		}

		for (int j = 0;j < (Config->GPU_C ? obuffercount : ibuffercount);j++)
		{
			cl_int tmp_flags = Config->GPU_C ? CL_MEM_READ_ONLY : (CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE);
			ocl_tmp_abuffers[i][j] = clCreateBuffer(ocl_contexts[i], tmp_flags, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (A tmp - Width: %lld Height: %lld)", (long long int) BufferWidth, (long long int) BufferHeight);

			ocl_tmp_bbuffers[i][j] = clCreateBuffer(ocl_contexts[i], tmp_flags, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (B tmp)");

			if (Config->GPU_C == 0)
			{
				ocl_tmp_abuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_abuffers[i][j], CL_TRUE, CL_MAP_WRITE, 0, BufferWidth * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping buffer (A)");

				ocl_tmp_bbuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_bbuffers[i][j], CL_TRUE, CL_MAP_WRITE, 0, BufferWidth * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping buffer (B)");
			}
		}

		for (int j = 0;j < num_bbuffers;j++)
		{
			cl_image_desc image_desc;
			memset(&image_desc, 0, sizeof(image_desc));
			image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
#ifdef CALDGEMM_TRANSPOSED_B
			image_desc.image_width = BufferWidth / 2;
			image_desc.image_height = BufferHeight;
			ocl_bbuffers[i][j] = clCreateImage(ocl_contexts[i], CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
#elif defined(CALDGEMM_TRANSPOSED_A)
			image_desc.image_width = BufferHeight / 2;
			image_desc.image_height = BufferWidth;
			ocl_bbuffers[i][j] = clCreateImage(ocl_contexts[i], CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
#endif
			if (ocl_error != CL_SUCCESS)
			{
				if (j < obuffercount)
				{
					CHKRET(ocl_error, "Error allocating device memory (B)");
				}
				else break;
			}
			
			bbuffers[i] = j + 1;
		}
		if (Config->Debug) fprintf(STD_OUT, "Allocated %d BBuffers on Device %d\n", bbuffers[i], i);

		for (int j = 0;j < 3 + 1;j++)
		{
			const char* sourceCode;
			switch (j)
			{
			case 0:
				sourceCode = OCLKernel;
				break;
			case 1:
				sourceCode = OCLKernelALPHA1;
				break;
			case 2:
				sourceCode = OCLKernelLinpack;
				break;
			case 3:
				sourceCode = OCLConvertKernel;
				break;
			}

			if (Config->PrintILKernel && i == 0)
			{
				fprintf(STD_OUT, "OpenCL Kernel %d:\n%s\n\n", j, sourceCode);
			}

			ocl_program[i][j] = clCreateProgramWithSource(ocl_contexts[i], 1, &sourceCode, NULL, &ocl_error);
			CHKRET(ocl_error, "Error creating program object");

			ocl_error = clBuildProgram(ocl_program[i][j], 1, &ocl_devices[i], Config->Disassemble ? "-save-temps=." : "", NULL, NULL);
			if (ocl_error != CL_SUCCESS)
			{
				fprintf(STD_OUT, "OpenCL Error while building program: %d\n", ocl_error);
				fprintf(STD_OUT, "OpenCL Kernel:\n\n%s\n\n", sourceCode);
				char build_log[16384];
				clGetProgramBuildInfo(ocl_program[i][j], ocl_devices[i], CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL);
				fprintf(STD_OUT, "Build Log:\n\n%s\n\n", build_log);
				return(1);
			}

			ocl_kernel[i][j] = clCreateKernel(ocl_program[i][j], j == 3 ? "oclconkernel" : "oclkernel", &ocl_error);
			CHKRET(ocl_error, "Error creating kernel");
		}
	}

	return(0);
}

int caldgemm_opencl::ReinitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ReinitDevices\n");
	fprintf(STD_OUT, "Reinit of OpenCL devices not supported yet\n");
	return(1);
}

int caldgemm_opencl::InitConstantData(double alpha)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL InitConstantData\n");
	return(0);
}

int caldgemm_opencl::ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ExecuteKernels\n");

	if (Config->Debug) fprintf(STD_OUT, "\tExecuting MM kernel (device %d obuffer %d, k=%lld m=%lld n=%lld)\n", Task.device, Task.j, (long long int) Task.k, (long long int) blockm, (long long int) blockn);
#ifdef REUSE_BBUFFERS
	if (!DGEMM_favor_m && buffersSwitchable)
	{
		const int buffer_pos = buffer_pointers_A[Task.device][blockm] % (buffer_pointers_A[Task.device][blockm] < bbuffers[Task.device] ? bbuffers[Task.device] : ibuffercount);
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 1, sizeof(cl_mem), &ocl_bbuffers[Task.device][buffer_pos]), "Error setting kernel memory A");
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 2, sizeof(cl_mem), &ocl_abuffers[Task.device][buffer_pointers_B[Task.device][blockn] % ibuffercount]), "Error setting kernel memory B");
	}
	else
#endif
	{
#ifdef REUSE_BBUFFERS
		const bool buffersSufficiant = buffer_pointers_B[Task.device][blockn] < bbuffers[Task.device];
#else
		const bool buffersSufficiant = false;
#endif
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 1, sizeof(cl_mem), &ocl_abuffers[Task.device][buffer_pointers_A[Task.device][blockm] % ibuffercount]), "Error setting kernel memory A");
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 2, sizeof(cl_mem), &ocl_bbuffers[Task.device][!buffersSufficiant ? (buffer_pointers_B[Task.device][blockn] % ibuffercount) : (buffer_pointers_B[Task.device][blockn] % bbuffers[Task.device])]), "Error setting kernel memory B");
	}

	int pitch, offset;
	int height1 = (int) (((size_t) blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height);
	int height2 = (int) (((size_t) blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height);
	if (Config->GPU_C == 0 || Config->DstMemory == 'g')
	{
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 0, sizeof(cl_mem), &ocl_cbuffers[Task.device][Task.j]), "Error setting kernel memory C");
		pitch = height1;
		offset = 0;
	}
	else
	{
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 0, sizeof(cl_mem), C_matrix_base_obj), "Error setting kernel memory C");
		pitch = C_pitch;
		offset = (C + blockn * Config->Height + blockm * Config->Height * C_pitch) - C_matrix_base;
	}

	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 3, sizeof(int), &height1), "Error setting kernel arg height1");
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 4, sizeof(int), &height2), "Error setting kernel arg height2");

	int width = Config->Width;
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 5, sizeof(int), &width), "Error setting kernel arg width");

	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 6, sizeof(double), &Alpha), "Error setting kernel arg alpha");
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 7, sizeof(double), &Beta), "Error setting kernel arg beta");

	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 8, sizeof(int), &pitch), "Error setting kernel arg pitch");
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 9, sizeof(int), &offset), "Error setting kernel arg offset");

	size_t local_size[2] = {GROUP_SIZE_X, GROUP_SIZE_Y};
	size_t global_size[2] = {GROUP_SIZE_X * GROUP_COUNT_X, GROUP_SIZE_Y * GROUP_COUNT_Y};
	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[Task.device][Task.j]);
		Timers.Kernel.Start();
	}
	if (Config->Debug) fprintf(STD_OUT, "MM Kernel: height1 %d height2 %d width %d alpha %f beta %f\n", height1, height2, width, Alpha, Beta);
	cl_event* kernel_event;
	if (Config->DstMemory == 'g') kernel_event = NULL;
	else kernel_event = &ocl_events[Task.device][Task.j];

	CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[Task.device][Task.j], ocl_kernel[Task.device][Task.kernel_num], 2, NULL, &global_size[0], &local_size[0], 0, NULL, kernel_event), "Error starting MM Kernel");

	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[Task.device][Task.j]);
		Timers.Kernel.Stop();
		Timers.CounterCopyFrom.Start();
	}

	if (Config->DstMemory == 'g')
	{
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {height1 * sizeof(double), height2, 1};
		if (Config->Debug) fprintf(STD_OUT, "Transfer C from GPU: region %d x %d\n", (int) region[0], (int) region[1]);
		CHKRET(clEnqueueReadBufferRect(ocl_command_queues[Task.device][Task.j], ocl_cbuffers[Task.device][Task.j], CL_FALSE, origin, origin, region, 0, 0, C_pitch * sizeof(double), 0, C + blockn * Config->Height + blockm * Config->Height * C_pitch, 0, NULL, &ocl_events[Task.device][Task.j]), "Error retrieving C\n");
		clFlush(ocl_command_queues[Task.device][Task.j]);
	}
	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[Task.device][Task.j]);
		Timers.CounterCopyFrom.Stop();
	}

	return(0);
}

int caldgemm_opencl::ExitRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ExitRuntime\n");

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < obuffercount;j++)
		{
			clReleaseCommandQueue(ocl_command_queues[i][j]);
		}
		clReleaseContext(ocl_contexts[i]);
	}

	return(0);
}

int caldgemm_opencl::FetchResult(int device, int j, int m, int n, int mustlock)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL FetchResult\n");
	return(0);
}

int caldgemm_opencl::CheckDMAQueue(int device, int forcej)
{
	return(0);
}

int caldgemm_opencl::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)
{
	if (Config->GPU_C) return(0);
	if (Config->Debug) fprintf(STD_OUT, "OPENCL RunMergeBuffers\n");
	double* src = ocl_tmp_cbuffers_ptr[device][j];
	for (int i = 0;i < height;i++)
	{
		for (int j = 0;j < width;j++)
		{
			dst[j] = src[j];
		}
		src += gpu_width;
		dst += pitch;
	}
	return(0);
}

int caldgemm_opencl::divideBuffer(double* src, size_t pitch_src, double* dest, size_t nSrcRows, size_t nSrcCols, bool transpose)
{
	if (Config->GPU_C) return(0);
	if (Config->Debug) fprintf(STD_OUT, "OPENCL divideBuffers\n");
	return(0);
}

int caldgemm_opencl::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	size_t origin[3] = {0, 0, 0};
	size_t region[3];
	region[2] = 1;

	if (Config->VerboseTiming) Timers.CounterCopyTo.Start();

	if (ocl_conversion_events_use[num_device][0])
	{
		WaitForEventAndRelease(&ocl_conversion_events[num_device][0]);
		ocl_conversion_events_use[num_device][0] = 0;
	}
	if (prepareM)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
		Timers.divideA++;

		cl_mem *dest_image;
		region[0] = (TransposeA ? Config->Height : Config->Width) * sizeof(double);
		region[1] = (TransposeA ? Config->Width : Config->Height);
		size_t pitch = A_pitch;
		void* src_ptr = A + blockm * Config->Height * (TransposeA ? 1 : A_pitch);
		
		if (!DGEMM_favor_m && buffersSufficiant0)
		{
			dest_image = &ocl_bbuffers[num_device][buffer_pointers_A[num_device][blockm] % (buffersSufficiant ? bbuffers[num_device] : ibuffercount)];
		}
		else
		{
			dest_image = &ocl_abuffers[num_device][next_buffer_A[num_device] % ibuffercount];
		}

#ifdef CALDGEMM_TRANSPOSED_B
		int arg_transpose = TransposeA;
#elif defined(CALDGEMM_TRANSPOSED_A)
		int arg_transpose = !TransposeA;
#endif
		if (Config->GPU_C == 0)
		{
			if (divideBuffer((double*) src_ptr, pitch, ocl_tmp_abuffers_ptr[num_device][next_buffer_A[num_device] % ibuffercount], region[1], region[0], arg_transpose)) return(1);
		}
		else
		{
			cl_mem dest_buffer_tmp = ocl_tmp_abuffers[num_device][j];
			if (Config->Debug) fprintf(STD_OUT, "Transfer A to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
			CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch * sizeof(double), 0, src_ptr, 0, NULL, NULL), "Error copying A");
			if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, A, 0");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), dest_image), "Error setting kernel arg, A, 1");
			int arg_width = region[0] / sizeof(double), arg_height = region[1];
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &arg_width), "Error setting kernel arg, A, 2");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &arg_height), "Error setting kernel arg, A, 3");
		
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, A, 4");

			size_t local_size[2] = {GROUP_SIZE_X, GROUP_SIZE_Y};
			size_t global_size[2] = {GROUP_SIZE_X * GROUP_COUNT_X, GROUP_SIZE_Y * GROUP_COUNT_Y};
			if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel A: x %d y %d (t: %d)\n", arg_width, arg_height, arg_transpose);
			CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][j], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], 0, NULL, &ocl_conversion_events[num_device][0]), "Error starting conversion kernel for A");
		}
		ocl_conversion_events_use[num_device][0] = 1;
		if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
	}

	if (ocl_conversion_events_use[num_device][1])
	{
		WaitForEventAndRelease(&ocl_conversion_events[num_device][1]);
		ocl_conversion_events_use[num_device][1] = 0;
	}
	if (prepareN)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tCopying part of B to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
		Timers.divideB++;

		cl_mem *dest_image;
		
		region[0] = (TransposeB ? Config->Width : Config->Height) * sizeof(double);
		region[1] = (TransposeB ? Config->Height : Config->Width);
		size_t pitch = B_pitch;
		void* src_ptr = B + blockn * Config->Height * (TransposeB ? B_pitch : 1);

		if (!DGEMM_favor_m && buffersSufficiant0)
		{
			dest_image = &ocl_abuffers[num_device][next_buffer_B[num_device] % ibuffercount];
		}
		else
		{
			dest_image = &ocl_bbuffers[num_device][buffersSufficiant ? (buffer_pointers_B[num_device][blockn] % bbuffers[num_device]) : (next_buffer_B[num_device] % ibuffercount)];
		}

#ifdef CALDGEMM_TRANSPOSED_B
		int arg_transpose = !TransposeB;
#elif defined(CALDGEMM_TRANSPOSED_A)
		int arg_transpose = TransposeB;
#endif

		if (Config->GPU_C == 0)
		{
			if (divideBuffer((double*) src_ptr, pitch, ocl_tmp_abuffers_ptr[num_device][next_buffer_B[num_device] % ibuffercount], region[1], region[0], arg_transpose)) return(1);
		}
		else
		{
			cl_mem dest_buffer_tmp = ocl_tmp_bbuffers[num_device][j];
			if (Config->Debug) fprintf(STD_OUT, "Transfer B to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
			CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch * sizeof(double), 0, src_ptr, 0, NULL, NULL), "Error copying B");
			if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, B, 0");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), dest_image), "Error setting kernel arg, B, 1");
			int arg_width = region[0] / sizeof(double), arg_height = region[1];
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &arg_width), "Error setting kernel arg, B, 2");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &arg_height), "Error setting kernel arg, B, 3");

			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, B, 4");

			size_t local_size[2] = {GROUP_SIZE_X, GROUP_SIZE_Y};
			size_t global_size[2] = {GROUP_SIZE_X * GROUP_COUNT_X, GROUP_SIZE_Y * GROUP_COUNT_Y};
			if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel B: x %d y %d\n", (int) arg_width, (int) arg_height);
			CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][j], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], 0, NULL, &ocl_conversion_events[num_device][1]), "Error starting conversion kernel for B");
		}
		ocl_conversion_events_use[num_device][1] = 1;
		if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
	}

	if (Config->GPU_C && Config->DstMemory == 'g')
	{
		Timers.divideC++;
		region[0] = (((size_t) blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height) * sizeof(double);
		region[1] = (((size_t) blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height);
		if (Config->Debug) fprintf(STD_OUT, "Transfer C to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
		CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], ocl_cbuffers[num_device][j], CL_FALSE, origin, origin, region, 0, 0, C_pitch * sizeof(double), 0, C + blockn * Config->Height + blockm * Config->Height * C_pitch, 0, NULL, NULL), "Error copying C");
		clFlush(ocl_command_queues[num_device][j]);
	}
	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[num_device][j]);
		Timers.CounterCopyTo.Stop();
	}

	return(0);
}

int caldgemm_opencl::ExitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ExitDevices\n");

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < ibuffercount;j++)
		{
			clReleaseMemObject(ocl_abuffers[i][j]);
		}
		for (int j = 0;j < obuffercount;j++)
		{
			if (Config->GPU_C == 0 || Config->DstMemory == 'g') clReleaseMemObject(ocl_cbuffers[i][j]);
		}
		for (int j = 0;j < (Config->GPU_C ? obuffercount : ibuffercount);j++)
		{
			if (Config->GPU_C == 0)
			{
				clEnqueueUnmapMemObject(ocl_command_queues[i][0], ocl_tmp_abuffers[i][j], ocl_tmp_abuffers_ptr[i][j], 0, NULL, NULL);
				clEnqueueUnmapMemObject(ocl_command_queues[i][0], ocl_tmp_bbuffers[i][j], ocl_tmp_bbuffers_ptr[i][j], 0, NULL, NULL);
				clFinish(ocl_command_queues[i][0]);
			}
			clReleaseMemObject(ocl_tmp_abuffers[i][j]);
			clReleaseMemObject(ocl_tmp_bbuffers[i][j]);
		}
		for (int j = 0;j < bbuffers[i];j++)
		{
			clReleaseMemObject(ocl_bbuffers[i][j]);
		}
		for (int j = 0;j < 3 + 1;j++)
		{
			clReleaseKernel(ocl_kernel[i][j]);
			clReleaseProgram(ocl_program[i][j]);
		}
	}
	return(0);
}

int caldgemm_opencl::UseOutputPthreads() {return(!Config->GPU_C);}
int caldgemm_opencl::UseInputPthreads() {return(!Config->GPU_C);}
int caldgemm_opencl::UseMutexPerDevice() {return(0);}

int caldgemm_opencl::RunCALDGEMM_Init()
{
	if (Config->DstMemory == 'c' && C_matrix_base == NULL)
	{
		fprintf(STD_OUT, "DstMemory = 'c' can only be used if C matrix memory was allocated using OpenCL memory before.\n");
		return(1);
	}
	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			ocl_conversion_events_use[i][j] = 0;
		}
	}
	return(0);
}

int caldgemm_opencl::RunCALDGEMM_Exit()
{
	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			if (ocl_conversion_events_use[i][j])
			{
				clReleaseEvent(ocl_conversion_events[i][j]);
			}
		}
	}
	return(0);
}

#define MAX_GPU_MEM_COUNT 64
struct gpu_mem_struct_opencl
{
	void* ptr;
	cl_mem mem_obj;
};
static gpu_mem_struct_opencl gpu_mem[MAX_GPU_MEM_COUNT];
static int nGPUMEM = 0;

double* caldgemm_opencl::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible, bool Cmatrix, bool interleave)
{
	if (gpuaccessible && Config->GPU_C)
	{
		if (nGPUMEM == MAX_GPU_MEM_COUNT)
		{
			fprintf(STD_OUT, "Cannot allocated more GPU memory, increase MAX_GPU_MEM_COUNT\n");
			return(0);
		}
		cl_int ocl_error;
		cl_int mem_flags = CL_MEM_READ_WRITE;
		mem_flags |= CL_MEM_ALLOC_HOST_PTR;
		if (0)
		{
#ifdef CL_MEM_USE_PERSISTENT_MEM_AMD
			mem_flags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
#else
			fprintf(STD_OUT, "WARNING: CL_MEM_USE_PERSISTENT_MEM_AMD flag not defined\n");
#endif
		}
		gpu_mem[nGPUMEM].mem_obj = clCreateBuffer(ocl_contexts[0], mem_flags, nDoubles * sizeof(double), NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error allocating memory (clCreateBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(0);
		}
		gpu_mem[nGPUMEM].ptr = clEnqueueMapBuffer(ocl_command_queues[0][0], gpu_mem[nGPUMEM].mem_obj, CL_TRUE, 0, 0, nDoubles * sizeof(double), 0, NULL, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error allocating memory (clEnqueueMapBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(0);
		}
		if (Cmatrix)
		{
			C_matrix_base = (double*) gpu_mem[nGPUMEM].ptr;
			C_matrix_base_obj = &gpu_mem[nGPUMEM].mem_obj;
		}
		return((double*) gpu_mem[nGPUMEM++].ptr);
	}
	else
	{
		return (caldgemm::AllocMemory(nDoubles, page_locked, huge_pages, gpuaccessible, Cmatrix, interleave));
	}
}

void caldgemm_opencl::FreeMemory(double* ptr, bool gpuaccessible)
{
	if (ptr == C_matrix_base) C_matrix_base = NULL;
	if (gpuaccessible && Config->GPU_C)
	{
		for (int i = 0;i < nGPUMEM;i++)
		{
			if (gpu_mem[i].ptr == (void*) ptr)
			{
				clEnqueueUnmapMemObject(ocl_command_queues[0][0], gpu_mem[i].mem_obj, gpu_mem[i].ptr, 0, NULL, NULL);
				clReleaseMemObject(gpu_mem[i].mem_obj);
				if (C_matrix_base_obj == &gpu_mem[nGPUMEM - 1].mem_obj) C_matrix_base_obj = &gpu_mem[i].mem_obj;
				gpu_mem[i] = gpu_mem[--nGPUMEM];
				return;
			}
		}
	}
	caldgemm::FreeMemory(ptr);
}
