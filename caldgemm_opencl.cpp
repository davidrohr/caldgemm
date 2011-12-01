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

#define OCL_KERNEL_PRE \
"#ifdef KHR_DP_EXTENSION\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"#else\n" \
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n" \
"#endif\n" \
"const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
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
"__kernel void oclkernel(__global const float4* iBuffer, __write_only image2d_t oBuffer, int width, int height, int transpose)\n"
"{\n"
"	int i, j;\n"
"	for (i = get_global_id(0);i < height / 2;i+=get_global_size(0))\n"
"	{\n"
"		for (j = get_global_id(1);j < width / 2;j+=get_global_size(1))\n"
"		{\n"
"			float4 tmp, tmp2;\n"
"			tmp = iBuffer[2 * i * width / 2 + j];\n"
"			tmp2 = iBuffer[2 * (i + 1) * width / 2 + j];\n"
"			int2 coord, coord2;\n"
"			float4 val, val2;\n"
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
"			write_imagef(oBuffer, coord, val);\n"
"			write_imagef(oBuffer, coord2, val2);\n"
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

#define ERRRET(...) {fprintf(STD_OUT, __VA_ARGS__);return(1);}
#define CHKRET(result, ...) \
	if (result != CL_SUCCESS) \
	{ \
		fprintf(STD_OUT, "OpenCL Error %d: %s\n", result, opencl_error_string(result)); \
		ERRRET(__VA_ARGS__); \
	}

caldgemm_opencl::caldgemm_opencl() : caldgemm()
{
}

caldgemm_opencl::~caldgemm_opencl()
{
}

#define WAITFOREVENT(eventnr, devicenr) { if (Config->Debug) fprintf(STD_OUT, "\tWaiting for event from device %d obuffer %d...\n", devicenr, eventnr); if (clWaitForEvents(1, &ocl_events[devicenr][eventnr]) != CL_SUCCESS) { fprintf(STD_OUT, "Error while waiting for event\n"); return(1);}}
int caldgemm_opencl::WaitForEvent(int a, int b, int) {WAITFOREVENT(a, b);return(0);}

int caldgemm_opencl::Initialize(int deviceNum, bool nocalinit)
{
	fprintf(STD_OUT, "OPENCL Initialice\n");
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
	if (deviceNum < 0) nDevices = 1;
	gpu_available = (nDevices > 0);

	for (int i = 0;i < nDevices;i++)
	{
		if (deviceNum >= 0) ocl_devices[i] = devices[deviceNum];
		else ocl_devices[i] = devices[i];
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
	fprintf(STD_OUT, "OPENCL ValidateRuntime\n");

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
	fprintf(STD_OUT, "OPENCL CheckDevices\n");
	return(0);
}

int caldgemm_opencl::InitDevices()
{
	fprintf(STD_OUT, "OPENCL InitDevices\n");

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
		for (int j = 0;j < 2;j++)
		{
			ocl_abuffers[i][j] = clCreateImage2D(ocl_contexts[i], CL_MEM_READ_WRITE, &ocl_image_format, BufferHeight / 2, BufferWidth, 0, NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (A)");
		}

		for (int j = 0;j < obuffercount;j++)
		{
			ocl_cbuffers[i][j] = clCreateBuffer(ocl_contexts[i], CL_MEM_READ_WRITE, BufferHeight * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (C)");

			ocl_tmp_abuffers[i][j] = clCreateBuffer(ocl_contexts[i], CL_MEM_READ_ONLY, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (A tmp)");

			ocl_tmp_bbuffers[i][j] = clCreateBuffer(ocl_contexts[i], CL_MEM_READ_ONLY, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (B tmp)");
		}

		for (int j = 0;j < num_bbuffers;j++)
		{
			ocl_bbuffers[i][j] = clCreateImage2D(ocl_contexts[i], CL_MEM_READ_WRITE, &ocl_image_format, BufferHeight / 2, BufferWidth, 0, NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (B)");
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

			ocl_error = clBuildProgram(ocl_program[i][j], 1, &ocl_devices[i], "", NULL, NULL);
			if (ocl_error != CL_SUCCESS)
			{
				fprintf(STD_OUT, "OpenCL Error while building program: %d\n", ocl_error);
				fprintf(STD_OUT, "OpenCL Kernel:\n\n%s\n\n", sourceCode);
				char build_log[16384];
				clGetProgramBuildInfo(ocl_program[i][j], ocl_devices[i], CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL);
				fprintf(STD_OUT, "Build Log:\n\n%s\n\n", build_log);
				return(1);
			}

			ocl_kernel[i][j] = clCreateKernel(ocl_program[i][j], "oclkernel", &ocl_error);
			CHKRET(ocl_error, "Error creating kernel");
		}
	}

	return(0);
}

int caldgemm_opencl::ReinitDevices()
{
	fprintf(STD_OUT, "OPENCL ReinitDevices\n");
	return(0);
}

int caldgemm_opencl::InitConstantData(double alpha)
{
	fprintf(STD_OUT, "OPENCL InitConstantData\n");
	return(0);
}

int caldgemm_opencl::ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn)
{
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {Config->Height / 2, Config->Height, 1};
	CHKRET(clEnqueueReadBufferRect(ocl_command_queues[Task.device][Task.j], ocl_cbuffers[Task.device][Task.j], CL_FALSE, origin, origin, region, 0, 0, C_pitch, 0, C + blockm + blockn * Config->Height * C_pitch, 0, NULL, &ocl_events[Task.device][Task.j]), "Error retrieving C\n");
	clFlush(ocl_command_queues[Task.device][Task.j]);

	fprintf(STD_OUT, "OPENCL ExecuteKernels\n");
	return(0);
}

int caldgemm_opencl::ExitRuntime()
{
	fprintf(STD_OUT, "OPENCL ExitRuntime\n");

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

int caldgemm_opencl::FetchResult(int device, int j, int m, int n)
{
	fprintf(STD_OUT, "OPENCL FetchResult\n");
	return(0);
}

int caldgemm_opencl::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)
{
	fprintf(STD_OUT, "OPENCL RunMergeBuffers\n");
	return(0);
}

int caldgemm_opencl::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0)
{
	fprintf(STD_OUT, "OPENCL DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	size_t origin[3] = {0, 0, 0};
	size_t region[3];
	region[2] = 1;

	if (Config->VerboseTiming) Timers.CounterCopyTo.Start();

	if (prepareM)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
		Timers.divideA++;

		cl_mem *dest_image;
		cl_mem dest_buffer_tmp = ocl_tmp_abuffers[num_device][j];
		region[0] = (TransposeA ? Config->Height : Config->Width) / 2;
		region[1] = (TransposeA ? Config->Width : Config->Height);
		size_t pitch = A_pitch;
		void* src_ptr = A + blockm * Config->Height * (TransposeA ? 1 : A_pitch);
		
		if (!DGEMM_favor_m && buffersSufficiant0)
		{
			dest_image = &ocl_bbuffers[num_device][buffer_pointers_A[num_device][blockm] % (buffersSufficiant ? bbuffers[num_device] : 2)];
		}
		else
		{
			dest_image = &ocl_abuffers[num_device][next_buffer_A[num_device] % 2];
		}

		CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch, 0, src_ptr, 0, NULL, NULL), "Error copying A");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, A, 0");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), &dest_image), "Error setting kernel arg, A, 1");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &region[0]), "Error setting kernel arg, A, 2");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &region[1]), "Error setting kernel arg, A, 3");
		int transpose = TransposeA;
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &transpose), "Error setting kernel arg, A, 4");

		size_t local_size[2] = {16, 16};
		size_t global_size[2] = {256, 256};
		CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][j], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], 0, NULL, NULL), "Error starting conversion kernel for A");
	}

	if (prepareN)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tCopying part of B to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
		Timers.divideB++;

		cl_mem *dest_image;
		cl_mem dest_buffer_tmp = ocl_tmp_bbuffers[num_device][j];
		region[0] = (TransposeB ? Config->Width : Config->Height) / 2;
		region[1] = (TransposeB ? Config->Height : Config->Width);
		size_t pitch = B_pitch;
		void* src_ptr = B + blockn * Config->Height * (TransposeB ? B_pitch : 1);

		if (!DGEMM_favor_m && buffersSufficiant0)
		{
			dest_image = &ocl_abuffers[num_device][next_buffer_B[num_device] % 2];
		}
		else
		{
			dest_image = &ocl_bbuffers[num_device][buffersSufficiant ? (buffer_pointers_B[num_device][blockn] % bbuffers[num_device]) : (next_buffer_B[num_device] % 2)];
		}

		CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch, 0, src_ptr, 0, NULL, NULL), "Error copying B");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, B, 0");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), &dest_image), "Error setting kernel arg, B, 1");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &region[0]), "Error setting kernel arg, B, 2");
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &region[1]), "Error setting kernel arg, B, 3");
		int transpose = !TransposeB;
		CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &transpose), "Error setting kernel arg, B, 4");

		size_t local_size[2] = {16, 16};
		size_t global_size[2] = {256, 256};
		CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][j], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], 0, NULL, NULL), "Error starting conversion kernel for B");
	}

	region[0] = Config->Height / 2;
	region[1] = Config->Height;
	Timers.divideC++;
	CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], ocl_cbuffers[num_device][j], CL_FALSE, origin, origin, region, 0, 0, C_pitch, 0, C + blockm + blockn * Config->Height * C_pitch, 0, NULL, NULL), "Error copying C");
	clFlush(ocl_command_queues[num_device][j]);

	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[num_device][j]);
		Timers.CounterCopyTo.Stop();
	}

	return(0);
}

int caldgemm_opencl::ExitDevices()
{
	fprintf(STD_OUT, "OPENCL ExitDevices\n");

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			clReleaseMemObject(ocl_abuffers[i][j]);
		}
		for (int j = 0;j < obuffercount;j++)
		{
			clReleaseMemObject(ocl_cbuffers[i][j]);
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

int caldgemm_opencl::UseOutputPthreads() {return(0);}
int caldgemm_opencl::UseInputPthreads() {return(0);}
int caldgemm_opencl::UseMutexPerDevice() {return(0);}

int caldgemm_opencl::reserve_cpu_cores()
{

	return(0);
}
