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
#include "caldgemm_common.h"
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

#ifndef _WIN32
#include <dlfcn.h>
#endif

const char* caldgemm_opencl::OCLConvertKernel =
OCL_KERNEL_PRE
"__kernel void oclconkernel(__global const uint4* __restrict const iBuffer, __global uint4* const oBuffer, int width, int height, int transpose)\n"
"{\n"
"	int i, j;\n"
"	for (i = get_global_id(1);i < height / 2;i+=get_global_size(1))\n"
"	{\n"
"		for (j = get_global_id(0);j < width / 2;j+=get_global_size(0))\n"
"		{\n"
"			uint4 tmp, tmp2;\n"
"			tmp = iBuffer[(2 * i) * (width / 2) + j];\n"
"			tmp2 = iBuffer[(2 * i + 1) * (width / 2) + j];\n"
"			uint4 val, val2;\n"
"			if (transpose)\n"
"			{\n"
"				uint4 val, val2;\n"
"				val.x = tmp.x;val.y = tmp.y;\n"
"				val.z = tmp2.x;val.w = tmp2.y;\n"
"				val2.x = tmp.z;val2.y = tmp.w;\n"
"				val2.z = tmp2.z;val2.w = tmp2.w;\n"
"				oBuffer[(2 * j) * (height / 2) + i] = val;\n"
"				oBuffer[(2 * j + 1) * (height / 2) + i] = val2;\n"
"			}\n"
"			else\n"
"			{\n"
"				oBuffer[(2 * i) * (width / 2) + j] = tmp;\n"
"				oBuffer[(2 * i + 1) * (width / 2) + j] = tmp2;\n"
"			}\n"
"		}\n"
"	}\n"
"}\n"
;

const char* caldgemm_opencl::OCLConvertKernelTex =
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

caldgemm::caldgemm_config_backend* caldgemm_opencl::create_caldgemm_config_backend()
{
	return(new caldgemm_config_backend_opencl);
}

caldgemm_opencl::caldgemm_config_backend_opencl::~caldgemm_config_backend_opencl() {}

int caldgemm_opencl::WaitForEventAndRelease(cl_event* pEvent, int lock)
{
	cl_int ocl_error;
	if (lock == -1) lock = (Config->ThreadSaveDriver == -1);
	if (Config->Debug) fprintf(STD_OUT, "\t\t\tOpenCL WaitForEventAndRelease: 0x%p\n", pEvent);
	if (lock)
	{
		cl_int status;
		do
		{
			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			if (clGetEventInfo(*pEvent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL) != CL_SUCCESS)
			{
				fprintf(STD_OUT, "Error querying event info\n");
				return(1);
			}
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
		} while (status != CL_COMPLETE);
	}
	else
	{
		if ((ocl_error = clWaitForEvents(1, pEvent)) != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error while waiting for event (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(1);
		}
	}
	if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
	if ((ocl_error = clReleaseEvent(*pEvent)) != CL_SUCCESS)
	{
		fprintf(STD_OUT, "Error releasing event (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
		return(1);
	}
	if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
	return(0);
}

int caldgemm_opencl::WaitForEvent(int a, int b, int mustlock)
{
	if (Config->Debug) fprintf(STD_OUT, "\tWaiting for event from device %d obuffer %d...\n", b, a);
	return(WaitForEventAndRelease(&ocl_events[b][a], mustlock || Config->ThreadSaveDriver == -1));
}

int caldgemm_opencl::Initialize(bool nocalinit)
{
	int deviceNum = Config->DeviceNum;
	if (!Config->Quiet) fprintf(STD_OUT, "Initializing CALDGEMM (OpenCL Runtime)\n");
	if (Config->Debug) fprintf(STD_OUT, "OPENCL Initialice\n");
	cl_int ocl_error;

	cl_uint num_devices;
	clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	nDevices = num_devices;
	if (nDevices == 0) ERRRET("No OpenCL device for this platform found\n");
	if (Config->Debug) fprintf(STD_OUT, "%d OpenCL devices found for this platform\n", nDevices);

	cl_device_id* devices = new cl_device_id[nDevices];
	if (devices == NULL) ERRRET("Memory allocation error\n");
	CHKRET(clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, nDevices, devices, NULL), "Error getting OpenCL devices\n");

	int gooddevices = 0;
	int cpu_found = 0;
	cl_device_id cpu_device;
	for (int i = 0;i < nDevices;i++)
	{
		char device_vendor[64], device_name[64];
		cl_device_type device_type;
		cl_uint nbits;
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, device_name, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, device_vendor, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, NULL);
		int device_ok = (device_type & CL_DEVICE_TYPE_GPU) && !(device_type & CL_DEVICE_TYPE_CPU);
		if (Config->Debug) fprintf(STD_OUT, "Device %d -> %d: %s %s (%d bits)\n", i, device_ok ? gooddevices : -1, device_vendor, device_name, nbits);
		if (device_ok)
		{
			devices[gooddevices++] = devices[i];
		}
		if (device_type & CL_DEVICE_TYPE_CPU)
		{
			cpu_found = 1;
			cpu_device = devices[i];
		}
	}

	if (cpu_found == 0)
	{
		ERRRET("No CPU OpenCL device found for mapping buffers\n");
	}
	if (gooddevices == 0)
	{
		ERRRET("No OpenCL GPU device found\n");
	}

	nDevices = gooddevices;
	if (nDevices > (signed) max_devices) nDevices = max_devices;
	if (nDevices > Config->NumDevices) nDevices = Config->NumDevices;

	if (deviceNum >= nDevices) ERRRET("OpenCL Device %d not available\n", deviceNum);
	if (deviceNum >= 0) nDevices = 1;
	gpu_available = (nDevices > 0);

	if (nDevices > 1 && !Config->MultiThread)
	{
		fprintf(STD_OUT, "Cannot use multiple devices without multithreading\n");
		nDevices = 1;
	}

	for (int i = 0;i < nDevices;i++)
	{
		if (deviceNum >= 0) ocl_devices[i] = devices[deviceNum];
		else ocl_devices[i] = devices[Config->DeviceNums[i]];
	}
	ocl_devices[nDevices] = cpu_device;
	delete[] devices;

	ocl_context = clCreateContext(NULL, nDevices + 1, ocl_devices, NULL, NULL, &ocl_error);
	CHKRET(ocl_error, "Error creating OpenCL context");

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < obuffercount;j++)
		{
			ocl_command_queues[i][j] = clCreateCommandQueue(ocl_context, ocl_devices[i], 0, &ocl_error);
			CHKRET(ocl_error, "Error creating OpenCL command queue");
		}
	}

	ocl_command_queue_cpu = clCreateCommandQueue(ocl_context, ocl_devices[nDevices], 0, &ocl_error);
	CHKRET(ocl_error, "Error creating OpenCL CPU command queue");

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

	if (Config->OpenCLPlatform >= (signed)num_platforms) ERRRET("OpenCL Platform %d not available\n", Config->OpenCLPlatform);
	ocl_platform = platforms[Config->OpenCLPlatform];
	delete[] platforms;

	if (Config->config_backend == NULL) Config->config_backend = new caldgemm_config_backend_opencl;
	config_backend = (caldgemm_config_backend_opencl*) Config->config_backend;

	if (config_backend->kernelLib != NULL)
	{
		if (Config->PrintILKernel)
		{
			fprintf(STD_OUT, "Cannot print kernel from 3ed party library\n");
		}

#ifdef _WIN32
		kernelLib = LoadLibrary(config_backend->kernelLib);
#else
		kernelLib = dlopen(config_backend->kernelLib, RTLD_LAZY|RTLD_GLOBAL);
#endif
		if (kernelLib == NULL)
		{
			fprintf(STD_OUT, "Error opening kernel library: %s\n", config_backend->kernelLib);
			return(1);
		}
#ifdef _WIN32
		kernelLibCreate = (cl_kernel (*) (cl_context*, int, cl_device_id*, int, int, int)) GetProcAddress(kernelLib, "kernelLibCreate");
		kernelLibQuerySettings = (void (*) (int*, int*, bool*, bool*, bool*, int*, int*)) GetProcAddress(kernelLib, "kernelLibQuerySettings");
#else
		kernelLibCreate = (cl_kernel (*)(cl_context*, int, cl_device_id*, int, int, int)) dlsym(kernelLib, "kernelLibCreate");
		kernelLibQuerySettings = (void (*) (int*, int*, bool*, bool*, bool*, int*, int*)) dlsym(kernelLib, "kernelLibQuerySettings");
#endif
		if (kernelLibCreate == NULL || kernelLibQuerySettings == NULL)
		{
			fprintf(STD_OUT, "Error getting function pointer from external library\n");
			return(1);
		}

		kernelLibQuerySettings(&KernelSettings.tiling_x, &KernelSettings.tiling_y, &KernelSettings.transposeA, &KernelSettings.transposeB, &KernelSettings.texture_buffers, &KernelSettings.group_size_x, &KernelSettings.group_size_y);
	}
	else
	{
		//Do not set default kernel settings
#ifdef CALDGEMM_TRANSPOSED_A
		KernelSettings.transposeA = true;
#else
		KernelSettings.transposeA = false;
#endif
#ifdef CALDGEMM_TRANSPOSED_B
		KernelSettings.transposeB = true;
#else
		KernelSettings.transposeB = false;
#endif
#ifdef OCL_USE_SIMPLE_BUFFERS
		KernelSettings.texture_buffers = false;
#else
		KernelSettings.texture_buffers = true;
#endif
		KernelSettings.tiling_x = OCL_TILING_X;
		KernelSettings.tiling_y = OCL_TILING_Y;
		KernelSettings.group_size_x = OCL_GROUP_SIZE_X;
		KernelSettings.group_size_y = OCL_GROUP_SIZE_Y;
		if (!(KernelSettings.transposeA ^ KernelSettings.transposeB))
		{
			fprintf(STD_OUT, "Must set either transposed A or transposed B\n");
			return(1);
		}
	}

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
			if (!KernelSettings.texture_buffers)
			{
				ocl_abuffers[i][j] = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			}
			else
			{
				cl_image_desc image_desc;
				memset(&image_desc, 0, sizeof(image_desc));
				image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
				if (KernelSettings.transposeB)
				{
					image_desc.image_width = BufferWidth / 2;
					image_desc.image_height = BufferHeight;
					ocl_abuffers[i][j] = clCreateImage(ocl_context, CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
				}
				else //must be transposeA
				{
					image_desc.image_width = BufferHeight / 2;
					image_desc.image_height = BufferWidth;
					ocl_abuffers[i][j] = clCreateImage(ocl_context, CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
				}
			}
			CHKRET(ocl_error, "Error allocating device memory (A)");
		}
		CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], ibuffercount, &ocl_abuffers[i][0], 0, 0, NULL, NULL), "Error migrating mem object");

		for (int j = 0;j < obuffercount;j++)
		{
			if (Config->DstMemory == 'g')
			{
				ocl_cbuffers[i][j] = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, BufferHeight * BufferHeight * sizeof(double), NULL, &ocl_error);
				CHKRET(ocl_error, "Error allocating device memory (C)");
				CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], 1, &ocl_cbuffers[i][j], 0, 0, NULL, NULL), "Error migrating mem object");
			}
			if (Config->GPU_C == 0)
			{
				ocl_tmp_cbuffers[i][j] = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BufferHeight * BufferHeight * sizeof(double), NULL, &ocl_error);
				CHKRET(ocl_error, "Error allocating host memory (C tmp)");

				ocl_tmp_cbuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_cbuffers[i][j], CL_TRUE, CL_MAP_READ, 0, BufferHeight * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping host memory (C tmp)");
				memset(ocl_tmp_cbuffers_ptr[i][j], 0, BufferHeight * BufferHeight * sizeof(double));
				
				if (Config->DstMemory == 'g')
				{
					CHKRET(clEnqueueWriteBuffer(ocl_command_queues[i][0], ocl_cbuffers[i][j], CL_TRUE, 0, BufferHeight * BufferHeight * sizeof(double), ocl_tmp_cbuffers_ptr[i][j], 0, NULL, NULL), "Error initializing GPU buffer with zero");
				}
			}
		}

		for (int j = 0;j < (Config->GPU_C ? obuffercount : ibuffercount);j++)
		{
			cl_int tmp_flags = Config->GPU_C ? CL_MEM_READ_ONLY : (CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE);
			ocl_tmp_abuffers[i][j] = clCreateBuffer(ocl_context, tmp_flags, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (A tmp - Width: %lld Height: %lld)", (long long int) BufferWidth, (long long int) BufferHeight);

			ocl_tmp_bbuffers[i][j] = clCreateBuffer(ocl_context, tmp_flags, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (B tmp)");

			if (Config->GPU_C == 0)
			{
				ocl_tmp_abuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_abuffers[i][j], CL_TRUE, CL_MAP_WRITE, 0, BufferWidth * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping buffer (A)");

				ocl_tmp_bbuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_bbuffers[i][j], CL_TRUE, CL_MAP_WRITE, 0, BufferWidth * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping buffer (B)");
			}
			else
			{
				CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], 1, &ocl_tmp_abuffers[i][j], 0, 0, NULL, NULL), "Error migrating mem object");
				CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], 1, &ocl_tmp_bbuffers[i][j], 0, 0, NULL, NULL), "Error migrating mem object");
			}
		}

		for (int j = 0;j < num_bbuffers;j++)
		{
			if (!KernelSettings.texture_buffers)
			{
				ocl_bbuffers[i][j] = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			}
			else
			{
				cl_image_desc image_desc;
				memset(&image_desc, 0, sizeof(image_desc));
				image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
				if (KernelSettings.transposeB)
				{
					image_desc.image_width = BufferWidth / 2;
					image_desc.image_height = BufferHeight;
					ocl_bbuffers[i][j] = clCreateImage(ocl_context, CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
				}
				else //must be transposeA
				{
					image_desc.image_width = BufferHeight / 2;
					image_desc.image_height = BufferWidth;
					ocl_bbuffers[i][j] = clCreateImage(ocl_context, CL_MEM_READ_WRITE, &ocl_image_format, &image_desc, NULL, &ocl_error);
				}
			}
			if (ocl_error != CL_SUCCESS)
			{
				if (j < obuffercount)
				{
					CHKRET(ocl_error, "Error allocating device memory (B)");
				}
				else break;
			}
			CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], 1, &ocl_bbuffers[i][j], 0, 0, NULL, NULL), "Error migrating mem object");
			
			bbuffers[i] = j + 1;
		}
		if (Config->Debug) fprintf(STD_OUT, "Allocated %d BBuffers on Device %d\n", bbuffers[i], i);
	}

	for (int j = 0;j < 3 + 1 + (Config->GPU_C ? 1 : 0);j++)
	{
		if (j != 3 && config_backend->kernelLib != NULL)
		{
			if (Config->PrintILKernel)
			{
				fprintf(STD_OUT, "Cannot print kernel from 3ed party library\n");
			}

			for (int i = 0;i < nDevices;i++)
			{
				ocl_kernel[i][j] = kernelLibCreate(&ocl_context, nDevices, ocl_devices, j, Config->Width, Config->GPU_C == 0);
				if (ocl_kernel[i][j] == 0)
				{
					fprintf(STD_OUT, "Error obtaining kernel from external library\n");
					return(1);
				}
			}
		}
		else
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
					sourceCode = KernelSettings.texture_buffers ? OCLConvertKernelTex : OCLConvertKernel;
					break;
				case 4:
					sourceCode = OCLKernel;
					break;
			}

			if (Config->PrintILKernel)
			{
				fprintf(STD_OUT, "OpenCL Kernel %d:\n%s\n\n", j, sourceCode);
			}

			ocl_program[j] = clCreateProgramWithSource(ocl_context, 1, &sourceCode, NULL, &ocl_error);
			CHKRET(ocl_error, "Error creating program object");

			ocl_error = clBuildProgram(ocl_program[j], nDevices, ocl_devices, Config->Disassemble ? "-save-temps=." : "", NULL, NULL);
			if (ocl_error != CL_SUCCESS)
			{
				fprintf(STD_OUT, "OpenCL Error while building program: %d\n", ocl_error);
				fprintf(STD_OUT, "OpenCL Kernel:\n\n%s\n\n", sourceCode);
				char build_log[16384];
				for (int i = 0;i < nDevices;i++)
				{
					clGetProgramBuildInfo(ocl_program[j], ocl_devices[i], CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL);
					fprintf(STD_OUT, "Build Log (device %d):\n\n%s\n\n", i, build_log);
				}
				return(1);
			}

			for (int i = 0;i < nDevices;i++)
			{
				ocl_kernel[i][j] = clCreateKernel(ocl_program[j], j == 3 ? "oclconkernel" : "oclkernel", &ocl_error);
				CHKRET(ocl_error, "Error creating kernel");
			}
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

	if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
	if (Config->Debug) fprintf(STD_OUT, "\tExecuting MM kernel (device %d obuffer %d, k=%lld m=%lld n=%lld)\n", Task.device, Task.j, (long long int) Task.k, (long long int) blockm, (long long int) blockn);
	if (Config->GPU_C && Task.kernel_num == 2) Task.kernel_num = 4;
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

	int pitch;
	cl_ulong offset; //uling should always be 64-bit in OpenCL
	int height1 = (int) (((size_t) blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height);
	int height2 = (int) (((size_t) blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height);
	if (Config->DstMemory == 'g')
	{
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 0, sizeof(cl_mem), &ocl_cbuffers[Task.device][Task.j]), "Error setting kernel memory C");
		pitch = height1;
		offset = 0;
	}
	else if (Config->GPU_C == 0)
	{
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 0, sizeof(cl_mem), &ocl_tmp_cbuffers[Task.device][Task.j]), "Error setting kernel memory C");
		pitch = height1;
		offset = 0;
	}
	else
	{
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 0, sizeof(cl_mem), C_matrix_base_obj), "Error setting kernel memory C");
		pitch = C_pitch;
		offset = (C + blockn * Config->Height + blockm * Config->Height * C_pitch) - C_matrix_base;
	}
	double beta = Config->GPU_C ? Beta : 0.;
	double alpha = Task.kernel_num == 2 ? 1. : Alpha;

	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 3, sizeof(int), &height1), "Error setting kernel arg height1");
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 4, sizeof(int), &height2), "Error setting kernel arg height2");

	int width = Config->Width;
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 5, sizeof(int), &width), "Error setting kernel arg width");

	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 6, sizeof(double), &alpha), "Error setting kernel arg alpha");
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 7, sizeof(double), &beta), "Error setting kernel arg beta");

	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 8, sizeof(int), &pitch), "Error setting kernel arg pitch");
	CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 9, sizeof(cl_ulong), &offset), "Error setting kernel arg offset");

	size_t local_size[2] = {(size_t) KernelSettings.group_size_x, (size_t) KernelSettings.group_size_y};
	size_t global_size[2] = {(size_t) height1 / KernelSettings.tiling_x, (size_t) height2 / KernelSettings.tiling_y};

	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[Task.device][Task.j]);
		Timers.Kernel.Start();
	}
	if (Config->Debug) fprintf(STD_OUT, "MM Kernel: height1 %d height2 %d width %d alpha %f beta %f pitch %d offset %lld\n", height1, height2, width, Alpha, Beta, pitch, (long long int) offset);
	cl_event* kernel_event;
	if (Config->DstMemory == 'g' && (Config->GPU_C || Config->ImplicitDriverSync)) kernel_event = NULL;
	else kernel_event = &ocl_events[Task.device][Task.j];

	CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[Task.device][Task.j], ocl_kernel[Task.device][Task.kernel_num], 2, NULL, &global_size[0], &local_size[0], 0, NULL, kernel_event), "Error starting MM Kernel");
	if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
	if (Config->VerboseTiming)
	{
		clFinish(ocl_command_queues[Task.device][Task.j]);
		Timers.Kernel.Stop();
		Timers.CounterCopyFrom.Start();
	}

	if (Config->DstMemory == 'g')
	{
		if (Config->GPU_C)
		{
			size_t origin[3] = {0, 0, 0};
			size_t region[3] = {(size_t) height1 * sizeof(double), (size_t) height2, 1};
			if (Config->Debug) fprintf(STD_OUT, "Transfer C from GPU: region %d x %d\n", (int) region[0], (int) region[1]);
			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			CHKRET(clEnqueueReadBufferRect(ocl_command_queues[Task.device][Task.j], ocl_cbuffers[Task.device][Task.j], CL_FALSE, origin, origin, region, 0, 0, C_pitch * sizeof(double), 0, C + blockn * Config->Height + blockm * Config->Height * C_pitch, 0, NULL, &ocl_events[Task.device][Task.j]), "Error retrieving C\n");
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
		}
		else if (Config->ImplicitDriverSync)
		{
			if (FetchResult(Task.device, Task.j, blockm, blockn)) {fprintf(STD_OUT, "Error copying from GPU\n"); return(1);}
		}
	}
	clFlush(ocl_command_queues[Task.device][Task.j]);
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
	}
	clReleaseCommandQueue(ocl_command_queue_cpu);
	clReleaseContext(ocl_context);

	return(0);
}

int caldgemm_opencl::FetchResult(int device, int j, int m, int n, int mustlock)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL FetchResult\n");
	if (Config->GPU_C == 0 && Config->DstMemory == 'g')
	{
		if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
		clEnqueueCopyBuffer(ocl_command_queues[device][j], ocl_cbuffers[device][j], ocl_tmp_cbuffers[device][j], 0, 0, Config->Height * Config->Height * sizeof(double), 0, NULL, &ocl_events[device][j]);
		clFlush(ocl_command_queues[device][j]);
		if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
		if (Config->VerboseTiming) clFinish(ocl_command_queues[device][j]);
	}
	return(0);
}

int caldgemm_opencl::CheckDMAQueue(int device, int forcej)
{
	return(0);
}

int caldgemm_opencl::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)	//gpu_* is not used, it is only relevant for partially used buffers in the CAL runtime!
{
	if (Config->GPU_C) return(0);
	if (Config->Debug) fprintf(STD_OUT, "OPENCL RunMergeBuffers (%d x %d) (%d x %d) (%d)\n", height, width, gpu_height, gpu_width, pitch);
	if (Config->SkipCPUProcessing) return(0);
	double* src = ocl_tmp_cbuffers_ptr[device][j];
	const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double
	const unsigned long long int double_minus_one = 0xBFF0000000000000;
#ifndef _NO_AVX
	if (reinterpret_cast<unsigned long long int &>(Beta) == double_one && reinterpret_cast<unsigned long long int &>(Alpha) == double_minus_one && (Config->ForceKernelVariant == -1 || Config->ForceKernelVariant == 2))
	{
		for (int i = 0;i < height;i++)
		{
			const double* __restrict__ saddr = &src[i * width];
			double* __restrict__ daddr = &dst[i * pitch];
			const int count = width / 16; //(4 unroll, 4 avx)
			for (int j = 0;j < count;j++)
			{
//				_mm_prefetch(CAST_FOR_MMPREFETCH (daddr + pitch), _MM_HINT_T0);
				_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + width), _MM_HINT_T0);;
				_mm256_store_pd(daddr, _mm256_sub_pd(_mm256_load_pd(daddr), _mm256_load_pd(saddr)));
				_mm256_store_pd(daddr + 4, _mm256_sub_pd(_mm256_load_pd(daddr + 4), _mm256_load_pd(saddr + 4)));
				_mm256_store_pd(daddr + 8, _mm256_sub_pd(_mm256_load_pd(daddr + 8), _mm256_load_pd(saddr + 8)));
				_mm256_store_pd(daddr + 12, _mm256_sub_pd(_mm256_load_pd(daddr + 12), _mm256_load_pd(saddr + 12)));
				daddr += 16;
				saddr += 16;
			}
		}
	}
	else
#endif
	{
		double my_alpha = (reinterpret_cast<unsigned long long int &>(Beta) == double_one && reinterpret_cast<unsigned long long int &>(Alpha) == double_minus_one && (Config->ForceKernelVariant == -1 || Config->ForceKernelVariant == 2)) ? -1. : 1.;
		for (int i = 0;i < height;i++)
		{
			for (int j = 0;j < width;j++)
			{
				dst[j] = Beta * dst[j] + my_alpha * src[j];
			}
			src += width;
			dst += pitch;
		}
	}
	return(0);
}

int caldgemm_opencl::divideBuffer(double* src, size_t pitch_src, double* dest, size_t nSrcRows, size_t nSrcCols, bool transpose)
{
	if (Config->GPU_C) return(0);
	if (Config->Debug) fprintf(STD_OUT, "OPENCL divideBuffers (%lld x %lld) (%lld)\n", (long long int) nSrcCols, (long long int) nSrcRows, (long long int) pitch_src);
	if (Config->SkipCPUProcessing) return(0);
	
	if (transpose)
	{
		for (unsigned int i = 0;i < nSrcRows;i += 8)
		{
			const double* __restrict__ saddr = &src[i * pitch_src];
			double* __restrict__ daddr = dest + i;
			for (unsigned int j = 0;j < nSrcCols / 2;j++)
			{
				__m128d x1 = _mm_load_pd(saddr);
				__m128d x2 = _mm_load_pd(saddr + pitch_src);
				__m128d x3 = _mm_load_pd(saddr + 2 * pitch_src);
				__m128d x4 = _mm_load_pd(saddr + 3 * pitch_src);
				__m128d x5 = _mm_load_pd(saddr + 4 * pitch_src);
				__m128d x6 = _mm_load_pd(saddr + 5 * pitch_src);
				__m128d x7 = _mm_load_pd(saddr + 6 * pitch_src);
				__m128d x8 = _mm_load_pd(saddr + 7 * pitch_src);
				_mm_stream_pd(daddr, _mm_unpacklo_pd(x1, x2));
				_mm_stream_pd(daddr + 2, _mm_unpacklo_pd(x3, x4));
				_mm_stream_pd(daddr + 4, _mm_unpacklo_pd(x5, x6));
				_mm_stream_pd(daddr + 6, _mm_unpacklo_pd(x7, x8));
				_mm_stream_pd(daddr + nSrcRows, _mm_unpackhi_pd(x1, x2));
				_mm_stream_pd(daddr + 2 + nSrcRows, _mm_unpackhi_pd(x3, x4));
				_mm_stream_pd(daddr + 4 + nSrcRows, _mm_unpackhi_pd(x5, x6));
				_mm_stream_pd(daddr + 6 + nSrcRows, _mm_unpackhi_pd(x7, x8));
				saddr += 2;
				daddr += 2 * nSrcRows;
			}
		}
	}
	else
	{
		for (unsigned int i = 0;i < nSrcRows;i++)
		{
			for (unsigned int j = 0;j < nSrcCols;j+=2)
			{
				_mm_stream_pd(dest + j, _mm_load_pd(src + j));
			}
			dest += nSrcCols;
			src += pitch_src;
		}
	}
	/*for (size_t i = 0;i < nSrcRows;i++)
	{
		for (size_t j = 0;j < nSrcCols;j++)
		{
			if (transpose)
			{
				dest[j * nSrcRows] = src[j];
			}
			else
			{
				dest[j] = src[j];
			}
		}
		if (transpose)
		{
			dest++;
		}
		else
		{
			dest += nSrcCols;
		}
		src += pitch_src;
	}*/
	return(0);
}

int caldgemm_opencl::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	const size_t origin[3] = {0, 0, 0};
	size_t region[3];
	region[2] = 1;
	const size_t HeightM = ((blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height);
	const size_t HeightN = ((blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height);

	if (Config->VerboseTiming && Config->GPU_C == 1) Timers.CounterCopyTo.Start();

	if (ocl_conversion_events_use[num_device][0])
	{
		WaitForEventAndRelease(&ocl_conversion_events[num_device][0]);
		ocl_conversion_events_use[num_device][0] = 0;
	}
	if (prepareM)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
		Timers.divideA++;

		int dest_image_id;
		cl_mem *dest_image;
		size_t pitch = A_pitch;
		void* src_ptr = A + blockm * Config->Height * (TransposeA ? 1 : A_pitch);
		
		if (!DGEMM_favor_m && buffersSufficiant0)
		{
			dest_image_id = buffer_pointers_A[num_device][blockm] % (buffersSufficiant ? bbuffers[num_device] : ibuffercount);
			dest_image = &ocl_bbuffers[num_device][dest_image_id];
		}
		else
		{
			dest_image_id = next_buffer_A[num_device] % ibuffercount;
			dest_image = &ocl_abuffers[num_device][dest_image_id];
		}

		const int arg_transpose = TransposeA ^ KernelSettings.transposeA;

		if (Config->GPU_C == 0)
		{
			if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer A (device = %d, k = %lld, context = %d, m = %lld, n = %lld, buffer = %d, transpose = %d)\n", num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn, next_buffer_A[num_device] % ibuffercount, arg_transpose);
			if (Config->VerboseTiming) Timers.CounterDivide.Start();
			if (divideBuffer((double*) src_ptr, pitch, ocl_tmp_abuffers_ptr[num_device][next_buffer_A[num_device] % ibuffercount], TransposeA ? Config->Width : HeightM, TransposeA ? HeightM : Config->Width, arg_transpose)) return(1);
			if (Config->VerboseTiming) Timers.CounterDivide.Stop();
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (device = %d, k = %lld, context = %d, m = %lld, n = %lld, buffer: %d->%d)\n", num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn, next_buffer_A[num_device] % ibuffercount, dest_image_id);
			if (KernelSettings.transposeA)
			{
				region[0] = Config->Width / 2;
				region[1] = HeightM;
			}
			else //must be transposeB
			{
				region[0] = HeightM / 2;
				region[1] = Config->Width;
			}

			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			if (Config->VerboseTiming) Timers.CounterCopyTo.Start();
			if (!KernelSettings.texture_buffers)
			{
				CHKRET(clEnqueueWriteBuffer(ocl_command_queues[num_device][j], *dest_image, CL_FALSE, 0, Config->Width * HeightM * sizeof(double), ocl_tmp_abuffers_ptr[num_device][next_buffer_A[num_device] % ibuffercount], 0, NULL, &ocl_conversion_events[num_device][0]), "Error copying A");
			}
			else
			{
				CHKRET(clEnqueueWriteImage(ocl_command_queues[num_device][j], *dest_image, CL_FALSE, origin, region, 0, 0, ocl_tmp_abuffers_ptr[num_device][next_buffer_A[num_device] % ibuffercount], 0, NULL, &ocl_conversion_events[num_device][0]), "Error copying A");
			}
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
			if (Config->VerboseTiming)
			{
				clFinish(ocl_command_queues[num_device][j]);
				Timers.CounterCopyTo.Stop();
			}
		}
		else
		{
			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			region[0] = (TransposeA ? HeightM : Config->Width) * sizeof(double);
			region[1] = (TransposeA ? Config->Width : HeightM);
			int arg_width = region[0] / sizeof(double), arg_height = region[1];
			cl_mem dest_buffer_tmp = ocl_tmp_abuffers[num_device][j];
			if (Config->Debug) fprintf(STD_OUT, "Transfer A to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
			CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch * sizeof(double), 0, src_ptr, 0, NULL, NULL), "Error copying A");
			if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, A, 0");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), dest_image), "Error setting kernel arg, A, 1");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &arg_width), "Error setting kernel arg, A, 2");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &arg_height), "Error setting kernel arg, A, 3");
		
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, A, 4");

			size_t local_size[2] = {GROUP_SIZE_X, GROUP_SIZE_Y};
			size_t global_size[2] = {GROUP_SIZE_X * GROUP_COUNT_X, GROUP_SIZE_Y * GROUP_COUNT_Y};
			if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel A: x %d y %d (t: %d)\n", arg_width, arg_height, arg_transpose);
			CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][j], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], 0, NULL, &ocl_conversion_events[num_device][0]), "Error starting conversion kernel for A");
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
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

		int dest_image_id;
		cl_mem *dest_image;
		
		size_t pitch = B_pitch;
		void* src_ptr = B + blockn * Config->Height * (TransposeB ? B_pitch : 1);

		if (!DGEMM_favor_m && buffersSufficiant0)
		{
			dest_image_id = next_buffer_B[num_device] % ibuffercount;
			dest_image = &ocl_abuffers[num_device][dest_image_id];
		}
		else
		{
			dest_image_id = buffersSufficiant ? (buffer_pointers_B[num_device][blockn] % bbuffers[num_device]) : (next_buffer_B[num_device] % ibuffercount);
			dest_image = &ocl_bbuffers[num_device][dest_image_id];
		}

		const int arg_transpose = TransposeB ^ KernelSettings.transposeB;

		if (Config->GPU_C == 0)
		{
			if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer B (device = %d, k = %lld, context = %d, m = %lld, n = %lld, buffer = %d, transpose = %d)\n", num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn, next_buffer_B[num_device] % ibuffercount, arg_transpose);
			if (Config->VerboseTiming) Timers.CounterDivide.Start();

			if (divideBuffer((double*) src_ptr, pitch, ocl_tmp_bbuffers_ptr[num_device][next_buffer_B[num_device] % ibuffercount], TransposeB ? HeightN : Config->Width, TransposeB ? Config->Width : HeightN, arg_transpose)) return(1);
			if (Config->VerboseTiming) Timers.CounterDivide.Stop();
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of B to GPU (device = %d, k = %lld, context = %d, m = %lld, n = %lld, buffer: %d->%d)\n", num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn, next_buffer_B[num_device] % ibuffercount, dest_image_id);

			if (KernelSettings.transposeB)
			{
				region[0] = HeightN / 2;
				region[1] = Config->Width;
			}
			else //must be transposeA
			{
				region[0] = Config->Width / 2;
				region[1] = HeightN;
			}
			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			if (Config->VerboseTiming) Timers.CounterCopyTo.Start();
			if (!KernelSettings.texture_buffers)
			{
				CHKRET(clEnqueueWriteBuffer(ocl_command_queues[num_device][j], *dest_image, CL_FALSE, 0, Config->Width * HeightN * sizeof(double), ocl_tmp_bbuffers_ptr[num_device][next_buffer_B[num_device] % ibuffercount], 0, NULL, &ocl_conversion_events[num_device][1]), "Error copying B");
			}
			else
			{
				CHKRET(clEnqueueWriteImage(ocl_command_queues[num_device][j], *dest_image, CL_FALSE, origin, region, 0, 0, ocl_tmp_bbuffers_ptr[num_device][next_buffer_B[num_device] % ibuffercount], 0, NULL, &ocl_conversion_events[num_device][1]), "Error copying B");
			}
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
			if (Config->VerboseTiming)
			{
				clFinish(ocl_command_queues[num_device][j]);
				Timers.CounterCopyTo.Stop();
			}
		}
		else
		{
			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			region[0] = (TransposeB ? Config->Width : HeightN) * sizeof(double);
			region[1] = (TransposeB ? HeightN : Config->Width);
			int arg_width = region[0] / sizeof(double), arg_height = region[1];
			cl_mem dest_buffer_tmp = ocl_tmp_bbuffers[num_device][j];
			if (Config->Debug) fprintf(STD_OUT, "Transfer B to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
			CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch * sizeof(double), 0, src_ptr, 0, NULL, NULL), "Error copying B");
			if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, B, 0");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), dest_image), "Error setting kernel arg, B, 1");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &arg_width), "Error setting kernel arg, B, 2");
			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &arg_height), "Error setting kernel arg, B, 3");

			CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, B, 4");

			size_t local_size[2] = {GROUP_SIZE_X, GROUP_SIZE_Y};
			size_t global_size[2] = {GROUP_SIZE_X * GROUP_COUNT_X, GROUP_SIZE_Y * GROUP_COUNT_Y};
			if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel B: x %d y %d\n", (int) arg_width, (int) arg_height);
			CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][j], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], 0, NULL, &ocl_conversion_events[num_device][1]), "Error starting conversion kernel for B");
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
		}
		ocl_conversion_events_use[num_device][1] = 1;
		if (Config->Debug && Config->VerboseTiming) clFinish(ocl_command_queues[num_device][j]);
	}

	if (Config->GPU_C && Config->DstMemory == 'g')
	{
		Timers.divideC++;
		region[0] = HeightN * sizeof(double);
		region[1] = HeightM;
		if (Config->Debug) fprintf(STD_OUT, "Transfer C to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
		if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
		CHKRET(clEnqueueWriteBufferRect(ocl_command_queues[num_device][j], ocl_cbuffers[num_device][j], CL_FALSE, origin, origin, region, 0, 0, C_pitch * sizeof(double), 0, C + blockn * Config->Height + blockm * Config->Height * C_pitch, 0, NULL, NULL), "Error copying C");
		clFlush(ocl_command_queues[num_device][j]);
		if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
	}
	if (Config->VerboseTiming && Config->GPU_C == 1)
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
			if (Config->DstMemory == 'g') clReleaseMemObject(ocl_cbuffers[i][j]);
			if (Config->GPU_C == 0)
			{
				clEnqueueUnmapMemObject(ocl_command_queues[i][0], ocl_tmp_cbuffers[i][j], ocl_tmp_cbuffers_ptr[i][j], 0, NULL, NULL);
				clFinish(ocl_command_queues[i][0]);
				clReleaseMemObject(ocl_tmp_cbuffers[i][j]);
			}
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
		for (int j = 0;j < 3 + 1 + (Config->GPU_C ? 1 : 0);j++)
		{
			clReleaseKernel(ocl_kernel[i][j]);
			if (config_backend->kernelLib == NULL && i == nDevices - 1) clReleaseProgram(ocl_program[j]);
		}
	}
	if (config_backend->kernelLib)
	{
#ifdef _WIN32
		FreeLibrary(kernelLib);
#else
		dlclose(kernelLib);
#endif
	}
	return(0);
}

int caldgemm_opencl::UseOutputPthreads() {return(!Config->GPU_C);}
int caldgemm_opencl::UseInputPthreads() {return(!Config->GPU_C);}
int caldgemm_opencl::UseMutexPerDevice() {return(0);}

int caldgemm_opencl::RunCALDGEMM_Init()
{
	if (Config->GPU_C && Config->DstMemory == 'c' && C_matrix_base == NULL)
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
static caldgemm_opencl::gpu_mem_struct_opencl gpu_mem[MAX_GPU_MEM_COUNT];
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
		gpu_mem[nGPUMEM].mem_obj = clCreateBuffer(ocl_context, mem_flags, nDoubles * sizeof(double), NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error allocating memory (clCreateBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(0);
		}

		gpu_mem[nGPUMEM].ptr = clEnqueueMapBuffer(ocl_command_queue_cpu, gpu_mem[nGPUMEM].mem_obj, CL_TRUE, 0, 0, nDoubles * sizeof(double), 0, NULL, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error allocating memory (clEnqueueMapBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(0);
		}
		for (int i = 0;i < nDevices;i++)
		{
			void* tmp_ptr = clEnqueueMapBuffer(ocl_command_queues[i][0], gpu_mem[nGPUMEM].mem_obj, CL_TRUE, 0, 0, nDoubles * sizeof(double), 0, NULL, NULL, &ocl_error);
			if (ocl_error != CL_SUCCESS || tmp_ptr != gpu_mem[nGPUMEM].ptr)
			{
				fprintf(STD_OUT, "Error allocating memory (clEnqueueMapBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
				return(0);
			}
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
				for (int j = 0;j < nDevices;j++)
				{
					clEnqueueUnmapMemObject(ocl_command_queues[j][0], gpu_mem[i].mem_obj, gpu_mem[i].ptr, 0, NULL, NULL);
				}
				clEnqueueUnmapMemObject(ocl_command_queue_cpu, gpu_mem[i].mem_obj, gpu_mem[i].ptr, 0, NULL, NULL);
				clReleaseMemObject(gpu_mem[i].mem_obj);
				if (C_matrix_base_obj == &gpu_mem[nGPUMEM - 1].mem_obj) C_matrix_base_obj = &gpu_mem[i].mem_obj;
				gpu_mem[i] = gpu_mem[--nGPUMEM];
				return;
			}
		}
	}
	caldgemm::FreeMemory(ptr);
}
