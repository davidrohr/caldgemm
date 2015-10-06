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
#include <algorithm>

#ifndef _WIN32
#include <syscall.h>
#include <unistd.h>
#include <sys/mman.h>
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000 /* arch specific */
#endif
#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#endif
#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#endif

#define OCL_KERNEL_PRE \
"#ifdef cl_amd_fp64\n" \
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n" \
"#else\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"#endif\n" \
"//const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n"

#ifdef CALDGEMM_OPENCL_EMULATE_STRIDED
inline cl_int clEnqueueReadBufferRectUse (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
	size_t host_row_pitch, size_t host_slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
	for (unsigned int i = 0;i < region[1];i++)
	{
		cl_int retVal = clEnqueueReadBuffer(command_queue, buffer, blocking_read, i * region[0], region[0], (char*) ptr + i * host_row_pitch, i == 0 ? num_events_in_wait_list : 0, i == 0 ? event_wait_list : NULL, i == region[1] - 1 ? event : NULL);
		if (retVal != CL_SUCCESS) return(retVal);
	}
	return(CL_SUCCESS);
}

inline cl_int clEnqueueWriteBufferRectUse (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
	size_t host_row_pitch, size_t host_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
	for (unsigned int i = 0;i < region[1];i++)
	{
		cl_int retVal = clEnqueueWriteBuffer(command_queue, buffer, blocking_read, i * region[0], region[0], (const char*) ptr + i * host_row_pitch, i == 0 ? num_events_in_wait_list : 0, i == 0 ? event_wait_list : NULL, i == region[1] - 1 ? event : NULL);
		if (retVal != CL_SUCCESS) return(retVal);
	}
	return(CL_SUCCESS);
}
#elif defined(CALDGEMM_OPENCL_USE_ORIGINAL_POINTERS)
inline cl_int clEnqueueReadBufferRectUse (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
	size_t host_row_pitch, size_t host_slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
	void* orig_ptr = NULL;
	size_t offset = 0;
	if (caldgemm_opencl::GetMemoryInfo(NULL, &orig_ptr, &offset, ptr)) return(CL_INVALID_MIP_LEVEL);
	size_t offset_origin[3] = {offset % host_row_pitch, offset / host_row_pitch, 0};
	return clEnqueueReadBufferRect(command_queue, buffer, blocking_read, buffer_origin, offset_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, orig_ptr, num_events_in_wait_list, event_wait_list, event);
}
inline cl_int clEnqueueWriteBufferRectUse (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
	size_t host_row_pitch, size_t host_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
	void* orig_ptr = NULL;
	size_t offset = 0;
	if (caldgemm_opencl::GetMemoryInfo(NULL, &orig_ptr, &offset, ptr)) return(CL_INVALID_MIP_LEVEL);
	size_t offset_origin[3] = {offset % host_row_pitch, offset / host_row_pitch, 0};
	return clEnqueueWriteBufferRect(command_queue, buffer, blocking_read, buffer_origin, offset_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, orig_ptr, num_events_in_wait_list, event_wait_list, event);
}
#else
#define clEnqueueWriteBufferRectUse clEnqueueWriteBufferRect
#define clEnqueueReadBufferRectUse clEnqueueReadBufferRect
#endif

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
#include <sys/mman.h>
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
	{ \
		const cl_int tmp_ocl_error = result; \
		if (tmp_ocl_error != CL_SUCCESS) \
		{ \
			fprintf(STD_OUT, __VA_ARGS__); \
			fprintf(STD_OUT, ":\n"); \
			fprintf(STD_OUT, "OpenCL Error %d: (%s: %d) %s\n", tmp_ocl_error, __FILE__, __LINE__, opencl_error_string(tmp_ocl_error)); \
			return(1); \
		} \
	}

caldgemm_opencl::caldgemm_opencl() : caldgemm()
{
}

caldgemm_opencl::~caldgemm_opencl()
{
}

caldgemm_opencl::caldgemm_config_backend_opencl::caldgemm_config_backend_opencl()
{
	size = sizeof(*this);
	kernelLib = NULL;
	allowCPUDevice = false;
}

caldgemm::caldgemm_config_backend* caldgemm_opencl::create_caldgemm_config_backend()
{
	return(new caldgemm_config_backend_opencl);
}

void caldgemm_opencl::caldgemm_config_backend_opencl::printConfig(caldgemm_config_backend* oldConfigA)
{
	caldgemm_config_backend_opencl* oldConfig = (caldgemm_config_backend_opencl*) oldConfigA;
	caldgemm_config_backend_opencl* const newConfig = this;
	caldgemm_config_backend_opencl* const myConfig = this;
	PRINT_CONFIG_STRING(kernelLib);
	PRINT_CONFIG_INT(allowCPUDevice);
}

caldgemm_opencl::caldgemm_config_backend_opencl::~caldgemm_config_backend_opencl() {}

int caldgemm_opencl::caldgemm_config_backend_opencl::ParseBackendOptions(unsigned int argc, char** argv)
{
	caldgemm::caldgemm_config* Config = NULL; //Should never be accessed if caldgemm_parse_parameterts.h is correct
#define CALDGEMM_PARAMETERS_BACKEND
#include "caldgemm_parse_parameters.h"
#undef CALDGEMM_PARAMETERS_BACKEND
	return(0);
}

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
	if (Config->SimpleGPUQueuing) return(0);
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
	CHKRET(clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices), "Error getting device IDs");
	nDevices = num_devices;
	if (nDevices == 0) ERRRET("No OpenCL device for this platform found\n");
	if (Config->Debug) fprintf(STD_OUT, "%d OpenCL devices found for this platform\n", nDevices);

	cl_device_id* devices = new cl_device_id[nDevices];
	if (devices == NULL) ERRRET("Memory allocation error\n");
	CHKRET(clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, nDevices, devices, NULL), "Error getting OpenCL devices");

	int gooddevices = 0;
	int cpu_found = 0;
	cl_device_id cpu_device;
	for (int i = 0;i < nDevices;i++)
	{
		char device_vendor[64], device_name[64];
		cl_device_type device_type;
		cl_uint nbits;
		CHKRET(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, device_name, NULL), "Error getting device info");
		CHKRET(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, device_vendor, NULL), "Error getting device info");
		CHKRET(clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL), "Error getting device info");
		CHKRET(clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, NULL), "Error getting device info");
		int device_ok = config_backend->allowCPUDevice ?
			(device_type & (CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU)) :
			(device_type & CL_DEVICE_TYPE_GPU) && !(device_type & CL_DEVICE_TYPE_CPU);
		if (Config->Debug) fprintf(STD_OUT, "Device %d -> %d: %s %s (%d bits)\n", i, device_ok ? gooddevices : -1, device_vendor, device_name, nbits);
		if (device_ok)
		{
			devices[gooddevices++] = devices[i];
		}
		else if (device_type & CL_DEVICE_TYPE_CPU)
		{
			cpu_found = 1;
			cpu_device = devices[i];
		}
	}

	if (cpu_found == 0)
	{
		if (config_backend->allowCPUDevice == false && Config->CPUInContext)
		{
			ERRRET("No CPU OpenCL device found for mapping buffers\n");
		}
		else
		{
			Config->CPUInContext = false;
		}
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
	if (!Config->UseGPU) nDevices = 0;
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
	ocl_context = clCreateContext(NULL, nDevices + (Config->CPUInContext ? 1 : 0), ocl_devices, NULL, NULL, &ocl_error);
	CHKRET(ocl_error, "Error creating OpenCL context");

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < (Config->AlternateSimpleQueuing ? 3 : obuffercount);j++)
		{
#ifdef CL_VERSION_2_0
			cl_queue_properties flags[] = {CL_QUEUE_PROPERTIES, 0, 0};
			if (Config->VerboseTiming || (Config->PipelinedOperation && CALDGEMM_OPENCL_PROFILED_PIPELINE)) flags[1] |= CL_QUEUE_PROFILING_ENABLE;
			ocl_command_queues[i][j] = clCreateCommandQueueWithProperties(ocl_context, ocl_devices[i], flags, &ocl_error);
#else
			cl_command_queue_properties flags = 0;
			if (Config->VerboseTiming || (Config->PipelinedOperation && CALDGEMM_OPENCL_PROFILED_PIPELINE)) flags |= CL_QUEUE_PROFILING_ENABLE;
			ocl_command_queues[i][j] = clCreateCommandQueue(ocl_context, ocl_devices[i], flags, &ocl_error);
#endif
			CHKRET(ocl_error, "Error creating OpenCL command queue");
		}
		
		if (Config->AsyncSideQueue)
		{
#ifdef CL_VERSION_2_0
			ocl_async_queue[i] = clCreateCommandQueueWithProperties(ocl_context, ocl_devices[i], NULL, &ocl_error);
#else
			ocl_async_queue[i] = clCreateCommandQueue(ocl_context, ocl_devices[i], 0, &ocl_error);
#endif
			CHKRET(ocl_error, "Error creating async OpenCL command queue");
		}
	}

	if (Config->CPUInContext)
	{
#ifdef CL_VERSION_2_0
		ocl_command_queue_cpu = clCreateCommandQueueWithProperties(ocl_context, ocl_devices[nDevices], NULL, &ocl_error);
#else
		ocl_command_queue_cpu = clCreateCommandQueue(ocl_context, ocl_devices[nDevices], 0, &ocl_error);
#endif
		CHKRET(ocl_error, "Error creating OpenCL CPU command queue");
	}
	AlternateLookaheadDoneMutex.Lock();

	return(0);
}

int caldgemm_opencl::ValidateRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ValidateRuntime\n");
	Config->MultiThreadDivide = false;

	if (Config->ThreadSaveDriver != -1) Config->ThreadSaveDriver = 1;
	if (Config->GPU_C == -1) Config->GPU_C = 1;
	if (Config->SimpleGPUQueuing)
	{
		if (Config->GPU_C == 0)
		{
			fprintf(STD_OUT, "SimpleGPUQueuing works only in combination with GPU_C\n");
			return(1);
		}
		if (Config->NoConcurrentKernels)
		{
			fprintf(STD_OUT, "Automatically disabling NoConcurrentKernels when SimpleGPUQueuing is active...\n");
			Config->NoConcurrentKernels = 0;
		}
	}

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
		CHKRET(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 64, platform_profile, NULL), "Error getting platform info");
		CHKRET(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 64, platform_version, NULL), "Error getting platform info");
		CHKRET(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 64, platform_name, NULL), "Error getting platform info");
		CHKRET(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 64, platform_vendor, NULL), "Error getting platform info");
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
		kernelLibQuerySettings = (void (*) (int*, int*, bool*, bool*, bool*, int*, int*, int*, int*)) GetProcAddress(kernelLib, "kernelLibQuerySettings");
		kernelLibTerminate = (void (*) ()) GetProcAddress(kernelLib, "kernelLibTerminate");
		kernelLibSuggestedMaxHeight = (size_t (*) ())  GetProcAddress(kernelLib, "suggestedMaxHeight");
		kernelLibGetAutoHeight = (size_t (*) (size_t, size_t, int, size_t)) GetProcAddress(kernelLib, "getAutoHeight");
		kernelLibModHeight = (void (*) (size_t, size_t)) GetProcAddress(kernelLib, "modHeight");
		kernelLibInitialize = (int (*) (cl_platform_id)) GetProcAddress(kernelLib, "kernelLibInitialize");
#else
		kernelLibCreate = (cl_kernel (*)(cl_context*, int, cl_device_id*, int, int, int)) dlsym(kernelLib, "kernelLibCreate");
		kernelLibQuerySettings = (void (*) (int*, int*, bool*, bool*, bool*, int*, int*, int*, int*)) dlsym(kernelLib, "kernelLibQuerySettings");
		kernelLibTerminate = (void (*) ()) dlsym(kernelLib, "kernelLibTerminate");
		kernelLibSuggestedMaxHeight = (size_t (*) ()) dlsym(kernelLib, "suggestedMaxHeight");
		kernelLibGetAutoHeight = (size_t (*) (size_t, size_t, int, size_t)) dlsym(kernelLib, "getAutoHeight");
		kernelLibModHeight = (void (*) (size_t, size_t)) dlsym(kernelLib, "modHeight");
		kernelLibInitialize = (int (*) (cl_platform_id)) dlsym(kernelLib, "kernelLibInitialize");
#endif
		if (kernelLibCreate == NULL || kernelLibQuerySettings == NULL || kernelLibTerminate == NULL || kernelLibSuggestedMaxHeight == NULL || kernelLibGetAutoHeight == NULL || kernelLibModHeight == NULL || kernelLibInitialize == NULL)
		{
			fprintf(STD_OUT, "Error getting function pointer from external library (%p %p %p)\n", kernelLibCreate, kernelLibQuerySettings, kernelLibTerminate);
			return(1);
		}
		if (kernelLibInitialize(ocl_platform))
		{
			fprintf(STD_OUT, "3rd Party DGEMM Library failed to initialize\n");
			return(1);
		}
		kernelLibQuerySettings(&KernelSettings.tiling_x, &KernelSettings.tiling_y, &KernelSettings.transposeA, &KernelSettings.transposeB, &KernelSettings.texture_buffers, &KernelSettings.group_size_x, &KernelSettings.group_size_y, &KernelSettings.min_tile_size, &KernelSettings.min_k);
		if (Config->Height == 0)
		{
			Config->Height = kernelLibSuggestedMaxHeight();
		}
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
		KernelSettings.min_tile_size = 32;
		KernelSettings.min_k = 4;
	}

	if (!(KernelSettings.transposeA ^ KernelSettings.transposeB))
	{
		fprintf(STD_OUT, "Must set either transposed A or transposed B\n");
		return(1);
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
	if (Config->max_bbuffers) num_bbuffers = Config->max_bbuffers;
	else if (Config->DstMemory == 'g') num_bbuffers =  max_bbuffers_g;
	else num_bbuffers = max_bbuffers;
	
	if (num_bbuffers > max_bbuffers)
	{
		fprintf(STD_OUT, "Requested number of bbuffers (%d) larger than max_bbuffers constant (%d)!", num_bbuffers, max_bbuffers);
		return(1);
	}

	SetupBufferSizes();

	cl_image_format ocl_image_format;
	ocl_image_format.image_channel_order = CL_RGBA;
	ocl_image_format.image_channel_data_type = CL_UNSIGNED_INT32;

	cpu_set_t tmpmask, oldtmpmask;
	sched_getaffinity(0, sizeof(oldtmpmask), &oldtmpmask);

	for (int i = 0;i < nDevices;i++)
	{
		if (Config->AllocMapping[i] != -1)
		{
			CPU_ZERO(&tmpmask);
			CPU_SET(Config->AllocMapping[i], &tmpmask);
			sched_setaffinity(0, sizeof(tmpmask), &tmpmask);
		}

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
		if (Config->DstMemory == 'g') CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], obuffercount, &ocl_cbuffers[i][0], 0, 0, NULL, NULL), "Error migrating mem object");

		for (int j = 0;j < (Config->GPU_C ? obuffercount : ibuffercount);j++)
		{
			cl_int tmp_flags = Config->GPU_C ? CL_MEM_READ_ONLY : (CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE);
			ocl_tmp_abuffers[i][j] = clCreateBuffer(ocl_context, tmp_flags, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
			CHKRET(ocl_error, "Error allocating device memory (A tmp - Width: %lld Height: %lld)", (long long int) BufferWidth, (long long int) BufferHeight);

			if (Config->GPU_C == 0 || Config->AlternateSimpleQueuing)
			{
				ocl_tmp_bbuffers[i][j] = clCreateBuffer(ocl_context, tmp_flags, BufferWidth * BufferHeight * sizeof(double), NULL, &ocl_error);
				CHKRET(ocl_error, "Error allocating device memory (B tmp)");
			}

			if (Config->GPU_C == 0)
			{

				ocl_tmp_abuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_abuffers[i][j], CL_TRUE, CL_MAP_WRITE, 0, BufferWidth * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping buffer (A)");

				ocl_tmp_bbuffers_ptr[i][j] = (double*) clEnqueueMapBuffer(ocl_command_queues[i][0], ocl_tmp_bbuffers[i][j], CL_TRUE, CL_MAP_WRITE, 0, BufferWidth * BufferHeight * sizeof(double), 0, NULL, NULL, &ocl_error);
				CHKRET(ocl_error, "Error mapping buffer (B)");
			}
		}
		if (Config->GPU_C)
		{
			CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], obuffercount, &ocl_tmp_abuffers[i][0], 0, 0, NULL, NULL), "Error migrating mem object");
			if (Config->AlternateSimpleQueuing)
			{
				CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], obuffercount, &ocl_tmp_bbuffers[i][0], 0, 0, NULL, NULL), "Error migrating mem object");
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
			ocl_error = clEnqueueMigrateMemObjects(ocl_command_queues[i][0], 1, &ocl_bbuffers[i][j], 0, 0, NULL, NULL);
			if (ocl_error != CL_SUCCESS)
			{
				if (j < obuffercount)
				{
					CHKRET(ocl_error, "Error migrating device memory (B)");
				}
				else break;
			}
			
			bbuffers[i] = j + 1;
		}
		if (Config->Debug) fprintf(STD_OUT, "Allocated %d BBuffers on Device %d\n", bbuffers[i], i);

		if (Config->AsyncSideQueue)
		{
			for (int j = 0;j < 4;j++)
			{
				ocl_async_buffers[i][j] = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, std::max<size_t>(BufferWidth, BufferHeight) * std::max<size_t>(BufferWidth, BufferHeight) * sizeof(double), NULL, &ocl_error);
				CHKRET(ocl_error, "Error allocating async device memory %d/%d", i, j);
			}
			CHKRET(clEnqueueMigrateMemObjects(ocl_command_queues[i][0], 4, &ocl_async_buffers[i][0], 0, 0, NULL, NULL), "Error migrating async mem object");
		}
		clFinish(ocl_command_queues[i][0]); //Finish migrating memory objects
	}

	sched_setaffinity(0, sizeof(oldtmpmask), &oldtmpmask);

	for (int j = 0;j < 3 + 1 + (Config->GPU_C ? 1 : 0) + (Config->AsyncDTRSM ? 2 : 0);j++)
	{
		if ((j != 3 || Config->Use3rdPartyTranspose) && config_backend->kernelLib != NULL)
		{
			if (Config->PrintILKernel)
			{
				fprintf(STD_OUT, "Cannot print kernel from 3ed party library\n");
			}

			for (int i = 0;i < nDevices;i++)
			{
				if (j < 5)
				{
					ocl_kernel[i][j] = kernelLibCreate(&ocl_context, nDevices, ocl_devices, j, Config->Width, Config->GPU_C == 0);
					if (ocl_kernel[i][j] == 0)
					{
						fprintf(STD_OUT, "Error obtaining kernel from external library\n");
						return(1);
					}
				}

				if (Config->AsyncSideQueue && (j == 0 || j == 3 || j >= 5))
				{
					int num = j >= 5 ? j - 3 : (j == 3 ? 1 : 0);
					ocl_async_kernel[i][num] = kernelLibCreate(&ocl_context, nDevices, ocl_devices, j, Config->Width, Config->GPU_C == 0);
					if (ocl_async_kernel[i][num] == 0)
					{
						fprintf(STD_OUT, "Error obtaining async kernel from external library\n");
						return(1);
					}
				}
			}
		}
		else
		{
			if (j == 5)
			{
				fprintf(STD_OUT, "AsyncDTRSM only supported with 3rd party lib\n");
				return(1);
			}
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

				if (Config->AsyncSideQueue && (j == 0 || j == 3))
				{
					ocl_async_kernel[i][j ? 1 : 0] = clCreateKernel(ocl_program[j], j == 3 ? "oclconkernel" : "oclkernel", &ocl_error);
					CHKRET(ocl_error, "Error creating async kernel");
				}
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
		cl_mem mem_c_matrix = 0;
		size_t matrix_offset = 0;
		if (GetMemoryInfo(&mem_c_matrix, NULL, &matrix_offset, C + blockn * Config->Height + blockm * Config->Height * C_pitch)) CHKRET(CL_INVALID_MIP_LEVEL, "Error obtaining memory info");
		CHKRET(clSetKernelArg(ocl_kernel[Task.device][Task.kernel_num], 0, sizeof(cl_mem), &mem_c_matrix), "Error setting kernel memory C");
		pitch = C_pitch;
		offset = matrix_offset / sizeof(C[0]);
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

	if (Config->Debug) fprintf(STD_OUT, "MM Kernel: height1 %d height2 %d width %d alpha %f beta %f pitch %d offset %lld\n", height1, height2, width, Alpha, Beta, pitch, (long long int) offset);
	cl_event* kernel_event;
	cl_event* need_retain_kernel_event = NULL;
	int need_release_kernel_event_after_transfer = 0;
	cl_event tmp_event;
	int wait_num_events = 0;
	cl_event* wait_event = NULL;
	cl_event simple_queue_event[2];

	cl_event* simple_queue_lookahead_event = NULL;

	int use_queue = Config->AlternateSimpleQueuing ? 2 : Task.j;
	if (Config->SimpleGPUQueuing)
	{
		if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && AlternateLookaheadTilesRemaining && blockm < AlternateLookaheadBlocksM)
		{
			simple_queue_lookahead_event = &AlternateLookaheadTilesRemaining_events[AlternateLookaheadTilesRemaining - 1];
		}

		if (Config->DstMemory == 'g')
		{
			kernel_event = NULL;
		}
		else
		{
			kernel_event = simple_queue_lookahead_event;
		}
		if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g')
		{
			simple_queue_event[wait_num_events++] = alternateSimpleQueueCopyCEvent[Task.device][Task.j];
		}
		else
		{
			if ((Config->AlternateSimpleQueuing || Task.j != simple_queue_events[Task.device][0][blockm].num_queue) && simple_queue_event_requested[Task.device][use_queue][0][blockm] == 0)
			{
				simple_queue_event[wait_num_events++] = simple_queue_events[Task.device][0][blockm].event;
				simple_queue_event_requested[Task.device][use_queue][0][blockm] = 1;
			}
			if ((Config->AlternateSimpleQueuing || Task.j != simple_queue_events[Task.device][1][blockn].num_queue) && simple_queue_event_requested[Task.device][use_queue][1][blockn] == 0)
			{
				simple_queue_event[wait_num_events++] = simple_queue_events[Task.device][1][blockn].event;
				simple_queue_event_requested[Task.device][use_queue][1][blockn] = 1;
			}
		}
		if (wait_num_events) wait_event = simple_queue_event;
		
		//Find whether we have to create an event, such that DGEMM_prepare_backend can check we are done
		int mb = (gpu_m + Config->Height - 1) / Config->Height;
		int nb = (gpu_n + Config->Height - 1) / Config->Height;
				
		if (DGEMM_favor_m ? (blockm != mb - 1) : (blockn != nb - 1))
		{
			size_t kklast = Task.k + (DGEMM_favor_m ? nb : mb);
			kklast -= kklast % (DGEMM_favor_m ? nb : mb);

			int num = 0;
			for (size_t kk = Task.k;kk < kklast;kk++)
			{
				if (tileDistribution[kk] == Task.device)
				{
					if (++num == ibuffercount) break;
				}
			}
			if (num < ibuffercount)
			{
				cl_event* ev = &simple_queue_event_kernels[Task.device][DGEMM_favor_m ? (buffer_pointers_A[Task.device][blockm] % ibuffercount) : (buffer_pointers_B[Task.device][blockn] % ibuffercount)][use_queue];
				if (Config->AlternateSimpleQueuing && *ev != NULL)
				{
					CHKRET(clReleaseEvent(*ev), "Error releasing event");
				}
				if (kernel_event == NULL)
				{
					kernel_event = ev;
				}
				else
				{
					need_retain_kernel_event = ev;
				}
			}
		}
	}
	else
	{
		if (Config->DstMemory == 'g' && (Config->GPU_C || Config->ImplicitDriverSync))
		{
			if (Config->NoConcurrentKernels) kernel_event = &tmp_event;
			else kernel_event = NULL;
		}
		else
		{
			kernel_event = &ocl_events[Task.device][Task.j];
		}

		if (Config->NoConcurrentKernels && last_device_kernel[Task.device] != 0)
		{
			wait_num_events = 1;
			wait_event = &last_device_kernel[Task.device];
		}
		else
		{
			wait_event = NULL;
		}
	}
	
	if (Config->VerboseTiming)
	{
		CHKRET(clFinish(ocl_command_queues[Task.device][use_queue]), "Error in clFinish");
		Timers.Kernel.Start();
	}
	if (Config->VerboseTiming || (Config->AlternateSimpleQueuing && Config->DstMemory == 'g'))
	{
		if (kernel_event == NULL)
		{
			kernel_event = &tmp_event;
			need_release_kernel_event_after_transfer = 1;
		}		
	}
	
	CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[Task.device][use_queue], ocl_kernel[Task.device][Task.kernel_num], 2, NULL, &global_size[0], &local_size[0], wait_num_events, wait_event, kernel_event), "Error starting MM Kernel");
	if (Config->VerboseTiming)
	{
		CHKRET(clFinish(ocl_command_queues[Task.device][use_queue]), "Error in clFinish");
		Timers.Kernel.Stop();
		cl_ulong start, end;
		CHKRET(clGetEventProfilingInfo(*kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL), "Error getting kernel profiling info");
		CHKRET(clGetEventProfilingInfo(*kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL), "Error getting kernel profiling info");
		Timers.device_kernel += end - start;
		Timers.CounterCopyFrom.Start();
	}
	if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g')
	{
		CHKRET(clReleaseEvent(alternateSimpleQueueCopyCEvent[Task.device][Task.j]), "Error releasing Event");
	}
	
	if (need_retain_kernel_event)
	{
		CHKRET(clRetainEvent(*kernel_event), "Error in clRetainEvent");
		*need_retain_kernel_event = *kernel_event;
	}

	if (Config->NoConcurrentKernels)
	{
		if (Config->DstMemory == 'g' && (Config->GPU_C || Config->ImplicitDriverSync))
		{
			if (last_device_kernel[Task.device] != 0) CHKRET(clReleaseEvent(last_device_kernel[Task.device]), "Error in clReleaseEvent");
		}
		last_device_kernel[Task.device] = *kernel_event;
	}
	if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);

	if (Config->DstMemory == 'g')
	{
		if (Config->GPU_C)
		{
			size_t origin[3] = {0, 0, 0};
			size_t region[3] = {(size_t) height1 * sizeof(double), (size_t) height2, 1};
			if (Config->Debug) fprintf(STD_OUT, "Transfer C from GPU: region %d x %d\n", (int) region[0], (int) region[1]);
			if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
			if (Config->AlternateSimpleQueuing)
			{
				CHKRET(clFlush(ocl_command_queues[Task.device][use_queue]), "Error in clFlush");
				use_queue = 1;
				wait_num_events = 1;
				wait_event = kernel_event;
			}
			else
			{
				wait_num_events = 0;
				wait_event = NULL;
			}
			int must_copy_simple_queue_event = 0;
			if (Config->AlternateSimpleQueuing)
			{
				if (simple_queue_lookahead_event == NULL)
				{
					if (alternateSimpleQueueCBuffferEvent[Task.device][Task.j].must_release) CHKRET(clReleaseEvent(alternateSimpleQueueCBuffferEvent[Task.device][Task.j].event), "Error releasing event");
					simple_queue_lookahead_event = &alternateSimpleQueueCBuffferEvent[Task.device][Task.j].event;
					alternateSimpleQueueCBuffferEvent[Task.device][Task.j].must_release = true;
				}
				else
				{
					must_copy_simple_queue_event = 1;
				}
			}
			CHKRET(clEnqueueReadBufferRectUse(ocl_command_queues[Task.device][use_queue], ocl_cbuffers[Task.device][Task.j], CL_FALSE, origin, origin, region, 0, 0, C_pitch * sizeof(double), 0, C + blockn * Config->Height + blockm * Config->Height * C_pitch, wait_num_events, wait_event, Config->SimpleGPUQueuing ? simple_queue_lookahead_event : &ocl_events[Task.device][Task.j]), "Error retrieving C");
			if (must_copy_simple_queue_event)
			{
				alternateSimpleQueueCBuffferEvent[Task.device][Task.j].event = *simple_queue_lookahead_event;
				alternateSimpleQueueCBuffferEvent[Task.device][Task.j].must_release = false;
			}
			if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
		}
		else if (Config->ImplicitDriverSync)
		{
			if (FetchResult(Task.device, Task.j, blockm, blockn)) {fprintf(STD_OUT, "Error copying from GPU\n"); return(1);}
		}
	}
	if (need_release_kernel_event_after_transfer) CHKRET(clReleaseEvent(*kernel_event), "Error releasing event");
	if (Config->PipelinedMidMarker && blockm * Config->Height >= Config->PipelinedMidMarker && MidMarker[Task.device][use_queue] == 0)
	{
		CHKRET(clEnqueueMarkerWithWaitList(ocl_command_queues[Task.device][use_queue], 0, NULL, &MidMarker[Task.device][Task.j]), "Error enqueuing OpenCL mid marker");
		if (Config->Debug) fprintf(STD_OUT, "Mid Marker Device %d queue %d block %d (Event %lld)\n", Task.device, Task.j, (int) blockm, (long long int) MidMarker[Task.device][Task.j]);
	}

	CHKRET(clFlush(ocl_command_queues[Task.device][use_queue]), "Error in clFlush");
	if (Config->VerboseTiming)
	{
		CHKRET(clFinish(ocl_command_queues[Task.device][use_queue]), "Error in clFinish");
		Timers.CounterCopyFrom.Stop();
	}

	return(0);
}

int caldgemm_opencl::ExitRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL ExitRuntime\n");

	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < (Config->AlternateSimpleQueuing ? 3 : obuffercount);j++)
		{
			CHKRET(clReleaseCommandQueue(ocl_command_queues[i][j]), "Error in clReleaseCommandQueue");
		}
	}
	if (Config->CPUInContext)
	{
		CHKRET(clReleaseCommandQueue(ocl_command_queue_cpu), "Error in clReleaseCommandQueue");
	}

	if (config_backend->kernelLib)
	{
		kernelLibTerminate();
#ifdef _WIN32
		FreeLibrary(kernelLib);
#else
		dlclose(kernelLib);
#endif
	}

	CHKRET(clReleaseContext(ocl_context), "Error in clReleaseContext");
	AlternateLookaheadDoneMutex.Unlock();

	return(0);
}

int caldgemm_opencl::FetchResult(int device, int j, int m, int n, int mustlock)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL FetchResult\n");
	if (Config->GPU_C == 0 && Config->DstMemory == 'g')
	{
		if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
		CHKRET(clEnqueueCopyBuffer(ocl_command_queues[device][j], ocl_cbuffers[device][j], ocl_tmp_cbuffers[device][j], 0, 0, Config->Height * Config->Height * sizeof(double), 0, NULL, &ocl_events[device][j]), "Error copying resulg from GPU to host");
		CHKRET(clFlush(ocl_command_queues[device][j]), "Error in clFlush");
		if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
		if (Config->VerboseTiming) CHKRET(clFinish(ocl_command_queues[device][j]), "Error in clFinish");
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

HighResTimer asynctimer;

int caldgemm_opencl::RunAsyncSingleTileDTRSM(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const size_t M, const size_t N, const double alpha, const double *A, const size_t lda, double *B, const size_t ldb)
{
	if (M == 0 || N == 0) return(0);
	
	if (!Config->AsyncDTRSM)
	{
		fprintf(STD_OUT, "Config DTRSM not enabled!\n");
		return(1);
	}

	size_t BufferSize = std::max<size_t>(BufferWidth, BufferHeight);
	const unsigned int K = (Side == CblasRight ? N : M);
	const unsigned int L = (Side == CblasRight ? M : N) & ~31;
	if (M < (size_t) Config->AsyncDTRSMThreshold || N < (size_t) Config->AsyncDTRSMThreshold || Order != CblasColMajor || Uplo != CblasUpper || ((Side != CblasLeft) ^ (TransA != CblasTrans)) || Diag != CblasUnit || K > BufferSize || K & 31)
	{
		cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, (double*) A, lda, B, ldb);
		return(0);
	}
	
	static int useDevice = 0;
	int useDeviceOffset, nAsyncDevices;
	if (nDevices < nDevicesInitialized && Config->AsyncSideQueueUseInactiveDeviceSet)
	{
		useDeviceOffset = nDevices;
		nAsyncDevices = nDevicesInitialized - nDevices;
	}
	else
	{
		useDeviceOffset = 0;
		nAsyncDevices = nDevicesInitialized;
	}
	if (useDevice >= nAsyncDevices) useDevice = 0;
	int useDeviceStart = useDevice;

	BufferSize *= BufferSize;
	
	const unsigned int TmpM = M & ~31;
	const unsigned int TmpN = N & ~31;
	unsigned int tmpL = (BufferSize / K) & ~31;
	unsigned int nTiles = (L + tmpL - 1) / tmpL;
	if (Config->AsyncSideQueueBalance)
	{
		int overlap = nTiles % nAsyncDevices;
		if (overlap)
		{
			nTiles += nAsyncDevices - overlap;
			tmpL = (L / nTiles);
			if ((tmpL & 31)) tmpL = (tmpL + 32) & ~31;
			nTiles = (L + tmpL - 1) / tmpL;
		}
	}
	const size_t origin[3] = {0, 0, 0};
	size_t region[3];
	region[2] = 1;
	
	
	for (unsigned int i = 0;i < nTiles;i++)
	{
		if (i == nTiles - 1) tmpL = L - i * tmpL;
		if (i < (unsigned int) nAsyncDevices)
		{
			region[0] = K * sizeof(double);
			region[1] = K;
			if (Config->Debug) fprintf(STD_OUT, "ASYNC Copying A to GPU, width: %lld, height: %lld\n", (long long int) region[0] / sizeof(double), (long long int) region[1]);
			CHKRET(clEnqueueWriteBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][0], CL_FALSE, origin, origin, region, 0, 0, lda * sizeof(double), 0, A, 0, NULL, NULL), "Error copying async A");
		}

		if (Side == CblasRight)
		{
			region[0] = tmpL * sizeof(double);
			region[1] = TmpN;
		}
		else
		{
			region[0] = TmpM * sizeof(double);
			region[1] = tmpL;
		}
		if (Config->Debug) fprintf(STD_OUT, "ASYNC Copying B to GPU, width: %lld, height: %lld\n", (long long int) region[0] / sizeof(double), (long long int) region[1]);
		CHKRET(clEnqueueWriteBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][1], CL_FALSE, origin, origin, region, 0, 0, ldb * sizeof(double), 0, B, 0, NULL, NULL), "Error copying async B");

		int kernelNum = Side == CblasRight ? 2 : 3;
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 0, sizeof(cl_uint), Side == CblasRight ? &tmpL : &TmpM), "Error setting kernel arg, 0");
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 1, sizeof(cl_uint), Side == CblasRight ? &TmpN : &tmpL), "Error setting kernel arg, 1");
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 2, sizeof(cl_double), &alpha), "Error setting kernel arg, 2");
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 3, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][0]), "Error setting kernel arg, 3");
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 4, sizeof(cl_uint), &K), "Error setting kernel arg, 4");
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 5, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][1]), "Error setting kernel arg, 5");
		CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 6, sizeof(cl_uint), Side == CblasRight ? &tmpL : &TmpM), "Error setting kernel arg, 6");
		
		size_t globalThreads[1] = {2 * tmpL};
		size_t localThreads[1]  = {64};
		CHKRET(clEnqueueNDRangeKernel(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_kernel[useDeviceOffset + useDevice][kernelNum], 1, NULL, globalThreads, localThreads, 0, NULL, NULL), "Error starting DTRSM Kernel");
		
		CHKRET(clEnqueueReadBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][1], CL_FALSE, origin, origin, region, 0, 0, ldb * sizeof(double), 0, B, 0, NULL, NULL), "Error retrieving async B");

		CHKRET(clFlush(ocl_async_queue[useDeviceOffset + useDevice]), "Error in clFlush");
		useDevice = (useDevice + 1) % nAsyncDevices;
		
		if (Side == CblasRight)
		{
			B += tmpL;
		}
		else
		{
			B += tmpL * ldb;
		}
	}
	
	if (Side == CblasRight)
	{
		if (M & 31) cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M & 31, N, alpha, (double*) A, lda, B, ldb);
	}
	else
	{
		if (N & 31) cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N & 31, alpha, (double*) A, lda, B, ldb);
	}
	
	for (int i = 0;i < std::min<int>(nTiles, nAsyncDevices);i++)
	{
		CHKRET(clFinish(ocl_async_queue[useDeviceOffset + useDeviceStart]), "Error in clFinish");
		useDeviceStart = (useDeviceStart + 1) % nAsyncDevices;
	}
	
	return(0);
}

int caldgemm_opencl::RunAsyncSingleTileDGEMM(const double* A, const double* B, double* C, double alpha, double beta, size_t m, size_t k, size_t n, size_t Apitch, size_t Bpitch, size_t Cpitch, bool orderColMajor, bool TransA, bool TransB)
{
	if (m == 0 || n == 0 || k == 0) return(0);
	
	if (!Config->AsyncSideQueue)
	{
		fprintf(STD_OUT, "Config Side Queue not enabled!\n");
		return(1);
	}

	double* tmp_D = NULL;

	asynctimer.ResetStart();
	int useCPU = 0; 
	
	while (true)
	{
		if (k % KernelSettings.min_k)
		{
			fprintf(STD_OUT, "Invalik k for async GPU DGEMM, running CPU DGEMM\n");
			useCPU = 1;
		}
		
		if (useCPU || m < (size_t) Config->AsyncDGEMMThreshold || n < (size_t) Config->AsyncDGEMMThreshold || k < (size_t) Config->AsyncDGEMMThreshold) //Does not make sense for too small matrices
		{
			cblas_dgemm(orderColMajor ? CblasColMajor : CblasRowMajor, TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans, m, n, k, alpha, (double*) A, Apitch, (double*) B, Bpitch, beta, C, Cpitch);
			useCPU = 1;
			break;
		}

		size_t BufferSize = std::max<size_t>(BufferWidth, BufferHeight);
		BufferSize *= BufferSize;
		if (orderColMajor)
		{
			const double* tmpD = A;
			A = B;
			B = tmpD;
			size_t tmpS = m;
			m = n;
			n = tmpS;
			tmpS = Apitch;
			Apitch = Bpitch;
			Bpitch = tmpS;
			bool tmpB = TransA;
			TransA = TransB;
			TransB = tmpB;
			orderColMajor = false;
		}
		
		bool tile_m = (n <= m);
		size_t tile_size_m, tile_size_n;
		int nTiles;
		
		if (tile_m)
		{
			tile_size_n = n - n % KernelSettings.min_tile_size;
			if (tile_size_n)
			{
				tile_size_m = BufferSize / std::max<size_t>(tile_size_n, k);
				tile_size_m -= tile_size_m % KernelSettings.min_tile_size;
				nTiles = (m - m % KernelSettings.min_tile_size + tile_size_m - 1) / tile_size_m;
			}
			else
			{
				tile_size_m = 0;
			}
		}
		else
		{
			tile_size_m = m - m % KernelSettings.min_tile_size;
			if (tile_size_m)
			{
				tile_size_n = BufferSize / std::max<size_t>(tile_size_m, k);
				tile_size_n -= tile_size_n % KernelSettings.min_tile_size;
				nTiles = (n - n % KernelSettings.min_tile_size + tile_size_n - 1) / tile_size_n;
			}
			else
			{
				tile_size_n = 0;
			}
		}
		if (tile_size_m == 0 || tile_size_n == 0 || tile_size_m * k > BufferSize || tile_size_n * k > BufferSize)
		{
			fprintf(STD_OUT, "Invalid Matrix Size for async GPU DGEMM, running CPU DGEMM\n");
			cblas_dgemm(CblasRowMajor, TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans, m, n, k, alpha, (double*) A, Apitch, (double*) B, Bpitch, beta, C, Cpitch);
			useCPU = 1;
			break;
		}

		if (Config->Verify)
		{
			tmp_D = new double[m * Cpitch];
			memcpy(tmp_D, C, m * Cpitch * sizeof(double));
			cblas_dgemm(orderColMajor ? CblasColMajor : CblasRowMajor, TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans, m, n, k, alpha, (double*) A, Apitch, (double*) B, Bpitch, beta, tmp_D, Cpitch);
		}

		if (Config->Debug) fprintf(STD_OUT, "Running ASYNC DGEMM\n");

		static int useDevice = 0;
		int useDeviceOffset, nAsyncDevices;
		if (nDevices < nDevicesInitialized && Config->AsyncSideQueueUseInactiveDeviceSet)
		{
			useDeviceOffset = nDevices;
			nAsyncDevices = nDevicesInitialized - nDevices;
		}
		else
		{
			useDeviceOffset = 0;
			nAsyncDevices = nDevicesInitialized;
		}
		if (useDevice >= nAsyncDevices) useDevice = 0;
		int useDeviceStart = useDevice;

		const size_t origin[3] = {0, 0, 0};
		size_t region[3];
		for (int i = 0;i < nTiles;i++)
		{
			size_t g_m, g_n;
			const double *g_A, *g_B;
			double *g_C;
			if (tile_m)
			{
				size_t Aoffset = i * tile_size_m;
				g_A = A + Aoffset * (TransA ? 1 : Apitch);
				g_B = B;
				g_C = C + Aoffset * Cpitch;
				g_n = tile_size_n;
				g_m = std::min<size_t>(tile_size_m, m - Aoffset);
				g_m -= g_m % KernelSettings.min_tile_size;
			}
			else
			{
				size_t Boffset = i * tile_size_n;
				g_A = A;
				g_B = B + Boffset * (TransB ? Bpitch : 1);
				g_C = C + Boffset;
				g_m = tile_size_m;
				g_n = std::min<size_t>(tile_size_n, n - Boffset);
				g_n -= g_n % KernelSettings.min_tile_size;
			}
			
			region[2] = 1;
			size_t local_size[2];
			size_t global_size[2];
			if (!Config->Use3rdPartyTranspose)
			{
				local_size[0] = GROUP_SIZE_X;
				local_size[1] = GROUP_SIZE_Y;
				global_size[0] = local_size[0] * GROUP_COUNT_X;
				global_size[1] = local_size[1] * GROUP_COUNT_Y;
			}
			
			if (i < nAsyncDevices || tile_m)
			{
				region[0] = (TransA ? g_m : k) * sizeof(double);
				region[1] = (TransA ? k : g_m);
				if (Config->Debug) fprintf(STD_OUT, "ASYNC Copying A to GPU, width: %lld, height: %lld\n", (long long int) region[0] / sizeof(double), (long long int) region[1]);
				int arg_transpose = TransA ^ KernelSettings.transposeA;
				CHKRET(clEnqueueWriteBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][arg_transpose || KernelSettings.texture_buffers ? 0 : 1], CL_FALSE, origin, origin, region, 0, 0, Apitch * sizeof(double), 0, g_A, 0, NULL, NULL), "Error copying async A");

				if (arg_transpose || KernelSettings.texture_buffers)
				{
					int arg_width = region[0] / sizeof(double);
					int arg_height = region[1];
					size_t local_size[2] = {GROUP_SIZE_X, GROUP_SIZE_Y};
					size_t global_size[2] = {GROUP_SIZE_X * GROUP_COUNT_X, GROUP_SIZE_Y * GROUP_COUNT_Y};
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 0, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][0]), "Error setting kernel arg, A, 0");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 1, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][1]), "Error setting kernel arg, A, 1");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 2, sizeof(int), &arg_width), "Error setting kernel arg, A, 2");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 3, sizeof(int), &arg_height), "Error setting kernel arg, A, 3");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, A, 4");
					if (Config->Debug) fprintf(STD_OUT, "ASYNC Running conversion kernel for A: transpose %d, width %d, height %d\n", arg_transpose, arg_width, arg_height);
					if (Config->Use3rdPartyTranspose)
					{
						local_size[0] = 16;
						local_size[1] = 16;
						global_size[0] = arg_width / 4;
						global_size[1] = arg_height / 4;
					}
					CHKRET(clEnqueueNDRangeKernel(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_kernel[useDeviceOffset + useDevice][1], 2, NULL, &global_size[0], &local_size[0], 0, NULL, NULL), "Error starting conversion kernel for async A");
				}
			}

			if (i < nAsyncDevices || !tile_m)
			{
				region[0] = (TransB ? k : g_n) * sizeof(double);
				region[1] = (TransB ? g_n : k);
				if (Config->Debug) fprintf(STD_OUT, "ASYNC Copying B to GPU, width: %lld, height: %lld\n", (long long int) region[0] / sizeof(double), (long long int) region[1]);
				int arg_transpose = TransB ^ KernelSettings.transposeB;
				CHKRET(clEnqueueWriteBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][arg_transpose || KernelSettings.texture_buffers ? 0 : 2], CL_FALSE, origin, origin, region, 0, 0, Bpitch * sizeof(double), 0, g_B, 0, NULL, NULL), "Error copying async B");

				if (arg_transpose || KernelSettings.texture_buffers)
				{
					int arg_width = region[0] / sizeof(double);
					int arg_height = region[1];
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 0, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][0]), "Error setting kernel arg, A, 0");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 1, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][2]), "Error setting kernel arg, A, 1");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 2, sizeof(int), &arg_width), "Error setting kernel arg, A, 2");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 3, sizeof(int), &arg_height), "Error setting kernel arg, A, 3");
					CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][1], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, A, 4");
					if (Config->Debug) fprintf(STD_OUT, "ASYNC Running conversion kernel for B: transpose %d, width %d, height %d\n", arg_transpose, arg_width, arg_height);
					if (Config->Use3rdPartyTranspose)
					{
						local_size[0] = 16;
						local_size[1] = 16;
						global_size[0] = arg_width / 4;
						global_size[1] = arg_height / 4;
					}
					CHKRET(clEnqueueNDRangeKernel(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_kernel[useDeviceOffset + useDevice][1], 2, NULL, &global_size[0], &local_size[0], 0, NULL, NULL), "Error starting conversion kernel for async B");
				}
			}

			size_t offset = 0;
			if (Config->DstMemory == 'g')
			{
				region[0] = g_n * sizeof(double);
				region[1] = g_m;
				CHKRET(clEnqueueWriteBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][3], CL_FALSE, origin, origin, region, 0, 0, Cpitch * sizeof(double), 0, g_C, 0, NULL, NULL), "Error copying async C");

				CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 0, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][3]), "Error setting kernel memory C");
			}
			else
			{
				cl_mem pmem;
				if (GetMemoryInfo(&pmem, NULL, &offset, g_C))
				{
					fprintf(STD_OUT, "C Matrix memory must be allocated via runtime for DstMemory = c\n");
					return(1);
				}
				offset /= sizeof(double);
				if (Config->Debug) fprintf(STD_OUT, "C Memory pointer offset: %lld\n", (long long int) offset);
				
				CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 0, sizeof(cl_mem), &pmem), "Error setting kernel memory C");
			}

			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 1, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][1]), "Error setting kernel memory A");
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 2, sizeof(cl_mem), &ocl_async_buffers[useDeviceOffset + useDevice][2]), "Error setting kernel memory B");

			int arg_m = g_m, arg_n = g_n, arg_k = k;
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 3, sizeof(int), &arg_n), "Error setting kernel arg height1");
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 4, sizeof(int), &arg_m), "Error setting kernel arg height2");
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 5, sizeof(int), &arg_k), "Error setting kernel arg width");

			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 6, sizeof(double), &alpha), "Error setting kernel arg alpha");
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 7, sizeof(double), &beta), "Error setting kernel arg beta");

			size_t pitch = Config->DstMemory == 'g' ? arg_n : Cpitch;
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 8, sizeof(int), &pitch), "Error setting kernel arg pitch");
			CHKRET(clSetKernelArg(ocl_async_kernel[useDeviceOffset + useDevice][0], 9, sizeof(cl_ulong), &offset), "Error setting kernel arg offset");

			local_size[0] = (size_t) KernelSettings.group_size_x;
			local_size[1] = (size_t) KernelSettings.group_size_y;
			global_size[0] = (size_t) g_n / KernelSettings.tiling_x;
			global_size[1] = (size_t) g_m / KernelSettings.tiling_y;

			CHKRET(clEnqueueNDRangeKernel(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_kernel[useDeviceOffset + useDevice][0], 2, NULL, &global_size[0], &local_size[0], 0, NULL, NULL), "Error starting MM Kernel");

			if (Config->DstMemory == 'g')
			{
				CHKRET(clEnqueueReadBufferRectUse(ocl_async_queue[useDeviceOffset + useDevice], ocl_async_buffers[useDeviceOffset + useDevice][3], CL_FALSE, origin, origin, region, 0, 0, Cpitch * sizeof(double), 0, g_C, 0, NULL, NULL), "Error retrieving async C");
			}

			CHKRET(clFlush(ocl_async_queue[useDeviceOffset + useDevice]), "Error in clFlush");
			useDevice = (useDevice + 1) % nAsyncDevices;
		}
		
		
		if (m % KernelSettings.min_tile_size)
		{
			cblas_dgemm(CblasRowMajor, TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans, m % KernelSettings.min_tile_size, n, k, alpha,
			    (double*) A + (m - m % KernelSettings.min_tile_size) * (TransA ? 1 : Apitch), Apitch,
			    (double*) B, Bpitch, beta,
			    C + (m - m % KernelSettings.min_tile_size) * Cpitch, Cpitch);
		}
		
		if (n % KernelSettings.min_tile_size)
		{
			cblas_dgemm(CblasRowMajor, TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans, m - m % KernelSettings.min_tile_size, n % KernelSettings.min_tile_size, k, alpha,
			    (double*) A, Apitch,
			    (double*) B + (n - n % KernelSettings.min_tile_size) * (TransB ? Bpitch : 1), Bpitch, beta,
			    C + (n - n % KernelSettings.min_tile_size), Cpitch);
		}

		for (int i = 0;i < std::min<int>(nTiles, nAsyncDevices);i++)
		{
			CHKRET(clFinish(ocl_async_queue[useDeviceOffset + useDeviceStart]), "Error in clFinish");
			useDeviceStart = (useDeviceStart + 1) % nAsyncDevices;
		}

		if (Config->Verify)
		{
			size_t errors = 0;
			size_t errorsrel[3];
			memset(errorsrel, 0, 3 * sizeof(size_t));

			for (size_t i = 0; i < m; i++)
			{
				for (size_t j = 0; j < n; j++)
				{
					if (!isDoubleEqual(C[i * Cpitch + j], tmp_D[i * Cpitch + j]))
					{
						if (errors < 5) fprintf(STD_OUT, "Error found at row %lld, col %lld: Expected: %3.5le, Found: %3.5le, Diff: %3.5le, Relative: %3.5le\n", (long long int) i, (long long int) j, tmp_D[i * Cpitch + j], C[i * Cpitch + j], tmp_D[i * Cpitch + j] - C[i * Cpitch + j], (tmp_D[i * Cpitch + j] - C[i * Cpitch + j]) / tmp_D[i * Cpitch + j]);
						++errors;
						if (fabs((C[i * Cpitch + j] - tmp_D[i * Cpitch + j]) / tmp_D[i * Cpitch + j]) > 0.05) errorsrel[0]++;
						else if (fabs((C[i * Cpitch + j] - tmp_D[i * Cpitch + j]) / tmp_D[i * Cpitch + j]) < 0.0001) errorsrel[2]++;
						else errorsrel[1]++;
						//printf("X");
					}
					else
					{
						//printf("-");
					}
				}
				//printf("\n");
			}
			if (errors)
			{
				fprintf(STD_OUT, "%lld elements were incorrect (Rel errors > 0.05: %lld, > 0.0001: %lld, rest: %lld)\n", (long long int) errors, (long long int) errorsrel[0], (long long int) errorsrel[1], (long long int) errorsrel[2]);
				if (errorsrel[0] == 0)
				{
					fprintf(STD_OUT, "Passed with Warnings!!!\n");
				}
				else
				{
					fprintf(STD_OUT, "FAILED\n");
				}
			}
			else
			{
				fprintf(STD_OUT, "PASSED\n");
			}

			delete[] tmp_D;
		}
		
		break;
	}
	
	if (Config->Debug) fprintf(STD_OUT, "ASYNC CALDGEMM (m = %6lld, n = %6lld, k = %6lld) %s - Time: %8.5f\n", (long long int) m, (long long int) n, (long long int) k, useCPU ? "CPU" : "GPU", asynctimer.GetCurrentElapsedTime());

	return(0);
}

inline void caldgemm_opencl::pipelinedModeSetStartBarriers(unsigned int num_device, int j, int &nTransferEvents, cl_event* transferEvents, bool &freeTransferEvents)
{
	if (Config->PipelinedOperation && finishData->running && nTransferEvents == 0 && pipelinedModeStartBarrierDone[num_device][Config->AlternateSimpleQueuing ? 0 : j] == 0)
	{
		for (int i = 0;i < obuffercount;i++)
		{
			if (Config->AlternateSimpleQueuing) i = Config->DstMemory == 'g' ? 1 : 2;
			if (Config->AlternateSimpleQueuing || i != j)
			{
				transferEvents[nTransferEvents++] = ((finishStructOpenCL*) finishData)->EndMarker[num_device][i];
				freeTransferEvents = false;
			}
			if (Config->AlternateSimpleQueuing) break;
		}
		pipelinedModeStartBarrierDone[num_device][Config->AlternateSimpleQueuing ? 0 : j] = 1;
	}
}


int caldgemm_opencl::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA)
{
	if (Config->Debug) fprintf(STD_OUT, "OPENCL DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	CALDGEMM_PREPARE_BACKEND_VARS1;

	const size_t origin[3] = {0, 0, 0};
	size_t region[3];
	region[2] = 1;

	if (Config->VerboseTiming && Config->GPU_C == 1) Timers.CounterCopyTo.Start();

	int nTransferEvents;
	cl_event transferEvents[obuffercount + 1];
	bool freeTransferEvents;
	
	bool flushKernel = false;
	
	for (int iMat = 0;iMat < 2;iMat++)
	{
		if (!Config->SimpleGPUQueuing && ocl_conversion_events_use[num_device][iMat])
		{
			WaitForEventAndRelease(&ocl_conversion_events[num_device][iMat]);
			ocl_conversion_events_use[num_device][iMat] = 0;
		}
		if (iMat ? prepareN : prepareM)
		{
			CALDGEMM_PREPARE_BACKEND_VARS2;
			cl_event* my_alternateSimpleQueueEvent_tmp_buffers = iMat ? alternateSimpleQueueEvent_tmp_bbuffers[num_device] : alternateSimpleQueueEvent_tmp_abuffers[num_device];
			double** my_ocl_tmp_buffers_ptr = iMat ? ocl_tmp_bbuffers_ptr[num_device] : ocl_tmp_abuffers_ptr[num_device];
			
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of %c to GPU (k = %lld, m = %lld, n = %lld)\n", myMat, (long long int) k, (long long int) blockm, (long long int) blockn);
			nTransferEvents = 0;
			freeTransferEvents = true;
			int forceFreeTransferEvent = -1;

			if (!access_bbuffers)
			{
				if (Config->SimpleGPUQueuing)
				{
					for (int i = 0;i < obuffercount;i++)
					{
						if (Config->AlternateSimpleQueuing && iMat == 0) i = 2;
						if (simple_queue_event_kernels[num_device][destbuffer][i] != NULL)
						{
							//printf("DEBUG: %cBuffer block%c %d Need to wait for device %d Buffer %d Context %d\n", myMat, iMat ? 'n' : 'm', (int) myblock, num_device, destbuffer, i);
							if (!Config->AlternateSimpleQueuing && i == j)
							{
								CHKRET(clReleaseEvent(simple_queue_event_kernels[num_device][destbuffer][i]), "Error in clReleaseEvent");
							}
							else
							{
								transferEvents[nTransferEvents++] = simple_queue_event_kernels[num_device][destbuffer][i];
							}
							simple_queue_event_kernels[num_device][destbuffer][i] = 0;
						}
					}
				}
			}
			cl_mem* dest_image = access_bbuffers ? &ocl_bbuffers[num_device][destbuffer] : &ocl_abuffers[num_device][destbuffer];
			
			int use_queue = Config->AlternateSimpleQueuing ? 0 : j;
			int arg_transpose = myTranspose ^ myKernelTranspose;

			if (Config->GPU_C == 0)
			{
				if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer %c (device = %d, k = %lld, context = %d, m = %lld, n = %lld, buffer = %d, transpose = %d)\n", myMat, num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn, my_next_buffer % ibuffercount, arg_transpose);
				if (Config->VerboseTiming) Timers.CounterDivide.Start();

				if (divideBuffer((double*) src_ptr, pitch, my_ocl_tmp_buffers_ptr[my_next_buffer % ibuffercount], TransposeA ? Config->Width : myHeight, TransposeA ? myHeight : Config->Width, arg_transpose)) return(1);
				if (Config->VerboseTiming) Timers.CounterDivide.Stop();
				if (Config->Debug) fprintf(STD_OUT, "\tCopying part of %c to GPU (device = %d, k = %lld, context = %d, m = %lld, n = %lld, buffer: %d->%d)\n", myMat, num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn, my_next_buffer % ibuffercount, destbuffer);
				if (KernelSettings.transposeA) //must be transposed A and not transposed b
				{
					region[0] = Config->Width / 2;
					region[1] = myHeight;
				}
				else //must be transposeB and not transposed a
				{
					region[0] = myHeight / 2;
					region[1] = Config->Width;
				}

				if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
				if (Config->VerboseTiming) Timers.CounterCopyTo.Start();
				if (!KernelSettings.texture_buffers)
				{
					CHKRET(clEnqueueWriteBuffer(ocl_command_queues[num_device][j], *dest_image, CL_FALSE, 0, Config->Width * myHeight * sizeof(double), my_ocl_tmp_buffers_ptr[my_next_buffer % ibuffercount], 0, NULL, &ocl_conversion_events[num_device][iMat]), "Error copying %c", myMat);
				}
				else
				{
					CHKRET(clEnqueueWriteImage(ocl_command_queues[num_device][j], *dest_image, CL_FALSE, origin, region, 0, 0, my_ocl_tmp_buffers_ptr[my_next_buffer % ibuffercount], 0, NULL, &ocl_conversion_events[num_device][iMat]), "Error copying %c", myMat);
				}
				if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
				if (Config->VerboseTiming)
				{
					CHKRET(clFinish(ocl_command_queues[num_device][j]), "Error in clFinish");
					Timers.CounterCopyTo.Stop();
				}
			}
			else
			{
				pipelinedModeSetStartBarriers(num_device, j, nTransferEvents, transferEvents, freeTransferEvents);
				if (Config->AlternateSimpleQueuing && (arg_transpose || KernelSettings.texture_buffers) && my_alternateSimpleQueueEvent_tmp_buffers[j] != 0)
				{
					if (!freeTransferEvents) forceFreeTransferEvent = nTransferEvents;
					transferEvents[nTransferEvents++] = my_alternateSimpleQueueEvent_tmp_buffers[j];
					my_alternateSimpleQueueEvent_tmp_buffers[j] = 0;
				}


				if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
				region[0] = (((bool) iMat ^ myTranspose) ? myHeight : Config->Width) * sizeof(double); //It is either transposeA or transposeB !
				region[1] = (((bool) iMat ^ myTranspose) ? Config->Width : myHeight);
				int arg_width = region[0] / sizeof(double), arg_height = region[1];
				cl_mem dest_buffer_tmp = iMat && Config->AlternateSimpleQueuing ? ocl_tmp_bbuffers[num_device][j] : ocl_tmp_abuffers[num_device][j];
				if (Config->Debug) fprintf(STD_OUT, "Transfer %c to GPU: region %d x %d\n", myMat, (int) region[0], (int) region[1]);
				if (arg_transpose == 0 && KernelSettings.texture_buffers == false) dest_buffer_tmp = *dest_image;

				cl_event* ev;
				cl_event ev2;
				if (Config->SimpleGPUQueuing)
				{
					if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g')
					{
						ev = NULL;
					}
					else
					{
						ev = &simple_queue_events[num_device][iMat][myblock].event;
						simple_queue_events[num_device][iMat][myblock].num_queue = j;
					}
				}
				else
				{
					ev = &ocl_conversion_events[num_device][iMat];
				}
				//printf("DEBUG: %cBuffer device %d buffer %d block%c %d wait for %d kernels\n", myMat, num_device, j, iMat ? 'n' : 'm', (int) myblock, nTransferEvents);

				CHKRET(clEnqueueWriteBufferRectUse(ocl_command_queues[num_device][use_queue], dest_buffer_tmp, CL_FALSE, origin, origin, region, 0, 0, pitch * sizeof(double), 0, src_ptr, nTransferEvents, nTransferEvents ? transferEvents : NULL, (arg_transpose == 0 && KernelSettings.texture_buffers == false) ? ev : (Config->AlternateSimpleQueuing ? &ev2 : NULL)), "Error copying %c", myMat);
				if (freeTransferEvents) for (int i = 0;i < nTransferEvents;i++) CHKRET(clReleaseEvent(transferEvents[i]), "Error releasing event %c", myMat);
				if (forceFreeTransferEvent >= 0) CHKRET(clReleaseEvent(transferEvents[forceFreeTransferEvent]), "Error releasing event %c", myMat);
				if (Config->Debug && Config->VerboseTiming) CHKRET(clFinish(ocl_command_queues[num_device][use_queue]), "Error in clFinish");
				if (arg_transpose || KernelSettings.texture_buffers)
				{
					CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 0, sizeof(cl_mem), &dest_buffer_tmp), "Error setting kernel arg, %c, 0", myMat);
					CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 1, sizeof(cl_mem), dest_image), "Error setting kernel arg, %c, 1", myMat);
					CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 2, sizeof(int), &arg_width), "Error setting kernel arg, %c, 2", myMat);
					CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 3, sizeof(int), &arg_height), "Error setting kernel arg, %c, 3", myMat);
					CHKRET(clSetKernelArg(ocl_kernel[num_device][3], 4, sizeof(int), &arg_transpose), "Error setting kernel arg, %c, 4", myMat);

					size_t local_size[2];
					size_t global_size[2];
					if (!Config->Use3rdPartyTranspose)
					{
						local_size[0] = GROUP_SIZE_X;
						local_size[1] = GROUP_SIZE_Y;
						global_size[0] = local_size[0] * GROUP_COUNT_X;
						global_size[1] = local_size[1] * GROUP_COUNT_Y;
					}
					else
					{
						local_size[0] = 16;
						local_size[1] = 16;
						global_size[0] = arg_width / 4;
						global_size[1] = arg_height / 4;
					}

					if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel %c: x %d y %d\n", myMat, (int) arg_width, (int) arg_height);
					if (Config->AlternateSimpleQueuing) use_queue = 2;
					int retain_ev = 0;
					if (Config->AlternateSimpleQueuing)
					{
						if (ev == NULL)
						{
							ev = &my_alternateSimpleQueueEvent_tmp_buffers[j];
						}
						else
						{
							retain_ev = 1;
						}
					}
					CHKRET(clEnqueueNDRangeKernel(ocl_command_queues[num_device][use_queue], ocl_kernel[num_device][3], 2, NULL, &global_size[0], &local_size[0], Config->AlternateSimpleQueuing ? 1 : 0, Config->AlternateSimpleQueuing ? &ev2 : NULL, ev), "Error starting conversion kernel for %c", myMat);
					flushKernel = true;
					if (retain_ev)
					{
						my_alternateSimpleQueueEvent_tmp_buffers[j] = *ev;
						clRetainEvent(*ev);
					}
					if (Config->AlternateSimpleQueuing) CHKRET(clReleaseEvent(ev2), "Error releasing event");
				}
				if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
			}
			ocl_conversion_events_use[num_device][iMat] = 1;
			if (Config->Debug && Config->VerboseTiming) CHKRET(clFinish(ocl_command_queues[num_device][use_queue]), "Error in clFinish");
		}
	}
	
	if (Config->GPU_C && Config->DstMemory == 'g')
	{
		int use_queue = Config->AlternateSimpleQueuing ? 0 : j;
		Timers.divideC++;
		region[0] = HeightN * sizeof(double);
		region[1] = HeightM;
		if (Config->Debug) fprintf(STD_OUT, "Transfer C to GPU: region %d x %d\n", (int) region[0], (int) region[1]);
		if (Config->ThreadSaveDriver == -1) pthread_mutex_lock(&globalDriverLock);
	
		if (Config->AlternateSimpleQueuing && alternateSimpleQueueCBuffferEvent[num_device][j].event != 0)
		{
			nTransferEvents = 1;
			transferEvents[0] = alternateSimpleQueueCBuffferEvent[num_device][j].event;
		}
		else
		{
			nTransferEvents = 0;
		}

		pipelinedModeSetStartBarriers(num_device, j, nTransferEvents, transferEvents, freeTransferEvents);

		CHKRET(clEnqueueWriteBufferRectUse(ocl_command_queues[num_device][use_queue], ocl_cbuffers[num_device][j], CL_FALSE, origin, origin, region, 0, 0, C_pitch * sizeof(double), 0, C + blockn * Config->Height + blockm * Config->Height * C_pitch, nTransferEvents, nTransferEvents ? transferEvents : NULL, Config->AlternateSimpleQueuing ? &alternateSimpleQueueCopyCEvent[num_device][j] : NULL), "Error copying C");
		if (Config->ThreadSaveDriver == -1) pthread_mutex_unlock(&globalDriverLock);
	}
	
	if (Config->AlternateSimpleQueuing)
	{
		if (prepareM || prepareN || (Config->GPU_C && Config->DstMemory == 'g')) CHKRET(clFlush(ocl_command_queues[num_device][0]), "Error in clFlush");
		if (flushKernel) CHKRET(clFlush(ocl_command_queues[num_device][2]), "Error in clFlush");
	}
	else
	{
		if (prepareM || prepareN || (Config->GPU_C && Config->DstMemory == 'g')) CHKRET(clFlush(ocl_command_queues[num_device][j]), "Error in clFlush");
	}

	if (Config->VerboseTiming && Config->GPU_C == 1)
	{
		for (int i = 0;i < 3;i++) //Queues 0 and 2 of AlternateSimpleQueueing for WriteBuffer and ExecuteKernel
		{
			if (i == 1) continue;
			if (!Config->AlternateSimpleQueuing) i = j;
			CHKRET(clFinish(ocl_command_queues[num_device][i]), "Error in clFinish");
			if (Config->AlternateSimpleQueuing) break;
		}
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
			CHKRET(clReleaseMemObject(ocl_abuffers[i][j]), "Error in clReleaseMemObject");
		}
		for (int j = 0;j < obuffercount;j++)
		{
			if (Config->DstMemory == 'g') CHKRET(clReleaseMemObject(ocl_cbuffers[i][j]), "Error in clReleaseMemObject");
			if (Config->GPU_C == 0)
			{
				CHKRET(clEnqueueUnmapMemObject(ocl_command_queues[i][0], ocl_tmp_cbuffers[i][j], ocl_tmp_cbuffers_ptr[i][j], 0, NULL, NULL), "Error in clEnqueueUnmapMemObject");
				CHKRET(clFinish(ocl_command_queues[i][0]), "Error in clFinish");
				CHKRET(clReleaseMemObject(ocl_tmp_cbuffers[i][j]), "Error in clReleaseMemObject");
			}
		}
		for (int j = 0;j < (Config->GPU_C ? obuffercount : ibuffercount);j++)
		{
			if (Config->GPU_C == 0)
			{
				CHKRET(clEnqueueUnmapMemObject(ocl_command_queues[i][0], ocl_tmp_abuffers[i][j], ocl_tmp_abuffers_ptr[i][j], 0, NULL, NULL), "Error in clEnqueueUnmapMemObject");
				CHKRET(clEnqueueUnmapMemObject(ocl_command_queues[i][0], ocl_tmp_bbuffers[i][j], ocl_tmp_bbuffers_ptr[i][j], 0, NULL, NULL), "Error in clEnqueueUnmapMemObject");
				CHKRET(clFinish(ocl_command_queues[i][0]), "Error in clFinish");
			}
			
			if (Config->GPU_C == 0 || Config->AlternateSimpleQueuing)
			{
				CHKRET(clReleaseMemObject(ocl_tmp_bbuffers[i][j]), "Error in clReleaseMemObject");
			}
			CHKRET(clReleaseMemObject(ocl_tmp_abuffers[i][j]), "Error in clReleaseMemObject");
		}
		for (int j = 0;j < bbuffers[i];j++)
		{
			CHKRET(clReleaseMemObject(ocl_bbuffers[i][j]), "Error in clReleaseMemObject");
		}
		for (int j = 0;j < 3 + 1 + (Config->GPU_C ? 1 : 0);j++)
		{
			CHKRET(clReleaseKernel(ocl_kernel[i][j]), "Error in clReleaseKernel");
			if (Config->AsyncSideQueue && j < 2 + (Config->AsyncDTRSM ? 2 : 0)) CHKRET(clReleaseKernel(ocl_async_kernel[i][j]), "Error in clReleaseKernel");
			if (config_backend->kernelLib == NULL && i == nDevices - 1) CHKRET(clReleaseProgram(ocl_program[j]), "Error in clReleaseProgram");
		}

		if (Config->AsyncSideQueue)
		{
			CHKRET(clReleaseCommandQueue(ocl_async_queue[i]), "Error in clReleaseCommandQueue");

			for (int j = 0;j < 4;j++)
			{
				CHKRET(clReleaseMemObject(ocl_async_buffers[i][j]), "Error in clReleaseMemObject");
			}
		}
	}
	return(0);
}

int caldgemm_opencl::UseOutputPthreads() {return(!Config->GPU_C);}
int caldgemm_opencl::UseInputPthreads() {return(!Config->GPU_C);}
int caldgemm_opencl::UseMutexPerDevice() {return(0);}
int caldgemm_opencl::AllowCPUFallback() {return(0);}


int caldgemm_opencl::CheckAlternateTilesRemainingSimpleQuieing()
{
	if (AlternateLookaheadTilesFull)
	{
		CHKRET(clWaitForEvents(AlternateLookaheadTilesFull, AlternateLookaheadTilesRemaining_events), "Error waiting for alternate lookahead tiles events");
	}
	AlternateLookaheadDoneMutex.Unlock();
	return(0);
}

void caldgemm_opencl::Preallocate()
{
	caldgemm::Preallocate();
	simple_queue_events[0][0] = new caldgemm_opencl_simple_queue_event[nDevices * (Config->PreallocData + Config->PreallocData)];
	simple_queue_event_requested[0][0][0] = new int[nDevices * obuffercount * (Config->PreallocData + Config->PreallocData)];
	memset(simple_queue_events[0][0], 0, nDevices * (Config->PreallocData + Config->PreallocData) * sizeof(caldgemm_opencl_simple_queue_event));
	memset(simple_queue_event_requested[0][0][0], 0, nDevices * obuffercount * (Config->PreallocData + Config->PreallocData) * sizeof(int));
	AlternateLookaheadTilesRemaining_events = new cl_event[Config->PreallocData * Config->PreallocData];
	memset(AlternateLookaheadTilesRemaining_events, 0, Config->PreallocData * Config->PreallocData * sizeof(cl_event));
}

void caldgemm_opencl::PreallocateFree()
{
	caldgemm::PreallocateFree();
	delete[] simple_queue_events[0][0];
	delete[] simple_queue_event_requested[0][0][0];
	delete[] AlternateLookaheadTilesRemaining_events;
}

void caldgemm_opencl::SetupSimpleQueue(size_t mb, size_t nb)
{
	if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g') return;
	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < obuffercount;j++)
		{
			for (int k = 0;k < 2;k++)
			{
				if (i || j || k)
				{
					simple_queue_event_requested[i][j][k] = &simple_queue_event_requested[0][0][0][(k ? mb : 0) + j * (mb + nb) + i * obuffercount * (mb + nb)];
					if (j == 0)
					{
						simple_queue_events[i][k] = &simple_queue_events[0][0][(k ? mb : 0) + i * (mb + nb)];
					}
				}
			}
		}
	}
}

int caldgemm_opencl::FinishDataInit()
{
	finishData = new finishStructOpenCL;
	return(finishData == NULL);
}

void caldgemm_opencl::FinishDataFill()
{
	if (Config->PipelinedOperation && !CPUOnlyRun && pipelinedRun)
	{
		memcpy(((finishStructOpenCL*) finishData)->StartMarker, StartMarker, sizeof(StartMarker));
		memcpy(((finishStructOpenCL*) finishData)->MidMarker, MidMarker, sizeof(MidMarker));
		memcpy(((finishStructOpenCL*) finishData)->EndMarker, EndMarker, sizeof(EndMarker));
		((finishStructOpenCL*) finishData)->MidMarkerDone = false;
		((finishStructOpenCL*) finishData)->EndMarkerDone = false;
	}
}

int caldgemm_opencl::RunCALDGEMM_Init()
{
	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			ocl_conversion_events_use[i][j] = 0;
		}
		if (Config->NoConcurrentKernels)
		{
			last_device_kernel[i] = 0;
		}
	}
	if (Config->SimpleGPUQueuing)
	{
		const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
		const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;

		if (((int) (DGEMM_favor_m ? nb : mb) + nDevices - 1) / nDevices > min_bbuffers)
		{
			fprintf(STD_OUT, "SimpleGPUQueuing can only work if [Number of BBuffers] * [Number of GPUs] > [Number of Blocks in one dimension] (bbuffers %d, devices %d, blocks %d)\n", min_bbuffers, nDevices, (int) (DGEMM_favor_m ? nb : mb));
			return(1);
		}

		if (!Config->PreallocData)
		{
			simple_queue_events[0][0] = new caldgemm_opencl_simple_queue_event[nDevices * (mb + nb)];
			simple_queue_event_requested[0][0][0] = new int[nDevices * obuffercount * (mb + nb)];
			if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && AlternateLookaheadTilesFull)
			{
				AlternateLookaheadTilesRemaining_events = new cl_event[AlternateLookaheadTilesFull];
			}
		}

		SetupSimpleQueue(mb, nb);
		if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g')
		{
			memset(alternateSimpleQueueCBuffferEvent, 0, nDevices * obuffercount * sizeof(alternateSimpleQueueCBuffferEventStruct));
		}
		else
		{
			memset(simple_queue_events[0][0], 0, nDevices * (mb + nb) * sizeof(caldgemm_opencl_simple_queue_event));
			memset(simple_queue_event_requested[0][0][0], 0, nDevices * obuffercount * (mb + nb) * sizeof(int));
		}
		if (Config->AlternateSimpleQueuing)
		{
			memset(alternateSimpleQueueEvent_tmp_abuffers, 0, nDevices * obuffercount * sizeof(cl_event));
			memset(alternateSimpleQueueEvent_tmp_bbuffers, 0, nDevices * obuffercount * sizeof(cl_event));
		}
		memset(&simple_queue_event_kernels[0][0][0], 0, nDevices * ibuffercount * obuffercount * sizeof(cl_event));
	}
	
	if (Config->PipelinedOperation && !CPUOnlyRun && pipelinedRun)
	{
		for (int i = 0;i < nDevices;i++)
		{
			for (int j = 0;j < obuffercount;j++)
			{
				CHKRET(clEnqueueMarkerWithWaitList(ocl_command_queues[i][j], 0, NULL, &StartMarker[i][j]), "Error enqueuing OpenCL marker");
			}
		}
		
		if (Config->PipelinedMidMarker)
		{
			memset(MidMarker, 0, sizeof(MidMarker));
		}
		if (Config->PipelinedOperation && finishData->running) memset(pipelinedModeStartBarrierDone, 0, sizeof(pipelinedModeStartBarrierDone));
	}
	return(0);
}

int caldgemm_opencl::RunCALDGEMM_Exit()
{
	if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && Config->SimpleGPUQueuing)
	{
		AlternateLookaheadDoneMutex.Lock();
		if (AlternateLookaheadTilesFull)
		{
			for (int i = 0;i < AlternateLookaheadTilesFull;i++)
			{
				CHKRET(clReleaseEvent(AlternateLookaheadTilesRemaining_events[i]), "Error releasing alternate lookahead tiles event");
			}
		}
	}

	if (Config->SimpleGPUQueuing)
	{
		if (!(Config->AlternateSimpleQueuing && Config->DstMemory == 'g'))
		{
			for (int i = 0;i < nDevices;i++)
			{
				for (int j = 0;j < 2;j++)
				{
					const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
					const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;

					for (int k = 0;k < (int) (j ? nb : mb);k++)
					{
						if (simple_queue_events[i][j][k].event != NULL) CHKRET(clReleaseEvent(simple_queue_events[i][j][k].event), "Error in clReleaseEvent");
					}
				}
			}
		}
		for (int i = 0;i < nDevices;i++)
		{
			for (int j = 0;j < obuffercount;j++)
			{
				if (!Config->AlternateSimpleQueuing || j == (Config->DstMemory == 'g' ? 1 : 2))
				{
					if (Config->PipelinedOperation && !CPUOnlyRun && pipelinedRun)
					{
						if (Config->PipelinedMidMarker && MidMarker[i][j] == 0)
						{
							CHKRET(clEnqueueMarkerWithWaitList(ocl_command_queues[i][j], 0, NULL, &MidMarker[i][j]), "Error enqueuing OpenCL marker");
							if (Config->Debug) fprintf(STD_OUT, "Enqueueing Fake Mid Marker at end (Device %d Queue %d) (Event %lld)\n", i, j, (long long int) MidMarker[i][j]);
						}
						CHKRET(clEnqueueMarkerWithWaitList(ocl_command_queues[i][j], 0, NULL, &EndMarker[i][j]), "Error enqueuing OpenCL marker");
					}
					else
					{
						CHKRET(clFinish(ocl_command_queues[i][j]), "Error in clFinish");
					}
				}
				if (Config->AlternateSimpleQueuing)
				{
					if (alternateSimpleQueueEvent_tmp_abuffers[i][j] != 0) CHKRET(clReleaseEvent(alternateSimpleQueueEvent_tmp_abuffers[i][j]), "Error releasing event");
					if (alternateSimpleQueueEvent_tmp_bbuffers[i][j] != 0) CHKRET(clReleaseEvent(alternateSimpleQueueEvent_tmp_bbuffers[i][j]), "Error releasing event");
				}
				if (Config->AlternateSimpleQueuing && alternateSimpleQueueCBuffferEvent[i][j].must_release)
				{
					CHKRET(clReleaseEvent(alternateSimpleQueueCBuffferEvent[i][j].event), "Error releasing event");
				}
				for (int k = 0;k < ibuffercount;k++)
				{
					if (simple_queue_event_kernels[i][k][j] != 0) CHKRET(clReleaseEvent(simple_queue_event_kernels[i][k][j]), "Error releasing event");
				}
			}
		}
	}
	else
	{
		for (int i = 0;i < nDevices;i++)
		{
			for (int j = 0;j < 2;j++)
			{
				if (ocl_conversion_events_use[i][j])
				{
					CHKRET(clReleaseEvent(ocl_conversion_events[i][j]), "Error in clReleaseEvent");
				}
			}
			if (Config->NoConcurrentKernels && last_device_kernel[i] != 0 && (Config->DstMemory == 'g' && (Config->GPU_C || Config->ImplicitDriverSync)))
			{
				CHKRET(clReleaseEvent(last_device_kernel[i]), "Error in clReleaseEvent");
			}
		}
	}
	if (Config->SimpleGPUQueuing && !Config->PreallocData)
	{
		delete[] simple_queue_events[0][0];
		delete[] simple_queue_event_requested[0][0][0];
		if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && AlternateLookaheadTilesFull)
		{
			delete[] AlternateLookaheadTilesRemaining_events;
		}
	}
	return(0);
}

int caldgemm_opencl::RunCALDGEMM_Finish()
{
	if (!finishData->running) return(0);
	cl_ulong gputime = 0;
	for (int i = 0;i < nDevices;i++)
	{
		if (!Config->AlternateSimpleQueuing)
		{
			CHKRET(clWaitForEvents(obuffercount, ((finishStructOpenCL*) finishData)->EndMarker[i]), "Error waiting to finish DGEMM (%d)", i);
		}
		else
		{
			CHKRET(clWaitForEvents(1, &((finishStructOpenCL*) finishData)->EndMarker[i][Config->DstMemory == 'g' ? 1 : 2]), "Error waiting to finish DGEMM (%d)", i);
		}

		cl_ulong minstart = ~0, maxend = 0; 
		for (int j = 0;j < obuffercount;j++)
		{
			if (!Config->AlternateSimpleQueuing || j == (Config->DstMemory == 'g' ? 1 : 2))
			{
				if (CALDGEMM_OPENCL_PROFILED_PIPELINE)
				{
					cl_ulong start, end;
					CHKRET(clGetEventProfilingInfo(((finishStructOpenCL*) finishData)->EndMarker[i][j], CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL), "Error getting event profiling info");
					CHKRET(clGetEventProfilingInfo(((finishStructOpenCL*) finishData)->StartMarker[i][j], CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL), "Error getting event profiling info");
					if (start < minstart) minstart = start;
					if (end > maxend) maxend = end;
				}
				CHKRET(clReleaseEvent(((finishStructOpenCL*) finishData)->StartMarker[i][j]), "Error in clReleaseEvent");
				CHKRET(clReleaseEvent(((finishStructOpenCL*) finishData)->EndMarker[i][j]), "Error in clReleaseEvent");
				if (Config->PipelinedMidMarker) CHKRET(clReleaseEvent(((finishStructOpenCL*) finishData)->MidMarker[i][j]), "Error in clReleaseEvent");
			}
		}
		if (maxend - minstart > gputime) gputime = maxend - minstart;
	}
	if (CALDGEMM_OPENCL_PROFILED_PIPELINE && gputime == 0)
	{
		fprintf(STD_OUT, "Error obtaining times from OpenCL runtime\n");
		return(1);
	}
	//if ((double) gputime * 1e-9 > finishData->GPUTimer)
	if (gputime > 0) finishData->GPUTimer = (double) gputime * 1e-9;
	if (finishData->GPUTimer > finishData->System) finishData->System = finishData->GPUTimer;
	return(0);
}

int caldgemm_opencl::CheckParams()
{
    return(0);
}

int caldgemm_opencl::WaitForCALDGEMMProgress(size_t n)
{
	if (Config->PipelinedOperation && finishData->running)
	{
		if (((finishStructOpenCL*) finishData)->EndMarkerDone) return(0);
		if (n && n < finishData->MidMarkerPos)
		{
			if (((finishStructOpenCL*) finishData)->MidMarkerDone)
			{
				if (Config->Debug) fprintf(STD_OUT, "Wait for Mid Marker (already done, Need %lld, marker %lld)\n", (long long int) n, (long long int) finishData->MidMarkerPos);
				return(0);
			}
			if (Config->Debug) fprintf(stderr, "Waiting for Mid Marker (Need %lld, marker %lld) ", (long long int) n, (long long int) finishData->MidMarkerPos);
			for (int i = 0;i < nDevices;i++)
			{
				if (Config->Debug)
				{
					fprintf(stderr, "Dev %d ", i);
					for (int j = 0;j < obuffercount;j++) fprintf(stderr, "%lld ", (long long int) ((finishStructOpenCL*) finishData)->MidMarker[i][j]);
				}
				if (!Config->AlternateSimpleQueuing)
				{
					CHKRET(clWaitForEvents(obuffercount, ((finishStructOpenCL*) finishData)->MidMarker[i]), "Error waiting for MidMarker");
				}
				else
				{
					CHKRET(clWaitForEvents(1, &((finishStructOpenCL*) finishData)->MidMarker[i][Config->DstMemory == 'g' ? 1 : 2]), "Error waiting for MidMarker");
				}
			}
			if (Config->Debug) fprintf(STD_OUT, "\nMid Marker Reached\n");
			((finishStructOpenCL*) finishData)->MidMarkerDone = true;
			return(0);
		}
		if (Config->Debug) fprintf(STD_OUT, "Waiting for End Marker (Need %lld, marker %lld)\n", (long long int) n, (long long int) gpu_n);
		for (int i = 0;i < nDevices;i++)
		{
			if (!Config->AlternateSimpleQueuing)
			{
				CHKRET(clWaitForEvents(obuffercount, ((finishStructOpenCL*) finishData)->EndMarker[i]), "Error waiting for EndMarker");
			}
			else
			{
				CHKRET(clWaitForEvents(1, &((finishStructOpenCL*) finishData)->EndMarker[i][Config->DstMemory == 'g' ? 1 : 2]), "Error waiting for EndMarker");
			}
		}
		if (Config->Debug) fprintf(STD_OUT, "End Marker Reached\n");
		((finishStructOpenCL*) finishData)->EndMarkerDone = true;
	}
	return(0);
}

#define MAX_GPU_MEM_COUNT 64
static caldgemm_opencl::gpu_mem_struct_opencl gpu_mem[MAX_GPU_MEM_COUNT];
static int nGPUMEM = 0;

int caldgemm_opencl::GetMemoryInfo(cl_mem* mem, void** ptr, size_t* offset, const void* addr)
{
	for (int i = 0;i < nGPUMEM;i++)
	{
		if (((size_t) addr >= (size_t) gpu_mem[i].ptr) && ((size_t) ((char*) addr - (char*) gpu_mem[i].ptr) < gpu_mem[i].size))
		{
			if (mem != NULL) *mem = gpu_mem[i].mem_obj;
			if (ptr != NULL) *ptr = gpu_mem[i].ptr;
			if (offset != NULL) *offset = ((char*) addr - (char*) gpu_mem[i].ptr);
			return(0);
		}
	}
	return(1);
}

double* caldgemm_opencl::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible, bool interleave)
{
	if (gpuaccessible && Config->GPU_C)
	{
		if (nGPUMEM == MAX_GPU_MEM_COUNT)
		{
			fprintf(STD_OUT, "Cannot allocated more GPU memory, increase MAX_GPU_MEM_COUNT\n");
			return(0);
		}
		if (interleave)
		{
			unsigned long nodemask = 0xffffff;
			if (syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8) != 0)
			{
				fprintf(STD_OUT, "Error setting memory policy\n");
			}
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
		if (huge_pages)
		{
			size_t psize = 2048 * 1024 / 8;
			if (nDoubles % psize) nDoubles += psize - nDoubles % psize;
		}
		gpu_mem[nGPUMEM].size = nDoubles * sizeof(double);
		gpu_mem[nGPUMEM].mem_obj = clCreateBuffer(ocl_context, mem_flags, nDoubles * sizeof(double), NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error allocating memory (clCreateBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(0);
		}

		/*
		NOT NECESSARY, can get pointer from GPU mapping
		gpu_mem[nGPUMEM].ptr = clEnqueueMapBuffer(ocl_command_queue_cpu, gpu_mem[nGPUMEM].mem_obj, CL_TRUE, 0, 0, nDoubles * sizeof(double), 0, NULL, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS)
		{
			fprintf(STD_OUT, "Error allocating memory (clEnqueueMapBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
			return(0);
		}*/
		if (page_locked)
		{
			//mlock(gpu_mem[nGPUMEM].ptr, nDoubles * sizeof(double));
		}
		for (int i = 0;i < nDevicesInitialized;i++)
		{
			fprintf(stderr, "%d", i);
			void* tmp_ptr = clEnqueueMapBuffer(ocl_command_queues[i][0], gpu_mem[nGPUMEM].mem_obj, CL_TRUE, 0, 0, nDoubles * sizeof(double), 0, NULL, NULL, &ocl_error);
			if (i == 0) gpu_mem[nGPUMEM].ptr = tmp_ptr;
			if (ocl_error != CL_SUCCESS || tmp_ptr != gpu_mem[nGPUMEM].ptr)
			{
				fprintf(STD_OUT, "Error allocating memory (clEnqueueMapBuffer) (%d: %s)\n", ocl_error, opencl_error_string(ocl_error));
				return(0);
			}
		}
		
		if (interleave)
		{
			if (syscall(SYS_set_mempolicy, MPOL_DEFAULT, NULL) != 0)
			{
				fprintf(STD_OUT, "Error setting memory policy\n");
			}
		}
		return((double*) gpu_mem[nGPUMEM++].ptr);
	}
	else
	{
		return (caldgemm::AllocMemory(nDoubles, page_locked, huge_pages, gpuaccessible, interleave));
	}
}

int caldgemm_opencl::FreeMemory(double* ptr, bool gpuaccessible)
{
	if (gpuaccessible && Config->GPU_C)
	{
		for (int i = 0;i < nGPUMEM;i++)
		{
			if (gpu_mem[i].ptr == (void*) ptr)
			{
				for (int j = 0;j < nDevicesInitialized;j++)
				{
					CHKRET(clEnqueueUnmapMemObject(ocl_command_queues[j][0], gpu_mem[i].mem_obj, gpu_mem[i].ptr, 0, NULL, NULL), "Error in clEnqueueUnmapMemObject");
					CHKRET(clFinish(ocl_command_queues[j][0]), "Error in clFinish");
				}
				//clEnqueueUnmapMemObject(ocl_command_queue_cpu, gpu_mem[i].mem_obj, gpu_mem[i].ptr, 0, NULL, NULL); //see above
				//CHKRET(clFinish(ocl_command_queue_cpu), "Error in clFinish");
				CHKRET(clReleaseMemObject(gpu_mem[i].mem_obj), "Error in clReleaseMemObject");
				gpu_mem[i] = gpu_mem[--nGPUMEM];
				return(0);
			}
		}
	}
	return(caldgemm::FreeMemory(ptr));
}

int caldgemm_opencl::CaldgemmCustomAutoHeight(size_t MaxGpuM, size_t MaxGpuN, int nDevices)
{
	if (config_backend->kernelLib != NULL)
	{
		size_t tmpHeight = kernelLibGetAutoHeight(MaxGpuM, MaxGpuN, nDevices, Config->Width);
		if (tmpHeight)
		{
			Config->Height = tmpHeight;
			return(1);
		}
	}
	return(0);
}
int caldgemm_opencl::CaldgemmCustomModHeight(size_t MOD_OVER, size_t MOD_GPU)
{
	if (config_backend->kernelLib != NULL)
	{
		kernelLibModHeight(MOD_OVER, MOD_GPU);
		return(1);
	}
	else
	{
		return(0);
	}
}
