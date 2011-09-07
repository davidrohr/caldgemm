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

#define ERRRET(...) {fprintf(STD_OUT, __VA_ARGS__);return(1);}

caldgemm_opencl::caldgemm_opencl() : caldgemm()
{
}

caldgemm_opencl::~caldgemm_opencl()
{
}

int caldgemm_opencl::WaitForEvent(int a, int b) {return(0);}

int caldgemm_opencl::Initialize(int deviceNum, bool nocalinit)
{
	fprintf(STD_OUT, "OPENCL Initialice\n");

	if (deviceNum < 0) deviceNum = 0;
	gpu_available = (nDevices > 0);

	cl_int ocl_error;

	cl_device_id* devices = new cl_device_id[nDevices];
	if (devices == NULL) ERRRET("Memory allocation error\n");
	if (clGetDeviceIDs(ocl_platform, CL_DEVICE_TYPE_ALL, nDevices, devices, NULL) != CL_SUCCESS) ERRRET("Error getting OpenCL devices\n");

	for (unsigned int i = 0;i < nDevices;i++)
	{
		char device_vendor[64], device_name[64];
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, device_name, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, device_vendor, NULL);
		if (Config->Debug) fprintf(STD_OUT, "Device %d: %s %s\n", i, device_vendor, device_name);
	}

	if (deviceNum >= nDevices) ERRRET("OpenCL Device %d not available\n", deviceNum);
	ocl_device = devices[deviceNum];
	delete[] devices;

	ocl_context = clCreateContext(NULL, 1, &ocl_device, NULL, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS) ERRRET("Error creating OpenCL context\n");

	ocl_command_queue = clCreateCommandQueue(ocl_context, ocl_device, 0, &ocl_error);
	if (ocl_error != CL_SUCCESS) ERRRET("Error creating OpenCL command queue\n");

	for (int i = 0;i < 2;i++)
	{
		ocl_buffers[i] = clCreateBuffer(ocl_context, i ? CL_MEM_WRITE_ONLY : CL_MEM_READ_ONLY, 1024 * 1024, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS) ERRRET("Error allocating device memory\n");
	}
	return(0);
}

int caldgemm_opencl::ValidateRuntime()
{
	fprintf(STD_OUT, "OPENCL ValidateRuntime\n");

	cl_int ocl_error;

	cl_uint num_platforms;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) ERRRET("Error getting OpenCL Platform Count\n");
	if (num_platforms == 0) ERRRET("No OpenCL Platform found\n");
	if (Config->Debug) fprintf(STD_OUT, "%d OpenCL Platforms found\n", num_platforms);

	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	if (platforms == NULL) ERRRET("Memory allocation error");
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) ERRRET("Error getting OpenCL Platforms\n");

	for (unsigned int i = 0;i < num_platforms;i++)
	{
		char platform_profile[64], platform_version[64], platform_name[64], platform_vendor[64];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 64, platform_profile, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 64, platform_version, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 64, platform_name, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 64, platform_vendor, NULL);
		if (Config->Debug) fprintf(STD_OUT, "Platform %d: (%s %s) %s %s\n", i, platform_profile, platform_version, platform_vendor, platform_name);
	}

	if (CalDGEMM_OpenCL_Platform >= num_platforms) ERRRET("OpenCL Platform %d not available\n", CalDGEMM_OpenCL_Platform);
	ocl_platform = platforms[CalDGEMM_OpenCL_Platform];
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
	fprintf(STD_OUT, "OPENCL ExecuteKernels\n");
	return(0);
}

int caldgemm_opencl::ExitRuntime()
{
	fprintf(STD_OUT, "OPENCL ExitRuntime\n");
	return(0);
}

int caldgemm_opencl::FetchResult(int device, int j, int m, int n)
{
	fprintf(STD_OUT, "OPENCL FetchResult\n");
	return(0);
}

int caldgemm_opencl::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers)
{
	fprintf(STD_OUT, "OPENCL RunMergeBuffers\n");
	return(0);
}

int caldgemm_opencl::DGEMM_prepare(size_t k, int j, unsigned int num_device)
{
	fprintf(STD_OUT, "OPENCL DGEMM_prepare\n");
	return(0);
}

int caldgemm_opencl::ExitDevices()
{
	fprintf(STD_OUT, "OPENCL ExitDevices\n");

	//clReleaseKernel(ocl_kernel);
	//clReleaseProgram(ocl_program);
	for (int i = 0;i < 2;i++)
	{
		clReleaseMemObject(ocl_buffers[i]);
	}
	clReleaseCommandQueue(ocl_command_queue);
	clReleaseContext(ocl_context);

	return(0);
}
