/**
 * Interface of the CALDGEMM library.
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

#ifndef CALDGEMM_OPENCL_H
#define CALDGEMM_OPENCL_H

#include <CL/opencl.h>

#include "caldgemm.h"

#if !defined(CALDGEMM_TRANSPOSED_A) & !defined(CALDGEMM_TRANSPOSED_B)
#error You must either defined CALDGEMM_TRANSPOSED_A or CALDGEMM_TRANSPOSED_B for the OpenCL backend
#endif

#ifndef _WIN32
#define HINSTANCE void*
#endif

class caldgemm_opencl : public caldgemm
{
public:
	caldgemm_opencl();
	virtual ~caldgemm_opencl();

	class caldgemm_config_backend_opencl : public caldgemm_config_backend
	{
	public:
		virtual ~caldgemm_config_backend_opencl();
		caldgemm_config_backend_opencl() {size = sizeof(*this);kernelLib = NULL;}
		char* kernelLib;
	};
	virtual caldgemm_config_backend* create_caldgemm_config_backend();
	

private:
	virtual int UseOutputPthreads();
	virtual int UseInputPthreads();
	virtual int UseMutexPerDevice();

	virtual int DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA);
	virtual	int Initialize (bool nocalinit);
	virtual int ValidateRuntime();
	virtual int CheckDevices();
	virtual int InitDevices();
	virtual int ReinitDevices();
	virtual int InitConstantData(double alpha);
	virtual int ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn);
	virtual int ExitRuntime();
	virtual int ExitDevices();
	virtual int WaitForEvent(int, int, int);
	virtual int FetchResult(int device, int j, int m, int n, int mustlock = 0);
	virtual int CheckDMAQueue(int device, int forcej = -1);
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch);
	virtual int RunCALDGEMM_Init();
	virtual int RunCALDGEMM_Exit();

	virtual double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible = false, bool Cmatrix = false, bool interleave = false);
	virtual void FreeMemory(double* ptr, bool gpuaccessible = false);

	cl_platform_id ocl_platform;
	cl_device_id ocl_devices[max_devices + 1]; //+1 for cpu
	cl_context ocl_context;
	cl_command_queue ocl_command_queues[max_devices][obuffercount];
	cl_command_queue ocl_command_queue_cpu;
	cl_mem ocl_abuffers[max_devices][ibuffercount];
	cl_mem ocl_bbuffers[max_devices][max_bbuffers];
	cl_mem ocl_cbuffers[max_devices][obuffercount];
	cl_mem ocl_tmp_abuffers[max_devices][ibuffercount > obuffercount ? ibuffercount : obuffercount];
	cl_mem ocl_tmp_bbuffers[max_devices][ibuffercount > obuffercount ? ibuffercount : obuffercount];
	cl_mem ocl_tmp_cbuffers[max_devices][obuffercount];
	cl_event ocl_events[max_devices][obuffercount];
	cl_program ocl_program[4];
	cl_kernel ocl_kernel[max_devices][5];

	double* ocl_tmp_abuffers_ptr[max_devices][ibuffercount];
	double* ocl_tmp_bbuffers_ptr[max_devices][ibuffercount];
	double* ocl_tmp_cbuffers_ptr[max_devices][obuffercount];

	cl_event ocl_conversion_events[max_devices][2];
	int ocl_conversion_events_use[max_devices][2];

	static const char *OCLKernel, *OCLKernelALPHA1, *OCLKernelLinpack, *OCLConvertKernel, *OCLConvertKernelTex;

	int WaitForEventAndRelease(cl_event* pEvent, int lock = -1);
	int divideBuffer(double* src, size_t pitch_src, double* dest, size_t nSrcRows, size_t nSrcCols, bool transpose);

	double* C_matrix_base;
	cl_mem* C_matrix_base_obj;

	static const int GROUP_SIZE_X = 16, GROUP_SIZE_Y = 16, GROUP_COUNT_X = 16, GROUP_COUNT_Y = 16; //Group size and count for conversion kernels.

	caldgemm_config_backend_opencl* config_backend;

	HINSTANCE kernelLib;
	cl_kernel (*kernelLibCreate) (cl_context* context, int nDevices, cl_device_id* devices, int kernelType, int k, int betazero);
	void (*kernelLibQuerySettings) (int* tiling_x, int* tiling_y, bool* transposeA, bool* transposeB, bool* texture_buffers, int* group_size_x, int* group_size_y);

public:
	struct gpu_mem_struct_opencl
	{
		void* ptr;
		cl_mem mem_obj;
	};
};

#endif
