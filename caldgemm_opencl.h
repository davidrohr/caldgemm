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

class caldgemm_opencl : public caldgemm
{
public:
	caldgemm_opencl();
	virtual ~caldgemm_opencl();

private:
	virtual int UseOutputPthreads();
	virtual int UseInputPthreads();
	virtual int UseMutexPerDevice();

	virtual int DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0, double* tmpBuffer);
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
	virtual int FetchResult(int device, int j, int m, int n);
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch);
	virtual int reserve_cpu_cores();
	virtual int RunCALDGEMM_Init();
	virtual int RunCALDGEMM_Exit();

	virtual double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible = false, bool Cmatrix = false);
	virtual void FreeMemory(double* ptr, bool gpuaccessible = false);

	cl_platform_id ocl_platform;
	cl_device_id ocl_devices[max_devices];
	cl_context ocl_contexts[max_devices];
	cl_command_queue ocl_command_queues[max_devices][obuffercount];
	cl_mem ocl_abuffers[max_devices][2];
	cl_mem ocl_bbuffers[max_devices][max_bbuffers];
	cl_mem ocl_cbuffers[max_devices][obuffercount];
	cl_mem ocl_tmp_abuffers[max_devices][obuffercount];
	cl_mem ocl_tmp_bbuffers[max_devices][obuffercount];
	cl_event ocl_events[max_devices][obuffercount];
	cl_program ocl_program[max_devices][4];
	cl_kernel ocl_kernel[max_devices][4];

	cl_event ocl_conversion_events[max_devices][2];
	int ocl_conversion_events_use[max_devices][2];

	static const char *OCLKernel, *OCLKernelALPHA1, *OCLKernelLinpack, *OCLConvertKernel;

	int WaitForEventAndRelease(cl_event* pEvent);

	double* C_matrix_base;
	cl_mem* C_matrix_base_obj;

	static const int GROUP_SIZE_X = 16, GROUP_SIZE_Y = 16, GROUP_COUNT_X = 16, GROUP_COUNT_Y = 16;
};

#endif
