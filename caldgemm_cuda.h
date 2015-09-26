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

#ifndef caldgemm_cuda_H
#define caldgemm_cuda_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cublas_v2.h>

#include "caldgemm.h"

class caldgemm_cuda : public caldgemm
{
public:
	caldgemm_cuda();
	virtual ~caldgemm_cuda();

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

	virtual double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible = false, bool interleave = false);
	virtual int FreeMemory(double* ptr, bool gpuaccessible = false);

	int cuda_devices[max_devices];
	cudaStream_t cuda_command_queues[max_devices][obuffercount];
	void* cuda_abuffers[max_devices][ibuffercount];
	void* cuda_bbuffers[max_devices][max_bbuffers];
	void* cuda_cbuffers[max_devices][obuffercount];
	void* cuda_tmp_abuffers[max_devices][obuffercount];
	void* cuda_tmp_bbuffers[max_devices][obuffercount];
	cudaEvent_t cuda_events[max_devices][obuffercount];
        cublasHandle_t cublas_handles[max_devices];

	cudaEvent_t cuda_conversion_events[max_devices][2];
	int cuda_conversion_events_use[max_devices][2];

	int WaitForEventAndRelease(cudaEvent_t* pEvent);

	static const int GROUP_SIZE_X = 16, GROUP_SIZE_Y = 16, GROUP_COUNT_X = 16, GROUP_COUNT_Y = 16;	//Group and block size for conversion kernels and for DGEMM kernel
};

#endif
