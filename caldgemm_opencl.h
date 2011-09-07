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

class caldgemm_opencl : public caldgemm
{
public:
	caldgemm_opencl();
	virtual ~caldgemm_opencl();

private:
	virtual int DGEMM_prepare(size_t k, int j, unsigned int num_device);
	virtual	int Initialize (int deviceNum, bool nocalinit);
	virtual int ValidateRuntime();
	virtual int CheckDevices();
	virtual int InitDevices();
	virtual int ReinitDevices();
	virtual int InitConstantData(double alpha);
	virtual int ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn);
	virtual int ExitRuntime();
	virtual int ExitDevices();
	virtual int WaitForEvent(int, int);
	virtual int FetchResult(int device, int j, int m, int n);
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers);

	cl_platform_id ocl_platform;
	cl_device_id ocl_device;
	cl_context ocl_context;
	cl_command_queue ocl_command_queue;
	cl_mem ocl_buffers[2];
	cl_program ocl_program;
	cl_kernel ocl_kernel;

	static const int CalDGEMM_OpenCL_Platform = 0;

	static const char *OCLKernel, *OCLKernelALPHA1, *OCLKernelLinpack, *OCLConvertKernel;
};

#endif
