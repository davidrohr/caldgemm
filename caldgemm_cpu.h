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

#ifndef caldgemm_cpu_H
#define caldgemm_cpu_H

#include "caldgemm.h"

class caldgemm_cpu : public caldgemm
{
public:
	caldgemm_cpu();
	virtual ~caldgemm_cpu();

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
	virtual int FetchResult(int device, int j, int m, int n);
	virtual int CheckDMAQueue(int device, int forcej = -1);
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch);
	virtual int reserve_cpu_cores();
	virtual int RunCALDGEMM_Init();
	virtual int RunCALDGEMM_Exit();
};

#endif
