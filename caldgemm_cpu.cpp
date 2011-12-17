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

#include "caldgemm_cpu.h"

caldgemm_cpu::caldgemm_cpu() : caldgemm()
{
}

caldgemm_cpu::~caldgemm_cpu()
{
}

int caldgemm_cpu::WaitForEvent(int a, int b, int)
{
	if (Config->Debug) fprintf(STD_OUT, "\tSkipping waiting for event from device %d obuffer %d...\n", b, a);
	return(0);
}

int caldgemm_cpu::Initialize(int deviceNum, bool nocalinit)
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU Initialice\n");

	nDevices = 0;
	gpu_available = 0;

	return(0);
}

int caldgemm_cpu::ValidateRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU ValidateRuntime\n");

	return(0);
}

int caldgemm_cpu::CheckDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU CheckDevices\n");
	return(0);
}

int caldgemm_cpu::InitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU InitDevices\n");

	return(0);
}

int caldgemm_cpu::ReinitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU ReinitDevices\n");
	return(0);
}

int caldgemm_cpu::InitConstantData(double alpha)
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU InitConstantData\n");
	return(0);
}

int caldgemm_cpu::ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn)
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU ExecuteKernels\n");

	fprintf(STD_OUT, "Error: DGEMMPrepareAndExecuteTask shoul never be executed for CALDGEMM_CPU\n");

	return(1);
}

int caldgemm_cpu::ExitRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU ExitRuntime\n");

	return(0);
}

int caldgemm_cpu::FetchResult(int device, int j, int m, int n)
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU FetchResult\n");
	return(0);
}

int caldgemm_cpu::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU RunMergeBuffers\n");
	return(0);
}

int caldgemm_cpu::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0)
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	return(0);
}

int caldgemm_cpu::ExitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CALDGEMM_CPU ExitDevices\n");

	return(0);
}

int caldgemm_cpu::UseOutputPthreads() {return(0);}
int caldgemm_cpu::UseInputPthreads() {return(0);}
int caldgemm_cpu::UseMutexPerDevice() {return(0);}

int caldgemm_cpu::reserve_cpu_cores()
{
	return(0);
}

int caldgemm_cpu::RunCALDGEMM_Init()
{
	return(0);
}

int caldgemm_cpu::RunCALDGEMM_Exit()
{
	return(0);
}

