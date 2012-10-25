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

#ifndef CALDGEMM_CAL_H
#define CALDGEMM_CAL_H

#include <cal.h>
#include <cal_ext.h>
#include <calcl.h>

#include <emmintrin.h>

#include "caldgemm.h"

class caldgemm_cal : public caldgemm
{
public:
	caldgemm_cal();
	virtual ~caldgemm_cal();

	virtual bool cpuUsed(int cpu);
	virtual double getMaxGPUTemperature();

private:
	int adl_util_initialized;
	virtual int UseOutputPthreads();
	virtual int UseInputPthreads();
	virtual int UseMutexPerDevice();

	unsigned int numInputs, numOutputs, numConstantBuffers;

#ifdef CALDGEMM_44
#ifdef CALDGEMM_SINGLE_BUFFER
	static const unsigned int dwBuffersA = 1;
#elif !defined(CALDGEMM_48) & !defined(CALDGEMM_DOUBLE_BUFFERS)
	static const unsigned int dwBuffersA = 2;
#else
	static const unsigned int dwBuffersA = 4;
#endif
#ifdef CALDGEMM_SINGLE_BUFFER
	static const unsigned int dwBuffersB = 1;
#elif !defined(CALDGEMM_84) & !defined(CALDGEMM_DOUBLE_BUFFERS)
	static const unsigned int dwBuffersB = 2;
#else
	static const unsigned int dwBuffersB = 4;
#endif
#else //CALDGEMM_44
#ifdef CALDGEMM_TRANSPOSED_A
	static const unsigned int dwBuffersA = 2;
#else
	static const unsigned int dwBuffersA = 8;
#endif
	static const unsigned int dwBuffersB = 2;
#endif //CALDGEMM_44

#ifdef CALDGEMM_USE_MEMEXPORT
	static const unsigned int dwBuffersC = 1;
#else
	static const unsigned int dwBuffersC = 8;
#endif

	struct BufferProperties
	{
		union
		{
			float*  ptr_float;
			unsigned int*   ptr_uint;
			int*    ptr_int;
			double* ptr_double;
			char*   ptr_char;
			void*   ptr_void;
		};
		unsigned int Width;
		unsigned int Height;
		unsigned int VectorSize;
		unsigned int DataSize;

		bool CALMemory;
		CALresource res;
		CALmem mem;
		CALmem dstMem;
		unsigned int pitch;
		CALresource tmpres;
		CALmem tmpmem;
		
		BufferProperties* conversionBuffer;
	};

	int divideBuffer(BufferProperties* dst, double* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers, bool transpose CALDGEMM_DIVBUFA);
	int mergeBuffers(double* dst, BufferProperties* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers);
	void checkCalPatch();
	void cal_init_constant_data(BufferProperties* &data, double alpha);
	virtual int DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA);
	virtual int RunCALDGEMM_Init();
	virtual int RunCALDGEMM_Exit();

	struct CALVersion {unsigned int major, minor, imp;};

	virtual	int Initialize (bool nocalinit);
	int SetupKernel(const char* ILKernel, CALmodule* module, CALcontext* ctx, unsigned int device_num, bool disassemble = false);
	int RunProgram(CALcontext* ctx, CALmodule* module, unsigned int Width, unsigned int Height, CALevent* event);
	int CleanupData(CALcontext* ctx, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext, unsigned int num_device);
	int Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext, unsigned int num_device);
	int SetupData(CALmodule* module, CALresource* &_Res, BufferProperties* &data, CALdevice* device, CALcontext* ctx, unsigned int numInputs, unsigned int numOutputs, unsigned int numConstantBuffers, CALname** ctxProgNames, int nContext, unsigned int num_device);
	int CopyDataFromGPU(int nDevice, CALresource* _Res, BufferProperties* data, unsigned int num, int nContext, size_t lastm, size_t lastn);
	int CopyDataToGPU(int nDevice, CALresource* _Res, BufferProperties* data, unsigned int num, int nContext, bool constants, BufferProperties* dest_data = NULL);
	int ValidateCALRuntime();

	BufferProperties* datas[max_devices][max_bbuffers];
	CALdevice devices[max_devices];
	CALcontext ctxs[max_devices];
	CALresource* resourceHandlers[max_devices][max_bbuffers];
	CALmodule modules[max_devices][kernel_count];
	CALmodule modulesConvert[max_devices];
	CALmodule fakeModule;
	CALname *progNames[max_devices][kernel_count];
	CALname progNamesConvert[max_devices][2 * dwBuffersA];
	CALevent events[max_devices][obuffercount];
	unsigned int device_nums[max_devices];

	static const char *ILKernel, *ILKernelALPHA1, *ILKernelLinpack, *ILFakeKernel, *ILConvertKernel;

	virtual int ValidateRuntime();
	virtual int CheckDevices();
	virtual int InitDevices();
	virtual int ReinitDevices();
	virtual int InitConstantData(double alpha);
	virtual int ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn);
	virtual int ExitRuntime();
	virtual int ExitDevices();
	virtual int WaitForEvent(int, int, int lock = 0);
	virtual int FetchResult(int device, int j, int m, int n);
	virtual int CheckDMAQueue(int device, int forcej = -1);
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch);
	virtual int reserve_cpu_cores();
};

#endif
