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

#include "caldgemm_config_load.h"

#include <cal.h>
#include <cal_ext.h>
#include <calcl.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <emmintrin.h>
#include <mm3dnow.h>
#include <pthread.h>
#include <signal.h>
typedef int blasint;
extern "C" {
#include <cblas.h>
}

#ifdef ATI_OS_WIN
#include <windows.h>
#endif

#ifdef ATI_OS_LINUX
#include <sys/time.h>
#endif

void* merge_wrapper(void* arg);
void* cblas_wrapper(void* arg);
void* linpack_wrapper(void* arg);

class HighResTimer {

public:
	HighResTimer();
	~HighResTimer();
	void Start();
	void Stop();
	void Reset();
	double GetElapsedTime();

private:

	double Frequency;
	double ElapsedTime;
	double StartTime;
};

class caldgemm
{
	friend void* merge_wrapper(void* arg);
	friend void* cblas_wrapper(void* arg);
	friend void* linpack_wrapper(void* arg);

public:

	caldgemm();
	~caldgemm();

	class caldgemm_config								//Run Parameters
	{
	public:
		caldgemm_config();

		bool AsyncDMA;
		bool AutoHeight;						//Automatically adjust height
		bool DivideToGPU;
		char DstMemory;
		int ImplicitDriverSync;
		bool DynamicSched;
		bool KeepBuffersMapped;
		bool MemPolicy;
		bool MultiThread;
		double GPURatio;
		bool UseCPU;
		bool UseGPU;

		int DeviceNum;

		bool Debug;
		bool DumpMatrix;
		unsigned int Iterations;
		int PinCPU;
		bool Verify;

		size_t Height;							//height of subblock od A, width of subblock of B
		size_t m, n;								//height of A, width of B, must be multiple of height
		size_t Width;							//k for matrix multiply

		bool Disassemble;
		bool PrintILKernel;

		bool AsyncTiming;
		bool DisplayTiming;					//Display Final Timing Information even when quiet
		bool NoPerformanceWarnings;			//Suppress also performance warnings, will usually be shown even in quiet mode
		const char* PreOut;
		bool Quiet;
		bool TabularTiming;
		bool VerboseTiming;

		int LinpackNodes;
		int MPIRank;
		int GPUClock;
		volatile size_t *LinpackSwapN;
		void (*linpack_factorize_function)();
		void (*linpack_broadcast_function)();
		void (*linpack_swap_function)();
	};
	
	//CALDGEMM interface functions
	//Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
	//Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = true, UseGPU = UseCPU = true, GPURatio = 0.66
	//m and n can be defined in the RunCALDGEMM call
	//The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
	int InitCALDGEMM(caldgemm_config* pInfo);
	int ExitCALDGEMM();
	int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = -1, size_t k = -1, size_t n = -1, size_t Apitch = -1, size_t Bpitch = -1, size_t Cpitch = -1, CBLAS_ORDER order = CblasRowMajor, CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans, int ExecuteLinpackCallbacks = 0);
	double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages);
	void FreeMemory(double* ptr);
	void ResetTimers();
	int broadcastcore();
	
	double avggflops;
	int avgngflops;

private:

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
	};
	
	static const int obuffercount = 3;				//Not cal context count but number of copies of data buffers etc.
	static const int max_outputthreads = CALDGEMM_OUTPUT_THREADS_SLOW;
	static const int vcpysize = 16;
	static const int kernel_count = 3;
#ifdef REUSE_BBUFFERS
	static const int max_bbuffers = 21;
	static const int max_bbuffers_g = 16;
#else
	static const int max_bbuffers = 3;
	static const int max_bbuffers_g = 3;
#endif	
	static const unsigned int max_devices = 8;
	int next_buffer_A[max_devices];
	int next_buffer_B[max_devices];
	int buffer_pointers_A[max_devices][2 * max_devices];
	int buffer_pointers_B[max_devices][2 * max_devices];

	int divideBuffer(BufferProperties* dst, double* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers, bool transpose);
	int mergeBuffers(double* dst, BufferProperties* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers);

	int DGEMM_prepare(size_t k, int j, unsigned int num_device);
	inline void DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn);
	inline void WaitForLASWP(size_t n);
	void checkCalPatch();
	void cal_init_constant_data(BufferProperties* &data, double alpha);
	void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2 = NULL);
	int cpuScheduler();
	int getcpumask(cpu_set_t* set);

	struct mergeParameters
	{
		caldgemm* cls;
		double* dst;
		BufferProperties* src;
		int nMergeThread;
		int nContext;
		int num_device;
		bool terminate;
		pthread_mutex_t mergeThreadMutex[2];
		size_t k;
	};
	mergeParameters mParam[max_devices][max_outputthreads];

	pthread_mutex_t obufferMutex[max_devices][obuffercount];

	struct structLinpackParameters
	{
		pthread_mutex_t linpackMutex[2];
		bool terminate;
	} linpackParameters;

	cpu_set_t oldcpumask;
	cpu_set_t gpumask;

	size_t gpu_m, gpu_n;

	bool caldgemm_initialized;

	pthread_mutex_t scheduleMutex;
	volatile long long int gpu_k_barrier, cpu_k_barrier;

	static const int max_linpack_callback_types = 3;

	double linpack_last_mn[max_linpack_callback_types];
	double linpackGPURatios[max_linpack_callback_types];
	double linpackBcastTime[max_linpack_callback_types];
	double linpackCPUDGEMMTime[max_linpack_callback_types];

#if (defined(CALDGEMM_TRANSPOSED_A) | defined(CALDGEMM_TRANSPOSED_B)) & !(defined(CALDGEMM_TRANSPOSED_A) & defined(CALDGEMM_TRANSPOSED_B))
	static const bool buffersSwitchable = true;
#else
	static const bool buffersSwitchable = false;
#endif

	struct CALVersion {unsigned int major, minor, imp;};

	int Initialize (int deviceNum);
	int SetupKernel(const char* ILKernel, CALmodule* module, CALcontext* ctx, unsigned int device_num, bool disassemble = false);
	int RunProgram(CALcontext* ctx, CALmodule* module, unsigned int Width, unsigned int Height, CALevent* event);
	int CleanupData(CALcontext* ctx, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext, unsigned int num_device);
	int Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext, unsigned int num_device);
	CALformat getFormat(unsigned int formatSize, unsigned int dataSize, bool isInt = false);
	unsigned int AnalyzeResults();
	int SetupData(CALmodule* module, CALresource* &_Res, BufferProperties* &data, CALdevice* device, CALcontext* ctx, unsigned int numInputs, unsigned int numOutputs, unsigned int numConstantBuffers, CALname** ctxProgNames, int nContext, unsigned int num_device);
	int CopyDataFromGPU(CALcontext* ctx, CALresource* _Res, BufferProperties* data, unsigned int num, CALevent* event, size_t lastm, size_t lastn);
	int CopyDataToGPU(CALcontext* ctx, CALresource* _Res, BufferProperties* data, unsigned int num, bool constants, CALevent* event, BufferProperties* dest_data = NULL);
	int ValidateCALRuntime();
	void displayMatrixTiming(const char* name);
	bool isDoubleEqual(double a, double b);

	struct TimerInfo
	{
		HighResTimer System, Kernel, CounterDivide, CounterMerge, CounterCopyTo, CounterCopyFrom, CPUTimer, GPUTimer, TotalCPUTimer, ATime, LinpackTimer1, LinpackTimer2, LinpackTimer3, BcastTimer;
		int divideA, divideB;
	} Timers;

	int DumpMatrix(double* A, double* B, double* C, double alpha, double beta, int m, int k, int n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB);

	double* A;
	double* B;
	double* C;

	double Alpha, Beta;

	size_t A_pitch, B_pitch, C_pitch;
	CBLAS_TRANSPOSE TransposeA;
	CBLAS_TRANSPOSE TransposeB;

#ifdef CALDGEMM_44
#if !defined(CALDGEMM_48)
	static const unsigned int dwBuffersA = 2;
#else
	static const unsigned int dwBuffersA = 4;
#endif
#if !defined(CALDGEMM_84)
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
	int bbuffers[max_devices];
	int outputthreads;

	size_t BufferHeight;						//Height to which the buffers were originally initialized
	size_t BufferWidth;							//Same for width

	caldgemm_config* Config;

	BufferProperties* datas[max_devices][max_bbuffers];
	unsigned int numInputs, numOutputs, numConstantBuffers;
	CALdevice devices[max_devices];
	CALcontext ctxs[max_devices];
	CALresource* resourceHandlers[max_devices][max_bbuffers];
	CALmodule modules[max_devices][kernel_count];
	CALmodule fakeModule;
	CALname *progNames[max_devices][kernel_count];
	CALevent events[max_devices][obuffercount];
	unsigned int device_nums[max_devices];
	int nDevices;

	static const char *ILKernel, *ILKernelALPHA1, *ILKernelLinpack, *ILFakeKernel, *ILKernelTorture;

	struct cblasParameters
	{
		caldgemm* cls;
		size_t cblas_size;
		size_t dynamic_run;						//Do an extra dynamic cblas run?, works also as m for the dynamic run
		size_t dynamic_size;					//n for dynamic run
		size_t cpu_k;							//k that cpu will take over from gpu in 3rd phase dynamic run
		size_t dynamic_run2;
		bool borders_done;
		bool terminate;
		pthread_mutex_t cblasMutex[2];
	};

	cblasParameters cParam;
	//For Verfify only
	double* D;

	//For Timing only
	bool CPUOnlyRun;
	int ExecLinpack;
	double gpu_ratio_used;
	double cpu_wait_time;

	bool DGEMM_split_m;							//Splitting direction for CPU/GPU
	bool DGEMM_favor_m;							//Direction of C matrix primary tiling

	size_t orig_m, orig_n;
	double *orig_a, *orig_b, *orig_c;
	
	int buffersMajor[max_devices];
	bool buffersMinor[max_devices][max_bbuffers];

	char hostname[256];							//Store hostname of node for host dependant debug code
	
	int conf_numprocs, conf_cpufreq, conf_numgpus, conf_gpufreq, conf_gpushaders;
};
