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

#ifndef CALDGEMM_H
#define CALDGEMM_H

#include "caldgemm_config_load.h"
#ifdef _WIN32

#ifdef INTEL_RUNTIME
#pragma warning(disable : 1786)
#pragma warning(disable : 1478)
#pragma warning(disable : 161)
#pragma warning(disable : 94)
#pragma warning(disable : 1229)
#endif //INTEL_RUNTIME

#ifdef VSNET_RUNTIME
#pragma warning(disable : 4616)
#pragma warning(disable : 4996)
#pragma warning(disable : 1684)
#endif //VSNET_RUNTIME 
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <emmintrin.h>
#ifdef _WIN32
#include "cmodules/pthread_mutex_win32_wrapper.h"
#include "cmodules/sched_affinity_win32_wrapper.h"
#else
#include <pthread.h>
#include <mm3dnow.h>
#endif

#ifdef ATI_OS_WIN
#include <windows.h>
#endif

#ifdef ATI_OS_LINUX
#include <sys/time.h>
#endif

template <class T> T mymin(const T a, const T b) {return(a < b ? a : b);}
template <class T> T mymax(const T a, const T b) {return(a > b ? a : b);}


#include "cmodules/timer.h"

class caldgemm
{
	static void* merge_wrapper(void* arg);
	static void* divide_wrapper(void* arg);
	static void* cblas_wrapper(void* arg);
	static void* linpack_wrapper(void* arg);
protected:
	static const unsigned int max_devices = 8;

public:

	caldgemm();
	virtual ~caldgemm();

	class caldgemm_config						//Run Parameters
	{
	public:
		caldgemm_config();

		bool AsyncDMA;							//Run DMA transfer and kernel execution in parallel
		bool DivideToGPU;						//Write preprocessed data difrectly to GPU
		char DstMemory;							//Dst memory of kernel on GPU (g) or CPU (c)
		int ImplicitDriverSync;					//Assume the CAL driver enforces an explicit sync when starting CAL kernel
		bool DynamicSched;						//Dynamically schedule CPU DGEMM
		bool SecondPhaseDynamicRuns;			//3rd phase in dynamic scheduling
		bool ThirdPhaseDynamicRuns;				//3rd phase in dynamic scheduling
		bool KeepBuffersMapped;					//Do not unmap CAL buffers before kernel execution
		bool MemPolicy;							//Set memory allocation policy to interleaved
		bool MultiThread;						//Use multiple threads
		bool MultiThreadDivide;					//Use multiple threads for DivideBuffer as well
		double GPURatio;						//Fraction of the matrix processed by GPU
		bool UseCPU;							//use CPU for DGEMM
		bool UseGPU;							//use GPUs for DGEMM

		int OpenCLPlatform;						//OpenCL Platform ID to use
		int DeviceNum;							//CAL Device to use (-1 for all devices)
		int NumDevices;							//Number of devices to use in parallel at max
		bool ImprovedScheduler;					//Tries to save bbuffers, replaces the round-robin scheduler

		bool Debug;								//Activate debig output
		bool DumpMatrix;						//Dump input matrix to file
		unsigned int Iterations;				//Run multiple iterations (for benchmark and debugging purpose only)
		bool Verify;							//Verify the result

		int GPUMapping[max_devices];			//Mapping of GPU devices to CPU cores. Affects DivideBuffer Threads, merge threads take the succeeding cores.
		int PinMainThread;						//Pin main thread to specific device. Default: Use the first GPU preprocessing core
		int PinCPU;								//Pin the GPU pre- and postprocessing threads to a CPU core, foreces all GPUMappings to PinCPU, -1 for disable
		bool SlowCPU;							//Try to put as many load as possible on the GPU as CPU is slow
		
		size_t Height;							//height of subblock od A, width of subblock of B
		size_t m, n;							//height of A, width of B, must be multiple of height
		size_t Width;							//k for matrix multiply
		bool AutoHeight;						//Automatically adjust height
		bool SmallTiles;						//ScheduleSmallTiles for alowing better GPU processing of the remainder parts

		bool Disassemble;						//Print the disassembled IL kernel
		bool PrintILKernel;						//Print the IL kernel source

		bool AsyncTiming;						//Print additional asynchronous timing information
		bool DisplayTiming;						//Display Final Timing Information even when quiet
		bool NoPerformanceWarnings;				//Suppress also performance warnings, will usually be shown even in quiet mode
		const char* PreOut;						//Prefix timing output with user defined string (for MPI-runs)
		bool Quiet;								//Quiet mode
		bool TabularTiming;						//Output a table with timing information
		bool VerboseTiming;						//Verbose timing information, disables asynchronous processing

		int LinpackNodes;						//Number of nodes contributing to MPI HPL run
		int MPIRank;							//MPI Rank to display in debug info
		int GPUClock;							//GPU clock of the device used (to display throttling information)
		
		int HPLFactorizeRestrictCPUs;			//Set 1 to restrct thread count to 8, 2 for dynamic restriction
		volatile size_t *LinpackSwapN;			//Current status of linpack pivoting process
		void (*linpack_factorize_function)();	//Linpack callback functions
		void (*linpack_broadcast_function)();
		void (*linpack_swap_function)();
	};
	
	//CALDGEMM interface functions
	//Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
	//Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = true, UseGPU = UseCPU = true, GPURatio = 0.66
	//m and n can be defined in the RunCALDGEMM call
	//The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
	int InitCALDGEMM(caldgemm_config* pInfo, bool nocalinit = false);
	int ExitCALDGEMM();
	int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = -1, size_t k = -1, size_t n = -1, size_t Apitch = -1, size_t Bpitch = -1, size_t Cpitch = -1, bool orderColMajor = false, bool TransA = false, bool TransB = false, int ExecuteLinpackCallbacks = 0);
	double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages);
	void FreeMemory(double* ptr);
	void ResetTimers();
	int broadcastcore();
	
	double avggflops;
	int avgngflops;

	bool cpuUsed(int cpu);

protected:

	virtual int UseOutputPthreads() = 0;
	virtual int UseInputPthreads() = 0;
	virtual int UseMutexPerDevice() = 0;

	struct BufferProperties;
	
	virtual int ValidateRuntime() = 0;
	virtual int CheckDevices() = 0;
	virtual int InitDevices() = 0;
	virtual int ReinitDevices() = 0;
	virtual int InitConstantData(double alpha) = 0;
	virtual int ExitRuntime() = 0;
	virtual int WaitForEvent(int, int, int lock = 0) = 0;
	virtual int FetchResult(int device, int j, int m, int n) = 0;
	virtual int ExitDevices() = 0;

	virtual	int Initialize (int deviceNum, bool nocalinit) = 0;
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch) = 0;

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
	int next_buffer_A[max_devices];
	int next_buffer_B[max_devices];
	int *buffer_pointers_A[max_devices];
	int *buffer_pointers_B[max_devices];
	
	pthread_mutex_t device_mutex[max_devices];

	int DGEMM_prepare(size_t k, int j, unsigned int num_device);
	virtual int DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0) = 0;
	inline void DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn)
	{
		if (DGEMM_favor_m)
		{
			const int nb = (gpu_n + Config->Height - 1) / Config->Height;
			blockn = k % nb;
			blockm = k / nb;
		}
		else
		{
			const int mb = (gpu_m + Config->Height - 1) / Config->Height;
			blockm = k % mb;
			blockn = k / mb;
		}
	}


	inline void WaitForLASWP(size_t n);
	void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2 = NULL);
	int cpuScheduler();
	int getcpumask(cpu_set_t* set);
	virtual int reserve_cpu_cores() = 0;
	int broadcast_cpu_core;

	struct mergeParameters
	{
		caldgemm* cls;
		double* dst;
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
	bool gpu_available;

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

	unsigned int AnalyzeResults();
	void displayMatrixTiming(const char* name);
	bool isDoubleEqual(double a, double b);

	struct TimerInfo
	{
		HighResTimer System, Kernel, CounterDivide, CounterMerge, CounterCopyTo, CounterCopyFrom, CPUTimer, GPUTimer, TotalCPUTimer, ATime, LinpackTimer1, LinpackTimer2, LinpackTimer3, BcastTimer;
		int divideA, divideB, divideC;
	} Timers;

	int DumpMatrix(double* A, double* B, double* C, double alpha, double beta, int m, int k, int n, int Apitch, int Bpitch, int Cpitch);

	double* A;
	double* B;
	double* C;

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	__attribute__((__may_alias__))
#endif
	double Alpha, Beta;

	size_t A_pitch, B_pitch, C_pitch;
	bool TransposeA;
	bool TransposeB;

	int bbuffers[max_devices];
	int outputthreads;

	size_t BufferHeight;						//Height to which the buffers were originally initialized
	size_t BufferWidth;							//Same for width
	size_t SmallTileHeight;						//Height of small tiles

	caldgemm_config* Config;

	int nDevices;

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

	struct DGEMMPrepareAndExecuteTask
	{
		struct DGEMMPrepareTask
		{
			size_t k;
			int j;
		} PrepareTasks[2];
		int k;
		int j;
		int device;
		int kernel_num;
		pthread_mutex_t mutex_start, mutex_finished;
	} DGEMMTasks[max_devices];
	int DGEMMPrepareAndExecute(caldgemm::DGEMMPrepareAndExecuteTask& Task);
	virtual int ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn) = 0;
	
	struct divideParameters
	{
		caldgemm* cls;
		int CPUCore;
		int nThread;
		int terminate;
		int reset;
		int curDevice;
	} dParam[max_devices];
	int divideThreads;

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
	int buffersMinor[max_devices][max_bbuffers];

	char hostname[256];							//Store hostname of node for host dependant debug code
	
	int conf_numprocs, conf_cpufreq, conf_numgpus, conf_gpufreq, conf_gpushaders;
};

#endif
