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

#include "cmodules/threadserver.h"


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
		int ThirdPhaseThreshold;				//Modifier to the number of remaining tiles required to start a third phase run
		bool KeepBuffersMapped;					//Do not unmap CAL buffers before kernel execution
		bool MemPolicy;							//Set memory allocation policy to interleaved
		bool MultiThread;						//Use multiple threads
		bool MultiThreadDivide;					//Use multiple threads for DivideBuffer as well
		bool ImprovedScheduler;					//Tries to save bbuffers, replaces the round-robin scheduler
		unsigned int ParallelDMA;						//Use multiple threads to handle GPU DMA, this is incompatible with DynamicSched, acivated if m > setting and setting != 0
		double GPURatio;						//Fraction of the matrix processed by GPU
		bool UseCPU;							//use CPU for DGEMM
		bool UseGPU;							//use GPUs for DGEMM

		int OpenCLPlatform;						//OpenCL Platform ID to use
		int DeviceNum;							//CAL Device to use (-1 for all devices)
		int NumDevices;							//Number of devices to use in parallel at max
		int DeviceNums[max_devices];			//Array of CAL devices to use (replaces DeviceNum for multiple devices). This translation is applied first, all other setting like GPU mappings are applied on top of this.

		bool Debug;								//Activate debig output
		bool DumpMatrix;						//Dump input matrix to file
		unsigned int Iterations;				//Run multiple iterations (for benchmark and debugging purpose only)
		bool Verify;							//Verify the result
		bool SkipCPUProcessing;					//Skip divide and merge buffer

		int GPUMapping[max_devices];			//Mapping of GPU devices to CPU cores. Affects DivideBuffer Threads, merge threads take the succeeding cores.
		int PostprocessMapping[max_devices];	//Mapping for postprocessing threads, default -1 = same mapping as GPU
		int AllocMapping[max_devices];			//Core (die with that core in fact) where the memory for dma transfer is allocated
		int DMAMapping[max_devices];			//Core for usage witgh ParallelDMA option
		int PinMainThread;						//Pin main thread to specific device. Default: Use the first GPU preprocessing core
		bool RepinDuringActiveWaitForEvent;		//Repin the Main CPU core that does the active wait for the event to the allocmapping of the GPU it waits for
		int SleepDuringActiveWait;				//Sleep for n usec between queries for GPU event, -1 disable
		bool ThreadSaveDriver;					//Assume GPU driver to be thread save
		int PinCPU;								//Pin the GPU pre- and postprocessing threads to a CPU core, foreces all GPUMappings to PinCPU, -1 for disable
		bool SlowCPU;							//Try to put as many load as possible on the GPU as CPU is slow
		int OutputThreads;						//Number of output threads
		int NumaPinning;						//Rotate pinning over NUMA nodes, better die utilization but perhaps worse L3 cache utilization.
		unsigned int AlternateLookahead;		//Alternate Lookahead implementation optimized for saving CPU cycles, set to an integer, AlternateLookahead is used as soon as n (since HPL is col major) is smaller than this value, 0 for disable
		
		size_t Height;							//height of subblock od A, width of subblock of B
		size_t Width;							//k for matrix multiply
		bool AutoHeight;						//Automatically adjust height
		int SmallTiles;							//ScheduleSmallTiles for alowing better GPU processing of the remainder parts

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
		
		int nExcludeCPUCores;					//CPU Cores to exlude
		int* ExcludeCPUCores;
	};
	
	//CALDGEMM interface functions
	//Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
	//Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = true, UseGPU = UseCPU = true, GPURatio = 0.66
	//m and n can be defined in the RunCALDGEMM call
	//The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
	int InitCALDGEMM(caldgemm_config* pInfo, bool nocalinit = false);
	int ExitCALDGEMM();
	int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = (size_t) -1, size_t k = (size_t) -1, size_t n = (size_t) -1, size_t Apitch = (size_t) -1, size_t Bpitch = (size_t) -1, size_t Cpitch = (size_t) -1, bool orderColMajor = false, bool TransA = false, bool TransB = false, int ExecuteLinpackCallbacks = 0);
	
	virtual double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible = false, bool Cmatrix = false);
	virtual void FreeMemory(double* ptr, bool gpuaccessible = false);

	void ResetTimers();
	int broadcastcore();
	
	double avggflops;
	int avgngflops;

	virtual bool cpuUsed(int cpu);
	
	void printConfig();

protected:

	static const int obuffercount = 3;				//Not cal context count but number of copies of data buffers etc.
	static const int max_outputthreads = CALDGEMM_OUTPUT_THREADS_SLOW;
	static const int vcpysize = 16;
	static const int kernel_count = 3;
#ifdef REUSE_BBUFFERS
	static const int max_bbuffers = 21;
	static const int max_bbuffers_g = 20;
#else
	static const int max_bbuffers = 3;
	static const int max_bbuffers_g = 3;
#endif	

	size_t matrix_m, matrix_n;

	int RunCALDGEMMMain(int parallelDevice = -1);
	int* tileDistribution;

	struct DGEMMPrepareAndExecuteTask
	{
		struct DGEMMPrepareTask
		{
			volatile size_t k;
			volatile int j;
		} PrepareTasks[2];
		volatile int k;
		volatile int j;
		int device;
		int kernel_num;
		pthread_mutex_t mutex_start, mutex_finished;
		int thread_running;
		volatile int* next_device;
		volatile int skip_device_to;
	} DGEMMTasks[max_devices];
	int DGEMMPrepareAndExecute(caldgemm::DGEMMPrepareAndExecuteTask& Task CALDGEMM_DIVBUFA);

	volatile bool DGEMMPrepareTaskEventReady[max_devices][obuffercount];

	struct BufferProperties;

	virtual int UseOutputPthreads() = 0;
	virtual int UseInputPthreads() = 0;
	virtual int UseMutexPerDevice() = 0;

	virtual int ValidateRuntime() = 0;
	virtual int CheckDevices() = 0;
	virtual int InitDevices() = 0;
	virtual int ReinitDevices() = 0;
	virtual int InitConstantData(double alpha) = 0;
	virtual int ExitRuntime() = 0;
	virtual int WaitForEvent(int, int, int lock = 0) = 0;
	virtual int FetchResult(int device, int j, int m, int n) = 0;
	virtual int ExitDevices() = 0;

	virtual	int Initialize (bool nocalinit) = 0;
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch) = 0;
	virtual int DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA) = 0;
	virtual int ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn) = 0;
	virtual int RunCALDGEMM_Init() = 0;
	virtual int RunCALDGEMM_Exit() = 0;

	virtual int reserve_cpu_cores() = 0;

	int next_buffer_A[max_devices];
	int next_buffer_B[max_devices];
	int *buffer_pointers_A[max_devices];
	int *buffer_pointers_B[max_devices];
	
	pthread_mutex_t device_mutex[max_devices];

	int DGEMM_prepare(size_t k, int j, unsigned int num_device CALDGEMM_DIVBUFA);
	double* divide_tmpBuffer;

	inline double* allocDivideBuffer()
	{
#ifdef CALDGEMM_DIVIDE_TRANSPOSE_TWOPHASE
		return(new double[CALDGEMM_TRANSPOSE_BLOCKING * mymax(Config->Width, Config->Height)]);
#elif defined(CALDGEMM_SHIFT_TEXTURE) && CALDGEMM_SHIFT_TEXTURE == 1
		return(new double[2 * CALDGEMM_DIVIDE_BLOCKING + 1]);
#else
		return(NULL);
#endif
	}

	inline void freeDivideBuffer(double* ptr)
	{
#if defined(CALDGEMM_DIVIDE_TRANSPOSE_TWOPHASE) || (defined(CALDGEMM_SHIFT_TEXTURE) && CALDGEMM_SHIFT_TEXTURE == 1)
		delete[] ptr;
#endif
	}
	
	inline void DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn)
	{
		if (DGEMM_favor_m)
		{
			const int nb = (int) ((gpu_n + Config->Height - 1) / Config->Height);
			blockn = k % nb;
			blockm = k / nb;
		}
		else
		{
			const int mb = (int) ((gpu_m + Config->Height - 1) / Config->Height);
			blockm = k % mb;
			blockn = k / mb;
		}
	}


	inline void WaitForLASWP(size_t n);
	void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2 = NULL);
	int cpuScheduler();
	int getcpumask(cpu_set_t* set);
	int broadcast_cpu_core;
	int main_blas_core;
	void ensure_omp_thread_pinning();

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

	pthread_mutex_t alternateLinpackMutex;
	volatile unsigned int AlternateLookaheadTilesRemaining;
	pthread_mutex_t tilesRemainingMutex;
	
	void CheckAlternateTilesRemaining(size_t m);

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

	struct divideParameters
	{
		caldgemm* cls;
		int CPUCore;
		int nThread;
		int terminate;
		int reset;
		volatile int curDevice;
	} dParam[max_devices];
	int divideThreads;

	cblasParameters cParam;

	void RunLinpackFactorization(int old_goto_threads, int require_threads);
	//For Verfify only
	double* D;

	class clsDMAParam : public qThreadParamCls<caldgemm>
	{

	};
	qThreadClsArray<caldgemm, clsDMAParam> DMAThreads;
	void DMA_wrapper(clsDMAParam* param);

	void* merge_wrapper_a(mergeParameters* par);
	void* divide_wrapper_a(divideParameters* par);
	void* cblas_wrapper_a(cblasParameters* par);
	void* linpack_wrapper_a();

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
