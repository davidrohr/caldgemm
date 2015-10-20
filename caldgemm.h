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
#include "cmodules/qsem.h"
#include "caldgemm_common.h"

template <class T> T mymin(const T a, const T b) {return(a < b ? a : b);}
template <class T> T mymax(const T a, const T b) {return(a > b ? a : b);}


#include "cmodules/timer.h"

class caldgemm
{
	static void* merge_wrapper(void* arg);
	static void* divide_wrapper(void* arg);
	static void* cblas_wrapper(void* arg);
	static void* linpack_broadcast_wrapper(void* arg);
public:
	static const unsigned int max_devices = 8;

public:

	caldgemm();
	virtual ~caldgemm();

	class caldgemm_config_backend
	{
	public:
		size_t size;
		caldgemm_config_backend() {size = sizeof(*this);}
		virtual ~caldgemm_config_backend();
		virtual int ParseBackendOptions(unsigned int argc, char** argv);
		virtual void printConfig(caldgemm_config_backend* oldConfig = NULL);
		virtual caldgemm_config_backend* Clone() const {return new caldgemm_config_backend(*this);}
	};

	class caldgemm_config						//Run Parameters
	{
	public:
		caldgemm_config();
		caldgemm_config(const caldgemm_config& other);
		~caldgemm_config() {
			if (config_backend) delete config_backend;
			free(argv_backend);
		}
		void InitBackendArgc();
		void AddBackendArgv(char* option);
		int InitializeBackendOptions();

		bool AsyncDMA;							//Run DMA transfer and kernel execution in parallel
		bool DivideToGPU;						//Write preprocessed data difrectly to GPU
		char DstMemory;							//Dst memory of kernel on GPU (g) or CPU (c)
		int ImplicitDriverSync;					//Assume the CAL driver enforces an explicit sync when starting CAL kernel
		unsigned int UseDMAFetchQueue;			//When starting a new kernel, ensure to start dma fetch for previous kernel beforehand. Used if UseDMAFetchQueue >= matrix_n
		bool DynamicSched;						//Dynamically schedule CPU DGEMM
		bool SecondPhaseDynamicRuns;			//3rd phase in dynamic scheduling
		bool ThirdPhaseDynamicRuns;				//3rd phase in dynamic scheduling
		int ThirdPhaseThreshold;				//Modifier to the number of remaining tiles required to start a third phase run
		bool KeepBuffersMapped;					//Do not unmap CAL buffers before kernel execution
		bool MemPolicy;							//Set memory allocation policy to interleaved
		bool MultiThread;						//Use multiple threads
		bool MultiThreadDivide;					//Use multiple threads for DivideBuffer as well
		bool ImprovedScheduler;					//Tries to save bbuffers, replaces the round-robin scheduler
		int ImprovedSchedulerBalance;			//Balancing Mode for Improved Scheduler
		bool SimpleGPUQueuing;					//Use a simpler scheduler that performs all command queueing based on device events.
		bool AlternateSimpleQueuing;			//Use variant of simple queuing, where always one queue is used for kernes, transfor to and from the device
		unsigned int ParallelDMA;				//Use multiple threads to handle GPU DMA, this is incompatible with DynamicSched, acivated if n >= setting and setting != 0, DMA cores defined by DMAMapping
		unsigned int GroupParallelDMA;			//Use in combination with ParallelDMA. Group devices with identical AllocMapping setting to one thread, at least one paralleDMA thread with that DMAMapping must exist. Activated if n < setting., make sure to have a preprocessing thread set to each CPU core used for this feature, the thread won't be used but it will ensure correct core pinning. -1 for always Grouped parallel DMA.
		double GPURatio;						//Fraction of the matrix processed by GPU
		double GPURatioDuringFact;				//Use modified GPU Ratio during factorization, works currently only with negative GPURatio
		double GPURatioMax;						//Max GPU ratio to use, if autocalculation exceeds this value, it is capped. This ensures the CPU always gets a certain part. (For the auto ratio calculation, the CPU needs a small part)
		double GPURatioMarginTime;				//Time margin used in auto calculation to ensure GPU time exceeds CPU time
		double GPURatioMarginTimeDuringFact;	//Same as bove dbut when Linpack Factorization is active
		double GPURatioLookaheadSizeMod;		//Incrase the virtual size of the lookahead part of the CPU DGEMM by this factor, as it is usually not running as efficiently as full dgemm. Only relevant with alternate lookahead disabled.
		int GPURatioPenalties;					//Apply penalties to the CPU part (0 disable, 1 apply penalty when CPU took to long, 2 apply penalty as well when CPU time is short in general)
		double GPURatioPenaltyFactor;			//Factor to apply
		unsigned int MinimizeCPUPart;			//Set GPURatio to 1.0 as soon as matrix n dimension is below this value
		int MinimizeCPUDuringFact;				//Always minimize CPU part during factorization
		bool UseCPU;							//use CPU for DGEMM
		bool UseGPU;							//use GPUs for DGEMM
		bool RereserveLinpackCPU;				//Use the Linpack CPU cores for DGEMM after they finished the broadcast
		int GPU_C;								//Store the C matrix on CPU, not every option is supported by every backend, -1 = auto detect
		int NoConcurrentKernels;				//Do not allow OpenCL to run multiple concurrent kernels in parallel.
		bool PipelinedOperation;				//Allows to queue two caldgemm calls in a pipeline. You need to run FinishCALDGEMM() to finish the iteration.
		bool PipelineDoubleBuffer;				//Use double buffers to better overlap pipeline steps
		size_t PipelinedMidMarker;				//Mid marker for pipelined operation

		int OpenCLPlatform;						//OpenCL Platform ID to use
		int DeviceNum;							//CAL Device to use (-1 for all devices)
		int NumDevices;							//Number of devices to use in parallel at max
		int NumActiveDevices;					//Initialize NumDevices but use ony NumActiveDevices for main queue.
		int DeviceNums[max_devices];			//Array of CAL devices to use (replaces DeviceNum for multiple devices). This translation is applied first, all other setting like GPU mappings are applied on top of this.
		int max_bbuffers;						//Limit the number of bbuffers
		int PreallocData;						//Preallocate buffers, set Prealloc to the maximum number of (mb/nb) blocks expected!
		int CPUInContext;						//Have the CPU as compute device in the runtime context

		bool Debug;								//Activate debig output
		bool DumpMatrix;						//Dump input matrix to file
		unsigned int Iterations;				//Run multiple iterations (for benchmark and debugging purpose only)
		bool Verify;							//Verify the result
		bool SkipCPUProcessing;					//Skip divide and merge buffer
		int ForceKernelVariant;					//Force a specific DGEMM kernel, default set to -1 for autoselect

		int GPUMapping[max_devices];			//Mapping of GPU devices to CPU cores. Affects DivideBuffer Threads, merge threads take the succeeding cores.
		int PostprocessMapping[max_devices];	//Mapping for postprocessing threads, default -1 = same mapping as GPU
		int AllocMapping[max_devices];			//Core (die with that core in fact) where the memory for dma transfer is allocated
		int DMAMapping[max_devices];			//Core for usage witgh ParallelDMA option
		int PinMainThread;						//Pin main thread to specific device. Default: Use the first GPU preprocessing core
		int PinDeviceRuntimeThreads;			//Pin threads of device runtime to this core, -1 for no pinning, -2 for same as main (default)
		int PinBroadcastThread;					//CPU core to pin broadcast thread to
		bool RepinDuringActiveWaitForEvent;		//Repin the Main CPU core that does the active wait for the event to the allocmapping of the GPU it waits for
		bool RepinMainThreadAlways;				//Superseedes the above setting. The main thread is always repinned to the allocmapping core of each GPU when working for this GPU
		int SpawnGPUThread;						//Spawn a GPU thread instead of a cblas thread, and perform cblas calls from calling thread. -2: disabled (default), -1: enabled, >= 0: define the CPU core to pin the caller thread to (PinMainThread will affect the GPU thread!)
		int SleepDuringActiveWait;				//Sleep for n usec between queries for GPU event, -1 disable
		int ThreadSaveDriver;					//Assume GPU driver to be thread save
		int PinCPU;								//Pin the GPU pre- and postprocessing threads to a CPU core, foreces all GPUMappings to PinCPU, -1 for disable
		int ForceNumCPUThreads;					//Limit the number of CPU threads to use
		int CPUCoreOffset;						//Offset all cpu core pinnings by this number
		bool SlowCPU;							//Try to put as many load as possible on the GPU as CPU is slow
		int OutputThreads;						//Number of output threads
		int NumaPinning;						//Rotate pinning over NUMA nodes, better die utilization but perhaps worse L3 cache utilization.
		unsigned int AlternateLookahead;		//Alternate Lookahead implementation optimized for saving CPU cycles, set to an integer, AlternateLookahead is used as soon as n (since HPL is col major) is smaller than this value, 0 for disable
		bool AsyncSideQueue;					//Create an asynchronous side queue to run small DGEMMs (without tiling) in parallel to a large DGEMM
		int AsyncSideQueueBalance;				//Balance workload of ASYNC Side queue among GPUs
		int AsyncSideQueueUseInactiveDeviceSet;	//If GPUs were disabled via SetNumberDevices for the main queue, the async side queue will use these disabled devices
		int AsyncDGEMMThreshold;				//Min size where GPU is used for async DGEMM
		int AsyncDTRSMThreshold;				//Same for DTRSM
		bool AsyncDTRSM;						//Allow side-queue to run DTRSM as well
		bool Use3rdPartyTranspose;				//Use transpose kernel fro 3rd party lib
		
		size_t Height;							//height of subblock od A, width of subblock of B
		size_t Width;							//k for matrix multiply
		bool AutoHeight;						//Automatically adjust height
		int SmallTiles;							//ScheduleSmallTiles for alowing better GPU processing of the remainder parts, 1 for aleays using MIN_TILE_SIZE for border tiles, 2 for automatic adaption between MIN_TILE_SIZE and Config->Height

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
		int (*HPLFactorizeRestrictCallback)(int matrix_n);		//Callback function to restrict number of cores used for factorization by the return value
		int LASWPSleep;							//Time in usec to sleep during checks whether LASWP is ready
		volatile size_t *LinpackSwapN;			//Current status of linpack pivoting process
		void (*linpack_factorize_function)();	//Linpack callback functions
		void (*linpack_broadcast_function)();
		void (*linpack_swap_function)();
		
		int nExcludeCPUCores;					//CPU Cores to exlude
		int* ExcludeCPUCores;
		int ShowConfig;							//Show CALDGEMM Config
		int ShowThreadPinning;					//Print thread pinning at each call

		int argc_backend;
		char** argv_backend;

		caldgemm_config_backend* config_backend;
	};

	virtual caldgemm_config_backend* create_caldgemm_config_backend();
	
	//CALDGEMM interface functions
	//Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
	//Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = true, UseGPU = UseCPU = true, GPURatio = 0.66
	//m and n can be defined in the RunCALDGEMM call
	//The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
	int InitCALDGEMM(caldgemm_config* pInfo, bool nocalinit = false);
	int ExitCALDGEMM();
	int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = (size_t) -1, size_t k = (size_t) -1, size_t n = (size_t) -1, size_t Apitch = (size_t) -1, size_t Bpitch = (size_t) -1, size_t Cpitch = (size_t) -1, bool orderColMajor = false, bool TransA = false, bool TransB = false, int ExecuteLinpackCallbacks = 0, int pipelined = 0);
	int FinishCALDGEMM(bool force = false);
	virtual int WaitForCALDGEMMProgress(size_t n);
	virtual int RunAsyncSingleTileDGEMM(const double* A, const double* B, double* C, double alpha, double beta, size_t m, size_t k, size_t n, size_t Apitch, size_t Bpitch, size_t Cpitch, bool orderColMajor, bool TransA, bool TransB);
	virtual int RunAsyncSingleTileDTRSM(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const size_t M, const size_t N, const double alpha, const double *A, const size_t lda, double *B, const size_t ldb);
	void SetNumberDevices(int n);
	int ParseParameters(unsigned int argc, char** argv, caldgemm_config* Config);
	int ParseParameters(char* params, caldgemm_config* Config);
	void ResetRatios();
	caldgemm_config* GetConfig() {return Config;}
	
	virtual double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible = false, bool interleave = false);
	virtual int FreeMemory(double* ptr, bool gpuaccessible = false);

	void ResetTimers();
	int broadcastcore();
	
	double avggflops;
	int avgngflops;

	virtual bool cpuUsed(int cpu);
	virtual double getMaxGPUTemperature();
	
	void printConfig(caldgemm_config* newConfig = NULL, caldgemm_config* oldConfig = NULL);
	void setMatrixDim(size_t m, size_t n) {matrix_m = m;matrix_n = n;}

protected:

	static const int obuffercount = 3; //Number of replicated of output data buffers, also named context within caldgemm (not to be mistaken for gpu context)
	static const int ibuffercount = 3;
	static const int max_outputthreads = CALDGEMM_OUTPUT_THREADS_SLOW;
	static const int vcpysize = 16;
	static const int kernel_count = 3;
#ifdef REUSE_BBUFFERS
	static const int max_bbuffers = 30;
	static const int max_bbuffers_g = 30;
#else
	static const int max_bbuffers = 3;
	static const int max_bbuffers_g = 3;
#endif	

	size_t matrix_m, matrix_n;

	struct DGEMMKernelSettingsStruct
	{
		int tiling_x;
		int tiling_y;
		int group_size_x;
		int group_size_y;
		bool transposeA;
		bool transposeB;
		bool texture_buffers;
		int min_tile_size;
		int min_k;
	};
	DGEMMKernelSettingsStruct KernelSettings;
	void SetDefaultKernelSettings();

	int RunCALDGEMMMain(int parallelDevice = -1);
	int* tileDistribution;
	int first_device_k[max_devices];

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
		qSem mutex_start, mutex_finished;
		int thread_running;
		volatile int* next_device;
		volatile int skip_device_to;
	} DGEMMTasks[max_devices];

	int DGEMMPrepareAndExecute(caldgemm::DGEMMPrepareAndExecuteTask& Task CALDGEMM_DIVBUFA);

	volatile bool DGEMMPrepareTaskEventReady[max_devices][obuffercount];

	struct BufferProperties;

	virtual int UseOutputPthreads() = 0;
	virtual int UseInputPthreads() = 0;
	virtual int AllowCPUFallback();
	virtual int UseMutexPerDevice() = 0;

	virtual int ValidateRuntime() = 0;
	virtual int CheckDevices() = 0;
	virtual int InitDevices() = 0;
	virtual int ReinitDevices() = 0;
	virtual int InitConstantData(double alpha) = 0;
	virtual int ExitRuntime() = 0;
	virtual int WaitForEvent(int, int, int lock = 0) = 0;
	virtual int FetchResult(int device, int j, int m, int n, int mustlock = 0) = 0;
	virtual int ExitDevices() = 0;

	virtual	int Initialize (bool nocalinit) = 0;
	virtual int RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch) = 0;
	virtual int DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA) = 0;
	virtual int ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn) = 0;
	virtual int RunCALDGEMM_Init();
	virtual int RunCALDGEMM_Exit();
	virtual int RunCALDGEMM_Finish();
	virtual int FinishDataInit();
	virtual void FinishDataFill();
	virtual int CheckParams();
	virtual void Preallocate();
	virtual void PreallocateFree();
	
	virtual int CaldgemmCustomAutoHeight(size_t MaxGpuM, size_t MaxGpuN, int nDevices);
	virtual int CaldgemmCustomModHeight(size_t MOD_OVER, size_t MOD_GPU);

	virtual int reserve_cpu_cores();
	
	void SetupBufferSizes();

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


	inline void WaitForLASWP(size_t blockm);
	void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2 = NULL);
	int cpuScheduler();
	int getcpumask(cpu_set_t* set);
	int broadcast_cpu_core;
	int main_blas_core;
	void ensure_omp_thread_pinning(const char* baseName);

	struct mergeParameters
	{
		caldgemm* cls;
		double* dst;
		int nMergeThread;
		int nContext;
		int num_device;
		bool terminate;
		qSem mergeThreadMutex[2];
		size_t k;
	};
	mergeParameters mParam[max_devices][max_outputthreads];

	qSem obufferMutex[max_devices][obuffercount];
	bool dma_pending[max_devices][obuffercount];

	struct structLinpackParameters
	{
		qSem linpackMutex[2];
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

	qSem alternateLookaheadMutex;
	int AlternateLookaheadTilesFull;
	volatile unsigned int AlternateLookaheadTilesRemaining;
	int AlternateLookaheadBlocksM;
	pthread_mutex_t tilesRemainingMutex;
	
	void CheckAlternateTilesRemaining(size_t m);
	virtual int CheckAlternateTilesRemainingSimpleQuieing();

	bool buffersSwitchable;

	unsigned int AnalyzeResults();
	void displayMatrixTiming(const char* name);
	bool isDoubleEqual(double a, double b);

	struct TimerInfo
	{
		HighResTimer System, Kernel, CounterDivide, CounterMerge, CounterCopyTo, CounterCopyFrom, CPUTimer, GPUTimer, TotalCPUTimer, ATime, LinpackTimer1, LinpackTimer2, LinpackTimer3, BcastTimer;
		int divideA, divideB, divideC;
		size_t device_kernel;
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
	int min_bbuffers;
	int outputthreads;

	size_t BufferHeight;						//Height to which the buffers were originally initialized
	size_t BufferWidth;							//Same for width
	size_t SmallTileHeight;						//Height of small tiles

	caldgemm_config* Config;

	int nDevices;
	int nDevicesInitialized;
	
	struct finishStruct
	{
		virtual ~finishStruct() {}
		size_t matrix_m, matrix_n, SmallTileHeight, orig_m, orig_n;
		double gpu_ratio_used, cpu_wait_time;
		int ExecLinpack;
		bool CPUOnlyRun, DGEMM_split_m;
		
		double System, CPUTimer, GPUTimer, TotalCPUTimer, LinpackTimer1, LinpackTimer2, LinpackTimer3, BcastTimer;
		int divideA, divideB, divideC;
		size_t device_kernel;

		size_t cblas_size;
		size_t dynamic_run;
		size_t dynamic_size;
		size_t cpu_k;
		size_t dynamic_run2;
		
		bool running;
		size_t MidMarkerPos;
	};
	finishStruct* finishData;

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
		qSem cblasMutex[2];
	};

	struct divideParameters
	{
		caldgemm* cls;
		int CPUCore;
		int nThread;
		int terminate;
		int reset;
		volatile int curDevice;
		int firstDevice;
	} dParam[max_devices];
	int divideThreads;

	cblasParameters cParam;

	void RunLinpackFactorization(int old_goto_threads, int& require_threads);
	
	double* D;									//For Verfify only

	class clsDMAParam : public qThreadParamCls<caldgemm>
	{
	};
	qThreadClsArray<caldgemm, clsDMAParam> DMAThreads;
	void DMA_wrapper(clsDMAParam* param);

	int caldgemm_part_cpu();
	int caldgemm_part_gpu();

	void* merge_wrapper_a(mergeParameters* par);
	void* divide_wrapper_a(divideParameters* par);
	void* cblas_wrapper_a(bool thread = false);
	void* linpack_broadcast_wrapper_a();

	pthread_mutex_t globalDriverLock;

	bool CPUOnlyRun;
	int ExecLinpack, pipelinedRun;
	int pipelineBuffer;
	double gpu_ratio_used;
	double cpu_wait_time;

	bool DGEMM_split_m;							//Splitting direction for CPU/GPU
	bool DGEMM_favor_m;							//Direction of C matrix primary tiling

	size_t orig_m, orig_n;
	double *orig_a, *orig_b, *orig_c;
	
	int buffersMajor[max_devices];
	int buffersMinor[max_devices][max_bbuffers];

	char hostname[256];							//Store hostname of node for host dependant debug code
	
	int conf_numprocs, conf_numprocs_real, conf_cpufreq, conf_numgpus, conf_gpufreq, conf_gpushaders;

	struct dma_fetch_queue_task
	{
		volatile size_t k;
		volatile int j;
		pthread_mutex_t mutex;
	};
	dma_fetch_queue_task dma_fetch_queue_tasks[max_devices];

	virtual int CheckDMAQueue(int device, int forcej = -1) = 0;
	
	bool warn_wrong_memory_allocation;
};

#endif
