/**
 * CPU side of CALDGEMM implementation.
 *
 * Copyright 2015:
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

#include "caldgemm.h"
#include "cmodules/qmalloc.h"
#include "cmodules/affinity.h"
#include "cmodules/qmath.h"
#include <algorithm>

#ifndef _WIN32
#include <syscall.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#else
extern "C"
{
	void ___chkstk() {}
	void __imp__cprintf() {}
}
#endif

#ifdef USE_OLD_HUGE_MALLOC
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

#include <math.h>
#include <emmintrin.h>

#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3

#ifndef SHM_HUGETLB
#define SHM_HUGETLB 04000
#endif

#ifdef USE_MKL
#include <mkl_service.h>
#endif

#include <math.h>

#include "cmodules/os_low_level_helper.h"

#ifdef _NO_AFFINITY
#define sched_setaffinity(a, b, c) 0
#endif

#if !defined(USE_GOTO_BLAS) | defined(_WIN32)
extern "C" {
extern int get_num_procs();
int get_num_procs()
{
	char* omp_threads = getenv("OMP_NUM_THREADS");
	if (omp_threads != NULL) return(atoi(omp_threads));
	return(get_number_of_cpu_cores());
}
}
#endif

extern "C" int HPL_CALDGEMM_gpu_height;
int HPL_CALDGEMM_gpu_height = 1024;

#ifdef DEBUG_MSG_TIMED
inline void printelapsedtime(bool reset = false)
{
    static int init = 1;
    static long long int begin;
    if (init == 1 || reset)
    {
		init = 0;
		timespec b;
		clock_gettime(CLOCK_REALTIME, &b);
		begin = (long long int) b.tv_sec * 1000000 + (long long int) b.tv_nsec / 1000;
    }
    timespec a;
    clock_gettime(CLOCK_REALTIME, &a);
    fprintf(STD_OUT, "%lld ", (long long int) a.tv_sec * 1000000 + (long long int) a.tv_nsec / 1000 - begin);
}
#define fprintf(file, ...) {printelapsedtime();fprintf(STD_OUT, __VA_ARGS__);}
#endif
//#define fprintf(file, ...) {fprintf(STD_OUT, "Thread %d ", gettid());fprintf(stderr, __VA_ARGS__);}

caldgemm::caldgemm_config_backend* caldgemm::create_caldgemm_config_backend()
{
	return(new caldgemm_config_backend);
}

void caldgemm::caldgemm_config_backend::printConfig(caldgemm::caldgemm_config_backend* oldConfig) {}

caldgemm::caldgemm_config_backend::~caldgemm_config_backend() {}

int caldgemm::caldgemm_config_backend::ParseBackendOptions(unsigned int argc, char** argv)
{
	if (argc > 1)
	{
		fprintf(STD_OUT, "Invalid Backend Options\n");
		return(1);
	}
	return(0);
}

void caldgemm::ResetRatios()
{
	for (int i = 0;i < caldgemm::max_linpack_callback_types;i++)
	{
		linpack_last_mn[i] = -1.;
		linpackGPURatios[i] = 1.;
		linpackBcastTime[i] = 0;
		linpackCPUDGEMMTime[i] = 0;
	}
}

caldgemm::caldgemm()
{
	caldgemm_initialized = false;
	ResetRatios();
	
	avggflops = 0;
	avgngflops = 0;
	
	conf_numprocs_real = get_number_of_cpu_cores();
	char* omp_threads = getenv("OMP_NUM_THREADS");
	if (omp_threads != NULL)
	{
		conf_numprocs = atoi(omp_threads);
	}
	else
	{
		conf_numprocs = conf_numprocs_real;
	}
	
	FILE* fp;
	fp = fopen("/proc/cpuinfo", "r");
	conf_cpufreq = 2100;
	if (fp)
	{
		char tmpbuffer[256];
		while (!feof(fp))
		{
			if (fgets(tmpbuffer, 255, fp) == 0) break;
			if (strncmp(tmpbuffer, "cpu MHz", 7) == 0)
			{
				float tmpval;
				char* ptr = tmpbuffer;
				while (*(ptr++) != ':');
				sscanf(ptr, "%f", &tmpval);
				conf_cpufreq = (int) tmpval;
				break;
			}
		}
		fclose(fp);
	}
	
	matrix_m = (size_t) -1;
	matrix_n = (size_t) -1;
	pipelineBuffer = 0;
	
	cParam.dynamic_size = 0; //Make Valgrind happy

	for (unsigned int i = 0;i < max_devices;i++)
	{
		dma_fetch_queue_tasks[i].k = (size_t) -1;
		for (int j = 0;j < obuffercount;j++)
		{
		    dma_pending[i][j] = false;
		}
	}
	
	conf_gpushaders = 0;
	conf_gpufreq = 0;
	
	warn_wrong_memory_allocation = true;
}

caldgemm::~caldgemm()
{
	if (caldgemm_initialized) ExitCALDGEMM();
}

caldgemm::caldgemm_config::caldgemm_config()
{
	static const char* EmptyOut = "";

	Verify = false;
	Disassemble = false;
	PrintILKernel = false;
	Quiet = true;
	DisplayTiming = false;
	DeviceNum = -1;
	ImprovedScheduler = false;
	ImprovedSchedulerBalance = 1;
	SimpleGPUQueuing = false;
	AlternateSimpleQueuing = false;
	AlternateSimpleQueuingMulti = false;
	NumDevices = max_devices;
	NumActiveDevices = 0;
	max_bbuffers = 0;
	OpenCLPlatform = 0;
	Width = 1024;
	Height = 0; //Auto Height, Initialize later
	AutoHeight = true;
	Iterations = 1;
	DstMemory = 'c';
	ImplicitDriverSync = -1;
	VerboseTiming = false;
	AsyncTiming = false;
	TabularTiming = false;
	Debug = false;
	MultiThread = true;
	MultiThreadDivide = true;
	RereserveLinpackCPU = false;
	UseGPU = true;
	UseCPU = true;
	GPURatio = -1.0;
	GPURatioDuringFact = 0.0;
	GPURatioMax = 1.0;
	GPURatioMarginTime = 0.0;
	GPURatioMarginTimeDuringFact = 0.3;
	GPURatioLookaheadSizeMod = 0.2;
	GPURatioPenalties = 1;
	GPURatioPenaltyFactor = 0.9;
	DynamicSched = true;
	ThirdPhaseDynamicRuns = true;
	SecondPhaseDynamicRuns = true;
	MemPolicy = true;
	DumpMatrix = false;
	DivideToGPU = false;
	AsyncDMA = true;
	KeepBuffersMapped = true;
	NoPerformanceWarnings = false;
	PinCPU = -1;
	ForceNumCPUThreads = 0;
	CPUCoreOffset = 0;
	PinMainThread = -1;
	SpawnGPUThread = -2;
	PinDeviceRuntimeThreads = -2;
	SlowCPU = false;
	LinpackNodes = 0;
	LinpackSwapN = NULL;
	HPLFactorizeRestrictCPUs = 2;
	HPLFactorizeRestrictCallback = NULL;
	MPIRank = -1;
	PreOut = EmptyOut;
	GPUClock = 0;
	SmallTiles = 0;
	ThreadSaveDriver = 0;
	SkipCPUProcessing = false;
	OutputThreads = -1;
	RepinDuringActiveWaitForEvent = 0;
	RepinMainThreadAlways = 0;
	SleepDuringActiveWait = -1;
	NumaPinning = false;
	ThirdPhaseThreshold = 0;
	AlternateLookahead = 0;
	ParallelDMA = 0;
	GroupParallelDMA = 0;
	LASWPSleep = 0;
	MinimizeCPUPart = 0;
	MinimizeCPUDuringFact = 0;
	PinBroadcastThread = -1;
	UseDMAFetchQueue = 0;
	GPU_C = -1;
	NoConcurrentKernels = 0;
	ForceKernelVariant = -1;
	PreallocData = 0;
	AsyncSideQueue = false;
	AsyncSideQueueBalance = 0;
	AsyncDGEMMThreshold = 480;
	AsyncDTRSMThreshold = 192;
	AsyncDTRSM = false;
	AsyncSideQueueUseInactiveDeviceSet = 0;
	Use3rdPartyTranspose = false;
	CPUInContext = 1;
	PipelinedOperation = false;
	PipelineDoubleBuffer = false;
	for (unsigned int i = 0;i < caldgemm::max_devices;i++)
	{
		GPUMapping[i] = 0;
		PostprocessMapping[i] = -1;
		AllocMapping[i] = -1;
		DMAMapping[i] = 0;
		DeviceNums[i] = i;
	}
	nExcludeCPUCores = 0;
	ExcludeCPUCores = NULL;
	ShowConfig = 0;
	ShowThreadPinning = 0;

	PipelinedMidMarker = 0;
	linpack_factorize_function = NULL;
	linpack_broadcast_function = NULL;
	linpack_swap_function = NULL;

	InitBackendArgc();
	config_backend = NULL;
}

caldgemm::caldgemm_config::caldgemm_config(const caldgemm::caldgemm_config& other)
{
	memcpy(this, &other, sizeof(*this));
	InitBackendArgc();
	if (other.config_backend)
	{
		config_backend = other.config_backend->Clone();
	}
	else
	{
		config_backend = NULL;
	}
}

void caldgemm::caldgemm_config::InitBackendArgc()
{
	argc_backend = 1;
	argv_backend = (char**) malloc(2 * sizeof(char*));
	argv_backend[0] = "backend_options";
	argv_backend[1] = NULL;
}

void caldgemm::caldgemm_config::AddBackendArgv(char* option)
{
	argv_backend = (char**) realloc(argv_backend, (argc_backend + 2) * sizeof(char*));
	argv_backend[argc_backend++] = option;
	argv_backend[argc_backend] = NULL;
}

int caldgemm::caldgemm_config::InitializeBackendOptions()
{
	int retVal = config_backend->ParseBackendOptions(argc_backend, argv_backend);
	free(argv_backend);
	InitBackendArgc();
	return(retVal);
}

int caldgemm::getcpumask(cpu_set_t* set)
{
    int retVal = 0;
    for (int i = 0;i < 24;i++)
    {
		if (CPU_ISSET(i, set)) retVal |= (1 << i);
    }
    return(retVal);
}

void caldgemm::print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2)
{
	fprintf(STD_OUT, "Matrix %lld x %lld, Subblocks %lld x %lld, Strides: %lld / %lld\n", (long long int) width, (long long int) height, (long long int) subx, (long long int) suby, (long long int) stridex, (long long int) stridey);
	for (size_t j = 0;j < height;j += stridey)
	{
		for (size_t jj = j;jj < j + suby && jj < height;jj++)
		{
			for (size_t i = 0;i < width;i += stridex)
			{
				for (size_t ii = i;ii < i + subx && ii < width;ii++)
				{
					if (M2 != NULL)
					{
						char tmpcolor[16] = "0";

						if (cParam.dynamic_run)
						{
							if (DGEMM_favor_m)
							{
								if (jj >= gpu_m - cParam.dynamic_run && ii >= gpu_n - cParam.dynamic_size) sprintf(tmpcolor, "01;33");
							}
							else
							{
								if (jj >= gpu_m - cParam.dynamic_size && ii >= gpu_n - cParam.dynamic_run) sprintf(tmpcolor, "01;33");
							}
						}

						if (DGEMM_split_m)	//favor splitting m because of consecutive memory
						{
							if (jj >= matrix_m - cParam.cblas_size || ii >= matrix_n - matrix_n % Config->Height) sprintf(tmpcolor, "01;34");
						}
						else
						{
							if (jj >= matrix_m - matrix_m % Config->Height || ii >= matrix_n - cParam.cblas_size) sprintf(tmpcolor, "01;34");
						}

						size_t k = ((gpu_m + Config->Height - 1) / Config->Height) * ((gpu_n + Config->Height - 1) / Config->Height);
						for (int l = 0;l < (int) cParam.dynamic_run2;l++)
						{
							k--;
							size_t cpublockm, cpublockn;
							DGEMM_getblocks(k, cpublockm, cpublockn);
							while ((DGEMM_favor_m ? (cpublockm * Config->Height >= gpu_m - cParam.dynamic_run && cpublockn * Config->Height >= gpu_n - cParam.dynamic_size) :
								(cpublockn * Config->Height >= gpu_n - cParam.dynamic_run && cpublockm * Config->Height >= gpu_m - cParam.dynamic_size)))
							{
								k--;
								DGEMM_getblocks(k, cpublockm, cpublockn);
							}
							if (jj / Config->Height == cpublockm && ii / Config->Height == cpublockn)
							{
								sprintf(tmpcolor, "01;35");
							}
						}

						int ok = isDoubleEqual(M[jj * pitch + ii], M2[jj * pitch + ii]);
#ifndef _WIN32
						fprintf(STD_OUT, "\33[%sm%d\33[%sm", ok ? "01;32" : "01;31", ok, tmpcolor);
#endif
						fprintf(STD_OUT, "%+10.3f\t", M[jj * pitch + ii]);
					}
					else
					{
						fprintf(STD_OUT, " %+10.3f\t", M[jj * pitch + ii]);
					}
				}
			}
#ifndef _WIN32
			fprintf(STD_OUT, "\33[0m");
#endif
			fprintf(STD_OUT, "\n");
		}
	}
	fprintf(STD_OUT, "Done\n");
}

void caldgemm::ensure_omp_thread_pinning(const char* baseName)
{
#ifndef USE_GOTO_BLAS
	if (!Config->UseCPU) return;
	if (Config->Debug) fprintf(STD_OUT, "Performing OpenMP Blas Thread Pinning\n");
	int* cpu_order = new int[conf_numprocs];
	if (Config->NumaPinning && conf_numprocs % 4 == 0)
	{
		cpu_order[0] = 0;
		int cpu_num = 1;
		
		int old_divider = conf_numprocs;
		if (Config->NumaPinning >= 2) old_divider /= 2;
		int divider = old_divider / 2;
		do
		{
			int cpu_num_end = cpu_num;
			for (int tmp_num = 0;tmp_num < cpu_num_end;tmp_num++)
			{
				cpu_order[cpu_num++] = cpu_order[tmp_num] + divider;
			}
			
			int cpu_num_end2 = cpu_num;
			for (int i = 1;i < old_divider / divider - 1;i++)
			{
				for (int tmp_num = cpu_num_end;tmp_num < cpu_num_end2;tmp_num++)
				{
					cpu_order[cpu_num++] = cpu_order[tmp_num] + 2 * i;
				}
			}
			
			old_divider = divider;
			divider = (divider % 2 == 0 && divider % 4 != 0 && divider > 2) ? 2 : divider / 2;
		} while (divider > 0);
		if (Config->NumaPinning >= 2)
		{
			for (int i = 0;i < conf_numprocs / 2;i++)
			{
				cpu_order[i + conf_numprocs / 2] = cpu_order[i] + conf_numprocs / 2;
			}
		}
		if (Config->Debug)
		{
			for (int i = 0;i < conf_numprocs;i++) fprintf(STD_OUT, "Numa ID %d Core %d\n", i, cpu_order[i]);
		}
	}
	else
	{
		if (Config->NumaPinning) fprintf(STD_OUT, "NUMA Pinning only available if number of processors is divisible by 4\n");
		for (int i = 0;i < conf_numprocs;i++) cpu_order[i] = i;
	}
	static int nInitialization = 0;
	nInitialization++;
	
	cpu_set_t oldaffinity;
	sched_getaffinity(0, sizeof(oldaffinity), &oldaffinity);
	
	cpu_set_t noaffinity;
	CPU_ZERO(&noaffinity);
	for (int i = 0;i < conf_numprocs;i++) CPU_SET(i + Config->CPUCoreOffset, &noaffinity);
	sched_setaffinity(0, sizeof(noaffinity), &noaffinity);

	setUnknownNames("Unknown - Before OMP Thread Creation");
#pragma omp parallel num_threads(conf_numprocs)
	{
		int thread_id = omp_get_thread_num();
		
		if (getThreadName(-1, NULL) == NULL)
		{
			char tmp[128];
			sprintf(tmp, "OpenMP Init %d %s%s%s Thread %d", nInitialization, baseName ? "(" : "", baseName ? baseName : "", baseName ? ")" : "", thread_id);
			setThreadName(tmp);
		}
		int localcore = thread_id * 2;

#pragma omp critical
		{
			int nFreeCores = 0;
			bool checkBroadcastCore = Config->ForceNumCPUThreads == 0 || broadcast_cpu_core < Config->ForceNumCPUThreads;
			if (thread_id == nFreeCores) localcore = main_blas_core;
			nFreeCores++;
			for (int i = 0;i < conf_numprocs;i++)
			{
				if (cpuUsed(cpu_order[i]) == false && (!checkBroadcastCore || cpu_order[i] != broadcast_cpu_core) && cpu_order[i] != main_blas_core)
				{
					if (thread_id == nFreeCores) localcore = cpu_order[i];
					nFreeCores++;
				}
			}
			if (checkBroadcastCore)
			{
				if (thread_id == nFreeCores) localcore = broadcast_cpu_core;
				nFreeCores++;
			}

			for (int j = 0;j < 2;j++)
			{
				for (int i = 0;i < conf_numprocs;i++)
				{
					if (cpuUsed(cpu_order[i]) && cpu_order[i] != main_blas_core)
					{
						size_t m = matrix_m, n = matrix_n;
						matrix_m = matrix_n = (size_t) -1;
						bool isDMACore = cpuUsed(cpu_order[i]);
						matrix_m = matrix_n = 0;
						if (cpuUsed(cpu_order[i])) isDMACore = false;
						matrix_m = m;
						matrix_n = n;
					
						if ((Config->ParallelDMA != 0 && isDMACore) ^ j)
						{
							if (thread_id == nFreeCores) localcore = cpu_order[i];
							nFreeCores++;
						}
					}
				}
			}
		}

		sched_setaffinity_set_core(localcore + Config->CPUCoreOffset);
		if (Config->Debug) fprintf(STD_OUT, "OpenMP BLAS thread %d pinned to core %d\n", thread_id, localcore);
	}
	setUnknownNames("Unknown OMP Thread");
	
	sched_setaffinity(0, sizeof(oldaffinity), &oldaffinity);
	delete[] cpu_order;
#endif
}

int caldgemm::CheckParams()
{
	if (Config->PipelinedOperation)
	{
		fprintf(STD_OUT, "Pipelined Mode not supported by backend!\n");
		return(1);
	}
	return(0);
}

int caldgemm::WaitForCALDGEMMProgress(size_t n)
{
	return(0);	//Default backend does not support pipelined mode, so we do not have to bother.
}

int caldgemm::InitCALDGEMM(caldgemm_config* pInfo, bool nocalinit)
{
	Config = pInfo;
	
	if (Config->ForceNumCPUThreads) conf_numprocs = Config->ForceNumCPUThreads;
#if defined(USE_GOTO_BLAS) & !defined(_WIN32)
	else conf_numprocs = get_num_procs();
#endif
	
#ifdef USE_GOTO_BLAS
	if (!Config->Quiet) fprintf(STD_OUT, "Initializing GotoBLAS\n");
	gotoblas_init();
#endif

	if (Config->Iterations > 1 && Config->UseCPU)
	{
		fprintf(STD_OUT, "ERROR: Multiple Iterations not supported with CPU enabled\n");
		return(1);
	}

#ifdef _WIN32
	strcpy(hostname, "Win32");
#else
	gethostname(hostname, 255);
#endif

#ifdef USE_GOTO_BLAS
	sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);		//GotoBLAS has its own thread pinning, store old value here.
#endif

	if (Config->PinCPU != -1)
	{
	    for (unsigned int i = 0;i < max_devices;i++) Config->GPUMapping[i] = Config->PinCPU;
	}

	CPU_ZERO(&gpumask);
	if (Config->PinMainThread == -1) Config->PinMainThread = Config->GPUMapping[0];
	CPU_SET(Config->PinMainThread + Config->CPUCoreOffset, &gpumask);

	if (Config->Debug) fprintf(STD_OUT, "Init Caldgemm, setting CPU mask %X\n", getcpumask(&gpumask));
	if (0 != sched_setaffinity(0, sizeof(gpumask), &gpumask))
	{
		fprintf(STD_OUT, "Error setting CPU affinity\n");
		return(1);
	}
	
	if (Config->SlowCPU)
	{
		Config->DynamicSched = false;
		Config->SmallTiles = 1;
	}
	if (SimpleQueuingAvailable() < 3 && Config->AlternateSimpleQueuingMulti)
	{
		fprintf(STD_OUT, "Alternate Simple Multi Queuing not supported by backend, disabling\n");
		Config->AlternateSimpleQueuingMulti = false;
	}
	if (SimpleQueuingAvailable() < 2 && Config->AlternateSimpleQueuing)
	{
		fprintf(STD_OUT, "Alternate Simple Queuing not supported by backend, disabling\n");
		Config->AlternateSimpleQueuing = false;
	}
	if (SimpleQueuingAvailable() < 1 && Config->SimpleGPUQueuing)
	{
		fprintf(STD_OUT, "Simple GPU Queuing not supported by backend, disabling\n");
		Config->SimpleGPUQueuing = false;
	}
	if (PipelinedModeAvailable() < 2 && Config->PipelineDoubleBuffer)
	{
		fprintf(STD_OUT, "Pipelined mode with double buffering not supported by backend, disabling\n");
		Config->PipelineDoubleBuffer = false;
	}
	if (PipelinedModeAvailable() < 1 && Config->PipelinedOperation)
	{
		fprintf(STD_OUT, "Pipelined operation not supported by backend, disabling\n");
		Config->PipelinedOperation = false;
		Config->PipelinedMidMarker = 0;
	}
	if (AsyncModeAvailable() < 2 && Config->AsyncDTRSM)
	{
		fprintf(STD_OUT, "Async Side-queue with DTRSM not supported by backend, disabling async DTRSM\n");
		Config->AsyncDTRSM = false;
	}
	if (AsyncModeAvailable() < 1 && Config->AsyncSideQueue)
	{
		fprintf(STD_OUT, "Async Side-queue not supported by backend, disabling\n");
		Config->AsyncSideQueue = false;
	}
	if (Config->AlternateSimpleQueuingMulti) Config->AlternateSimpleQueuing = true;
	if (Config->AlternateSimpleQueuing) Config->SimpleGPUQueuing = true;
	if (!Config->SimpleGPUQueuing && Config->PipelinedOperation)
	{
		fprintf(STD_OUT, "Pipeline Operation requires SimpleGPUQueuing!\n");
		return(1);
	}
	if (Config->SimpleGPUQueuing && !Config->GPU_C)
	{
		fprintf(STD_OUT, "Simple GPU Queuing requires GPU_C!\n");
		return(1);
	}
	if (!Config->PipelinedOperation)
	{
		Config->PipelinedMidMarker = 0;
		Config->PipelineDoubleBuffer = false;
	}
	if (Config->MultiThread == false) Config->MultiThreadDivide = false;
	if (Config->MultiThread == false || !Config->UseCPU) Config->SpawnGPUThread = -2;
	if (Config->ParallelDMA || Config->SimpleGPUQueuing) Config->ImprovedScheduler = true;
	if ((Config->AsyncSideQueue || Config->SimpleGPUQueuing) && (Config->GPU_C == 0 || UseInputPthreads() || UseOutputPthreads()))
	{
		fprintf(STD_OUT, "ASYNC Side queue / Simple GPU Queuing can only work with GPU_C\n");
		Config->AsyncSideQueue = false;
	}
	if (!Config->AsyncSideQueue) Config->AsyncDTRSM = false;

	setThreadName(Config->SpawnGPUThread == -2 ? "Main (GPU)" : "Main (CPU)");

#ifndef USE_GOTO_BLAS
	if (Config->ParallelDMA && Config->linpack_broadcast_function && (Config->ParallelDMA > Config->AlternateLookahead || Config->DynamicSched))
	{
		fprintf(STD_OUT, "WARNING: There is a possible thread-pinning collision when using Parallel DMA in multi-node HPL if either Dynamic Scheduling is activated or ParallelDMA > AlternateLookahead\n");
	}
#endif
	if (CheckParams()) return(1);

	if (ValidateRuntime()) return(1);
	if (Config->Height == 0) Config->Height = 4096; //Runtime did not set suggested value, so we use the default
	if (Config->ImplicitDriverSync == -1) Config->ImplicitDriverSync = 1;
	buffersSwitchable = (KernelSettings.transposeA ^ KernelSettings.transposeB);
	if (Config->Debug) fprintf(STD_OUT, "Initializing Backend\n");
	setUnknownNames("Unknown - Before Runtime Initialization");
	
	if (Config->PinDeviceRuntimeThreads != -2)
	{
		cpu_set_t affinity;
		CPU_ZERO(&affinity);
		if (Config->PinDeviceRuntimeThreads == -1) for (int i = 0;i < conf_numprocs;i++) CPU_SET(i + Config->CPUCoreOffset, &affinity);
		else CPU_SET(Config->PinDeviceRuntimeThreads + Config->CPUCoreOffset, &affinity);
		if (0 != sched_setaffinity(0, sizeof(affinity), &affinity))
		{
			fprintf(STD_OUT, "Error setting CPU affinity\n");
			return(1);
		}
	}
	
	if (Initialize(nocalinit) || !Config->UseGPU)
	{
		gpu_available = false;
	}

	if (!gpu_available)
	{
		if (!AllowCPUFallback()) return(1);
		if (!Config->Quiet && Config->UseGPU) fprintf(STD_OUT, "No GPU available, falling back to CPU\n");
		nDevices = 0;
		Config->UseGPU = 0;
		Config->UseCPU = 1;
		Config->KeepBuffersMapped = 0;
	}

	if (Config->PinDeviceRuntimeThreads != -2 && 0 != sched_setaffinity(0, sizeof(gpumask), &gpumask))
	{
		fprintf(STD_OUT, "Error setting CPU affinity\n");
		return(1);
	}
	
	if (Config->ParallelDMA && Config->GroupParallelDMA)
	{
		for (int i = 0;i < nDevices;i++)
		{
			if (Config->AllocMapping[i] == -1)
			{
				fprintf(STD_OUT, "Error during initialization, GroupParallelDMA activated but AllocMapping not set for GPU %d\n", i);
				return(1);
			}
			bool found = false;
			for (int j = 0;j < nDevices;j++)
			{
				if (Config->DMAMapping[j] == Config->AllocMapping[i])
				{
					found = true;
					break;
				}
			}
			if (found == false)
			{
				fprintf(STD_OUT, "Error during initialization, No DMAMapping thread found that maps to the AllocMapping of GPU %d\n", i);
				return(1);
			}
		}
	}

	if (CheckDevices()) return(1);

	outputthreads = Config->OutputThreads == -1 ? (Config->KeepBuffersMapped || Config->DstMemory == 'g' ? CALDGEMM_OUTPUT_THREADS : CALDGEMM_OUTPUT_THREADS_SLOW) : Config->OutputThreads;

	if (Config->UseGPU && InitDevices()) return(1);
	
	min_bbuffers = max_bbuffers;
	for (int i = 0;i < nDevices;i++)
	{
		if (bbuffers[i] < min_bbuffers) min_bbuffers = bbuffers[i];
	}
	if (!Config->Quiet)
	{
		if (nDevices)
		{
			fprintf(STD_OUT, "Running on %d devices with %d bbuffers (%s)\n", nDevices, min_bbuffers, hostname);
		}
		else
		{
			fprintf(STD_OUT, "Running on CPU only (%s)\n", hostname);
		}
	}

	int thread = (Config->PinDeviceRuntimeThreads >= 0 ? Config->PinDeviceRuntimeThreads : Config->PinMainThread) + Config->CPUCoreOffset;
	setUnknownAffinity(1, &thread);
	setUnknownNames("Device Runtime");

	if (Config->PinBroadcastThread == -1)
	{
		int linpackCPU = 0;
		while (linpackCPU < conf_numprocs)
		{
			if (cpuUsed(linpackCPU) == false) break;
			linpackCPU++;
		}
		if (linpackCPU >= conf_numprocs) linpackCPU = 0;
		broadcast_cpu_core = linpackCPU;
	}
	else
	{
		broadcast_cpu_core = Config->PinBroadcastThread;
	}
	if (Config->Debug) fprintf(STD_OUT, "Broadcast CPU core set to %d\n", broadcast_cpu_core);

#ifndef USE_GOTO_BLAS		//If we do not use GotoBLAS thread pinning determine main blas thread only after determining GPU devices to avoid collisions. Store the thread afterward as for GotoBLAS.
	if (Config->UseCPU)
	{
		if (Config->SpawnGPUThread >= 0)
		{
			main_blas_core = Config->SpawnGPUThread;
			if (Config->PinBroadcastThread == -1 && main_blas_core == broadcast_cpu_core)
			{
				fprintf(STD_OUT, "Your pinning of the Main CPU thread (Config->SpawnGPUThread) collides with autoselected linpack blas core, please set Config->PinBroadcastThread!");
				return(1);
			}
		}
		else
		{
			main_blas_core = 0;
			while ((cpuUsed(main_blas_core) || broadcast_cpu_core == main_blas_core) && main_blas_core < conf_numprocs - 1) main_blas_core++;
		}
	}
	else
	{
		main_blas_core = Config->PinMainThread;
	}
	if (Config->Debug) fprintf(STD_OUT, "Pinning Main OpenMP BLAS thread to core %d\n", main_blas_core);
	sched_setaffinity_set_core(main_blas_core + Config->CPUCoreOffset);

	sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);		//As for GotoBLAS above, store pinning here
#else	//Set main blas core for GotoBLAS
	for (int i = 0;i < conf_numprocs;i++)
	{
		main_blas_core = 0;
		if (CPU_ISSET(i, &oldcpumask))
		{
			main_blas_core = i;
			break;
		}
	}
#endif

	if (Config->MultiThread && UseOutputPthreads())
	{
		for (int device_num = 0;device_num < nDevices;device_num++)
		{
			for (int i = 0;i < (Config->OutputThreads == -1 ? max_outputthreads : Config->OutputThreads);i++)
			{
				mParam[device_num][i].num_device = device_num;
				mParam[device_num][i].cls = this;
				mParam[device_num][i].terminate = false;
				mParam[device_num][i].nMergeThread = i;
				pthread_t thr;
				pthread_create(&thr, NULL, merge_wrapper, &mParam[device_num][i]);

				while (mParam[device_num][i].mergeThreadMutex[0].Trylock() != EBUSY) mParam[device_num][i].mergeThreadMutex[0].Unlock();
			}
		}
	}
	
	if (Config->MultiThread && UseMutexPerDevice())
	{
		for (int i = 0;i < nDevices;i++)
		{
			pthread_mutex_init(&device_mutex[i], NULL);
		}
	}

	sched_setaffinity(0, sizeof(gpumask), &gpumask);
	
#ifdef CALDGEMM_DIVIDE_STATIC_BUFFER
	divide_tmpBuffer = allocDivideBuffer();
#endif

	if (Config->AlternateLookahead)
	{
		pthread_mutex_init(&tilesRemainingMutex, NULL);
		alternateLookaheadMutex.Lock();
	}

	if (Config->MultiThread)
	{
		linpackParameters.terminate = false;
		linpackParameters.linpackMutex[1].Lock();
		pthread_t thr;
		pthread_create(&thr, NULL, linpack_broadcast_wrapper, this);
		if (Config->Debug) fprintf(STD_OUT, "Waiting for linpack slave to start\n");
		while (linpackParameters.linpackMutex[0].Trylock() != EBUSY) linpackParameters.linpackMutex[0].Unlock();
		pthread_mutex_init(&scheduleMutex, NULL);
		
		divideThreads = 0;
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int i = 0;i < nDevices;i++)
			{
				DGEMMTasks[i].mutex_start.Lock();
				DGEMMTasks[i].mutex_finished.Lock();
				if (Config->GPUMapping[i] == Config->PinMainThread) continue;
				int found = 0;
				for (int j = 0;j < i;j++)
				{
					if (Config->GPUMapping[i] == Config->GPUMapping[j])
					{
						found = 1;
						break;
					}
				}
				if (found == 0)
				{
					pthread_t thr;
					dParam[divideThreads].cls = this;
					dParam[divideThreads].CPUCore = Config->GPUMapping[i];
					dParam[divideThreads].nThread = divideThreads;
					dParam[divideThreads].terminate = 0;
					pthread_create(&thr, NULL, divide_wrapper, &dParam[divideThreads]);
					DGEMMTasks[divideThreads].mutex_finished.Lock();
					divideThreads++;
				}
			}
		}
	}

	for (int l = 0;l < nDevices;l++)
	{
		for (int i = 0;i < obuffercount;i++) DGEMMPrepareTaskEventReady[l][i] = false;
		DGEMMTasks[l].thread_running = 0;
		DGEMMTasks[l].skip_device_to = -1;
		DGEMMTasks[l].device = l;
	}

	if (Config->Debug) fprintf(STD_OUT, "Using %d CPU cores at %d MHz, %d GPUs of %d shaders at %d MHz\n", conf_numprocs, conf_cpufreq, nDevices, conf_gpushaders, conf_gpufreq);
	ensure_omp_thread_pinning(Config->SpawnGPUThread != -2 ? NULL : "Main");

	if (Config->UseCPU)
	{
		cParam.cls = this;
		cParam.terminate = false;
		cParam.cblasMutex[0].Lock();
		if (Config->MultiThread)
		{
			pthread_t thr;
			pthread_create(&thr, NULL, cblas_wrapper, &cParam);
			if (Config->Debug) fprintf(STD_OUT, "Waiting for cblas slave to start\n");
			while (cParam.cblasMutex[1].Trylock() != EBUSY) cParam.cblasMutex[1].Unlock();
		}
	}
	
	if (Config->ParallelDMA && nDevices)
	{
		DMAThreads.SetNumberOfThreads(nDevices - 1, this, &caldgemm::DMA_wrapper, 1, &Config->DMAMapping[1]);
	}

	if (Config->ThreadSaveDriver == -1)
	{
		pthread_mutex_init(&globalDriverLock, NULL);
	}
	if (Config->UseDMAFetchQueue)
	{
		for (int i = 0;i < nDevices;i++)
		{
			pthread_mutex_init(&dma_fetch_queue_tasks[i].mutex, NULL);
		}
	}
#ifndef _WIN32
	if (Config->UseGPU && Config->UseCPU)
	{
		for (int i = 0;i < conf_numprocs;i++)
		{
			if (CPU_ISSET(i, &oldcpumask) && cpuUsed(i)) fprintf(STD_OUT, "WARNING: Core %d used by GotoBLAS main thread and CALDGEMM, be sure not to use CPU and GPU at the same time!\n", i);
		}
	}
#endif

	if (Config->MemPolicy)
	{
#ifdef _WIN32

#else
		unsigned long nodemask = 0xffffff;
		syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
#endif
	}

	if (Config->PreallocData)
	{
		if (Preallocate()) return(1);
	}
	
	/*fprintf(STD_OUT, "Setting FIFO scheduler\n");
	sched_param param;
	sched_getparam(0, &param);
	param.sched_priority = 1;
	if (0 != sched_setscheduler(0, SCHED_FIFO, &param))
	{
	fprintf(STD_OUT, "Error setting scheduler\n");
	return(1);
	}*/
	//setpriority(PRIO_PROCESS, 0, -20);
	
	if (Config->Debug) fprintf(STD_OUT, "Caldgemm Init complete, setting CPU mask %X\n", getcpumask(&oldcpumask));
	sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);
	
	goto_set_num_threads(conf_numprocs);
	
	if (FinishDataInit()) return(1);
	finishData->running = false;

	for (int i = 0;i < nDevices;i++) for (int j = 0;j < 2;j++) DGEMMTasks[i].PrepareTasks[j].j = DGEMMTasks[i].PrepareTasks[j].k = 0; //Fix valgrind warning
	cParam.cblas_size = cParam.dynamic_run = 0;
	
	nDevicesInitialized = nDevices;
	if (Config->NumActiveDevices > 0 && Config->NumActiveDevices < nDevices) nDevices = Config->NumActiveDevices;

	caldgemm_initialized = true;
	
	if (Config->ShowConfig) printConfig();

	return(0);
}

int caldgemm::broadcastcore()
{
	return(broadcast_cpu_core);
}

bool caldgemm::cpuUsed(int cpu)
{
	if (Config->UseGPU && cpu == Config->PinMainThread) return(true);

	for (int i = 0;i < nDevices;i++)
	{
		if (UseInputPthreads())
		{
			int procsreq = 1;
			for (int j = i;j < nDevices;j++)
			{
				if (Config->GPUMapping[i] == Config->GPUMapping[j] && Config->PostprocessMapping[j] == -1) procsreq += outputthreads;
			}
			if ((Config->MultiThreadDivide ? (cpu >= Config->GPUMapping[i]) : (cpu > Config->GPUMapping[i])) && cpu < Config->GPUMapping[i] + procsreq) return(true);
		}

		if (UseOutputPthreads())
		{
			if (Config->PostprocessMapping[i] != -1 && cpu >= Config->PostprocessMapping[i] && cpu < Config->PostprocessMapping[i] + outputthreads) return(true);
		}

		if (Config->ParallelDMA && matrix_n >= Config->ParallelDMA)
		{
			if (((matrix_n < Config->GroupParallelDMA || (signed) Config->GroupParallelDMA == -1) ? Config->AllocMapping[i] : Config->DMAMapping[i]) == cpu) return(true);
		}
	}
	
	for (int i = 0;i < Config->nExcludeCPUCores;i++) if (Config->ExcludeCPUCores[i] == cpu) return(true);

	if (Config->PinDeviceRuntimeThreads == cpu) return(true);

	return(false);
}

int caldgemm::reserve_cpu_cores()
{
	int nthreads = 0;
	int mainfound = 0;
	if (UseOutputPthreads() || UseInputPthreads() || Config->ParallelDMA || Config->GroupParallelDMA)
	{
		for (int i = 0;i < nDevices;i++)
		{
			int offset = 0;
			for (int j = 0;j < i;j++)
			{
				if (Config->GPUMapping[i] == Config->GPUMapping[j] && Config->PostprocessMapping[j] != -1) offset++;
			}
			if (matrix_n >= Config->ParallelDMA && Config->ParallelDMA != 0)
			{
				if (matrix_n < Config->GroupParallelDMA)
				{
					if (Config->AllocMapping[i] != Config->PinMainThread)
					{
						bool found = false;
						for (int j = 0;j < i;j++)
						{
							if (Config->AllocMapping[j] == Config->AllocMapping[i])
							{
								found = true;
								break;
							}
						}
						if (!found)
						{
							caldgemm_goto_reserve_cpu(Config->AllocMapping[i], 1);
							if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for Grouped DMA Thread\n", Config->AllocMapping[i]);
							nthreads++;
						}
					}
				}
				else if (i)
				{
					caldgemm_goto_reserve_cpu(Config->DMAMapping[i], 1);
					if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for DMA Thread\n", Config->DMAMapping[i]);
					nthreads++;
				}
			}
			else if (offset == 0 && Config->MultiThreadDivide && UseInputPthreads())
			{
				caldgemm_goto_reserve_cpu(Config->GPUMapping[i], 1);
				if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for DivideBuffer\n", Config->GPUMapping[i]);
				nthreads++;
				if (Config->GPUMapping[i] == Config->PinMainThread) mainfound = 1;
			}

			if (UseOutputPthreads())
			{
				for (int j = 0;j < outputthreads;j++)
				{
					const int merge_core = Config->PostprocessMapping[i] == -1 ? (Config->GPUMapping[i] + 1 + offset * outputthreads + j) : (Config->PostprocessMapping[i] + j);
					caldgemm_goto_reserve_cpu(merge_core, 1);
					if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for MergeBuffer\n", merge_core);
				}
				nthreads += outputthreads;
			}
		}
	}

	if (mainfound == 0 || !Config->MultiThreadDivide)
	{
		caldgemm_goto_reserve_cpu(Config->PinMainThread, 1);
		if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for Main Thread\n", Config->PinMainThread);
		if (Config->ForceNumCPUThreads == 0 || Config->PinMainThread < Config->ForceNumCPUThreads) nthreads++;
	}

	for (int i = 0;i < Config->nExcludeCPUCores;i++)
	{
		caldgemm_goto_reserve_cpu(Config->ExcludeCPUCores[i], 1);
		if (Config->Debug) fprintf(STD_OUT, "Excluding Core %d\n", Config->ExcludeCPUCores[i]);
	}
	if (Config->ForceNumCPUThreads) nthreads += Config->nExcludeCPUCores;

	if (Config->PinDeviceRuntimeThreads >= 0)
	{
		caldgemm_goto_reserve_cpu(Config->PinDeviceRuntimeThreads, 1);
		if (Config->ForceNumCPUThreads == 0 || Config->PinDeviceRuntimeThreads < Config->ForceNumCPUThreads) nthreads++;
		nthreads++;
	}

	if (Config->Debug) fprintf(STD_OUT, "Reserved %d cores\n", nthreads);
	return(nthreads);
}

void caldgemm::DMA_wrapper(caldgemm::clsDMAParam* par)
{
	{
		char tmpName[32];
		sprintf(tmpName, "DMA Thread %d", par->threadNum);
		setThreadName(tmpName);
	}
	if (Config->Debug) fprintf(STD_OUT, "DMA wrapper thread %d running\n", par->threadNum);
	while(par->WaitForTask())
	{
		if (Config->Debug) fprintf(STD_OUT, "DMA wrapper thread %d starting processing\n", par->threadNum);
		RunCALDGEMMMain(par->threadNum);
	}
	if (Config->Debug) fprintf(STD_OUT, "DMA wrapper thread %d terminating\n", par->threadNum);
}

void* caldgemm::linpack_broadcast_wrapper(void* arg)
{
	return ((caldgemm*) arg)->linpack_broadcast_wrapper_a();
}

void* caldgemm::linpack_broadcast_wrapper_a()
{
	setThreadName("Linpack Broadcast Wrapper");
	if (Config->Debug) fprintf(STD_OUT, "Linpack broadcast helper thread started\n");

	int linpackCPU = broadcast_cpu_core;
	if (linpackCPU >= conf_numprocs_real) linpackCPU = 0;
	if (Config->Debug) fprintf(STD_OUT, "Linpack Thread, core %d\n", linpackCPU);
	sched_setaffinity_set_core(linpackCPU + Config->CPUCoreOffset);

	linpackParameters.linpackMutex[0].Lock();
	while (linpackParameters.linpackMutex[0].Lock() == 0 && linpackParameters.terminate == false)
	{
		Timers.LinpackTimer2.Start();
		Config->linpack_broadcast_function();
		Timers.LinpackTimer2.Stop();
		Timers.BcastTimer.Start();

		linpackParameters.linpackMutex[1].Unlock();
	}

	if (Config->Debug) fprintf(STD_OUT, "linpack slave terminating\n");

	linpackParameters.linpackMutex[1].Unlock();

	pthread_exit(NULL);
	return(NULL);
}

int caldgemm::cpuScheduler()
{
	int retVal = 0;
	if (Config->UseCPU && Config->MultiThread && Config->DynamicSched && (Config->ParallelDMA == 0 || Config->ParallelDMA > matrix_n))
	{
		const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
		const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;
		size_t nBlocks = mb * nb;

		pthread_mutex_lock(&scheduleMutex);
		const size_t k = gpu_k_barrier == -1 ? 0 : gpu_k_barrier;

		if ((size_t) gpu_k_barrier < nBlocks - 1)
		{
			size_t blockm, blockn;
			DGEMM_getblocks(k, blockm, blockn);

			if (cParam.dynamic_run == 0 && Config->SecondPhaseDynamicRuns)
			{
				cParam.dynamic_size = ((1.0f - gpu_ratio_used) * (float) (nBlocks - k - 1) + 0.5) * Config->Height;
				if (cParam.dynamic_size > (nBlocks - k - 1) * Config->Height) cParam.dynamic_size = (nBlocks - k - 1) * Config->Height;
				if (cParam.dynamic_size > Config->Height)
				{
					cParam.dynamic_run = 1 + cParam.dynamic_size / mymin(gpu_m, gpu_n);
					cParam.dynamic_size /= cParam.dynamic_run;
					cParam.dynamic_size -= cParam.dynamic_size % Config->Height;
					cParam.dynamic_run *= Config->Height;
					if (cParam.dynamic_size && (DGEMM_favor_m ? gpu_n : gpu_m) % Config->Height)
					{
						const size_t adjustment = Config->Height - (DGEMM_favor_m ? gpu_n : gpu_m) % Config->Height;
						if (Config->Debug) fprintf(STD_OUT, "Adjusting second phase run size for small tiles: %lld - %lld = %lld\n", (long long int) cParam.dynamic_size, (long long int) adjustment, (long long int) cParam.dynamic_size - adjustment);
						cParam.dynamic_size -= adjustment;
					}
					if (cParam.dynamic_run && (DGEMM_favor_m ? gpu_m : gpu_n) % Config->Height)
					{
						const size_t adjustment = Config->Height - (DGEMM_favor_m ? gpu_m : gpu_n) % Config->Height;
						if (Config->Debug) fprintf(STD_OUT, "Adjusting second phase run row size for small tiles: %lld - %lld = %lld\n", (long long int) cParam.dynamic_run, (long long int) adjustment, (long long int) cParam.dynamic_run - adjustment);
						cParam.dynamic_run -= adjustment;
					}

					while (DGEMM_favor_m ? (blockm * Config->Height >= gpu_m - cParam.dynamic_run && blockn * Config->Height >= gpu_n - cParam.dynamic_size) :
						(blockn * Config->Height >= gpu_n - cParam.dynamic_run && blockm * Config->Height >= gpu_m - cParam.dynamic_size))
					{
						if (cParam.dynamic_run > Config->Height)
						{
							cParam.dynamic_run -= Config->Height;
							cParam.dynamic_size = mymin(gpu_m, gpu_n);
						}
						else
						{
							if (cParam.dynamic_size > Config->Height)
							{
								cParam.dynamic_size -= Config->Height;
							}
							else
							{
								cParam.dynamic_run = cParam.dynamic_size = 0;
							}
						}
						if (Config->Debug) fprintf(STD_OUT, "cParam dynamic size reduced to: %lld blockrows (%lld), %lld blocks (%lld)\n", (long long int) cParam.dynamic_run / Config->Height, (long long int) cParam.dynamic_run, (long long int) cParam.dynamic_size / Config->Height, (long long int) cParam.dynamic_size);
					}

					if (nBlocks >= 256 && nBlocks - k - 1 > 16 && cParam.dynamic_run == Config->Height && cParam.dynamic_size < mymin(gpu_m, gpu_n)) cParam.dynamic_size += Config->Height;

					if (!Config->Quiet) fprintf(STD_OUT, "Scheduling Additional CPU DGEMM Run over %lld blockrows (%lld), %lld blocks (%lld)\n", (long long int) cParam.dynamic_run / Config->Height, (long long int) cParam.dynamic_run, (long long int) cParam.dynamic_size / Config->Height, (long long int) cParam.dynamic_size);
					retVal = 1;
				}
				else
				{
					cParam.dynamic_size = 0;
					goto TryThirdRun;
				}
			}
			else
			{
TryThirdRun:
				if (Config->ThirdPhaseDynamicRuns)
				{
					size_t test_cpu_k = cpu_k_barrier - 1;
					size_t cpublockm, cpublockn;
					DGEMM_getblocks(test_cpu_k, cpublockm, cpublockn);
					while (test_cpu_k > k && (DGEMM_favor_m ? (cpublockm * Config->Height >= gpu_m - cParam.dynamic_run && cpublockn * Config->Height >= gpu_n - cParam.dynamic_size) :
						(cpublockn * Config->Height >= gpu_n - cParam.dynamic_run && cpublockm * Config->Height >= gpu_m - cParam.dynamic_size)))
					{
						test_cpu_k--;
						DGEMM_getblocks(test_cpu_k, cpublockm, cpublockn);
					}
					if ((long long int) test_cpu_k > 0 && (signed) k <= (signed) test_cpu_k - 2 * nDevices + Config->ThirdPhaseThreshold)
					{
						if (!Config->Quiet) fprintf(STD_OUT, "Scheduling dynamic 3rd phase run, CPU taking tile %lld (k=%lld,m=%lld,n=%lld) from GPU (GPU k = %lld)\n", (long long int) test_cpu_k, (long long int) k, (long long int) cpublockm, (long long int) cpublockn, (long long int) gpu_k_barrier);
						cParam.dynamic_run2++;
						cParam.cpu_k = test_cpu_k;
						cpu_k_barrier = test_cpu_k;
						retVal = 1;
					}
				}
			}
		}
		pthread_mutex_unlock(&scheduleMutex);
	}
	return(retVal);
}

void caldgemm::RunLinpackFactorization(int old_goto_threads, int& require_threads)
{
	const CBLAS_TRANSPOSE TransposeA = this->TransposeA ? CblasTrans : CblasNoTrans;
	const CBLAS_TRANSPOSE TransposeB = this->TransposeB ? CblasTrans : CblasNoTrans;
	const size_t A_pitch_use = (this->TransposeA ? 1 : A_pitch);

	if (ExecLinpack >= 2)
	{
		if (Config->AlternateLookahead > matrix_n)
		{
			if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tWaiting for GPUs to finish initial DGEMM part to start Linpack factorization\n");
			alternateLookaheadMutex.Lock();
			if (Config->SimpleGPUQueuing)
			{
				CheckAlternateTilesRemainingSQ();
			}
			_mm_mfence();
		}
		else
		{
			if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tDoing initial cblas runs to prepare Linpack factorization\n");
			Timers.CPUTimer.Start();
			cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->Width, matrix_n, Config->Width, Alpha, A - Config->Width * A_pitch_use, A_pitch, B, B_pitch, Beta, C - Config->Width * C_pitch, C_pitch);
			Timers.CPUTimer.Stop();
		}
		if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tStarting Linpack factorization\n");
		if (Config->HPLFactorizeRestrictCPUs == 1)
		{
			if (8 < old_goto_threads - require_threads) goto_set_num_threads(8);
		}
		else if (Config->HPLFactorizeRestrictCPUs >= 2)
		{
			caldgemm_goto_restrict_cpus(Config->HPLFactorizeRestrictCPUs);
		}
		Timers.LinpackTimer1.Start();
		Config->linpack_factorize_function();
		Timers.LinpackTimer1.Stop();
		if (Config->HPLFactorizeRestrictCPUs >= 2) caldgemm_goto_restrict_cpus(0);
	}

	if (Config->LinpackNodes > 1)
	{
		if (Config->MultiThread)
		{
			caldgemm_goto_reserve_cpu(broadcast_cpu_core, 1);
			if (Config->ForceNumCPUThreads == 0 || broadcast_cpu_core < Config->ForceNumCPUThreads) require_threads++;

			linpackParameters.linpackMutex[0].Unlock();
		}
		else
		{
			Timers.LinpackTimer2.Start();
			Config->linpack_broadcast_function();
			Timers.LinpackTimer2.Stop();
		}
	}
	goto_set_num_threads(old_goto_threads - require_threads);
}

void* caldgemm::cblas_wrapper(void* arg)
{
	return ((cblasParameters*) arg)->cls->cblas_wrapper_a(true);
}

int caldgemm::caldgemm_part_cpu()
{
	const size_t A_pitch_use = (TransposeA ? 1 : A_pitch);
	const size_t B_pitch_use = (TransposeB ? B_pitch : 1);
	const CBLAS_TRANSPOSE TransposeA = this->TransposeA ? CblasTrans : CblasNoTrans;
	const CBLAS_TRANSPOSE TransposeB = this->TransposeB ? CblasTrans : CblasNoTrans;
	if (!Config->Quiet) fprintf(STD_OUT, "\t\tSlave thread starting cblas (m: %lld, n: %lld, cblas_size: %lld (%lld), dynamic: %lld/%lld, cpu_k: %lld)\n", (long long int) matrix_m, (long long int) matrix_n, (long long int) cParam.cblas_size, (long long int) Config->Height, (long long int) cParam.dynamic_run, (long long int) cParam.dynamic_size, (long long int) cParam.cpu_k);

	int old_goto_threads = conf_numprocs;

	int require_threads_base = reserve_cpu_cores();
		
	if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu \n", require_threads_base);
	if (old_goto_threads > require_threads_base)
	{
		goto_set_num_threads(old_goto_threads - require_threads_base);
	}
	else
	{
		goto_set_num_threads(1);
		caldgemm_goto_reserve_cpus(0);
	}

	Timers.TotalCPUTimer.Start();
	Timers.LinpackTimer3.Start();
	bool cpus_restricted = false;
	if (Config->HPLFactorizeRestrictCPUs >= 2 && (Config->LinpackSwapN != NULL || (ExecLinpack && Config->AlternateLookahead <= matrix_n)))
	{
		caldgemm_goto_restrict_cpus(Config->HPLFactorizeRestrictCPUs);
		cpus_restricted = true;
	}
	if (Config->LinpackSwapN != NULL)
	{
		Config->linpack_swap_function();
	}
	Timers.LinpackTimer3.Stop();

	if (Config->HPLFactorizeRestrictCallback != NULL) require_threads_base += Config->HPLFactorizeRestrictCallback(matrix_n);
	int require_threads = require_threads_base;

	if ((ExecLinpack && Config->AlternateLookahead <= matrix_n) || ExecLinpack == 1)
	{
		RunLinpackFactorization(old_goto_threads, require_threads);
	}
		
	if (cpus_restricted)
	{
		caldgemm_goto_restrict_cpus(0);
		if (old_goto_threads > require_threads)
		{
			goto_set_num_threads(old_goto_threads - require_threads);
		}
		else
		{
			goto_set_num_threads(1);
		}
	}


	Timers.CPUTimer.Start();
	bool linpackfinished = false;
	do
	{
		if (cParam.dynamic_run2)
		{
			size_t blockm, blockn;
			DGEMM_getblocks(cParam.cpu_k, blockm, blockn);
			VT_USER_START_A("CPU DGEMM Phase 3");
			cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, blockm == gpu_m / Config->Height ? (gpu_m % Config->Height) : Config->Height, blockn == gpu_n / Config->Height ? (gpu_n % Config->Height) : Config->Height, Config->Width, Alpha, A + blockm * Config->Height * A_pitch_use, A_pitch, B + blockn * Config->Height * B_pitch_use, B_pitch, Beta, C + blockm * Config->Height * C_pitch + blockn * Config->Height, C_pitch);
			VT_USER_END_A("CPU DGEMM Phase 3");
		}
		else
		{
			if (cParam.dynamic_run)
			{
				VT_USER_START_A("CPU DGEMM Phase 2");
				if (DGEMM_favor_m)
				{
					cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cParam.dynamic_run, cParam.dynamic_size, Config->Width, Alpha, A + (gpu_m - cParam.dynamic_run) * A_pitch_use, A_pitch, B + (gpu_n - cParam.dynamic_size) * B_pitch_use, B_pitch, Beta, C + (gpu_m - cParam.dynamic_run) * C_pitch + gpu_n - cParam.dynamic_size, C_pitch);
				}
				else
				{
					cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cParam.dynamic_size, cParam.dynamic_run, Config->Width, Alpha, A + (gpu_m - cParam.dynamic_size) * A_pitch_use, A_pitch, B + (gpu_n - cParam.dynamic_run) * B_pitch_use, B_pitch, Beta, C + (gpu_m - cParam.dynamic_size) * C_pitch + gpu_n - cParam.dynamic_run, C_pitch);
				}
				VT_USER_END_A("CPU DGEMM Phase 2");
			}

			size_t cblas2;
			if (Config->RereserveLinpackCPU)
			{
				if (ExecLinpack && Config->LinpackNodes > 1 && Config->MultiThread && (((double) matrix_m * (double) matrix_n) - linpack_last_mn[ExecLinpack]) / linpack_last_mn[ExecLinpack] < 0.3 && linpackCPUDGEMMTime[ExecLinpack] - linpackBcastTime[ExecLinpack] > 5.0)
				{
					cblas2 = (double) (DGEMM_split_m ? matrix_n : matrix_m) * (linpackBcastTime[ExecLinpack] + 3.0) / linpackCPUDGEMMTime[ExecLinpack];
					if (!Config->Quiet) fprintf(STD_OUT, "Splitting CPU DGEMM for later enabling additional cores, cblas2=%lld\n", (long long int) cblas2);
				}
				else
				{
					cblas2 = 0;
				}
				if (cblas2 % 8) cblas2 += 8 - cblas2 % 8;
			}
			else
			{
				cblas2 = 0;
			}

			if (DGEMM_split_m)	//favor splitting m because of consecutive memory
			{
				if (matrix_n != gpu_n && cParam.borders_done == false)
				{
					VT_USER_START_A("CPU DGEMM Borders");
					cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, matrix_m - cParam.cblas_size, matrix_n - gpu_n, Config->Width, Alpha, A, A_pitch, B + gpu_n * B_pitch_use, B_pitch, Beta, C + gpu_n, C_pitch);
					VT_USER_END_A("CPU DGEMM Borders");
				}
					
				if (ExecLinpack >= 2 && cParam.borders_done == false && Config->AlternateLookahead > matrix_n)
				{
					Timers.CPUTimer.Stop();
					RunLinpackFactorization(old_goto_threads, require_threads);
					Timers.CPUTimer.Start();
				}
					
				if (cParam.dynamic_run == 0)
				{
					VT_USER_START_A("CPU DGEMM Phase 1");
					if (cblas2)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cParam.cblas_size, cblas2, Config->Width, Alpha, A + (matrix_m - cParam.cblas_size) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (matrix_m - cParam.cblas_size) * C_pitch, C_pitch);

						if (linpackParameters.linpackMutex[1].Trylock() == EBUSY)
						{
							if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
						}
						else
						{
							Timers.BcastTimer.Stop();
							if (!Config->NoPerformanceWarnings && Timers.BcastTimer.GetElapsedTime() > 1.0) fprintf(STD_OUT, "Bcast core idle for %2.4f seconds\n", Timers.BcastTimer.GetElapsedTime());

							int require_threads_new = require_threads_base;
							if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
							if (old_goto_threads > require_threads_new)
							{
								goto_set_num_threads(old_goto_threads - require_threads_new);
								caldgemm_goto_reserve_cpu(broadcast_cpu_core, 0);
							}
							else
							{
								goto_set_num_threads(1);
								caldgemm_goto_reserve_cpus(0);
							}
							linpackfinished = true;
						}
					}
					cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cParam.cblas_size, matrix_n - cblas2, Config->Width, Alpha, A + (matrix_m - cParam.cblas_size) * A_pitch_use, A_pitch, B + cblas2 * B_pitch_use, B_pitch, Beta, C + (matrix_m - cParam.cblas_size) * C_pitch + cblas2, C_pitch);
					VT_USER_END_A("CPU DGEMM Phase 1");
				}
			}
			else
			{
				if (cParam.dynamic_run == 0)
				{
					VT_USER_START_A("CPU DGEMM Phase 1");
					if (cblas2)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cblas2, cParam.cblas_size, Config->Width, Alpha, A, A_pitch, B + (matrix_n - cParam.cblas_size) * B_pitch_use, B_pitch, Beta, C + matrix_n - cParam.cblas_size, C_pitch);
							
						if (linpackParameters.linpackMutex[1].Trylock() == EBUSY)
						{
							if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
						}
						else
						{
							int require_threads_new = require_threads_base;
							if (old_goto_threads > require_threads_new)
							{
								if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
								goto_set_num_threads(old_goto_threads - require_threads_new);
								caldgemm_goto_reserve_cpu(broadcast_cpu_core, 0);
							}
							else
							{
								goto_set_num_threads(1);
								caldgemm_goto_reserve_cpus(0);
							}
							linpackfinished = true;
						}
					}
					cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, matrix_m - cblas2, cParam.cblas_size, Config->Width, Alpha, A + cblas2 * A_pitch_use, A_pitch, B + (matrix_n - cParam.cblas_size) * B_pitch_use, B_pitch, Beta, C + cblas2 * C_pitch + matrix_n - cParam.cblas_size, C_pitch);
					VT_USER_END_A("CPU DGEMM Phase 1");
				}

				if (ExecLinpack >= 2 && cParam.borders_done == false && Config->AlternateLookahead > matrix_n)
				{
					Timers.CPUTimer.Stop();
					RunLinpackFactorization(old_goto_threads, require_threads);
					Timers.CPUTimer.Start();
				}
					
				if (matrix_m != gpu_m && cParam.borders_done == false)
				{
					VT_USER_START_A("CPU DGEMM Borders");
					cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, matrix_m - gpu_m, matrix_n - cParam.cblas_size, Config->Width, Alpha, A + gpu_m * A_pitch_use, A_pitch, B, B_pitch, Beta, C + gpu_m * C_pitch, C_pitch);
					VT_USER_END_A("CPU DGEMM Borders");
				}
			}
		}

		cParam.borders_done = true;
		if (Config->Debug) fprintf(STD_OUT, "cblas run completed\n");
	} while (cpuScheduler());
	Timers.CPUTimer.Stop();

	if (linpackfinished == false && ExecLinpack && Config->MultiThread && Config->LinpackNodes > 1)
	{
		linpackParameters.linpackMutex[1].Lock();
	}
	Timers.TotalCPUTimer.Stop();
	goto_set_num_threads(old_goto_threads);
	caldgemm_goto_reserve_cpus(0);

	return(0);
}

int caldgemm::caldgemm_part_gpu()
{
	const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
	const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;
	const size_t nBlocks = mb * nb;

	if (Config->Debug)
	{
		if (DGEMM_favor_m)
		{
			fprintf(STD_OUT, "Favoring m direction, %lld blocks (%lld x %lld) (mb x nb)\n", (long long int) nBlocks, (long long int) mb, (long long int) nb);
		}
		else
		{
			fprintf(STD_OUT, "Not favoring m direction, %lld blocks (%lld x %lld) (mb x nb)\n", (long long int) nBlocks, (long long int) mb, (long long int) nb);
		}
	}

	if (!Config->NoPerformanceWarnings && (buffersSwitchable ? mymin(nb, mb) : nb) > (size_t) (bbuffers[0] * nDevices)) fprintf(STD_OUT, "WARNING: Insufficient buffers for Input Matrices, retransfer required\n");

	Timers.GPUTimer.Start();

	for (unsigned int i = 0; i < Config->Iterations; ++i)
	{
		AlternateLookaheadBlocksM = (std::min<size_t>(Config->Width, gpu_m) - 1) / Config->Height + 1;
		AlternateLookaheadTilesRemaining = AlternateLookaheadTilesFull = nb * AlternateLookaheadBlocksM;

		if (Config->ImprovedScheduler)
		{
			if (!Config->PreallocData) tileDistribution = new int[nBlocks];
			for (int l = 0;l < nDevices;l++) first_device_k[l] = -1;
				
			size_t block_correction_factor = 0;
			if (Config->Height > CALDGEMM_MIN_CORRECTION_SIZE && Config->SmallTiles && Config->ImprovedSchedulerBalance == 1)
			{
				size_t undersize;
				size_t scaler;
				if (DGEMM_favor_m)
				{
					undersize = gpu_n % Config->Height;
					scaler = mb;
				}
				else
				{
					undersize = gpu_m % Config->Height;
					scaler = nb;
				}
				if (undersize)
				{
					if (undersize < CALDGEMM_MIN_CORRECTION_SIZE) undersize = CALDGEMM_MIN_CORRECTION_SIZE;
					block_correction_factor = (Config->Height - undersize) * scaler / Config->Height;
				}
			}
			bool balance2 = Config->ImprovedSchedulerBalance == 2 && (DGEMM_favor_m ? gpu_n : gpu_m) % Config->Height;
			int mb_use, nb_use, nBlocks_use;
			if (balance2)
			{
				mb_use = DGEMM_favor_m ? mb : (mb - 1);
				nb_use = DGEMM_favor_m ? (nb - 1) : nb;
				nBlocks_use = mb_use * nb_use;
			}
			else
			{
				mb_use = mb;
				nb_use = nb;
				nBlocks_use = nBlocks;
			}
			//size_t numt[4] = {0,0,0,0}, sizet[4] = {0,0,0,0};
			for (size_t l = 0;l < nBlocks;l++)
			{
				size_t blockn, blockm;
				int k;
				if (DGEMM_favor_m)
				{
					blockn = l % nb;
					blockm = l / nb;
					if (balance2 && blockn == nb - 1)
					{
						tileDistribution[l] = nDevices - 1 - nDevices * blockm / mb;
					}
					else
					{
						k = blockn * mb_use + blockm;
						tileDistribution[l] = std::min<int>(nDevices * k / (nBlocks_use - block_correction_factor), nDevices - 1);
					}
				}
				else
				{
					blockm = l % mb;
					blockn = l / mb;
					if (balance2 && blockm == mb - 1)
					{
						tileDistribution[l] = nDevices - 1 - nDevices * blockn / nb;
					}
					else
					{
						k = blockn + blockm * nb_use;
						tileDistribution[l] = std::min<int>(nDevices * k / (nBlocks_use - block_correction_factor), nDevices - 1);
					}
				}
				/*numt[tileDistribution[l]]++;
				size_t height1 = (int) (((size_t) blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height);
				size_t height2 = (int) (((size_t) blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height);
				sizet[tileDistribution[l]] += height1 * height2;*/
				if (first_device_k[tileDistribution[l]] == -1) first_device_k[tileDistribution[l]] = l;

				//if (Config->Debug) fprintf(STD_OUT, "Tile %lld (%lld / %lld) processed by device %d\n", (long long int) l, (long long int) blockm, (long long int) blockn, tileDistribution[l]);
			}
				
			//for (int l = 0;l < 4;l++) fprintf(STD_OUT, "TILESTAT %d: %3lld - %lld\n", l, (long long int) numt[l], (long long int) sizet[l]);
			//fprintf(STD_OUT, "TILESIZE %lld (factor %lld - miss %lld)\n", (long long int) (Config->Height * Config->Height), (long long int) block_correction_factor, (long long int) ((sizet[2] - sizet[3]) / (Config->Height * Config->Height)));
				
			if (Config->Debug)
			{
				for (size_t l = 0;l < nBlocks;l++)
				{
					fprintf(STD_OUT, "%d ", tileDistribution[l]);
					if ((l + 1) % (DGEMM_favor_m ? nb : mb) == 0) fprintf(STD_OUT, "\n");
				}
			}
		}

		for (int ii = 0;ii < nDevices;ii++)
		{
			buffersMajor[ii] = -1;
			for (int j = 0;j < bbuffers[ii];j++) buffersMinor[ii][j] = -1;
			next_buffer_A[ii] = 0;
			next_buffer_B[ii] = 0;
		}

		if (Config->PreallocData && ((int) mb > Config->PreallocData || (int) nb > Config->PreallocData))
		{
			fprintf(STD_OUT, "Value of PreallocData too small for current block count! (mb %d nb %d pre %d)", (int) mb, (int) nb, Config->PreallocData);
			return(1);
		}

		if (RunCALDGEMM_Init()) return(1);

		if (Config->ParallelDMA != 0 && matrix_n >= Config->ParallelDMA)
		{
			DMAThreads.Start();
			RunCALDGEMMMain(0);
			DMAThreads.Sync();
		}
		else
		{
			if (RunCALDGEMMMain()) return(1);
		}
		if (RunCALDGEMM_Exit()) return(0);

		if (Config->ImprovedScheduler)
		{
			if (!Config->PreallocData) delete[] tileDistribution;
		}

		if(Config->Verify && i < Config->Iterations - 1) AnalyzeResults();
	}
	Timers.GPUTimer.Stop();

	if (Config->MultiThread && Config->UseCPU)
	{
		Timers.ATime.Reset();
		Timers.ATime.Start();
	}

	if (Config->SpawnGPUThread == -2)
	{
		if (Config->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&oldcpumask));
		sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);
	}

	return(0);
}

void* caldgemm::cblas_wrapper_a(bool thread)
{
	if (thread)
	{
		setThreadName(Config->SpawnGPUThread != -2 ? "GPU Wrapper" : "CBLAS Wrapper");
		if (Config->Debug) fprintf(STD_OUT, "Cblas helper thread started\n");
		if (Config->SpawnGPUThread == -2)
		{
			ensure_omp_thread_pinning("CBLAS");
		}

		if (Config->SpawnGPUThread != -2)
		{
			sched_setaffinity(0, sizeof(gpumask), &gpumask);
		}
		else if (Config->GPUMapping[0] + outputthreads * nDevices + 1 >= conf_numprocs)
		{
			sched_setaffinity_set_core(0 + Config->CPUCoreOffset);
		}
		else
		{
			if (Config->Debug) fprintf(STD_OUT, "Cblas thread Thread, setting CPU mask %X\n", getcpumask(&oldcpumask));
			sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);
		}
		cParam.cblasMutex[1].Lock();
	}

	while (cParam.cblasMutex[1].Lock() == 0 && cParam.terminate == false)
	{
		if (Config->SpawnGPUThread != -2)
		{
			caldgemm_part_gpu();
		}
		else
		{
			caldgemm_part_cpu();
		}

		if (Config->Debug) fprintf(STD_OUT, "\t\tUnlocking cblasmutex 0\n");
		cParam.cblasMutex[0].Unlock();
		if (!thread) break;
	}

	if (thread)
	{
		if (Config->Debug) fprintf(STD_OUT, "blas slave terminating\n");

		cParam.cblasMutex[0].Unlock();
		pthread_exit(NULL);
	}
	return(NULL);
}

void* caldgemm::divide_wrapper(void* arg)
{
	return ((divideParameters*) arg)->cls->divide_wrapper_a((divideParameters*) arg);
}

void* caldgemm::divide_wrapper_a(divideParameters* par)
{
	if (Config->Debug) fprintf(STD_OUT, "Divide Thread %d for core %d started\n", par->nThread, par->CPUCore);
	{
		char tmp[128];
		sprintf(tmp, "Divide %d", par->nThread);
		setThreadName(tmp);
	}
	sched_setaffinity_set_core(par->CPUCore + Config->CPUCoreOffset);
	
	par->curDevice = -1;
	for (int i = 0;i < nDevices;i++)
	{
		if (Config->GPUMapping[i] == par->CPUCore)
		{
			if (par->curDevice == 1) par->curDevice = i;
			DGEMMTasks[i].next_device = &par->curDevice;
		}
	}

	double* tmpBuffer = allocDivideBuffer();

	for (int i = 0;i < nDevices;i++)
	{
		if (Config->GPUMapping[i] == par->CPUCore)
		{
			par->firstDevice = i;
			break;
		}
	}

	int mutex_to_unlock = par->nThread;
	int i = 0;
	while (true)
	{
		if (Config->GPUMapping[i] == par->CPUCore)
		{
			par->reset = 0;
			par->curDevice = i;
			DGEMMTasks[mutex_to_unlock].mutex_finished.Unlock();
			if (Config->Debug) fprintf(STD_OUT, "Divide Thread %d on Core %d waiting to operate on device %d\n", par->nThread, par->CPUCore, i);

			DGEMMTasks[i].mutex_start.Lock();
			if (par->terminate) break;
			if (par->reset)
			{
				if (Config->Debug) fprintf(STD_OUT, "Divide Thread %d resetting\n", par->nThread);
				i = par->firstDevice;
				mutex_to_unlock = i;
				continue;
			}
			
			if (DGEMMTasks[i].skip_device_to != -1)
			{
				//fprintf(STD_OUT, "Skipping device %d, switching to %d\n", i, DGEMMTasks[i].skip_device_to);
				const int oldi = i;
				i = DGEMMTasks[i].skip_device_to;
				DGEMMTasks[oldi].skip_device_to = -1;
				DGEMMTasks[i].mutex_start.Lock();
			}
			
			if (Config->Debug) fprintf(STD_OUT, "Divide Thread for device %d Starting processing (k = %d)\n", i, DGEMMTasks[i].k);
			DGEMMPrepareAndExecute(DGEMMTasks[i] CALDGEMM_DIVBUFB);
			
			mutex_to_unlock = i;
		}
		i = (i + 1) % nDevices;
	}

	freeDivideBuffer(tmpBuffer);

	if (Config->Debug) fprintf(STD_OUT, "Divide Thread %d for Core %d terminating\n", par->nThread, par->CPUCore);

	DGEMMTasks[par->nThread].mutex_finished.Unlock();
	pthread_exit(NULL);
	return(NULL);
}

void* caldgemm::merge_wrapper(void* arg)
{
	return ((mergeParameters*) arg)->cls->merge_wrapper_a((mergeParameters*) arg);
}

void* caldgemm::merge_wrapper_a(mergeParameters* par)
{
	{
		char tmp[128];
		sprintf(tmp, "Merge %d/%d", par->num_device, par->nMergeThread);
		setThreadName(tmp);
	}

	if (Config->Debug) fprintf(STD_OUT, "Merger Thread %d started\n", par->nMergeThread);

	int merge_core;
	
	if (Config->PostprocessMapping[par->num_device] == -1)
	{
		merge_core = Config->GPUMapping[par->num_device] + par->nMergeThread + 1;
		for (int i = 0;i < par->num_device;i++)
		{
			if (Config->GPUMapping[i] == Config->GPUMapping[par->num_device]) merge_core += outputthreads;
		}
	}
	else
	{
		merge_core = Config->PostprocessMapping[par->num_device] + par->nMergeThread;
	}
	if (Config->Debug) fprintf(STD_OUT, "Merge Thread %d, core %d\n", par->nMergeThread, merge_core);
	sched_setaffinity_set_core(merge_core % conf_numprocs_real + Config->CPUCoreOffset);

	//HighResTimer mergeTimer;

	par->mergeThreadMutex[0].Lock();
	while (par->mergeThreadMutex[0].Lock() == 0 && par->terminate == false)
	{
		if (Config->Debug) fprintf(STD_OUT, "\t\tSlave thread %d (device %d) starting merge process for obuffer %d (k = %lld)\n", par->nMergeThread, par->num_device, par->nContext, (long long int) par->k);
		size_t blockm, blockn;
		DGEMM_getblocks(par->k, blockm, blockn);
		/*if (Config->Debug)
		{
		    mergeTimer.Reset();
		    mergeTimer.Start();
		}*/
		RunMergeBuffers(par->dst, par->num_device, par->nContext, (blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height, (blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height, BufferHeight, BufferHeight, C_pitch);
		/*if (Config->Debug)
		{
		    mergeTimer.Stop();
		    fprintf(STD_OUT, "\t\tMerge time: %2.3f\n", mergeTimer.GetElapsedTime());
		}*/
		if (!Config->SimpleGPUQueuing) CheckAlternateTilesRemaining(blockm);
		
		if (Config->Debug) fprintf(STD_OUT, "\t\tUnlocking mutex device %d obuffer %d (Slavethread %d)\n", par->num_device, par->nContext, par->nMergeThread);
		obufferMutex[par->num_device][par->nContext].Unlock();
		par->mergeThreadMutex[1].Unlock();
	}
	if (Config->Debug) fprintf(STD_OUT, "merge slave %d terminating\n", par->nMergeThread);
	par->mergeThreadMutex[1].Unlock();
	pthread_exit(NULL);
	return(NULL);
}

int caldgemm::DumpMatrix(double* a, double* b, double* c, double alpha, double beta, int tmp_m, int tmp_k, int tmp_n, int Apitch, int Bpitch, int Cpitch)
{
	int i = 0;
	char filename[256];
	FILE* fp = NULL;
	do
	{
		if (fp) fclose(fp);
		sprintf(filename, "dump%d.out", i++);
	} while ((fp = fopen(filename, "r")) != NULL && i < 100);
	if (i == 100)
	{
		if (fp) fclose(fp);
		return(1);
	}
	fp = fopen(filename, "w+b");
	int nWritten = 0;
	nWritten += fwrite(&a, sizeof(a), 1, fp);
	nWritten += fwrite(&b, sizeof(b), 1, fp);
	nWritten += fwrite(&c, sizeof(c), 1, fp);
	nWritten += fwrite(&alpha, sizeof(alpha), 1, fp);
	nWritten += fwrite(&beta, sizeof(beta), 1, fp);
	nWritten += fwrite(&tmp_m, sizeof(tmp_m), 1, fp);
	nWritten += fwrite(&tmp_k, sizeof(tmp_k), 1, fp);
	nWritten += fwrite(&tmp_n, sizeof(tmp_n), 1, fp);
	nWritten += fwrite(&Apitch, sizeof(Apitch), 1, fp);
	nWritten += fwrite(&Bpitch, sizeof(Bpitch), 1, fp);
	nWritten += fwrite(&Cpitch, sizeof(Cpitch), 1, fp);
	for (i = 0;i < tmp_m;i++)
	{
		nWritten += fwrite(a + i * Apitch, sizeof(double), tmp_k, fp);
	}
	for (i = 0;i < tmp_k;i++)
	{
		nWritten += fwrite(b + i * Bpitch, sizeof(double), tmp_n, fp);
	}
	fclose(fp);
	if (nWritten == 0) return(1);
	return(0);
}

void caldgemm::WaitForLASWP(size_t blockm)
{
	if (Config->LinpackSwapN != NULL)
	{
		int shown = false;
		size_t need = (blockm + 1) * Config->Height;
		if (need > gpu_m) need = gpu_m;
		if (ExecLinpack >= 2 && Config->AlternateLookahead <= matrix_n) need += Config->Width;
		//if (Config->Debug) fprintf(STD_OUT, "Checking LASWP / DTRSM... current: %lld need: %lld\n", (long long int) *Config->LinpackSwapN, (long long int) need);
		while (*Config->LinpackSwapN < need)
		{
			if (Config->Debug && shown == false)
			{
				fprintf(STD_OUT, "Waiting for LASWP / DTRSM... current: %lld need: %lld\n", (long long int) *Config->LinpackSwapN, (long long int) need);
				shown = true;
			}
#ifdef _WIN32
			if (Config->LASWPSleep) Sleep(Config->LASWPSleep / 1000);
#else
			if (Config->LASWPSleep) usleep(Config->LASWPSleep);
#endif
		}
	}
}

int caldgemm::CheckAlternateTilesRemainingSQ()
{
	return(0);
}

void caldgemm::CheckAlternateTilesRemaining(size_t m)
{
	if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && AlternateLookaheadTilesRemaining)
	{
		//if (Config->Debug) fprintf(STD_OUT, "Checking Alternate Tiles: m = %lld - Remaining = %d\n", (long long int) m, (int) AlternateLookaheadTilesRemaining);
		if ((int) m < AlternateLookaheadBlocksM)
		{
			pthread_mutex_lock(&tilesRemainingMutex);
			if (--AlternateLookaheadTilesRemaining == 0)
			{
				if (Config->Debug) fprintf(STD_OUT, "GPU done with initial part, factorization may start\n");
				alternateLookaheadMutex.Unlock();
			}
			pthread_mutex_unlock(&tilesRemainingMutex);
		}
	}
}

int caldgemm::Preallocate()
{
	for (int l = 0;l < nDevices;l++)
	{
		buffer_pointers_A[l] = new int[Config->PreallocData];
		buffer_pointers_B[l] = new int[Config->PreallocData];
		memset(buffer_pointers_A[l], 0, Config->PreallocData * sizeof(int));
		memset(buffer_pointers_B[l], 0, Config->PreallocData * sizeof(int));
	}
	tileDistribution = new int[Config->PreallocData * Config->PreallocData];
	memset(tileDistribution, 0, Config->PreallocData * Config->PreallocData * sizeof(int));
	return(0);
}

int caldgemm::PreallocateFree()
{
	for (int l = 0;l < nDevices;l++)
	{
		delete[] buffer_pointers_A[l];
		delete[] buffer_pointers_B[l];
	}
	delete[] tileDistribution;
	return(0);
}

void caldgemm::SetNumberDevices(int n)
{
	nDevices = n;
	if (nDevices <= 0) nDevices = 1;
	if (nDevices > nDevicesInitialized) nDevices = nDevicesInitialized;
}

int caldgemm::RunAsyncSingleTileDGEMM(const double* A, const double* B, double* C, double alpha, double beta, size_t m, size_t k, size_t n, size_t Apitch, size_t Bpitch, size_t Cpitch, bool orderColMajor, bool TransA, bool TransB)
{
	fprintf(STD_OUT, "Async Queue not supported by backend\n");
	return(1);
}

int caldgemm::RunAsyncSingleTileDTRSM(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const size_t M, const size_t N, const double alpha, const double *A, const size_t lda, double *B, const size_t ldb)
{
	fprintf(STD_OUT, "Async Queue not supported by backend\n");
	return(1);
}

int caldgemm::RunCALDGEMMMain(int parallelDevice)
{
	const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
	const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;
	const size_t nBlocks = mb * nb;

	//Check for double == 1.0 is unsafe and causes compiler warning
	const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double

#if defined(CALDGEMM_44) && !defined(CALDGEMM_USE_MEMEXPORT)
	const unsigned long long int double_minus_one = 0xBFF0000000000000;
	const int kernel_num = Config->ForceKernelVariant != -1 ? Config->ForceKernelVariant : (((Config->Width == BufferWidth && reinterpret_cast<unsigned long long int &>(reinterpret_cast<char &>(Beta)) == double_one && reinterpret_cast<unsigned long long int &>(reinterpret_cast<char &>(Alpha)) == double_minus_one) ? 2 : (reinterpret_cast<unsigned long long int &>(reinterpret_cast<char &>(Alpha)) == double_one)));
#else
	const int kernel_num = Config->ForceKernelVariant != -1 ? Config->ForceKernelVariant : ((reinterpret_cast<unsigned long long int &>(Alpha) == double_one));
#endif
	if ((Config->Debug) && Config->UseGPU) fprintf(STD_OUT, "Using Kernel %d (alpha=0x%llX (%2.3f), width = %lld)\n", kernel_num, (reinterpret_cast<long long int &>(Alpha)), Alpha, (long long int) Config->Width);

	int oldj[max_devices];
	int j[max_devices];
	int iMergeThread[max_devices];
	size_t blockm = 0, blockn = 0;
	unsigned long long int lastk[max_devices];
	size_t nextk = 0;
	size_t next_device_k[max_devices];
	int ImprovedSchedPhase1 = Config->ImprovedScheduler;
	int forcePreparation[max_devices];

	int myUseDevice = 0;
	int myNDevices;
	int myDevices[max_devices] = {0};
	if (parallelDevice == -1)
	{
		myNDevices = nDevices;
		for (int i = 0;i < nDevices;i++) myDevices[i] = i;
	}
	else if (matrix_n >= Config->GroupParallelDMA)
	{
		myNDevices = 1;
		myDevices[0] = parallelDevice;
	}
	else
	{
		myNDevices = 0;
		for (int i = 0;i < nDevices;i++)
		{
			if (Config->AllocMapping[i] == Config->DMAMapping[parallelDevice]) myDevices[myNDevices++] = i;
		}
		if (myNDevices == 0) return(0);
	}
	int use_device = myDevices[myUseDevice];

	for (int tl = 0;tl < myNDevices;tl++)
	{
		int l = myDevices[tl];
		next_device_k[l] = (!Config->ImprovedScheduler || first_device_k[l] == -1) ? 0 : first_device_k[l];
		j[l] = 0;
		iMergeThread[l] = 0;
		lastk[l] = -1;
		forcePreparation[l] = 0;
		if (!Config->PreallocData)
		{
			buffer_pointers_A[l] = new int[mb];
			buffer_pointers_B[l] = new int[nb];
		}
		
		for (size_t ll = 0;ll < mb;ll++) buffer_pointers_A[l][ll] = -1;
		for (size_t ll = 0;ll < nb;ll++) buffer_pointers_B[l][ll] = -1;
	}

	bool cpu_k_barrier_hit = false;
	if (gpu_n && gpu_m)
	{
		int currentPinning = Config->PinMainThread;
#ifdef CALDGEMM_LOOP_DETECTION
		int loop_detect = -1, loop_detect2 = -1;
#endif
		for (size_t k = 0;k < nBlocks + 2 * myNDevices;k++)
		{
restartkloop:
			//fprintf(STD_OUT, "!!!!! k %lld nd k %lld nextk %lld\n", (long long int) k, (long long int) next_device_k[use_device], (long long int) nextk);
			if (Config->ImprovedScheduler && !ImprovedSchedPhase1 && tileDistribution[next_device_k[use_device]] < 0) next_device_k[use_device] = 0;
			if (next_device_k[use_device] != 0) k = next_device_k[use_device];
			else if (nextk && nextk >= k) k = nextk + 1;
			if (next_device_k[use_device] >= nBlocks) next_device_k[use_device] = 0;
			if (k > nextk) nextk = k;

			if (k < nBlocks)
			{
				if (ImprovedSchedPhase1)
				{
					while (k < nBlocks && tileDistribution[k] != use_device)
					{
						if (Config->Debug) fprintf(STD_OUT, "Skipping tile %lld (m=%lld n=%lld) for device %d, will be processed by device %d\n", (long long int) k, (long long int) blockm, (long long int) blockn, use_device, tileDistribution[k]);
						k++;
					}
					if (k == nBlocks && parallelDevice == -1 && (Config->DynamicSched || (signed) nBlocks < 2 * nDevices)) goto endimprovedphase;
					if (k >= nBlocks)
					{
						next_device_k[use_device] = 0;
						if(!((obuffercount > 1) ? ((signed) lastk[use_device] != -1) : (k < nBlocks))) break;
					}
				}
#ifdef CALDGEMM_LOOP_DETECTION
				if (loop_detect2 == (signed) k)
				{
					fprintf(STD_OUT, "SCHEDULING ERROR A: Loop Detected, device = %d, k = %lld, next_device_k = %lld, nextk = %lld, ImprovedSched = %d, Phase1 = %d\n", use_device, (long long int) k, (long long int) next_device_k[use_device], (long long int) nextk, (int) Config->ImprovedScheduler, (int) ImprovedSchedPhase1);
					exit(1);
				}
				loop_detect2 = k;
#endif
			}
			
			if (k < nBlocks)
			{
				if (Config->ImprovedScheduler)
				{
					if (k >= nBlocks || tileDistribution[k] < 0)
					{
						if (Config->Debug)
						{
							DGEMM_getblocks(k, blockm, blockn);
							fprintf(STD_OUT, "Tile %lld (m=%lld n=%lld) already processed, skipping\n", (long long int) k, (long long int) blockm, (long long int) blockn);
						}
#ifdef CALDGEMM_LOOP_DETECTION
						if (loop_detect == (signed) k)
						{
							fprintf(STD_OUT, "SCHEDULING ERROR B: Loop Detected, k = %lld, next_device_k = %lld, nextk = %lld, ImprovedSched = %d, Phase1 = %d\n", (long long int) k, (long long int) next_device_k[use_device], (long long int) nextk, (int) Config->ImprovedScheduler, (int) ImprovedSchedPhase1);
							exit(1);
						}
						loop_detect = k;
#endif
						next_device_k[use_device] = 0;
						continue;
					}
				}
#ifdef CALDGEMM_LOOP_DETECTION
				loop_detect = loop_detect2 = -1;
#endif
				DGEMM_getblocks(k, blockm, blockn);

				if (cParam.dynamic_run)
				{
					if (DGEMM_favor_m)
					{
						if (blockm * Config->Height >= gpu_m - cParam.dynamic_run && blockn * Config->Height >= gpu_n - cParam.dynamic_size)
						{
							if (Config->Debug) fprintf(STD_OUT, "GPU skipping k = %lld (m=%lld n=%lld) (Dynamic Run 2nd Phase)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
							next_device_k[use_device] = 0;
							continue;
						}
					}
					else
					{
						if (blockn * Config->Height >= gpu_n - cParam.dynamic_run && blockm * Config->Height >= gpu_m - cParam.dynamic_size)
						{
							if (Config->Debug) fprintf(STD_OUT, "GPU skipping k = %lld (m=%lld n=%lld)(Dynamic Run 2nd Phase)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
							next_device_k[use_device] = 0;
							continue;
						}
					}
				}

				if (Config->MultiThread) pthread_mutex_lock(&scheduleMutex);
				if ((signed) k < cpu_k_barrier)
				{
					if ((signed int) k > (signed int) gpu_k_barrier)
					{
						gpu_k_barrier = k;
					}
				}
				else
				{
					if (Config->Debug) fprintf(STD_OUT, "gpu_k %lld (m=%lld n=%lld) reached cpu_k_barrier %lld, skipping remaining k (Dynamic Run 3rd Phase)\n", (long long int) k, (long long int) blockm, (long long int) blockn, (long long int) cpu_k_barrier);

					k = nBlocks;
					if (nextk < nBlocks) nextk = nBlocks;
					next_device_k[use_device] = 0;
					cpu_k_barrier_hit = true;
				}
				if (Config->MultiThread) pthread_mutex_unlock(&scheduleMutex);
			}

			if (ImprovedSchedPhase1 && k >= nBlocks && parallelDevice == -1 && (Config->DynamicSched || (signed) nBlocks < 2 * nDevices))
			{
endimprovedphase:
				if (Config->Debug) fprintf(STD_OUT, "First improved scheduling phase ended\n");
				ImprovedSchedPhase1 = 0;
				k = nextk = 0;
				for (int l = 0;l < nDevices;l++)
				{
					next_device_k[l] = 0;
					forcePreparation[l] = 1;
				}
				goto restartkloop;
			}

			if (Config->RepinMainThreadAlways && currentPinning != Config->AllocMapping[use_device])
			{
				sched_setaffinity_set_core(Config->AllocMapping[use_device] + Config->CPUCoreOffset);
				if (Config->Debug) fprintf(STD_OUT, "Repinning to %d\n", Config->AllocMapping[use_device]);
				currentPinning = Config->AllocMapping[use_device];
			}

			if (k < nBlocks)
			{
				if (Config->Debug) fprintf(STD_OUT, "Iteration k = %lld, m = %lld, n = %lld (device %d obuffer %d)\n", (long long int) k, (long long int) blockm, (long long int) blockn, use_device, j[use_device]);

				if (Config->MultiThreadDivide && parallelDevice == -1 && Config->GPUMapping[use_device] != Config->PinMainThread && UseInputPthreads() && DGEMMTasks[use_device].thread_running)
				{
					DGEMMTasks[use_device].thread_running = 0;
					if (Config->Debug) fprintf(STD_OUT, "Waiting for divide thread for device %d (k=%lld lastk = %lld j=%d)\n", use_device, (long long int) k, lastk[use_device], oldj[use_device]);

					int tmpval = DGEMMTasks[use_device].mutex_finished.Trylock();
					if (tmpval == EBUSY)
					{
						int tmp_device = *(DGEMMTasks[use_device].next_device);
						if (tmp_device != use_device && DGEMMTasks[tmp_device].thread_running == 0)
						{
							if (Config->Debug) fprintf(STD_OUT, "Divide thread waiting for wrong device, skipping device %d\n", tmp_device);
							DGEMMTasks[tmp_device].skip_device_to = use_device;
							DGEMMTasks[tmp_device].mutex_start.Unlock();
						}
						DGEMMTasks[use_device].mutex_finished.Lock();
					}
					else if (tmpval) fprintf(STD_OUT, "ERROR locking mutex_finished: %s - %d\n", __FILE__, __LINE__);

					if (Config->Debug) fprintf(STD_OUT, "Main thread: Divide thread for device %d finished\n", use_device);
				}

				DGEMMPrepareAndExecuteTask& Task = DGEMMTasks[use_device];
				Task.PrepareTasks[0].j = Task.PrepareTasks[1].j = -1;
				Task.kernel_num = kernel_num;
				Task.k = k;
				Task.j = j[use_device];

				if (next_device_k[use_device] == 0 || (signed) lastk[use_device] == -1 || obuffercount == 1 || Config->AsyncDMA == false || forcePreparation[use_device])
				{
					Task.PrepareTasks[0].k = k;
					Task.PrepareTasks[0].j = j[use_device];
					if (Config->ImprovedScheduler && !ImprovedSchedPhase1)
					{
						if ((size_t) buffersMajor[use_device] != (DGEMM_favor_m ? blockm : blockn))
						{
							if (Config->Debug) fprintf(STD_OUT, "Resetting favored directions buffers for device %d\n", use_device);
							buffersMajor[use_device] = -1;
						}
					}
					forcePreparation[use_device] = 0;
				}
				if (obuffercount > 1 && (signed) lastk[use_device] != -1 && Config->AsyncDMA && k + (myNDevices - myUseDevice - 1) % myNDevices + 1 < nBlocks && cpu_k_barrier_hit == false)
				{
					if (ImprovedSchedPhase1) nextk = k + 1;
					else nextk++;
					size_t nextblockm, nextblockn;
					DGEMM_getblocks(nextk, nextblockm, nextblockn);
					if (cParam.dynamic_run || Config->ImprovedScheduler)
					{
						while ( nextk < nBlocks && (
							(cParam.dynamic_run && (DGEMM_favor_m ? (nextblockm * Config->Height >= gpu_m - cParam.dynamic_run && nextblockn * Config->Height >= gpu_n - cParam.dynamic_size) :
							(nextblockn * Config->Height >= gpu_n - cParam.dynamic_run && nextblockm * Config->Height >= gpu_m - cParam.dynamic_size))) ||
							(Config->ImprovedScheduler && tileDistribution[nextk] < 0) ||
							(ImprovedSchedPhase1 && tileDistribution[nextk] != use_device)
							)
							)
						{
							nextk++;
							DGEMM_getblocks(nextk, nextblockm, nextblockn);
						}
					}
					if ((signed) nextk < cpu_k_barrier)
					{
						Task.PrepareTasks[1].k = nextk;
						Task.PrepareTasks[1].j = (j[use_device] + 1) % obuffercount;
					}
					next_device_k[use_device] = nextk;
				}
				else
				{
					if (ImprovedSchedPhase1)
					{
						next_device_k[use_device] = k + 1;
						forcePreparation[use_device] = 1;
					}
					else
					{
						next_device_k[use_device] = 0;
					}
				}

				if (Config->ImprovedScheduler) tileDistribution[k] = -1;

				if (Config->MultiThreadDivide && parallelDevice == -1 && Config->GPUMapping[use_device] != Config->PinMainThread && UseInputPthreads() && cpu_k_barrier_hit == false)
				{
					if (Config->Debug) fprintf(STD_OUT, "Starting PrepareAndExecute task on divide thread for device %d (k = %lld)\n", use_device, (long long int) k);
					DGEMMTasks[use_device].mutex_start.Unlock();
					DGEMMTasks[use_device].thread_running = 1;
				}
				else
				{
#ifdef CALDGEMM_DIVIDE_STATIC_BUFFER
					double* __restrict__ tmpBuffer = divide_tmpBuffer;
#endif
					if (DGEMMPrepareAndExecute(Task CALDGEMM_DIVBUFB)) return(1);
				}
			}
			if (obuffercount == 1)
			{
				oldj[use_device] = j[use_device];
				lastk[use_device] = k;
			}
			if ((obuffercount > 1) ? ((signed) lastk[use_device] != -1) : (k < nBlocks))
			{
				if (nBlocks <= k && (signed) lastk[use_device] < cpu_k_barrier && Config->MultiThreadDivide && parallelDevice == -1 && Config->GPUMapping[use_device] != Config->PinMainThread && UseInputPthreads() && DGEMMTasks[use_device].thread_running)
				{
					DGEMMTasks[use_device].thread_running = 0;
					if (Config->Debug) fprintf(STD_OUT, "Waiting for divide thread for device %d (late phase, k=%lld lastk = %lld j=%d)\n", use_device, (long long int) k, lastk[use_device], oldj[use_device]);
					int tmpval = DGEMMTasks[use_device].mutex_finished.Trylock();
					if (tmpval == EBUSY)
					{
						int tmp_device = *(DGEMMTasks[use_device].next_device);
						if (tmp_device != use_device && DGEMMTasks[tmp_device].thread_running == 0)
						{
							if (Config->Debug) fprintf(STD_OUT, "Divide thread waiting for wrong device (late phase), skipping device %d\n", tmp_device);
							DGEMMTasks[tmp_device].skip_device_to = use_device;
							DGEMMTasks[tmp_device].mutex_start.Unlock();
						}
						DGEMMTasks[use_device].mutex_finished.Lock();
					}
					else if (tmpval) fprintf(STD_OUT, "ERROR trylocking mutex_finished: %s - %d\n", __FILE__, __LINE__);
				}
				size_t lastm, lastn;
				DGEMM_getblocks(lastk[use_device], lastm, lastn);
				int must_lock = 0;
				if (Config->ThreadSaveDriver != 1)
				{
					if (parallelDevice >= 0)
					{
							must_lock = 1;
					}
					else if (Config->MultiThreadDivide) for (int ii = 0;ii < nDevices;ii++)
					{
						if (Config->GPUMapping[ii] != Config->PinMainThread)
						{
							must_lock = 1;
							break;
						}
					}
				}

				if ((signed long int) lastk[use_device] != -1 && lastk[use_device] < nBlocks)
				{
					while (DGEMMPrepareTaskEventReady[use_device][oldj[use_device]] == false);
					DGEMMPrepareTaskEventReady[use_device][oldj[use_device]] = false;
					if (WaitForEvent(oldj[use_device], use_device, must_lock)) return(1);
					if (Config->Debug && Config->GPU_C == 0) fprintf(STD_OUT, "Processing Output (Iteration %lld) for device %d tile %lld (m = %lld, n = %lld)\n", (long long int) k, use_device, (long long int) lastk[use_device], (long long int) lastm, (long long int) lastn);
					if (Config->UseDMAFetchQueue >= matrix_n && Config->DstMemory == 'g')
					{
						if (CheckDMAQueue(use_device, oldj[use_device])) return(1);
					}
					else if (Config->ImplicitDriverSync == 0 && Config->DstMemory == 'g')
					{
						if (FetchResult(use_device, oldj[use_device], lastm, lastn, Config->MultiThread && UseMutexPerDevice())) {fprintf(STD_OUT, "Error copying from GPU\n");return(1);}
						if (WaitForEvent(oldj[use_device], use_device)) return(1);
					}
				}
				if (Config->VerboseTiming) Timers.CounterMerge.Start();

				if (k == nBlocks + 2 * myNDevices - 1 || Config->MultiThread == false || UseOutputPthreads() == 0)
				{
					if (lastk[use_device] < nBlocks)
					{
						if (Config->Debug && Config->GPU_C == 0) fprintf(STD_OUT, "\tMerging buffer (device %d, obuffer %d, k = %lld, main thread)\n", use_device, oldj[use_device], (long long int) lastk[use_device]);
						if (RunMergeBuffers(C + lastn * Config->Height + lastm * C_pitch * Config->Height, use_device, oldj[use_device], (lastn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height, (lastm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height, BufferHeight, BufferHeight, C_pitch)) {fprintf(STD_OUT, "Error merging\n"); return(1);}
						if (!Config->SimpleGPUQueuing) CheckAlternateTilesRemaining(lastm);
						if (Config->Debug) fprintf(STD_OUT, "Main thread unlocking obuffer mutex device %d obuffer %d\n", use_device, oldj[use_device]);
						if (Config->MultiThread && UseOutputPthreads()) obufferMutex[use_device][oldj[use_device]].Unlock();
					}
					if (Config->MultiThread && UseOutputPthreads())
					{
						for (int l = 0;l < obuffercount;l++)
						{
							for (int tll = 0;tll < myNDevices;tll++)
							{
								int ll = myDevices[tll];
								if ((ll != use_device || l != oldj[ll]) && (signed) lastk[ll] != -1)
								{
									if (Config->Debug) fprintf(STD_OUT, "Waiting to finish merge process for device %d obuffer %d\n", ll, l);
									obufferMutex[ll][l].Lock();
									obufferMutex[ll][l].Unlock();
								}
							}
						}
					}
				}
				else if (lastk[use_device] < nBlocks)
				{
					if (Config->AsyncTiming)
					{
						Timers.ATime.Reset();
						Timers.ATime.Start();
					}
					mParam[use_device][iMergeThread[use_device]].mergeThreadMutex[1].Lock();

					if (Config->AsyncTiming)
					{
						Timers.ATime.Stop();
						if ((!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001) || Config->Debug) fprintf(STD_OUT, "\t\tWARNING: Wait Time for merge thread: %1.5f\n", Timers.ATime.GetElapsedTime());
					}
					if (Config->Debug) fprintf(STD_OUT, "\t\tUnlocking outputthread mutex %d to process device %d obuffer %d\n", iMergeThread[use_device], use_device, oldj[use_device]);
					mParam[use_device][iMergeThread[use_device]].nContext = oldj[use_device];
					mParam[use_device][iMergeThread[use_device]].dst = C + (lastn * Config->Height + lastm * C_pitch * Config->Height);
					mParam[use_device][iMergeThread[use_device]].k = lastk[use_device];
					mParam[use_device][iMergeThread[use_device]].mergeThreadMutex[0].Unlock();
					iMergeThread[use_device] = (iMergeThread[use_device] + 1) % outputthreads;
				}

				if (Config->VerboseTiming) Timers.CounterMerge.Stop();
			}
			oldj[use_device] = j[use_device];
			j[use_device] = (j[use_device] + 1) % obuffercount;
			lastk[use_device] = k;
			if (Config->MultiThread)
			{
				myUseDevice = (myUseDevice + 1) % myNDevices;
				use_device = myDevices[myUseDevice];
			}
		}
		if (currentPinning != Config->PinMainThread)
		{
			sched_setaffinity(0, sizeof(gpumask), &gpumask);
		}
	}
	if (Config->MultiThreadDivide && parallelDevice == -1 && UseInputPthreads())
	{
		for (int l = 0;l < divideThreads;l++)
		{
			if (dParam[l].curDevice != dParam[l].firstDevice)
			{
				dParam[l].reset = 1;
				DGEMMTasks[dParam[l].curDevice].mutex_start.Unlock();
				DGEMMTasks[dParam[l].firstDevice].mutex_finished.Lock();
			}
		}
	}

	if (Config->PreallocData == 0)
	{
		for (int tl = 0;tl < myNDevices;tl++)
		{
			int l = myDevices[tl];
			delete[] buffer_pointers_A[l];
			delete[] buffer_pointers_B[l];
		}
	}
	return(0);
}

int caldgemm::RunCALDGEMM(double* a, double* b, double* c, double alpha, double beta, size_t tmp_m, size_t tmp_k, size_t tmp_n, size_t Apitch, size_t Bpitch, size_t Cpitch, bool orderColMajor, bool TransA, bool TransB, int ExecuteLinpackCallbacks, int pipelined)
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "Caldgemm not initialized, aborting DGEMM run\n");
		return(1);
	}

#ifdef DEBUG_MSG_TIMED	
	if (Config->Debug) printelapsedtime("Resetting Timer\n");
#endif

	if (tmp_m == 0 || tmp_k == 0 || tmp_n == 0)
	{
		if (Config->LinpackSwapN != NULL)
		{
			HPL_CALDGEMM_gpu_height = 0;
			Config->linpack_swap_function();
			Config->LinpackSwapN = 0;
		}

		if (ExecuteLinpackCallbacks)
		{
			Timers.LinpackTimer1.Start();
			Config->linpack_factorize_function();
			Timers.LinpackTimer1.Stop();
			if (Config->LinpackNodes > 1)
			{
				Timers.LinpackTimer2.Start();
				Config->linpack_broadcast_function();
				Timers.LinpackTimer2.Stop();
			}
		}
		return(0);		//Do Nothing
	}

	bool forceCPU = false;
	bool forceReinit = false;
	double GPURatio;
	int old_outputthreads = outputthreads;

	size_t MaxGpuM, MaxGpuN; //Maximal values of m and n that can be given to GPU, This is below m,n if ExecuteLinpackCallback = true

	A = a;
	B = b;
	C = c;
	Alpha = alpha;
	Beta = beta;
	matrix_m = tmp_m;
	matrix_n = tmp_n;
	if ((signed) tmp_k != -1) Config->Width = tmp_k;
	
	A_pitch = ((signed) Apitch != -1) ? Apitch : Config->Width;
	B_pitch = ((signed) Bpitch != -1) ? Bpitch : matrix_n;
	C_pitch = ((signed) Cpitch != -1) ? Cpitch : matrix_n;
	ResetTimers();

	if (orderColMajor)
	{
		double* tmpd;
		size_t tmpi;
		bool tmpt;
		tmpd = A; A = B; B = tmpd;
		tmpi = matrix_m; matrix_m = matrix_n; matrix_n = tmpi;
		tmpi = A_pitch; A_pitch = B_pitch; B_pitch = tmpi;
		tmpt = TransA;TransA = TransB;TransB = tmpt;
	}

	if (!Config->Quiet) fprintf(STD_OUT, "Starting DGEMM Run m=%lld k=%lld n=%lld Alpha=%f Beta=%f LDA=0x%lx LDB=0x%lx LDC=0x%lx At=%d Bt=%d ColMajor=%d (A=0x%llx, B=0x%llx, C=0x%llx, (C-A=%lld, (C-B)/w=%lld), Linpack=%d)\n", (long long int) matrix_m, (long long int) Config->Width, (long long int) matrix_n, Alpha, Beta, A_pitch, B_pitch, C_pitch, (int) (TransA), (int) (TransB), (int) (orderColMajor), (long long int) A, (long long int) B, (long long int) C, (long long int) ((size_t) C - (size_t) A) / sizeof(double), (long long int) ((size_t) C - (size_t) B) / sizeof(double) / Config->Width, (int) ExecuteLinpackCallbacks);

	TransposeA = TransA;
	TransposeB = TransB;    
	ExecLinpack = ExecuteLinpackCallbacks;
	pipelinedRun = pipelined;
	orig_m = matrix_m;
	orig_n = matrix_n;
	orig_a = A;
	orig_b = B;
	orig_c = C;

	if (Config->Verify)
	{
		if (Config->PipelinedOperation)
		{
				fprintf(STD_OUT, "PipelinedOperation cannot be used in combination with Verify!\n");
				return(1);
		}
		D = new double[(size_t) matrix_m * (size_t) C_pitch];
		if (D == NULL)
		{
			fprintf(STD_OUT, "Memory allocation error\n");
			return(1);
		}
		memcpy(D, C, matrix_m * C_pitch * sizeof(double));
	}

	if (Config->DumpMatrix) DumpMatrix(A, B, C, Alpha, Beta, matrix_m, Config->Width, matrix_n, A_pitch, B_pitch, C_pitch);

	Timers.System.Start();
	
	if (ExecLinpack >= 2 && Config->AlternateLookahead <= matrix_n)
	{
		if (matrix_m < Config->Width)
		{
			MaxGpuM = 0;
		}
		else
		{
			MaxGpuM = matrix_m - Config->Width;
		}
	}
	else
	{
		MaxGpuM = matrix_m;
	}
	MaxGpuN = matrix_n;

#ifndef TESTMODE    
	//Check if the GPU can/shall process the required dgemm task
	if (Config->Iterations > 1 || !Config->UseCPU);
	else if (Config->Width % 8 || Config->Width < 256) forceCPU = true;
	else if (MaxGpuM < Config->Height / 2 || MaxGpuN < Config->Height / 2) forceCPU = true;
#ifdef _WIN32
	else if (Alpha == 0.) forceCPU = true;
#else
	else if (__fpclassify(Alpha) == FP_ZERO) forceCPU = true;
#endif
	else if (((size_t) A) & (vcpysize - 1) || ((size_t) B) & (vcpysize - 1) || ((size_t) C) & (vcpysize - 1) ||
		A_pitch & (vcpysize / sizeof(double) - 1) || B_pitch & (vcpysize / sizeof(double) - 1) || C_pitch & (vcpysize / sizeof(double) - 1))
	{
		fprintf(STD_OUT, "Input addresses not aligned correctly: A 0x%llX B 0x%llX C 0x%llX Pitch 0x%llX 0x%llX 0x%llX\n", (long long int) A, (long long int) B, (long long int) C, (long long int) A_pitch, (long long int) B_pitch, (long long int) C_pitch);
		forceCPU = true;
	}
#endif

	if (Config->AutoHeight)
	{
#ifdef CALDGEMM_CUSTOM_AUTO_HEIGHT
#include CALDGEMM_CUSTOM_AUTO_HEIGHT
#else
		if (CaldgemmCustomAutoHeight(MaxGpuM, MaxGpuN, nDevices) == 0)
		{
			if (ExecLinpack >= 2 && !Config->SmallTiles)
			{
				if (MaxGpuM < 1024 || MaxGpuN < 1024)
				{
					Config->Height = 512;
				}
				else if (MaxGpuM < 2048 || MaxGpuN < 2048 || (MaxGpuM * MaxGpuN < 13 * 14 * 1024 * 1024 && mymax(MaxGpuN, MaxGpuM) % 2048 >= 1024) || (MaxGpuM * MaxGpuN < 16 * 1024 * 1024))
				{
					Config->Height = 1024;
				}
				else if (MaxGpuM < 3072 || MaxGpuN < 3072 || (MaxGpuM * MaxGpuN < 20 * 21 * 1024 * 1024 && mymax(MaxGpuN, MaxGpuM) % 3072 >= 2048) || (MaxGpuM * MaxGpuN < 120 * 1024 * 1024))
				{
					Config->Height = 2048;
				}
				else if (MaxGpuM < 4096 || MaxGpuN < 4096 || MaxGpuM * MaxGpuN < 27 * 28 * 1024 * 1024)
				{
					Config->Height = 3072;
				}
				else
				{
					Config->Height = 4096;
				}
			}
			else
			{
				if (MaxGpuM < 1024 || MaxGpuN < 1024)
				{
					Config->Height = 512;
				}
				else if (MaxGpuM < 2048 || MaxGpuN < 2048 || MaxGpuM * MaxGpuN < (size_t) nDevices * 16 * 1024 * 1024)
				{
					Config->Height = 1024;
				}
				else if (MaxGpuM < 3072 || MaxGpuN < 3072 || MaxGpuM * MaxGpuN < (size_t) nDevices * 120 * 1024 * 1024)
				{
					Config->Height = 2048;
				}
				else if (MaxGpuM < 4096 || MaxGpuN < 4096 || MaxGpuM * MaxGpuN < (size_t) nDevices * 40 * 40 * 1024 * 1024)
				{
					Config->Height = 3072;
				}
				else
				{
					Config->Height = 4096;
				}
				while (Config->SlowCPU && !Config->SmallTiles && Config->Height > 1024 && (MaxGpuM % Config->Height > 1024 || MaxGpuN % Config->Height > 1024)) Config->Height -= 1024;
			}
		}
#endif
		if (Config->Height > BufferHeight) Config->Height = BufferHeight;
		if (Config->Height % KernelSettings.min_tile_size)
		{
			Config->Height = Config->Height > (size_t) KernelSettings.min_tile_size ? (Config->Height - Config->Height % KernelSettings.min_tile_size) : KernelSettings.min_tile_size;
		}
		if (Config->Debug)  fprintf(STD_OUT, "Using Height %lld of max %lld\n", (long long int) Config->Height, (long long int) BufferHeight);
	}
	HPL_CALDGEMM_gpu_height = Config->Height;

	if (Config->UseGPU && (Config->Width > BufferWidth || Config->Height > BufferHeight)) forceReinit = true;
	if (Config->UseCPU)
	{
		if (Config->UseGPU == false || (forceReinit && (long long int) MaxGpuM * (long long int) MaxGpuN * (long long int) Config->Width < (long long int) 24 * 1024 * 1024 * 1024) || (Config->Width < 1024 && Config->Height < 1024) || (ExecLinpack && matrix_m < Config->Width)) forceCPU = true;
	}

	AlternateLookaheadTilesFull = 0;

	if (forceCPU)
	{
		if (Config->Debug) fprintf(STD_OUT, "Running CPU only DGEMM\n");
		if (Config->ShowThreadPinning) printThreadPinning();

		if (Config->LinpackSwapN != NULL)
		{
			HPL_CALDGEMM_gpu_height = 0;
			Config->linpack_swap_function();
		}
		if (ExecLinpack)
		{
			size_t usewidth = Config->Width > matrix_m ? matrix_m : Config->Width;
			Timers.CPUTimer.Start();
			cblas_dgemm(CblasRowMajor, TransposeA ? CblasTrans : CblasNoTrans, TransposeB ? CblasTrans : CblasNoTrans, usewidth, matrix_n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
			Timers.CPUTimer.Stop();

			if (Config->Debug) fprintf(STD_OUT, "DGEMM was running on CPU only, executing linpack callback functions\n");
			Timers.LinpackTimer1.Start();
			Config->linpack_factorize_function();
			Timers.LinpackTimer1.Stop();
			if (Config->LinpackNodes > 1)
			{
				Timers.LinpackTimer2.Start();
				Config->linpack_broadcast_function();
				Timers.LinpackTimer2.Stop();
			}

			matrix_m -= usewidth;
			A += usewidth * (TransposeA ? 1 : A_pitch);
			C += usewidth * (C_pitch);
		}
		Timers.CPUTimer.Start();

		goto_set_num_threads(conf_numprocs);
		if (matrix_m) cblas_dgemm(CblasRowMajor, TransposeA ? CblasTrans : CblasNoTrans, TransposeB ? CblasTrans : CblasNoTrans, matrix_m, matrix_n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
		Timers.CPUTimer.Stop();
		CPUOnlyRun = true;
	}
	else
	{
		CPUOnlyRun = false;

		if (ExecLinpack)
		{
			outputthreads = mymin(CALDGEMM_OUTPUT_THREADS_SLOW, outputthreads + CALDGEMM_EXTRA_OUTPUT_THREADS_LINPACK);
		}

		if (Config->SpawnGPUThread == -2)
		{
			if (Config->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&gpumask));
			sched_setaffinity(0, sizeof(cpu_set_t), &gpumask);
		}

		if (forceReinit)
		{
			fprintf(STD_OUT, "WARNING: Reinit for increased buffer width / height\n");
			fprintf(STD_OUT, "Reinit not yet implemented correctly, exiting");
			exit(1);
			if (ReinitDevices()) return(1);
		}

		InitConstantData(alpha);

		if (Config->SlowCPU || matrix_n < Config->MinimizeCPUPart || (Config->MinimizeCPUDuringFact && ExecLinpack >= 2) || Config->GPURatio >= 1.0)
		{
			GPURatio = 1.0;
		}
		else
		{
			if (Config->GPURatio <= -0.999) //Auto determination (code must be adapted for each CPU / GPU config)
			{
				//Optimal ratio found using combined runs
				if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 5000000000) GPURatio = 0.75;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 600000000) GPURatio = 0.74;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 500000000) GPURatio = 0.73;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 200000000) GPURatio = 0.73;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 100000000) GPURatio = 0.72;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 7000000) GPURatio = 0.70;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 5000000) GPURatio = 0.67;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 2500000) GPURatio = 0.60;
				else if ((long long int) MaxGpuM * (long long int) MaxGpuN > (long long int) 1000000) GPURatio = 0.55;
				else GPURatio = 0.50;
				if (Config->Width < 1024) GPURatio *= (double) Config->Width / (double) 1024;
				if (Config->Height < 1024) GPURatio *= (double) Config->Height / (double) 1024 * (double) Config->Height / (double) 1024;
		
				const int require_threads = outputthreads * nDevices + 1 + (ExecLinpack && Config->LinpackNodes > 1);
				const double CPUscale = (double) (conf_cpufreq * mymax(conf_numprocs - require_threads, 1)) / (double) (2100 * (24 - require_threads));
				const double GPUscale = (double) nDevices * conf_gpushaders * conf_gpufreq / (double) (850 * 20 * 64);
				if (Config->Debug) fprintf(STD_OUT, "GPU Curve Ration: %1.3f, CPUScale %1.3f, GPUScale %1.3f\n", GPURatio, CPUscale, GPUscale);
				GPURatio = GPUscale * GPURatio / (GPUscale * GPURatio + (1.0 - GPURatio) * CPUscale);
		
				if (Config->Debug) fprintf(STD_OUT, "GPURatio automatically set to %1.3f\n", GPURatio);
				if (GPURatio > 1.) GPURatio = 1.0;
				if ((matrix_n + 4) % 4096 < 8 && GPURatio > 0.5) GPURatio = 1. - 0.95 * (1. - GPURatio);
			}
			else
			{
				if (ExecLinpack > 1 && Config->GPURatioDuringFact > 0) GPURatio = Config->GPURatioDuringFact;
				else GPURatio = fabs(Config->GPURatio);
			}

			if (ExecLinpack && (Config->GPURatio < 0 || GPURatio < 0.99) && !Config->SlowCPU)
			{
				if (Config->GPURatio <= -0.99) //Auto determination
				{
					if (ExecLinpack > 1) GPURatio = 1.0 - (1.0 - GPURatio) * 0.80 * Config->Width / 1024;
					else GPURatio = 1.0 - (1.0 - GPURatio) * 0.90;
					if (GPURatio > 1.0) GPURatio = 1.0;
				}
				if (linpack_last_mn[ExecLinpack] > 0 && (((double) MaxGpuM * (double) MaxGpuN) - linpack_last_mn[ExecLinpack]) / linpack_last_mn[ExecLinpack] < 0.3 && linpackGPURatios[ExecLinpack] > 0.0001)
				{
					GPURatio = linpackGPURatios[ExecLinpack];
					if (Config->Debug||1) fprintf(STD_OUT, "Taking GPU Ratio from table, entry %d, val %2.3f\n", ExecLinpack, 100 * GPURatio);
				}
				else
				{
					linpackGPURatios[ExecLinpack] = GPURatio;
					if (Config->Debug||1) fprintf(STD_OUT, "Initializing ratio table entry %d with %2.3f\n", ExecLinpack, 100 * GPURatio);
				}
			}
			if (Config->GPURatioMax > 0 && GPURatio > Config->GPURatioMax) GPURatio = Config->GPURatioMax;;
			if (Config->GPURatio < 0 && Config->GPURatio > -0.99)
			{
				double threshold = (ExecLinpack > 1 && Config->GPURatioDuringFact > 0.) ? Config->GPURatioDuringFact : -Config->GPURatio;
				if (GPURatio < threshold) GPURatio = threshold;
			}
			//if (Config->AlternateLookahead > matrix_n) GPURatio = 1. - (1. - GPURatio) * 0.88;
		}

		gpu_ratio_used = GPURatio;

		if (ExecLinpack >= 2 && Config->AlternateLookahead <= matrix_n)
		{
			matrix_m -= Config->Width;
			A += Config->Width * (TransposeA ? 1 : A_pitch);
			C += Config->Width * (C_pitch);
			HPL_CALDGEMM_gpu_height += Config->Width;
		}

		cParam.dynamic_run = 0;
		cParam.dynamic_run2 = 0;
		cParam.borders_done = false;
		SmallTileHeight = (Config->SmallTiles == 1 ? KernelSettings.min_tile_size : Config->Height);
recalculate_ratio:
		if (Config->UseCPU == true && Config->UseGPU == true)
		{
			if ((DGEMM_split_m = ((Config->LinpackSwapN == NULL && (ExecLinpack == 0 || Config->AlternateLookahead <= matrix_n)) ? (matrix_m >= matrix_n) : 1)))
			{
				size_t virtualm = matrix_m + (matrix_n % SmallTileHeight) * matrix_m / matrix_n;
				if (ExecLinpack >= 2 && Config->AlternateLookahead <= matrix_n) virtualm += Config->Width * (Config->GPURatioLookaheadSizeMod + (float) matrix_m / matrix_n);
				gpu_m = GPURatio * (float) virtualm + (SmallTileHeight - 1);
				if (gpu_m > matrix_m)
				{
					if (Config->SmallTiles == 2 && SmallTileHeight > (size_t) KernelSettings.min_tile_size)
					{
						if (SmallTileHeight > 1024) SmallTileHeight = 1024;
						else SmallTileHeight = KernelSettings.min_tile_size;
						goto recalculate_ratio;
					}
					gpu_m = matrix_m;
				}
				gpu_m -= gpu_m % SmallTileHeight;
				cParam.cblas_size = matrix_m - gpu_m;
				gpu_n = matrix_n;
				gpu_n -= gpu_n % SmallTileHeight;
				if (Config->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld, Tilesize %lld\n", (long long int) gpu_m, (long long int) gpu_n, (long long int) matrix_m - gpu_m, (long long int) gpu_n, (long long int) SmallTileHeight);
			}
			else
			{
				size_t virtualn = matrix_n + (matrix_m % SmallTileHeight) * matrix_n / matrix_m;
				if (ExecLinpack >= 2 && Config->AlternateLookahead <= matrix_n) virtualn += Config->Width * (Config->GPURatioLookaheadSizeMod + (float) matrix_n / matrix_m);
				gpu_n = GPURatio * (float) virtualn + (SmallTileHeight - 1);
				if (gpu_n > matrix_n)
				{
					if (Config->SmallTiles == 2 && SmallTileHeight > (size_t) KernelSettings.min_tile_size)
					{
						if (SmallTileHeight > 1024) SmallTileHeight = 1024;
						else SmallTileHeight = KernelSettings.min_tile_size;
						goto recalculate_ratio;
					}
					gpu_n = matrix_n;
				}
				gpu_n -= gpu_n % SmallTileHeight;
				cParam.cblas_size = matrix_n - gpu_n;
				gpu_m = matrix_m;
				gpu_m -= gpu_m % SmallTileHeight;
				if (Config->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld, Tilesize %lld\n", (long long int) gpu_m, (long long int) gpu_n, (long long int) matrix_m, (long long int) matrix_n - gpu_n, (long long int) SmallTileHeight);
			}

			const size_t over_m = gpu_m % Config->Height, over_n = gpu_n % Config->Height;
			if (over_m < CALDGEMM_MIN_TILE_DIM2) gpu_m -= over_m;
			else
			{
#ifdef CALDGEMM_CUSTOM_HEIGHT_MOD
#define MOD_OVER over_m
#define MOD_GPU gpu_m
#include CALDGEMM_CUSTOM_HEIGHT_MOD
#undef MOD_OVER
#undef MOD_GPU
#else
			CaldgemmCustomModHeight(over_m, gpu_m);
#endif
			}
			if (over_n < CALDGEMM_MIN_TILE_DIM2) gpu_n -= over_n;
			else
			{
#ifdef CALDGEMM_CUSTOM_HEIGHT_MOD
#define MOD_OVER over_n
#define MOD_GPU gpu_n
#include CALDGEMM_CUSTOM_HEIGHT_MOD
#undef MOD_OVER
#undef MOD_GPU
#else
				CaldgemmCustomModHeight(over_n, gpu_n);
#endif
			}

			cParam.cblas_size = DGEMM_split_m ? (matrix_m - gpu_m) : (matrix_n - gpu_n);
		}
		else
		{
			if (warn_wrong_memory_allocation && (Config->GPU_C || Config->DstMemory == 'c'))
			{
				warn_wrong_memory_allocation = false; //Only warn once
				fprintf(STD_OUT, "WARNING, you are using GPU_C or '-o g' option, but apparently you did not use CALDGEMM memory allocation with gpu_accessible feature ('-_' is missing).\nYou must take care to allocate GPU accessible memory yourself, or this can lead to invalid memory accesses.\n");
			}
			
			DGEMM_split_m = 0;
			if (matrix_n % SmallTileHeight || matrix_m % SmallTileHeight)
			{
				fprintf(STD_OUT, "Invalid matrix size for GPU only (%lld %% %lld = %lld, %lld %% %lld = %lld)\n", (long long int) matrix_n, (long long int) SmallTileHeight, (long long int) matrix_n % SmallTileHeight, (long long int) matrix_m, (long long int) SmallTileHeight, (long long int) matrix_m % SmallTileHeight);
				return(1);
			}
			if (ExecLinpack)
			{
				fprintf(STD_OUT, "Linpack callbacks in CALDGEMM are only possible with UseCPU = true!\n");
				return(1);
			}
			gpu_n = matrix_n;
			gpu_m = matrix_m;
		}

		DGEMM_favor_m = (Config->LinpackSwapN == NULL && (ExecLinpack == 0 || Config->AlternateLookahead <= matrix_n)) ? (gpu_m >= gpu_n) : 1;

		if (!Config->Quiet) fprintf(STD_OUT, "Ratio %f - gpu_m %lld gpu_n %lld - Split %c Favor %c - Height %lld (/ %lld), Min Tiling %lld (%lld, %lld)\n", GPURatio, (long long int) gpu_m, (long long int) gpu_n, DGEMM_split_m ? 'm' : 'n', DGEMM_favor_m ? 'm' : 'n', (long long int) Config->Height, (long long int) BufferHeight, (long long int) SmallTileHeight, (long long int) (gpu_m % Config->Height), (long long int) (gpu_n % Config->Height));

		if (Config->ShowThreadPinning) printThreadPinning();

		const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
		const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;
		const size_t nBlocks = mb * nb;
		cParam.cpu_k = nBlocks;
		cpu_k_barrier = nBlocks;
		gpu_k_barrier = -1;
		if (Config->UseCPU)
		{
			if (!Config->MultiThread)
			{
				cblas_wrapper_a();
			}
			cParam.cblasMutex[1].Unlock();
		}
		else if (Config->LinpackSwapN != NULL)
		{
			HPL_CALDGEMM_gpu_height = 0;
			Config->linpack_swap_function();
		}

		if (Config->SpawnGPUThread != -2)
		{
			if (caldgemm_part_cpu()) return(1);
		}
		else
		{
			if (caldgemm_part_gpu()) return(1);
		}

		if (Config->UseCPU)
		{
			if (Config->Debug) fprintf(STD_OUT, "Waiting for CPU DGEMM to finish\n");
			cParam.cblasMutex[0].Lock();
			if (Config->MultiThread)
			{
				Timers.ATime.Stop();
				cpu_wait_time = Timers.ATime.GetElapsedTime();
				if (Config->DynamicSched && !Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() >= 0.15 && cParam.cblas_size > 0)
				{
					fprintf(STD_OUT, "WARNING: CPU synchronisation took %2.4f sec\n", Timers.ATime.GetElapsedTime());
				}
				else if (Config->Debug)
				{
					fprintf(STD_OUT, "CPU synchronisation took %2.4f sec\n", Timers.ATime.GetElapsedTime());
				}
			}
		}
	}
	if (Config->LinpackSwapN != NULL) *Config->LinpackSwapN = 0;
	outputthreads = old_outputthreads;

	Timers.System.Stop();
	if (Config->Debug) fprintf(STD_OUT, "DGEMM Run Complete\n");
	
	if (finishData->running)
	{
			if (!Config->Quiet) fprintf(STD_OUT, "Waiting for previous pipelined DGEMM iteration to finish\n");
			int retVal = FinishCALDGEMM();
			if (retVal) return(retVal);
	}
	
	finishData->matrix_m = matrix_m; finishData->matrix_n = matrix_n; finishData->SmallTileHeight = SmallTileHeight; finishData->orig_m = orig_m; finishData->orig_n = orig_n;
	finishData->gpu_ratio_used = gpu_ratio_used; finishData->cpu_wait_time = cpu_wait_time;
	finishData->ExecLinpack = ExecLinpack;
	finishData->CPUOnlyRun = CPUOnlyRun; finishData->DGEMM_split_m = DGEMM_split_m;
	
	finishData->System = Timers.System.GetElapsedTime(); finishData->CPUTimer = Timers.CPUTimer.GetElapsedTime(); finishData->GPUTimer = Timers.GPUTimer.GetElapsedTime(); finishData->TotalCPUTimer = Timers.TotalCPUTimer.GetElapsedTime();
	finishData->LinpackTimer1 = Timers.LinpackTimer1.GetElapsedTime(); finishData->LinpackTimer2 = Timers.LinpackTimer2.GetElapsedTime(); finishData->LinpackTimer3 = Timers.LinpackTimer3.GetElapsedTime(); finishData->BcastTimer = Timers.BcastTimer.GetElapsedTime();
	
	finishData->divideA = Timers.divideA; finishData->divideB = Timers.divideB; finishData->divideC = Timers.divideC;
	finishData->device_kernel = Timers.device_kernel;

	finishData->cblas_size = cParam.cblas_size;
	finishData->dynamic_run = cParam.dynamic_run;
	finishData->dynamic_size = cParam.dynamic_size;
	finishData->cpu_k = cParam.cpu_k;
	finishData->dynamic_run2 = cParam.dynamic_run2;
	finishData->MidMarkerPos = Config->PipelinedMidMarker;
	FinishDataFill();

	if (Config->PipelineDoubleBuffer) pipelineBuffer ^= 1;

	if (Config->PipelinedOperation && !CPUOnlyRun && pipelinedRun)
	{
		finishData->running = true;
		return(0);
	}
	else
	{
		finishData->running = false;
		return(FinishCALDGEMM(true));
	}
}

int caldgemm::FinishDataInit()
{
	finishData = new finishStruct;
	return(finishData == NULL);
}

void caldgemm::FinishDataFill(){}

int caldgemm::FinishCALDGEMM(bool force)
{
	if (!(force || finishData->running)) return(0);
	if (Config->PipelinedOperation)
	{
		int retVal = RunCALDGEMM_Finish();
		finishData->running = false;

		if (retVal) return(retVal);
	}
#ifdef TESTMODE
	print_submatrices(C, 12, 24, C_pitch, 1, 1, 1, 1);
#endif

	if (!Config->NoPerformanceWarnings && Config->DynamicSched && Config->UseCPU && Config->UseGPU && !finishData->CPUOnlyRun && fabs(finishData->TotalCPUTimer - finishData->GPUTimer) > 1.0)
	{
		fprintf(STD_OUT, "WARNING: Bad GPU / CPU Splitting: GPU Time: %2.4f, CPU Time: %2.4f (m = %lld, n = %lld)\n", finishData->GPUTimer, finishData->TotalCPUTimer, (long long int) finishData->matrix_m, (long long int) finishData->matrix_n);
	}
	displayMatrixTiming("caldgemm");
	if (Config->Verify)
	{
		A = orig_a;
		B = orig_b;
		C = orig_c;
		matrix_m = orig_m;
		matrix_n = orig_n;
		AnalyzeResults();
		delete[] D;
	}

	if (finishData->ExecLinpack)
	{
		if (Config->GPURatioPenalties >= 2)
		{
			if (finishData->CPUTimer < 2.0)
			{
				finishData->gpu_ratio_used = 1. - Config->GPURatioPenaltyFactor * (1. - finishData->gpu_ratio_used);
			}
			if (finishData->ExecLinpack >= 2 && finishData->GPUTimer - finishData->LinpackTimer1 < 1.0)
			{
				finishData->gpu_ratio_used = 1. - Config->GPURatioPenaltyFactor * (1. - finishData->gpu_ratio_used);
			}
		}
		if (Config->GPURatioPenalties >= 1)
		{
			if (finishData->cpu_wait_time >= 0.05)
			{
				finishData->gpu_ratio_used = 1. - Config->GPURatioPenaltyFactor * (1. - finishData->gpu_ratio_used);
			}
		}
		const double tmpratio = finishData->cpu_wait_time > 0.15 ? 0.0 : 0.5;
		const double newratio = tmpratio * linpackGPURatios[finishData->ExecLinpack] + (1.0 - tmpratio) * finishData->gpu_ratio_used;
		if (Config->Debug) fprintf(STD_OUT, "updating ratio table entry %d (old: %2.3f, new: %2.3f, factor: %2.3f) => %2.3f\n", finishData->ExecLinpack, 100 * linpackGPURatios[finishData->ExecLinpack], 100 * finishData->gpu_ratio_used, tmpratio, 100 * newratio);

		linpackGPURatios[finishData->ExecLinpack] = newratio;
		linpackCPUDGEMMTime[finishData->ExecLinpack] = finishData->CPUTimer;
		linpackBcastTime[finishData->ExecLinpack] = finishData->LinpackTimer2;
		linpack_last_mn[finishData->ExecLinpack] = (double) finishData->orig_m * (double) finishData->orig_n;
	}

	return(0);
}

int caldgemm::RunCALDGEMM_Finish() {return(0);}

int caldgemm::DGEMMPrepareAndExecute(caldgemm::DGEMMPrepareAndExecuteTask& Task CALDGEMM_DIVBUFA)
{
	if (Config->MultiThread && UseMutexPerDevice()) pthread_mutex_lock(&device_mutex[Task.device]);
	if (Config->Debug) fprintf(STD_OUT, "DGEMMPrepareAndExecute device %d k1 %d j1 %d k2 %d j2 %d\n", Task.device, (int) Task.PrepareTasks[0].k, Task.PrepareTasks[0].j, (int) Task.PrepareTasks[1].k, Task.PrepareTasks[1].j);
	for (int l = 0;l < 2;l++)
	{
		if (Task.PrepareTasks[l].j != -1)
		{
			if (DGEMM_prepare(Task.PrepareTasks[l].k, Task.PrepareTasks[l].j, Task.device CALDGEMM_DIVBUFB)) return(1);
		}
	}
	
	if (Config->MultiThread && UseOutputPthreads())
	{
		if (Config->Debug) fprintf(STD_OUT, "\tLocking obuffer mutex %d/%d\n", Task.device, Task.j);
		if (Config->AsyncTiming)
		{
			Timers.ATime.Reset();
			Timers.ATime.Start();
		}
		if (Config->MultiThread && UseMutexPerDevice()) pthread_mutex_unlock(&device_mutex[Task.device]);
		obufferMutex[Task.device][Task.j].Lock();
		if (Config->MultiThread && UseMutexPerDevice()) pthread_mutex_lock(&device_mutex[Task.device]);
		if (Config->AsyncTiming)
		{
			Timers.ATime.Stop();
			if ((!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001) || Config->Debug) fprintf(STD_OUT, "\t\tWait Time for output buffer: %1.5f\n", Timers.ATime.GetElapsedTime());
		}
	}
	size_t blockm, blockn;
	DGEMM_getblocks(Task.k, blockm, blockn);

	if (buffer_pointers_A[Task.device][blockm] < 0 || buffer_pointers_B[Task.device][blockn] < 0)
	{
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING, Buffer falsified by previous iteration, need to retransfer (ptr_a = %d, ptr_b = %d)\n", buffer_pointers_A[Task.device][blockm], buffer_pointers_B[Task.device][blockn]);
		if (DGEMM_prepare(Task.k, Task.j, Task.device CALDGEMM_DIVBUFB)) return(1);
	}
	if (ExecuteKernels(Task, blockm, blockn)) return(1);
	if (Config->SimpleGPUQueuing) CheckAlternateTilesRemaining(blockm);
	DGEMMPrepareTaskEventReady[Task.device][Task.j] = true;
	if (Config->MultiThread && UseMutexPerDevice()) pthread_mutex_unlock(&device_mutex[Task.device]);
	return(0);
}

void caldgemm::SetupBufferSizes()
{
	if (Config->Height % KernelSettings.min_tile_size)
	{
		int new_tile_size = Config->Height > (size_t) KernelSettings.min_tile_size ? (Config->Height - Config->Height % KernelSettings.min_tile_size) : KernelSettings.min_tile_size;
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "Default buffer height %d does not fit tile size of %d, adjusting height to %d\n", (int) Config->Height, KernelSettings.min_tile_size, new_tile_size);
		Config->Height = new_tile_size;
	}
	if (Config->Width % KernelSettings.min_k)
	{
		int new_k = Config->Width > (size_t) KernelSettings.min_k ? (Config->Width - Config->Width % KernelSettings.min_k) : KernelSettings.min_k;
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "Default buffer width %d does not fit minimum k value of %d, adjusting width to %d\n", (int) Config->Width, KernelSettings.min_k, new_k);
		Config->Width = new_k;
	}
	BufferHeight = Config->Height;
	BufferWidth = Config->Width;
}

int caldgemm::ExitCALDGEMM()
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "CALDGEMM not initialized, cannot uninitialize!\n");
		return(1);
	}
	nDevices = nDevicesInitialized;
	if (Config->Debug) fprintf(STD_OUT, "Uninitializing CALDGEMM\n");
	delete finishData;
	if (Config->PreallocData) if (PreallocateFree()) return(1);

	if (Config->UseGPU && ExitDevices()) return(1);
	if (Config->MultiThread && UseOutputPthreads())
	{
		for (int num_device = 0;num_device < nDevices;num_device++)
		{
			for (int i = 0;i < (Config->OutputThreads == -1 ? max_outputthreads : Config->OutputThreads);i++)
			{
				if (Config->Debug) fprintf(STD_OUT, "Trying to terminate merge slave %d\n", i);
				mParam[num_device][i].terminate = true;
				mParam[num_device][i].mergeThreadMutex[1].Lock();
				mParam[num_device][i].mergeThreadMutex[0].Unlock();
			}
		}
	}

	ExitRuntime();

	if (Config->UseCPU && Config->UseGPU)
	{
		if (Config->Debug) fprintf(STD_OUT, "Trying to terminate blas slave\n");
		cParam.terminate = true;
		if (Config->MultiThread)
		{
			cParam.cblasMutex[1].Unlock();
			if (Config->Debug) fprintf(STD_OUT, "Waiting for blas threads to terminate\n");
			cParam.cblasMutex[0].Lock();
		}
		for (int i = 0;i < 2;i++) cParam.cblasMutex[i].Unlock();
	}

	if (Config->AlternateLookahead)
	{
		if (pthread_mutex_destroy(&tilesRemainingMutex)) fprintf(STD_OUT, "ERROR destroying tilesRemainingMutex: %s - %d\n", __FILE__, __LINE__);
	}

	if (Config->MultiThread)
	{
		if (Config->Debug) fprintf(STD_OUT, "Trying to terminate linpack slave\n");
		linpackParameters.terminate = true;
		linpackParameters.linpackMutex[0].Unlock();
		if (Config->Debug) fprintf(STD_OUT, "Waiting for linpack slave to terminate\n");
		linpackParameters.linpackMutex[1].Lock();
		for (int i = 0;i < 2;i++) linpackParameters.linpackMutex[i].Unlock();

		if (UseOutputPthreads())
		{
			if (Config->Debug) fprintf(STD_OUT, "Waiting for merge threads to terminate\n");
			for (int i = 0;i < (Config->OutputThreads == -1 ? max_outputthreads : Config->OutputThreads);i++)
			{
				for (int num_device = 0;num_device < nDevices;num_device++)
				{
					mParam[num_device][i].mergeThreadMutex[1].Lock();
				}
			}
		}
		
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int i = 0;i < divideThreads;i++)
			{
				dParam[i].terminate = 1;
				DGEMMTasks[dParam[i].curDevice].mutex_start.Unlock();
				if (Config->Debug) fprintf(STD_OUT, "Waiting for divide threads to terminate\n");
				DGEMMTasks[i].mutex_finished.Lock();
			}
		}
	}

	if (Config->MultiThread)
	{
		if (UseOutputPthreads())
		{
			for (int num_device = 0;num_device < nDevices;num_device++)
			{
				for (int i = 0;i < (Config->OutputThreads == -1 ? max_outputthreads : Config->OutputThreads);i++)
				{
					for (int j = 0;j < 2;j++)
					{
						mParam[num_device][i].mergeThreadMutex[j].Unlock();
					}
				}
			}
		}
		if (pthread_mutex_destroy(&scheduleMutex)) fprintf(STD_OUT, "ERROR destroying schedule mutex\n");
		
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int i = 0;i < nDevices;i++)
			{
				DGEMMTasks[i].mutex_start.Unlock();
				DGEMMTasks[i].mutex_finished.Unlock();
			}
		}
		if (Config->MultiThread && UseMutexPerDevice())
		{
			for (int i = 0;i < nDevices;i++)
			{
				if (pthread_mutex_destroy(&device_mutex[i])) fprintf(STD_OUT, "ERROR destroying device_mutex: %s - %d\n", __FILE__, __LINE__);
			}
		}
	}

	if (Config->ThreadSaveDriver == -1)
	{
		pthread_mutex_destroy(&globalDriverLock);
	}
	if (Config->UseDMAFetchQueue)
	{
		for (int i = 0;i < nDevices;i++)
		{
			pthread_mutex_destroy(&dma_fetch_queue_tasks[i].mutex);
		}
	}

#ifdef CALDGEMM_DIVIDE_STATIC_BUFFER
	freeDivideBuffer(divide_tmpBuffer);
#endif

	caldgemm_initialized = false;
	return(0);
}

void caldgemm::ResetTimers()
{
	//Reset Timers
	Timers.System.Reset();
	Timers.Kernel.Reset();
	Timers.CounterDivide.Reset();
	Timers.CounterMerge.Reset();
	Timers.CounterCopyTo.Reset();
	Timers.CounterCopyFrom.Reset();
	Timers.CPUTimer.Reset();
	Timers.TotalCPUTimer.Reset();
	Timers.GPUTimer.Reset();
	Timers.divideA = Timers.divideB = Timers.divideC = 0;
	Timers.LinpackTimer1.Reset();
	Timers.LinpackTimer2.Reset();
	Timers.LinpackTimer3.Reset();
	Timers.BcastTimer.Reset();
	Timers.device_kernel = 0;
}

#define MAX_HUGE_ADDRESSES 256
double* huge_page_addresses[MAX_HUGE_ADDRESSES];
int nHugeAddresses = 0;
#ifndef HUGE_PAGESIZE
#define HUGE_PAGESIZE (1024 * 2048)
#endif

double* caldgemm::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible, bool interleave)
{
#ifndef USE_OLD_HUGE_MALLOC
	return((double*) qmalloc::qMalloc(nDoubles * sizeof(double), huge_pages, false, page_locked, NULL, interleave));
#else
#ifdef WASTE_MEMORY
	nDoubles += 40 * 1024 * 1024;
#endif
	double* ptr;
#ifndef _WIN32
	if (huge_pages)
	{
		if (nHugeAddresses >= MAX_HUGE_ADDRESSES - 1)
		{
			fprintf(STD_OUT, "No more huge_page memory available, increase MAX_HUGE_ADDRESSES\n");
			return(NULL);
		}
		int shmid;
		void *address = NULL;

		if (Config->Debug)  fprintf(STD_OUT, "Running Huge Maloc\n");

		if ((shmid = shmget(IPC_PRIVATE, (nDoubles * sizeof(double) + HUGE_PAGESIZE) & ~(HUGE_PAGESIZE - 1), SHM_HUGETLB | IPC_CREAT | 0600)) < 0)
		{
			fprintf(STD_OUT, "Memory allocation error (shmget).\n");
			return(NULL);
		}

		ptr = (double*) shmat(shmid, NULL, SHM_RND);
		if ((long long int) address == -1)
		{
			fprintf(STD_OUT, "Memory allocation error (shmat).\n");
			return(NULL);
		}

		shmctl(shmid, IPC_RMID, NULL);

		if (page_locked && shmctl(shmid, SHM_LOCK, NULL) == -1)
		{
			fprintf(STD_OUT, "ERROR locking HugePage Memory\n");
			shmdt((void*) ptr);
			return(NULL);
		}

		huge_page_addresses[nHugeAddresses++] = ptr;

	}
	else
#endif
	{
#ifdef _WIN32
		ptr = (double*) VirtualAlloc(NULL, nDoubles * sizeof(double), MEM_COMMIT, PAGE_READWRITE);
#else
		ptr = new double[nDoubles];
#endif
	}
	if (ptr == NULL) return(NULL);
#ifdef WASTE_MEMORY
	nDoubles -= 40 * 1024 * 1024;
	ptr += 20 * 1024 * 1024;
#endif
	if (!huge_pages && page_locked)
	{
#ifdef _WIN32
		size_t minp, maxp;
		HANDLE pid = GetCurrentProcess();
		if (GetProcessWorkingSetSize(pid, &minp, &maxp) == 0) fprintf(STD_OUT, "Error getting minimum working set size\n");
		if (SetProcessWorkingSetSize(pid, minp + nDoubles * sizeof(double), maxp + nDoubles * sizeof(double)) == 0) fprintf(STD_OUT, "Error settings maximum working set size\n");
		if (VirtualLock(ptr, nDoubles * sizeof(double)) == 0)
#else
		if (mlock(ptr, nDoubles * sizeof(double)))
#endif
		{
			fprintf(STD_OUT, "ERROR locking Pages\n");
			if (!huge_pages)
			{
#ifdef _WIN32
				DWORD err = GetLastError();
				fprintf(STD_OUT, "Error Number: %d\n", err);
				VirtualFree(ptr, 0, MEM_RELEASE);
#else
				delete[] ptr;
#endif
			}
			return(NULL);
		}
	}
	return(ptr);
#endif
}

int caldgemm::FreeMemory(double* ptr, bool gpuaccessible)
{
#ifndef USE_OLD_HUGE_MALLOC
	qmalloc::qFree(ptr);
#else
#ifdef WASTE_MEMORY
	ptr -= 20 * 1024 * 1024;
#endif
#ifndef _WIN32
	for (int i = 0;i < nHugeAddresses;i++)
	{
		if (huge_page_addresses[i] == ptr)
		{
			shmdt((void*) ptr);
			huge_page_addresses[i] = huge_page_addresses[--nHugeAddresses];
			return;
		}
	}
#endif

#ifdef _WIN32
	VirtualFree(ptr, 0, MEM_RELEASE);
#else
	delete[] ptr;
#endif
#endif
	return(0);
}

void caldgemm::displayMatrixTiming(const char* name)
{
	double gflops_CPU = (double) 1e-09 * finishData->orig_m * finishData->orig_n * (2 * Config->Width + 2) * (double) Config->Iterations / finishData->System;
	avggflops = ((double) avgngflops * avggflops + gflops_CPU) / (double) (avgngflops + 1);
	avgngflops++;
	if (!Config->Quiet || (Config->DisplayTiming /*&& matrix_m * matrix_n >= 16 * 24 * 1024 * 1024*/)) fprintf(STD_OUT, "%sProgram: %s Sizes - A: %lldx%lld B: %lldx%lld C:%lldx%lld (Host: %s) System Time %2.3f System Gflops %2.3f\n", Config->PreOut, name, 
		(long long int) finishData->orig_m, (long long int) Config->Width, (long long int) Config->Width, (long long int) finishData->orig_n, (long long int) finishData->orig_m, (long long int) finishData->orig_n, hostname, finishData->System, gflops_CPU);
	if (Config->UseCPU == true && Config->UseGPU == true)
	{
		double flopsc, flopsg;
		if (finishData->CPUOnlyRun)
		{
			flopsc = (double) 1e-09 * finishData->orig_m * finishData->orig_n * (2 * Config->Width + 2) * Config->Iterations / finishData->CPUTimer;
			flopsg = 0.0;
		}
		else if (finishData->DGEMM_split_m)
		{
			flopsc = (double) 1e-09 * (finishData->dynamic_run * finishData->dynamic_size + finishData->cblas_size * finishData->matrix_n + (finishData->matrix_n % finishData->SmallTileHeight) * (finishData->matrix_m - finishData->cblas_size) + finishData->dynamic_run2 * Config->Height * Config->Height + (finishData->ExecLinpack >= 2 && Config->AlternateLookahead <= finishData->matrix_n ? Config->Width * finishData->matrix_n : 0)) * (2 * Config->Width + 2) * Config->Iterations / finishData->CPUTimer;
			flopsg = (double) 1e-09 * ((finishData->matrix_m - finishData->cblas_size) * (finishData->matrix_n - finishData->matrix_n % finishData->SmallTileHeight) - finishData->dynamic_run * finishData->dynamic_size - finishData->dynamic_run2 * Config->Height * Config->Height) * (2 * Config->Width + 2) * Config->Iterations / finishData->GPUTimer;
		}
		else
		{
			flopsc = (double) 1e-09 * (finishData->dynamic_run * finishData->dynamic_size + finishData->cblas_size * finishData->matrix_m + (finishData->matrix_m % finishData->SmallTileHeight) * (finishData->matrix_n - finishData->cblas_size) + finishData->dynamic_run2 * Config->Height * Config->Height + (finishData->ExecLinpack >= 2 && Config->AlternateLookahead <= finishData->matrix_n ? Config->Width * finishData->matrix_n : 0)) * (2 * Config->Width + 2) * Config->Iterations / finishData->CPUTimer;
			flopsg = (double) 1e-09 * ((finishData->matrix_n - finishData->cblas_size) * (finishData->matrix_m - finishData->matrix_m % finishData->SmallTileHeight) - finishData->dynamic_run * finishData->dynamic_size - finishData->dynamic_run2 * Config->Height * Config->Height) * (2 * Config->Width + 2) * Config->Iterations / finishData->GPUTimer;
		}
		
		if (Config->GPUClock && finishData->matrix_m * finishData->matrix_n >= 24 * 24 * 1024 * 1024 && flopsg <= (double) 460 * (double) Config->GPUClock / (double) 850 - (double) 20)
		{
			fprintf(STD_OUT, "%sThrottling: %s (%2.3f GFlops)\n", Config->PreOut, hostname, flopsg);
		}

		//const double gpu_ratio_used_new = std::min(1.0, flopsg / (flopsc * (Timers.System.GetElapsedTime() - Timers.LinpackTimer1.GetElapsedTime() - (ExecLinpack > 1 ? Config->GPURatioMarginTimeDuringFact : Config->GPURatioMarginTime) - Timers.LinpackTimer3.GetElapsedTime()) / Timers.System.GetElapsedTime() + flopsg));
		double gpu_ratio_used_new = mymin(1.0, flopsg / (flopsc * (finishData->CPUTimer - (finishData->ExecLinpack > 1 ? Config->GPURatioMarginTimeDuringFact : Config->GPURatioMarginTime)) / finishData->TotalCPUTimer + flopsg));
		if (gpu_ratio_used_new < 0) finishData->gpu_ratio_used = 1.;
		
		if (!Config->Quiet || (Config->DisplayTiming /*&& matrix_m * matrix_n >= 16 * 24 * 1024 * 1024*/))
		{
			char timingoutputbase[1024];
			char *timingoutput = timingoutputbase;
			timingoutput += sprintf(timingoutput, "%sGPU Time %2.4f (%2.4f Gflops)   CPU Time %2.4f (%2.4f Gflops)", Config->PreOut, finishData->GPUTimer, flopsg, finishData->CPUTimer, flopsc);
			if (finishData->ExecLinpack) timingoutput += sprintf(timingoutput, "   Linpack Time: %2.4f (%d, %2.4f, %2.4f)  Total CPU Time: %2.4f", finishData->LinpackTimer1, finishData->ExecLinpack, finishData->LinpackTimer2, finishData->LinpackTimer3, finishData->TotalCPUTimer);
			if (Config->TabularTiming)
			{
				timingoutput += sprintf(timingoutput, " --- GPU Ratio - Real: %2.3f Corrected: %2.3f Guessed: %2.3f , m*n: %.1E, CPU Wait Time: %2.3f", (flopsg / (flopsc + flopsg)), gpu_ratio_used_new, finishData->gpu_ratio_used, (double) (finishData->matrix_m * finishData->matrix_n), finishData->cpu_wait_time > 0.001 ? finishData->cpu_wait_time : (finishData->TotalCPUTimer - finishData->GPUTimer));
			}
			sprintf(timingoutput, "\n");
			fwrite(timingoutputbase, 1, strlen(timingoutputbase), STD_OUT);
		}
		finishData->gpu_ratio_used = gpu_ratio_used_new;
	}
	if ((!Config->Quiet || (Config->DisplayTiming /*&& matrix_n * matrix_m >= 16 * 24 * 1024 * 1024*/)) && Config->VerboseTiming)
	{
		double gflops = (double) 1e-09 * matrix_m * matrix_n * (2 * Config->Width - 1) * (double)Config->Iterations / Timers.Kernel.GetElapsedTime();
#ifdef CALDGEMM_BENCHMARK_KERNEL
		gflops *= (double) CALDGEMM_BENCHMARK_KERNEL;
#endif
		double copyto = Config->DivideToGPU ? 0 : ((double) 1e-09 * ((Config->Height * Timers.divideA + Config->Height * Timers.divideB) * Config->Width + Timers.divideC * Config->Height * Config->Height) * sizeof(double) * (double)Config->Iterations / Timers.CounterCopyTo.GetElapsedTime());
		double copyfrom = Config->DstMemory == 'g' ? ((double) 1e-09 * matrix_m * matrix_n * sizeof(double) * (double)Config->Iterations / Timers.CounterCopyFrom.GetElapsedTime()) : 0;
		double copyMerge = Config->MultiThread || UseOutputPthreads() == 0 ? 0 :((double) 1e-09 * matrix_m * matrix_n * sizeof(double) * (double)Config->Iterations / Timers.CounterMerge.GetElapsedTime());
		double copyDivide = UseInputPthreads() ? (double) 1e-09 * (Config->Height * Timers.divideA + Config->Height * Timers.divideB) * Config->Width * sizeof(double) * (double)Config->Iterations / Timers.CounterDivide.GetElapsedTime() : 0;
		fprintf(STD_OUT, "Times:  Kernel                    Divide (%d,%d)            Merge                   Copy To                 Copy From\n", Timers.divideA, Timers.divideB);
		fprintf(STD_OUT, "        %2.4f (%2.4f Gflops)  %2.4f (%2.4f GB/s)    %2.4f (%2.4f GB/s)    %2.4f (%2.4f GB/s)    %2.4f (%2.4f Gb/s)\n", Timers.Kernel.GetElapsedTime(), gflops, Timers.CounterDivide.GetElapsedTime(), copyDivide, Timers.CounterMerge.GetElapsedTime(), copyMerge, Timers.CounterCopyTo.GetElapsedTime(), copyto, Timers.CounterCopyFrom.GetElapsedTime(), copyfrom);
		double gflops_device = 0;
		if (Timers.device_kernel)
		{
			gflops_device = (double) matrix_m * matrix_n * (2 * Config->Width - 1) * (double)Config->Iterations / (double) Timers.device_kernel;
			fprintf(STD_OUT, "        %2.4f (%2.4f Gflops)\n", (double) Timers.device_kernel * 1e-09, gflops_device);
		}
		if (Config->TabularTiming)
		{
			fprintf(STD_OUT, "TIMES:\tw\t%lld\th\t%lld\tkernel\t%2.4f / %2.4f\tdivide\t%2.4f\tmerge\t%2.4f\tcopyto\t%2.4f\tcopyfr\t%2.4f\n", (long long int) Config->Width, (long long int) Config->Height, gflops, gflops_device, copyDivide, copyMerge, copyto, copyfrom);
		}
	}
}

unsigned int caldgemm::AnalyzeResults()
{
	size_t errors = 0;
	size_t total = 0;
	
	if (!Config->Quiet) fprintf(STD_OUT, "Verifying results can take a long time on large matrices.\n");
	HighResTimer Timer;
	Timer.Reset();
	Timer.Start();
	cblas_dgemm(CblasRowMajor, TransposeA ? CblasTrans : CblasNoTrans, TransposeB ? CblasTrans : CblasNoTrans, matrix_m, matrix_n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, D, C_pitch);
	Timer.Stop();
	if (!Config->Quiet) fprintf(STD_OUT, "CPU Time: %f Gflops: %f\n", Timer.GetElapsedTime(), (double)1e-09 * 2 * matrix_m * matrix_n * Config->Width / Timer.GetElapsedTime());
	
#ifdef TESTMODE
	fprintf(STD_OUT, "Reference Matrix:\n");
	print_submatrices(D, 12, 24, C_pitch, 1, 1, 1, 1);
#endif

	int nblocksm = 0;
	int* errortiles = NULL;
	if (Config->Height)
	{
		nblocksm = matrix_m / Config->Height + 1;
		errortiles = (int*) malloc((matrix_n / Config->Height + 1) * nblocksm * sizeof(int));
		memset(errortiles, 0, (matrix_n / Config->Height + 1) * nblocksm * sizeof(int));
	}
	size_t errorsrel[3];
	memset(errorsrel, 0, 3 * sizeof(size_t));

	for (size_t i=0; i < matrix_m; i++)
	{
		for (size_t j=0; j < matrix_n; j++)
		{
			if (!isDoubleEqual(C[i * C_pitch + j],D[i * C_pitch + j]))
			{
				if (errors < 5) fprintf(STD_OUT, "Error found at row %lld, col %lld: Expected: %3.5le, Found: %3.5le, Diff: %3.5le, Relative: %3.5le\n", (long long int) i, (long long int) j, D[i * C_pitch + j], C[i * C_pitch + j], D[i * C_pitch + j] - C[i * C_pitch + j], (D[i * C_pitch + j] - C[i * C_pitch + j]) / D[i * C_pitch + j]);
				++errors;
				if (Config->Height) errortiles[j / Config->Height * nblocksm + i / Config->Height]++;
				if (fabs((C[i * C_pitch + j] - D[i * C_pitch + j]) / D[i * C_pitch + j]) > 0.05) errorsrel[0]++;
				else if (fabs((C[i * C_pitch + j] - D[i * C_pitch + j]) / D[i * C_pitch + j]) < 0.0001) errorsrel[2]++;
				else errorsrel[1]++;
			}
			++total;
		}
	}
	if (errors)
	{
		fprintf(STD_OUT, "%lld out of %lld elements were incorrect (Rel errors > 0.05: %lld, > 0.0001: %lld, rest: %lld)\n", (long long int) errors, (long long int) total, (long long int) errorsrel[0], (long long int) errorsrel[1], (long long int) errorsrel[2]);
		if (errorsrel[0] == 0)
		{
			fprintf(STD_OUT, "Passed with Warnings!!!\n");
		}
		else
		{
			fprintf(STD_OUT, "FAILED (Host %s)\n", hostname);
		}
	}
	else if (!Config->Quiet)
	{
		fprintf(STD_OUT, "Passed!\n");
	}
	if (!Config->NoPerformanceWarnings && (errors || Config->Debug) && Config->Height)
	{
		fprintf(STD_OUT, "GPU output matrix\n");
		print_submatrices(C, matrix_n, matrix_m, C_pitch, 1, 1, Config->Height, Config->Height);
		fprintf(STD_OUT, "Reference matrix\n");
		print_submatrices(D, matrix_n, matrix_m, C_pitch, 1, 1, Config->Height, Config->Height, C);
	}

	if (!Config->NoPerformanceWarnings && errors && Config->Height)
	{
		fprintf(STD_OUT, "Number of errors in tiles\n");
		for (size_t i = 0;i < matrix_m;i += Config->Height)
		{
			for (size_t j = 0;j < matrix_n;j += Config->Height)
			{
				fprintf(STD_OUT, "%8d\t", errortiles[j / Config->Height * nblocksm + i / Config->Height]);
			}
			fprintf(STD_OUT, "\n");
		}
	}

	if (Config->Height) free(errortiles);
		
	return(errors == 0);
}

bool caldgemm::isDoubleEqual(double a, double b)
{
	if (!qIsFinite(a) || !qIsFinite(b)) return(false);
	double valmax = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
	if (valmax < 1e-15)
	{
		return(fabs(a - b) < 1e16);
	}
	else if (valmax < 1e-9)
	{
		return(fabs((a - b)/valmax) < 5e-2);
	}
	else if(valmax < 1e-8)
	{
		return (fabs((a-b)/valmax) < 1e-3);
	}
	else
	{
		return (fabs((a-b)/valmax) < 1e-4);
	}
}

int caldgemm::DGEMM_prepare(size_t k, int j, unsigned int num_device CALDGEMM_DIVBUFA)
{
#ifdef CALDGEMM_BENCHMARK_KERNEL
	return(0);
#endif
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	bool buffersSufficiant0, buffersSufficiant;
#ifdef REUSE_BBUFFERS
	if (DGEMM_favor_m)
	{
		buffersSufficiant0 = true;
		buffersSufficiant = next_buffer_B[num_device] < bbuffers[num_device];
	}
	else
	{
		buffersSufficiant0 = buffersSwitchable;
		buffersSufficiant = buffersSwitchable && next_buffer_A[num_device] < bbuffers[num_device];
	}
#else
	buffersSufficiant0 = buffersSufficiant = false;
#endif

	if (Config->Debug) fprintf(STD_OUT, "Running Preprocessing device = %d k = %lld\n", num_device, (long long int) k);
	//if (Config->Debug) fprintf(STD_OUT, "device %d Favor %d major %d minor %d blockm %d blockn %d\n", (int) num_device, (int) DGEMM_favor_m, (int) buffersMajor[num_device], (int) buffersMinor[num_device][DGEMM_favor_m ? blockn : blockm], (int) blockm, (int) blockn);
	
	const bool prepareM = DGEMM_favor_m ? (buffersMajor[num_device] < (signed long long int) blockm) : (!buffersSufficiant0 || buffer_pointers_A[num_device][blockm] == -1);
	const bool prepareN = DGEMM_favor_m ? (!buffersSufficiant0 || buffer_pointers_B[num_device][blockn] == -1) : (buffersMajor[num_device] < (signed long long int) blockn);

	if (prepareM)
	{
		WaitForLASWP(blockm);
		if (DGEMM_favor_m) buffersMajor[num_device] = blockm;
		else if (buffersSufficiant0)
		{
			const int buffer_pos = next_buffer_A[num_device] % (buffersSufficiant ? bbuffers[num_device] : ibuffercount);
			if (buffersMinor[num_device][next_buffer_A[num_device] % bbuffers[num_device]] != -1)
			{
				static bool bbuffer_warning_shown = false;
				if (Config->Debug || !(Config->NoPerformanceWarnings || bbuffer_warning_shown))
				{
					bbuffer_warning_shown = true;
					fprintf(STD_OUT, "WARNING: Insufficient BBuffers, replacing blockm %d by %d in buffer %d\n", buffersMinor[num_device][buffer_pos], (int) blockm, buffer_pos);
				}
				buffer_pointers_A[num_device][buffersMinor[num_device][buffer_pos]] = -1;
				
			}
			buffersMinor[num_device][buffer_pos] = blockm;
		}
		buffer_pointers_A[num_device][blockm] = next_buffer_A[num_device];
	}
	else if (Config->Debug) fprintf(STD_OUT, "\tSkipping preprocessing part of A (device = %d, k = %lld, j = %d, m = %lld, n = %lld)\n", num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn);

	if (prepareN)
	{
		if (!DGEMM_favor_m) buffersMajor[num_device] = blockn;
		else if (buffersSufficiant0)
		{
			const int buffer_pos = next_buffer_B[num_device] % (buffersSufficiant ? bbuffers[num_device] : ibuffercount);
			if (buffersMinor[num_device][buffer_pos] != -1)
			{
				static bool bbuffer_warning_shown = false;
				if (Config->Debug || !(Config->NoPerformanceWarnings || bbuffer_warning_shown))
				{
					bbuffer_warning_shown = true;
					fprintf(STD_OUT, "WARNING: Insufficient BBuffers, replacing blockn %d by %d in buffer %d\n", buffersMinor[num_device][buffer_pos], (int) blockn, buffer_pos);
				}
				buffer_pointers_B[num_device][buffersMinor[num_device][buffer_pos]] = -1;
			}
			buffersMinor[num_device][buffer_pos] = blockn;
		}
		buffer_pointers_B[num_device][blockn] = next_buffer_B[num_device];
	}
	else if (Config->Debug) fprintf(STD_OUT, "\tSkipping preprocessing part of B (device = %d, k = %lld, j = %d, m = %lld, n = %lld)\n", num_device, (long long int) k, j, (long long int) blockm, (long long int) blockn);

	if (prepareM || prepareN) dma_pending[num_device][j] = true;
	if(DGEMM_prepare_backend(k, j, num_device, prepareM, prepareN, buffersSufficiant, buffersSufficiant0 CALDGEMM_DIVBUFB)) return(1);

	if (prepareM) next_buffer_A[num_device]++;
	if (prepareN) next_buffer_B[num_device]++;

	return(0);
}

void caldgemm::printConfig(caldgemm::caldgemm_config* newConfig, caldgemm::caldgemm_config* oldConfig)
{
	caldgemm_config* myConfig = newConfig ? newConfig : Config;

	PRINT_CONFIG_INT(AsyncDMA);
	PRINT_CONFIG_INT(PipelinedOperation);
	PRINT_CONFIG_INT(PipelinedMidMarker);
	PRINT_CONFIG_INT(PipelineDoubleBuffer);
	PRINT_CONFIG_INT(DivideToGPU);
	PRINT_CONFIG_CHAR(DstMemory);
	PRINT_CONFIG_INT(ImplicitDriverSync);
	PRINT_CONFIG_INT(UseDMAFetchQueue);
	PRINT_CONFIG_INT(DynamicSched);
	PRINT_CONFIG_INT(SecondPhaseDynamicRuns);
	PRINT_CONFIG_INT(ThirdPhaseDynamicRuns);
	PRINT_CONFIG_INT(ThirdPhaseThreshold);
	PRINT_CONFIG_INT(KeepBuffersMapped);
	PRINT_CONFIG_INT(MemPolicy);
	PRINT_CONFIG_INT(MultiThread);
	PRINT_CONFIG_INT(MultiThreadDivide);
	PRINT_CONFIG_INT(ImprovedScheduler);
	PRINT_CONFIG_INT(ImprovedSchedulerBalance);
	PRINT_CONFIG_INT(SimpleGPUQueuing);
	PRINT_CONFIG_INT(AlternateSimpleQueuing);
	PRINT_CONFIG_INT(AlternateSimpleQueuingMulti);
	PRINT_CONFIG_INT(ParallelDMA);
	PRINT_CONFIG_INT(GroupParallelDMA);
	PRINT_CONFIG_DOUBLE(GPURatio);
	PRINT_CONFIG_DOUBLE(GPURatioDuringFact);
	PRINT_CONFIG_DOUBLE(GPURatioMax);
	PRINT_CONFIG_DOUBLE(GPURatioMarginTime);
	PRINT_CONFIG_DOUBLE(GPURatioMarginTimeDuringFact);
	PRINT_CONFIG_DOUBLE(GPURatioLookaheadSizeMod);
	PRINT_CONFIG_INT(GPURatioPenalties);
	PRINT_CONFIG_DOUBLE(GPURatioPenaltyFactor);
	PRINT_CONFIG_INT(MinimizeCPUPart);
	PRINT_CONFIG_INT(MinimizeCPUDuringFact);
	PRINT_CONFIG_INT(UseCPU);
	PRINT_CONFIG_INT(UseGPU);
	PRINT_CONFIG_INT(RereserveLinpackCPU);
	PRINT_CONFIG_INT(GPU_C);
	PRINT_CONFIG_INT(NoConcurrentKernels);

	PRINT_CONFIG_INT(OpenCLPlatform);
	PRINT_CONFIG_INT(DeviceNum);
	PRINT_CONFIG_INT(NumDevices);
	PRINT_CONFIG_INT(NumActiveDevices);
	PRINT_CONFIG_LOOP_INT(DeviceNums, NumDevices);
	PRINT_CONFIG_INT(max_bbuffers);
	PRINT_CONFIG_INT(PreallocData);
	PRINT_CONFIG_INT(CPUInContext);

	PRINT_CONFIG_INT(Debug);
	PRINT_CONFIG_INT(DumpMatrix);
	PRINT_CONFIG_INT(Iterations);
	PRINT_CONFIG_INT(Verify);
	PRINT_CONFIG_INT(SkipCPUProcessing);
	PRINT_CONFIG_INT(ForceKernelVariant);

	PRINT_CONFIG_LOOP_INT(GPUMapping, NumDevices);
	PRINT_CONFIG_LOOP_INT(PostprocessMapping, NumDevices);
	PRINT_CONFIG_LOOP_INT(AllocMapping, NumDevices);
	PRINT_CONFIG_LOOP_INT(DMAMapping, NumDevices);

	PRINT_CONFIG_INT(PinMainThread);
	PRINT_CONFIG_INT(PinDeviceRuntimeThreads);
	PRINT_CONFIG_INT(PinBroadcastThread);
	PRINT_CONFIG_INT(RepinDuringActiveWaitForEvent);
	PRINT_CONFIG_INT(RepinMainThreadAlways);
	PRINT_CONFIG_INT(SpawnGPUThread);
	PRINT_CONFIG_INT(SleepDuringActiveWait);
	PRINT_CONFIG_INT(ThreadSaveDriver);
	PRINT_CONFIG_INT(PinCPU);
	PRINT_CONFIG_INT(ForceNumCPUThreads);
	PRINT_CONFIG_INT(CPUCoreOffset);
	PRINT_CONFIG_INT(SlowCPU);
	PRINT_CONFIG_INT(OutputThreads);
	PRINT_CONFIG_INT(NumaPinning);
	PRINT_CONFIG_INT(AlternateLookahead);
	PRINT_CONFIG_INT(AsyncSideQueue);
	PRINT_CONFIG_INT(AsyncSideQueueBalance);
	PRINT_CONFIG_INT(AsyncDGEMMThreshold);
	PRINT_CONFIG_INT(AsyncDTRSMThreshold);
	PRINT_CONFIG_INT(AsyncDTRSM);
	PRINT_CONFIG_INT(AsyncSideQueueUseInactiveDeviceSet);
	PRINT_CONFIG_INT(Use3rdPartyTranspose);

	PRINT_CONFIG_INT(Height);
	PRINT_CONFIG_INT(Width);
	PRINT_CONFIG_INT(AutoHeight);
	PRINT_CONFIG_INT(SmallTiles);

	PRINT_CONFIG_INT(Disassemble);
	PRINT_CONFIG_INT(PrintILKernel);

	PRINT_CONFIG_INT(AsyncTiming);
	PRINT_CONFIG_INT(DisplayTiming);
	PRINT_CONFIG_INT(NoPerformanceWarnings);
	PRINT_CONFIG_STRING(PreOut);
	PRINT_CONFIG_INT(Quiet);
	PRINT_CONFIG_INT(TabularTiming);
	PRINT_CONFIG_INT(VerboseTiming);

	PRINT_CONFIG_INT(LinpackNodes);
	PRINT_CONFIG_INT(MPIRank);
	PRINT_CONFIG_INT(GPUClock);

	PRINT_CONFIG_INT(HPLFactorizeRestrictCPUs);
	PRINT_CONFIG_INT(LASWPSleep);
	PRINT_CONFIG_LOOP_INT(ExcludeCPUCores, nExcludeCPUCores);
	PRINT_CONFIG_INT(ShowConfig);
	PRINT_CONFIG_INT(ShowThreadPinning);

	PRINT_CONFIG_INT_THIS(BufferWidth);
	PRINT_CONFIG_INT_THIS(BufferHeight);
	
	if (myConfig->config_backend && (oldConfig == NULL || oldConfig->config_backend))
	{
		myConfig->config_backend->printConfig(oldConfig ? oldConfig->config_backend : NULL);
	}
}

double caldgemm::getMaxGPUTemperature()
{
    return(0.);
}

int caldgemm::RunCALDGEMM_Init()
{
	return(0);
}

int caldgemm::RunCALDGEMM_Exit()
{
	return(0);
}

void caldgemm::SetDefaultKernelSettings()
{
#ifdef CALDGEMM_TRANSPOSED_A
	KernelSettings.transposeA = true;
#else
	KernelSettings.transposeA = false;
#endif
#ifdef CALDGEMM_TRANSPOSED_B
	KernelSettings.transposeB = true;
#else
	KernelSettings.transposeB = false;
#endif
	KernelSettings.texture_buffers = true;
	KernelSettings.tiling_x = TILING_X;
	KernelSettings.tiling_y = TILING_Y;
	KernelSettings.group_size_x = 16;
	KernelSettings.group_size_y = 16;
	KernelSettings.min_tile_size = CALDGEMM_MIN_TILE_DIM;
	KernelSettings.min_k = 4;
}

int caldgemm::CaldgemmCustomAutoHeight(size_t MaxGpuM, size_t MaxGpuN, int nDevices) {return 0;}
int caldgemm::CaldgemmCustomModHeight(size_t MOD_OVER, size_t MOD_GPU) {return 0;}

int caldgemm::ParseParameters(unsigned int argc, char** argv, caldgemm_config* Config)
{
#include "caldgemm_parse_parameters.h"
	return(0);
}

int caldgemm::ParseParameters(char* params, caldgemm_config* Config)
{
	if (Config->Debug) fprintf(STD_OUT, "Parsing CALDGEMM Parameters: '%s'\n", params);
	char* tmpParams = new char[strlen(params) + 1]; //This memory will be leaked, in case of string parameters we need to keep a copy, and we do not know how long params will live.
	strcpy(tmpParams, params);
	int argc = 1;
	char** argv = new char*[strlen(params) / 2 + 3];
	char* tmppos = tmpParams;
	argv[0] = "caldgemm";
	while (*tmppos != 0)
	{
		while (*tmppos == ' ' || *tmppos == '	') tmppos++;
		if (*tmppos == 0) break;
		argv[argc++] = tmppos;
		while (*tmppos != ' ' && *tmppos != '	' && *tmppos != 0) tmppos++;
		if (*tmppos) *(tmppos++) = 0;
	}
	argv[argc] = NULL;
	int retVal = ParseParameters(argc, argv, Config);
	delete[] argv;
	retVal |= Config->InitializeBackendOptions();
	return(retVal);
}

int caldgemm::AllowCPUFallback() {return(1);}
int caldgemm::SimpleQueuingAvailable() {return(0);}
int caldgemm::PipelinedModeAvailable() {return(0);}
int caldgemm::AsyncModeAvailable() {return(0);}

bool caldgemm::NeedSimpleQueueKernelEvent(int blockm, int blockn, int k, int device)
{
	int mb = (gpu_m + Config->Height - 1) / Config->Height;
	int nb = (gpu_n + Config->Height - 1) / Config->Height;

	if (DGEMM_favor_m ? (blockm != mb - 1) : (blockn != nb - 1))
	{
		int kklast = k + (DGEMM_favor_m ? nb : mb);
		kklast -= kklast % (DGEMM_favor_m ? nb : mb);

		int num = 0;
		for (int kk = k;kk < kklast;kk++)
		{
			if (tileDistribution[kk] == device)
			{
				if (++num == ibuffercount) break;
			}
		}
		if (num < ibuffercount)
		{
			return(true);
		}
	}
	return(false);
}

#ifndef USE_GOTO_BLAS
static int caldgemm_restrict_cpus = 0;
static int current_num_threads = get_num_procs();

void cblas_dscala(blasint N, double alpha, double *X, blasint incX)
{
	int oldthreads = 0;
	if (caldgemm_restrict_cpus > 2 && current_num_threads > 8)
	{
		oldthreads = current_num_threads;
		omp_set_num_threads(8);
	}
	cblas_dscal(N, alpha, X, incX);
	if (oldthreads) omp_set_num_threads(oldthreads);
}

void cblas_daxpya(blasint n, double alpha, double *x, blasint incx, double *y, blasint incy)
{
	int oldthreads = 0;
	if (caldgemm_restrict_cpus > 2 && current_num_threads > 8)
	{
		oldthreads = current_num_threads;
		omp_set_num_threads(8);
	}
	cblas_daxpy(n, alpha, x, incx, y, incy);
	if (oldthreads) omp_set_num_threads(oldthreads);
}

void cblas_dgemma(CBLAS_ENUM CBLAS_ORDER Order, CBLAS_ENUM CBLAS_TRANSPOSE TransA, CBLAS_ENUM CBLAS_TRANSPOSE TransB, blasint M, blasint N, blasint K, double alpha, double *A, blasint lda, double *B, blasint ldb, double beta, double *C, blasint ldc)
{
	int oldthreads = 0;
	if (caldgemm_restrict_cpus)
	{
		int nthreads = 0;
		long long int tflops = (long long int) M * (long long int) N * (long long int) K;
		if (tflops <= 16384) nthreads = 1;
		else if (tflops <= 65536) nthreads = 2;
		else if (tflops < 200000 || (tflops > 2000000 && tflops < 4000000)) nthreads = 3;
		else if (tflops <= 2000000) nthreads = 4;
		else if (tflops <= 26542080) nthreads = 8;
		else if (tflops <= 56623104) nthreads = 12;
		else if (tflops <= 89915392) nthreads = 16;
		else if (tflops <= 262144000) nthreads = 20;

		if (nthreads && nthreads < current_num_threads)
		{
			oldthreads = current_num_threads;
			omp_set_num_threads(nthreads);
		}
	}
	cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	if (oldthreads) omp_set_num_threads(oldthreads);
}

void cblas_dgemva(CBLAS_ENUM CBLAS_ORDER order,  CBLAS_ENUM CBLAS_TRANSPOSE trans,  blasint m, blasint n, double alpha, double  *a, blasint lda,  double  *x, blasint incx,  double beta,  double  *y, blasint incy)
{
	int oldthreads = 0;
	if (caldgemm_restrict_cpus)
	{
		int nthreads = 0;
		if (n >= 4 * m)
		{
			long long int tflops = (long long int) n * 64;
			if (tflops <= 458752) nthreads = 4;
			else if (tflops <= 655360) nthreads = 8;
		}
		else
		{
			long long int tflops = (long long int) m * (long long int) n;
			if (tflops < 102400) nthreads = 1;
			else if (tflops < 3686400) nthreads = 3;
			else nthreads = 4;
		}
		if (caldgemm_restrict_cpus > 2 && nthreads > 8) nthreads = 8;
		if (nthreads && nthreads < current_num_threads)
		{
			oldthreads = current_num_threads;
			omp_set_num_threads(nthreads);
		}
	}
	cblas_dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
	if (oldthreads) omp_set_num_threads(oldthreads);
}

void cblas_dtrsma(CBLAS_ENUM CBLAS_ORDER Order, CBLAS_ENUM CBLAS_SIDE Side, CBLAS_ENUM CBLAS_UPLO Uplo, CBLAS_ENUM CBLAS_TRANSPOSE TransA, CBLAS_ENUM CBLAS_DIAG Diag, blasint M, blasint N, double alpha, double *A, blasint lda, double *B, blasint ldb)
{
	int oldthreads = 0;
	if (caldgemm_restrict_cpus)
	{
		int nthreads = 0;
		long long int tflops = (long long int) N * (long long int) N * (long long int) M;
		if (tflops <= 32768) nthreads = 1;
		else if (tflops <= 110592) nthreads = 3;
		else if (tflops <= 100000000) nthreads = 4;
		else if (tflops <= 1000000000) nthreads = 16;
		if (caldgemm_restrict_cpus > 2 && nthreads > 8) nthreads = 8;
		if (nthreads && nthreads < current_num_threads)
		{
			oldthreads = current_num_threads;
			omp_set_num_threads(nthreads);
		}
	} 
	cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
	if (oldthreads) omp_set_num_threads(oldthreads);
}

void caldgemm_goto_restrict_cpus(int val)
{
	caldgemm_restrict_cpus = val;
}

void goto_set_num_threads(int num)
{
	current_num_threads = num;
	omp_set_num_threads(num);
#ifdef USE_MKL
	mkl_set_num_threads(num);
#endif
}

#endif

// vim: ts=4 sw=4 noet sts=4 tw=100
