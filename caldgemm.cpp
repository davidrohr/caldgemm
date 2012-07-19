/**
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

#include "caldgemm.h"
#include "caldgemm_common.h"
#include "cmodules/qmalloc.h"

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

#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3

#ifndef SHM_HUGETLB
#define SHM_HUGETLB 04000
#endif

#if !defined(USE_GOTO_BLAS) | defined(_WIN32)
#include "cmodules/os_low_level_helper.h"
extern "C" {
extern int get_num_procs();
int get_num_procs()
{
	return(get_number_of_cpu_cores());
}
}
#endif

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
    fprintf(stderr, "%lld ", (long long int) a.tv_sec * 1000000 + (long long int) a.tv_nsec / 1000 - begin);
}
#define fprintf(file, ...) {printelapsedtime();fprintf(stderr, __VA_ARGS__);}
#endif

caldgemm::caldgemm()
{
	caldgemm_initialized = false;
	for (int i = 0;i < caldgemm::max_linpack_callback_types;i++)
	{
		linpack_last_mn[i] = -1.;
		linpackGPURatios[i] = 1.;
		linpackBcastTime[i] = 0;
		linpackCPUDGEMMTime[i] = 0;
	}

	avggflops = 0;
	avgngflops = 0;
	
	conf_numprocs = get_num_procs();
	
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
	NumDevices = max_devices;
	OpenCLPlatform = 0;
	Width = 1024;
	Height = 4096;
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
	UseGPU = true;
	UseCPU = true;
	GPURatio = -1.0;
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
	PinMainThread = -1;
	SlowCPU = false;
	m = 0;
	n = 0;
	LinpackNodes = 0;
	LinpackSwapN = NULL;
	HPLFactorizeRestrictCPUs = 2;
	MPIRank = -1;
	PreOut = EmptyOut;
	GPUClock = 0;
	SmallTiles = 0;
	ThreadSaveDriver = false;
	SkipCPUProcessing = false;
	OutputThreads = -1;
	RepinDuringActiveWaitForEvent = 0;
	SleepDuringActiveWait = -1;
	NumaPinning = false;
	ThirdPhaseThreshold = 0;
	for (unsigned int i = 0;i < caldgemm::max_devices;i++)
	{
		GPUMapping[i] = 0;
		PostprocessMapping[i] = -1;
		AllocMapping[i] = -1;
		DeviceNums[i] = i;
	}
	nExcludeCPUCores = 0;
	ExcludeCPUCores = NULL;
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
							if (jj >= Config->m - cParam.cblas_size || ii >= Config->n - Config->n % Config->Height) sprintf(tmpcolor, "01;34");
						}
						else
						{
							if (jj >= Config->m - Config->m % Config->Height || ii >= Config->n - cParam.cblas_size) sprintf(tmpcolor, "01;34");
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
						fprintf(STD_OUT, "%+10.3lf\t", M[jj * pitch + ii]);
					}
					else
					{
						fprintf(STD_OUT, " %+10.3lf\t", M[jj * pitch + ii]);
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

void caldgemm::ensure_omp_thread_pinning()
{
#ifndef USE_GOTO_BLAS
	if (Config->Debug) fprintf(STD_OUT, "Performing OpenMP Blas Thread Pinning\n");
	int* cpu_order = new int[conf_numprocs];
	if (Config->NumaPinning && conf_numprocs % 4 == 0)
	{
		cpu_order[0] = 0;
		int cpu_num = 1;
		
		int divider = conf_numprocs / 2;
		int old_divider = conf_numprocs;
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

#pragma omp parallel num_threads(conf_numprocs)
	{
		int thread_id = omp_get_thread_num();
		cpu_set_t localset;
		int localcore = thread_id * 2;

		int nFreeCores = 0;
		if (thread_id == nFreeCores) localcore = main_blas_core;
		nFreeCores++;
		for (int i = 0;i < conf_numprocs;i++)
		{
			if (cpuUsed(cpu_order[i]) == false && cpu_order[i] != broadcast_cpu_core && cpu_order[i] != main_blas_core)
			{
				if (thread_id == nFreeCores) localcore = cpu_order[i];
				nFreeCores++;
			}
		}
		if (thread_id == nFreeCores) localcore = broadcast_cpu_core;
		nFreeCores++;
		for (int i = 0;i < conf_numprocs;i++)
		{
			if (cpuUsed(cpu_order[i]) && cpu_order[i] != main_blas_core)
			{
				if (thread_id == nFreeCores) localcore = cpu_order[i];
				nFreeCores++;
			}
		}

		CPU_ZERO(&localset);
		CPU_SET(localcore, &localset);
		sched_setaffinity(0, sizeof(localset), &localset);
		if (Config->Debug) fprintf(STD_OUT, "OpenMP BLAS thread %d pinned to core %d\n", thread_id, localcore);
	}
	delete[] cpu_order;
#endif
}

int caldgemm::InitCALDGEMM(caldgemm_config* pInfo, bool nocalinit)
{
	Config = pInfo;

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

#ifndef _WIN32
	if (Config->UseGPU)
	{
		for (int i = 0;i < conf_numprocs;i++)
		{
			if (CPU_ISSET(i, &oldcpumask) && cpuUsed(i)) fprintf(STD_OUT, "WARNING: Core %d used by GotoBLAS main thread and CALDGEMM, be sure not to use CPU and GPU at the same time!\n", i);
		}
	}
#endif

	if (Config->PinCPU != -1)
	{
	    for (unsigned int i = 0;i < max_devices;i++) Config->GPUMapping[i] = Config->PinCPU;
	}

	CPU_ZERO(&gpumask);
	if (Config->PinMainThread == -1) Config->PinMainThread = Config->GPUMapping[0];
	CPU_SET(Config->PinMainThread, &gpumask);

#ifdef USE_GOTO_BLAS
	sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);		//GotoBLAS has its own thread pinning, store old value here.
#endif

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
	if (Config->MultiThread == false) Config->MultiThreadDivide = false;

	if (ValidateRuntime()) return(1);

	if (Config->Debug) fprintf(STD_OUT, "Initializing CAL\n");
	if (Config->UseGPU == 0 || Initialize(nocalinit))
	{
		gpu_available = false;
	}
	if (!gpu_available)
	{
		if (!Config->Quiet && Config->UseGPU) fprintf(STD_OUT, "No GPU available, falling back to CPU\n");
		nDevices = 0;
		Config->UseGPU = 0;
		Config->UseCPU = 1;
		Config->KeepBuffersMapped = 0;
	}

	if (CheckDevices()) return(1);

	outputthreads = Config->OutputThreads == -1 ? (Config->KeepBuffersMapped || Config->DstMemory == 'g' ? CALDGEMM_OUTPUT_THREADS : CALDGEMM_OUTPUT_THREADS_SLOW) : Config->OutputThreads;
	
#ifndef USE_GOTO_BLAS		//If we do not use GotoBLAS thread pinning determine main blas thread only after determining GPU devices to avoid collisions. Store the thread afterward as for GotoBLAS.
	main_blas_core = 0;
	while (cpuUsed(main_blas_core) && main_blas_core < get_num_procs() - 1) main_blas_core++;
	if (Config->Debug) fprintf(STD_OUT, "Pinning Main OpenMP BLAS thread to core %d\n", main_blas_core);
	cpu_set_t blasset;
	CPU_ZERO(&blasset);
	CPU_SET(main_blas_core, &blasset);
	sched_setaffinity(0, sizeof(blasset), &blasset);

	sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);		//GotoBLAS has its own thread pinning, store old value here.
#endif

	if (InitDevices()) return(1);
	
	int min_bbuffers = max_bbuffers;
	for (int i = 0;i < nDevices;i++)
	{
		if (bbuffers[i] < min_bbuffers) min_bbuffers = bbuffers[i];
	}
	if (!Config->Quiet) fprintf(STD_OUT, "Running on %d devices with %d bbuffers\n", nDevices, min_bbuffers);

	if (Config->MultiThread && UseOutputPthreads())
	{
		for (int device_num = 0;device_num < nDevices;device_num++)
		{
			for (int i = 0;i < obuffercount;i++)
			{
				pthread_mutex_init(&obufferMutex[device_num][i], NULL);
			}

			for (int i = 0;i < max_outputthreads;i++)
			{
				mParam[device_num][i].num_device = device_num;
				mParam[device_num][i].cls = this;
				mParam[device_num][i].terminate = false;
				mParam[device_num][i].nMergeThread = i;
				for (int j = 0;j < 2;j++) pthread_mutex_init(&mParam[device_num][i].mergeThreadMutex[j], NULL);
				pthread_t thr;
				pthread_create(&thr, NULL, merge_wrapper, &mParam[device_num][i]);

				while (pthread_mutex_trylock(&mParam[device_num][i].mergeThreadMutex[0]) != EBUSY) if (pthread_mutex_unlock(&mParam[device_num][i].mergeThreadMutex[0])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			}
		}
	}
	
	if (Config->MultiThreadDivide && UseMutexPerDevice())
	{
	    for (int i = 0;i < nDevices;i++)
	    {
		pthread_mutex_init(&device_mutex[i], NULL);
	    }
	}

	cpu_set_t tmpmask;
	CPU_ZERO(&tmpmask);
	CPU_SET(Config->PinMainThread, &tmpmask);
	sched_setaffinity(0, sizeof(tmpmask), &tmpmask);

	int linpackCPU = 0;
	while (linpackCPU < conf_numprocs)
	{
		if (cpuUsed(linpackCPU) == false && linpackCPU != main_blas_core) break;
		linpackCPU++;
	}
	if (linpackCPU >= conf_numprocs) linpackCPU = 0;
	broadcast_cpu_core = linpackCPU;
	if (Config->Debug) fprintf(STD_OUT, "Broadcast CPU core set to %d\n", linpackCPU);

	if (Config->MultiThread)
	{
		linpackParameters.terminate = false;
		for (int j = 0;j < 2;j++) pthread_mutex_init(&linpackParameters.linpackMutex[j], NULL);
		if (pthread_mutex_lock(&linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
		pthread_t thr;
		pthread_create(&thr, NULL, linpack_wrapper, this);
		if (Config->Debug) fprintf(STD_OUT, "Waiting for linpack slave to start\n");
		while (pthread_mutex_trylock(&linpackParameters.linpackMutex[1]) != EBUSY) if (pthread_mutex_unlock(&linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		pthread_mutex_init(&scheduleMutex, NULL);
		
		divideThreads = 0;
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int i = 0;i < nDevices;i++)
			{
				pthread_mutex_init(&DGEMMTasks[i].mutex_start, NULL);
				if (pthread_mutex_lock(&DGEMMTasks[i].mutex_start)) fprintf(STD_OUT, "ERROR locking divide start mutex (%d)\n", i);
				DGEMMTasks[i].thread_running = 0;
				DGEMMTasks[i].skip_device_to = -1;
				pthread_mutex_init(&DGEMMTasks[i].mutex_finished, NULL);
				if (pthread_mutex_lock(&DGEMMTasks[i].mutex_finished)) fprintf(STD_OUT, "ERROR locking divide finish mutex (%d)\n", i);
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
					if (pthread_mutex_lock(&DGEMMTasks[divideThreads].mutex_finished)) fprintf(STD_OUT, "ERROR locking divide finish mutex (%d)\n", divideThreads);
					divideThreads++;
				}
			}
		}
	}

	if (Config->Debug) fprintf(STD_OUT, "Using %d CPU cores at %d MHz, %d GPUs of %d shaders at %d MHz\n", conf_numprocs, conf_cpufreq, nDevices, conf_gpushaders, conf_gpufreq);
	ensure_omp_thread_pinning();

	if (Config->UseCPU)
	{
		cParam.cls = this;
		cParam.terminate = false;
		for (int j = 0;j < 2;j++) pthread_mutex_init(&cParam.cblasMutex[j], NULL);
		if (pthread_mutex_lock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (Config->MultiThread)
		{
			pthread_t thr;
			pthread_create(&thr, NULL, cblas_wrapper, &cParam);
			if (Config->Debug) fprintf(STD_OUT, "Waiting for cblas slave to start\n");
			while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		}
	}

	if (Config->MemPolicy)
	{
#ifdef _WIN32

#else
		unsigned long nodemask = 0xffffff;
		syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
#endif
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

	caldgemm_initialized = true;

	return(0);
}

int caldgemm::broadcastcore()
{
	return(broadcast_cpu_core);
}

bool caldgemm::cpuUsed(int cpu)
{
	if (cpu == Config->PinMainThread) return(true);
	for (int i = 0;i < Config->nExcludeCPUCores;i++) if (Config->ExcludeCPUCores[i] == cpu) return(true);
	return(false);
}

void* caldgemm::linpack_wrapper(void* arg)
{
	caldgemm* cls = (caldgemm*) arg;
	volatile caldgemm::caldgemm_config* Config = cls->Config;
	if (Config->Debug) fprintf(STD_OUT, "Linpack helper thread started\n");

	cpu_set_t linpack_mask;
	CPU_ZERO(&linpack_mask);
	
	int linpackCPU = cls->broadcast_cpu_core;
	if (linpackCPU >= cls->conf_numprocs) linpackCPU = 0;
	CPU_SET(linpackCPU, &linpack_mask);
	if (Config->Debug) fprintf(STD_OUT, "Linpack Thread, setting CPU mask %X\n", cls->getcpumask(&linpack_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &linpack_mask);

	if (pthread_mutex_lock(&cls->linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&cls->linpackParameters.linpackMutex[0]) == 0 && cls->linpackParameters.terminate == false)
	{
		cls->Timers.LinpackTimer2.Start();
		Config->linpack_broadcast_function();
		cls->Timers.LinpackTimer2.Stop();
		cls->Timers.BcastTimer.Start();

		if (pthread_mutex_unlock(&cls->linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}

	if (Config->Debug) fprintf(STD_OUT, "linpack slave terminating\n");
	pthread_exit(NULL);
	return(NULL);
}

int caldgemm::cpuScheduler()
{
	int retVal = 0;
	if (Config->UseCPU && Config->MultiThread && Config->DynamicSched)
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
						fprintf(STD_OUT, "Adjusting second phase run size for small tiles: %lld - %lld = %lld\n", (long long int) cParam.dynamic_size, (long long int) adjustment, (long long int) cParam.dynamic_size - adjustment);
						cParam.dynamic_size -= adjustment;
					}
					if (cParam.dynamic_run && (DGEMM_favor_m ? gpu_m : gpu_n) % Config->Height)
					{
						const size_t adjustment = Config->Height - (DGEMM_favor_m ? gpu_m : gpu_n) % Config->Height;
						fprintf(STD_OUT, "Adjusting second phase run row size for small tiles: %lld - %lld = %lld\n", (long long int) cParam.dynamic_run, (long long int) adjustment, (long long int) cParam.dynamic_run - adjustment);
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



void* caldgemm::cblas_wrapper(void* arg)
{
	volatile caldgemm::cblasParameters* par = (caldgemm::cblasParameters*) arg;
	volatile caldgemm::caldgemm_config* Config = par->cls->Config;

	if (Config->Debug) fprintf(STD_OUT, "Cblas helper thread started\n");

	par->cls->ensure_omp_thread_pinning();

	if (Config->Debug) fprintf(STD_OUT, "Cblas thread Thread, setting CPU mask %X\n", par->cls->getcpumask(&par->cls->oldcpumask));
	
	if (Config->GPUMapping[0] + par->cls->outputthreads * par->cls->nDevices + 1 >= par->cls->conf_numprocs)
	{
		cpu_set_t tmp_mask;
		CPU_ZERO(&tmp_mask);
		CPU_SET(0, &tmp_mask);
		sched_setaffinity(0, sizeof(tmp_mask), &tmp_mask);
	}
	else
	{
		sched_setaffinity(0, sizeof(par->cls->oldcpumask), &par->cls->oldcpumask);
	}
	
	if (Config->MultiThread) if (pthread_mutex_lock(&par->cls->cParam.cblasMutex[1])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&par->cls->cParam.cblasMutex[1]) == 0 && par->terminate == false)
	{
		const double Alpha = par->cls->Alpha;
		const double Beta = par->cls->Beta;
		double* const A = par->cls->A;
		double* const B = par->cls->B;
		double* const C = par->cls->C;
		const size_t A_pitch = par->cls->A_pitch;
		const size_t B_pitch = par->cls->B_pitch;
		const size_t C_pitch = par->cls->C_pitch;
		const size_t A_pitch_use = (par->cls->TransposeA ? 1 : A_pitch);
		const size_t B_pitch_use = (par->cls->TransposeB ? B_pitch : 1);
		const CBLAS_TRANSPOSE TransposeA = par->cls->TransposeA ? CblasTrans : CblasNoTrans;
		const CBLAS_TRANSPOSE TransposeB = par->cls->TransposeB ? CblasTrans : CblasNoTrans;
		if (!Config->Quiet) fprintf(STD_OUT, "\t\tSlave thread starting cblas (m: %lld, n: %lld, cblas_size: %lld (%lld), dynamic: %lld/%lld, cpu_k: %lld)\n", (long long int) Config->m, (long long int) Config->n, (long long int) par->cblas_size, (long long int) Config->Height, (long long int) par->dynamic_run, (long long int) par->dynamic_size, (long long int) par->cpu_k);


		int old_goto_threads = get_num_procs();

		int require_threads_base = par->cls->reserve_cpu_cores();
		int require_threads = require_threads_base;
		
		if (par->cls->ExecLinpack && par->cls->Config->LinpackNodes > 1)
		{
			caldgemm_goto_reserve_cpu(par->cls->broadcast_cpu_core, 1);
			require_threads++;
		}
		if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu (/ Linpack)\n", require_threads);
		if (old_goto_threads > require_threads)
		{
			goto_set_num_threads(old_goto_threads - require_threads);
		}
		else
		{
			goto_set_num_threads(1);
			caldgemm_goto_reserve_cpus(0);
		}

		par->cls->Timers.TotalCPUTimer.Start();
		par->cls->Timers.LinpackTimer3.Start();
		if (Config->LinpackSwapN != NULL)
		{
			if (Config->HPLFactorizeRestrictCPUs == 1)
			{
			    if (8 < old_goto_threads - require_threads) goto_set_num_threads(8);
			}
			else if (Config->HPLFactorizeRestrictCPUs >= 2)
			{
			    caldgemm_goto_restrict_cpus(Config->HPLFactorizeRestrictCPUs);
			}
			Config->linpack_swap_function();
			if (Config->HPLFactorizeRestrictCPUs >= 2)
			{
			    caldgemm_goto_restrict_cpus(0);
			}
			if (old_goto_threads > require_threads)
			{
				goto_set_num_threads(old_goto_threads - require_threads);
			}
			else
			{
				goto_set_num_threads(1);
			}
		}
		par->cls->Timers.LinpackTimer3.Stop();

		if (par->cls->ExecLinpack)
		{
			if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tDoing initial cblas runs to prepare Linpack factorization\n");
			par->cls->Timers.CPUTimer.Start();
			cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->Width, Config->n, Config->Width, Alpha, A - Config->Width * A_pitch_use, A_pitch, B, B_pitch, Beta, C - Config->Width * C_pitch, C_pitch);
			par->cls->Timers.CPUTimer.Stop();
#ifndef NO_ASYNC_LINPACK
			if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tStarting Linpack factorization\n");
			if (Config->HPLFactorizeRestrictCPUs == 1)
			{
			    if (8 < old_goto_threads - require_threads) goto_set_num_threads(8);
			}
			else if (Config->HPLFactorizeRestrictCPUs >= 2)
			{
			    caldgemm_goto_restrict_cpus(Config->HPLFactorizeRestrictCPUs);
			}
			par->cls->Timers.LinpackTimer1.Start();
			Config->linpack_factorize_function();
			par->cls->Timers.LinpackTimer1.Stop();
			if (Config->HPLFactorizeRestrictCPUs >= 2) caldgemm_goto_restrict_cpus(0);
			goto_set_num_threads(old_goto_threads - require_threads);

			if (par->cls->Config->LinpackNodes > 1)
			{
				if (Config->MultiThread)
				{
					pthread_mutex_unlock(&par->cls->linpackParameters.linpackMutex[0]);
				}
				else
				{
					par->cls->Timers.LinpackTimer2.Start();
					Config->linpack_broadcast_function();
					par->cls->Timers.LinpackTimer2.Stop();
				}
			}
#endif
		}

		par->cls->Timers.CPUTimer.Start();
		bool linpackfinished = false;
		do
		{
			if (par->dynamic_run2)
			{
				size_t blockm, blockn;
				par->cls->DGEMM_getblocks(par->cpu_k, blockm, blockn);
				cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, blockm == par->cls->gpu_m / Config->Height ? (par->cls->gpu_m % Config->Height) : Config->Height, blockn == par->cls->gpu_n / Config->Height ? (par->cls->gpu_n % Config->Height) : Config->Height, Config->Width, Alpha, A + blockm * Config->Height * A_pitch_use, A_pitch, B + blockn * Config->Height * B_pitch_use, B_pitch, Beta, C + blockm * Config->Height * C_pitch + blockn * Config->Height, C_pitch);
			}
			else
			{
				if (par->dynamic_run)
				{
					if (par->cls->DGEMM_favor_m)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->dynamic_run, par->dynamic_size, Config->Width, Alpha, A + (par->cls->gpu_m - par->dynamic_run) * A_pitch_use, A_pitch, B + (par->cls->gpu_n - par->dynamic_size) * B_pitch_use, B_pitch, Beta, C + (par->cls->gpu_m - par->dynamic_run) * C_pitch + par->cls->gpu_n - par->dynamic_size, C_pitch);
					}
					else
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->dynamic_size, par->dynamic_run, Config->Width, Alpha, A + (par->cls->gpu_m - par->dynamic_size) * A_pitch_use, A_pitch, B + (par->cls->gpu_n - par->dynamic_run) * B_pitch_use, B_pitch, Beta, C + (par->cls->gpu_m - par->dynamic_size) * C_pitch + par->cls->gpu_n - par->dynamic_run, C_pitch);
					}
				}

				size_t cblas2;
#ifdef RERESERVE_LINPACK_CPUS
				if (par->cls->ExecLinpack && par->cls->Config->LinpackNodes > 1 && Config->MultiThread && (((double) Config->m * (double) Config->n) - par->cls->linpack_last_mn[par->cls->ExecLinpack]) / par->cls->linpack_last_mn[par->cls->ExecLinpack] < 0.3 && par->cls->linpackCPUDGEMMTime[par->cls->ExecLinpack] - par->cls->linpackBcastTime[par->cls->ExecLinpack] > 5.0)
				{
					cblas2 = (double) (par->cls->DGEMM_split_m ? Config->n : Config->m) * (par->cls->linpackBcastTime[par->cls->ExecLinpack] + 3.0) / par->cls->linpackCPUDGEMMTime[par->cls->ExecLinpack];
					if (!Config->Quiet) fprintf(STD_OUT, "Splitting CPU DGEMM for later enabling additional cores, cblas2=%lld\n", (long long int) cblas2);
				}
				else
				{
					cblas2 = 0;
				}
				if (cblas2 % 8) cblas2 += 8 - cblas2 % 8;
#else
				cblas2 = 0;
#endif

				if (par->cls->DGEMM_split_m)	//favor splitting m because of consecutive memory
				{
					if (par->dynamic_run == 0)
					{
						if (cblas2)
						{
							cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->cblas_size, cblas2, Config->Width, Alpha, A + (Config->m - par->cblas_size) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Config->m - par->cblas_size) * C_pitch, C_pitch);

							if (pthread_mutex_trylock(&par->cls->linpackParameters.linpackMutex[1]) == EBUSY)
							{
								if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
							}
							else
							{
								par->cls->Timers.BcastTimer.Stop();
								if (!Config->NoPerformanceWarnings && par->cls->Timers.BcastTimer.GetElapsedTime() > 1.0) fprintf(STD_OUT, "Bcast core idle for %2.4lf seconds\n", par->cls->Timers.BcastTimer.GetElapsedTime());

								int require_threads_new = require_threads_base;
								if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
								if (old_goto_threads > require_threads_new)
								{
									goto_set_num_threads(old_goto_threads - require_threads_new);
									caldgemm_goto_reserve_cpu(par->cls->broadcast_cpu_core, 0);
								}
								else
								{
									goto_set_num_threads(1);
									caldgemm_goto_reserve_cpus(0);
								}
								linpackfinished = true;
							}
						}
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->cblas_size, Config->n - cblas2, Config->Width, Alpha, A + (Config->m - par->cblas_size) * A_pitch_use, A_pitch, B + cblas2 * B_pitch_use, B_pitch, Beta, C + (Config->m - par->cblas_size) * C_pitch + cblas2, C_pitch);
					}

					if (Config->n % par->cls->SmallTileHeight && par->borders_done == false)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m - par->cblas_size, Config->n % par->cls->SmallTileHeight, Config->Width, Alpha, A, A_pitch, B + (Config->n - Config->n % par->cls->SmallTileHeight) * B_pitch_use, B_pitch, Beta, C + Config->n - Config->n % par->cls->SmallTileHeight, C_pitch);
					}
				}
				else
				{
					if (par->dynamic_run == 0)
					{
						if (cblas2)
						{
							cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cblas2, par->cblas_size, Config->Width, Alpha, A, A_pitch, B + (Config->n - par->cblas_size) * B_pitch_use, B_pitch, Beta, C + Config->n - par->cblas_size, C_pitch);
							
							if (pthread_mutex_trylock(&par->cls->linpackParameters.linpackMutex[1]) == EBUSY)
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
									caldgemm_goto_reserve_cpu(par->cls->broadcast_cpu_core, 0);
								}
								else
								{
									goto_set_num_threads(1);
									caldgemm_goto_reserve_cpus(0);
								}
								linpackfinished = true;
							}
						}
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m - cblas2, par->cblas_size, Config->Width, Alpha, A + cblas2 * A_pitch_use, A_pitch, B + (Config->n - par->cblas_size) * B_pitch_use, B_pitch, Beta, C + cblas2 * C_pitch + Config->n - par->cblas_size, C_pitch);
					}

					if (Config->m % par->cls->SmallTileHeight && par->borders_done == false)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m % par->cls->SmallTileHeight, Config->n - par->cblas_size, Config->Width, Alpha, A + (Config->m - Config->m % par->cls->SmallTileHeight) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Config->m - Config->m % par->cls->SmallTileHeight) * C_pitch, C_pitch);
					}
				}
			}
			par->borders_done = true;
			if (Config->Debug) fprintf(STD_OUT, "cblas run completed\n");
		} while (par->cls->cpuScheduler());
		par->cls->Timers.CPUTimer.Stop();

#ifndef NO_ASYNC_LINPACK
		if (linpackfinished == false && par->cls->ExecLinpack && Config->MultiThread && par->cls->Config->LinpackNodes > 1)
		{
			pthread_mutex_lock(&par->cls->linpackParameters.linpackMutex[1]);
		}
#endif
		par->cls->Timers.TotalCPUTimer.Stop();
		goto_set_num_threads(old_goto_threads);
		caldgemm_goto_reserve_cpus(0);

		if (Config->Debug) fprintf(STD_OUT, "\t\tUnlocking cblasmutex 0\n");
		if (pthread_mutex_unlock(&par->cls->cParam.cblasMutex[0])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (!Config->MultiThread) break;
	}
	if (Config->Debug) fprintf(STD_OUT, "blas slave terminating\n");
	if (Config->MultiThread)
	{
		pthread_exit(NULL);
	}
	return(NULL);
}

void* caldgemm::divide_wrapper(void* arg)
{
	caldgemm::divideParameters* par = (caldgemm::divideParameters*) arg;
	if (par->cls->Config->Debug) fprintf(STD_OUT, "Divide Thread %d for core %d started\n", par->nThread, par->CPUCore);
	cpu_set_t divide_mask;
	CPU_ZERO(&divide_mask);
	CPU_SET(par->CPUCore, &divide_mask);
	sched_setaffinity(0, sizeof(cpu_set_t), &divide_mask);
	
	par->curDevice = -1;
	for (int i = 0;i < par->cls->nDevices;i++)
	{
		if (par->cls->Config->GPUMapping[i] == par->CPUCore)
		{
			if (par->curDevice == 1) par->curDevice = i;
			par->cls->DGEMMTasks[i].next_device = &par->curDevice;
		}
	}

	if (pthread_mutex_unlock(&par->cls->DGEMMTasks[par->nThread].mutex_finished)) fprintf(STD_OUT, "ERROR unlocking divide finish mutex (%d)\n", par->nThread);
	int i = 0;
	while (true)
	{
		if (par->cls->Config->GPUMapping[i] == par->CPUCore)
		{
			par->reset = 0;
			if (par->cls->Config->Debug) fprintf(STD_OUT, "Divide Thread %d on Core %d waiting to operate on device %d\n", par->nThread, par->CPUCore, i);
			par->curDevice = i;
			if (pthread_mutex_lock(&par->cls->DGEMMTasks[i].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			if (par->terminate) break;
			if (par->reset)
			{
				if (par->cls->Config->Debug) fprintf(STD_OUT, "Divide Thread %d resetting\n", par->nThread);
				i = 0;
				continue;
			}
			
			if (par->cls->DGEMMTasks[i].skip_device_to != -1)
			{
				//fprintf(STD_OUT, "Skipping device %d, switching to %d\n", i, par->cls->DGEMMTasks[i].skip_device_to);
				const int oldi = i;
				i = par->cls->DGEMMTasks[i].skip_device_to;
				par->cls->DGEMMTasks[oldi].skip_device_to = -1;
				if (pthread_mutex_lock(&par->cls->DGEMMTasks[i].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			}
			
			if (par->cls->Config->Debug) fprintf(STD_OUT, "Divide Thread for device %d Starting processing (k = %d)\n", i, par->cls->DGEMMTasks[i].k);
			par->cls->DGEMMPrepareAndExecute(par->cls->DGEMMTasks[i]);
			
			if (pthread_mutex_unlock(&par->cls->DGEMMTasks[i].mutex_finished)) fprintf(STD_OUT, "ERROR unlocking divide finish mutex (%d): %s - %d\n", i, __FILE__, __LINE__);
		}
		i = (i + 1) % par->cls->nDevices;
	}

	if (par->cls->Config->Debug) fprintf(STD_OUT, "Divide Thread %d for Core %d terminating\n", par->nThread, par->CPUCore);
	pthread_exit(NULL);
	return(NULL);
}

void* caldgemm::merge_wrapper(void* arg)
{
	caldgemm::mergeParameters* par = (caldgemm::mergeParameters*) arg;

	if (par->cls->Config->Debug) fprintf(STD_OUT, "Merger Thread %d started\n", par->nMergeThread);

	cpu_set_t merge_mask;
	CPU_ZERO(&merge_mask);
	int merge_core;
	
	if (par->cls->Config->PostprocessMapping[par->num_device] == -1)
	{
		merge_core = par->cls->Config->GPUMapping[par->num_device] + par->nMergeThread + 1;
		for (int i = 0;i < par->num_device;i++)
		{
			if (par->cls->Config->GPUMapping[i] == par->cls->Config->GPUMapping[par->num_device]) merge_core += par->cls->outputthreads;
		}
	}
	else
	{
		merge_core = par->cls->Config->PostprocessMapping[par->num_device] + par->nMergeThread;
	}
	CPU_SET(merge_core % par->cls->conf_numprocs, &merge_mask);
	if (par->cls->Config->Debug) fprintf(STD_OUT, "Merge Thread %d, setting CPU mask %X\n", par->nMergeThread, par->cls->getcpumask(&merge_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &merge_mask);

	HighResTimer mergeTimer;

	if (pthread_mutex_lock(&par->mergeThreadMutex[0])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&par->mergeThreadMutex[0]) == 0 && par->terminate == false)
	{
		if (par->cls->Config->Debug) fprintf(STD_OUT, "\t\tSlave thread %d (device %d) starting merge process for obuffer %d (k = %lld)\n", par->nMergeThread, par->num_device, par->nContext, (long long int) par->k);
		size_t blockm, blockn;
		par->cls->DGEMM_getblocks(par->k, blockm, blockn);
		if (par->cls->Config->Debug)
		{
		    mergeTimer.Reset();
		    mergeTimer.Start();
		}
		par->cls->RunMergeBuffers(par->dst, par->num_device, par->nContext, (blockn == par->cls->gpu_n / par->cls->Config->Height) ? (par->cls->gpu_n % par->cls->Config->Height) : par->cls->Config->Height, (blockm == par->cls->gpu_m / par->cls->Config->Height) ? (par->cls->gpu_m % par->cls->Config->Height) : par->cls->Config->Height, par->cls->BufferHeight, par->cls->BufferHeight, par->cls->C_pitch);
		if (par->cls->Config->Debug)
		{
		    mergeTimer.Stop();
		    fprintf(STD_OUT, "\t\tMerge time: %2.3lf\n", mergeTimer.GetElapsedTime());
		}
		if (par->cls->Config->Debug) fprintf(STD_OUT, "\t\tUnlocking mutex device %d obuffer %d (Slavethread %d)\n", par->num_device, par->nContext, par->nMergeThread);
		if (pthread_mutex_unlock(&par->cls->obufferMutex[par->num_device][par->nContext])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (pthread_mutex_unlock(&par->mergeThreadMutex[1])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}
	if (par->cls->Config->Debug) fprintf(STD_OUT, "merge slave %d terminating\n", par->nMergeThread);
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

void caldgemm::WaitForLASWP(size_t n)
{
	if (Config->LinpackSwapN != NULL)
	{
		int shown = false;
		while (*Config->LinpackSwapN < (n + 1) * Config->Height + (ExecLinpack ? Config->Width : 0) && *Config->LinpackSwapN < gpu_m)
		{
			if (Config->Debug && shown == false)
			{
				fprintf(STD_OUT, "Waiting for LASWP / DTRSM... %lld of %lld\n", (long long int) *Config->LinpackSwapN, (long long int) (n + 1) * Config->Height);
				shown = true;
			}
		}
	}
}

int caldgemm::RunCALDGEMM(double* a, double* b, double* c, double alpha, double beta, size_t tmp_m, size_t tmp_k, size_t tmp_n, size_t Apitch, size_t Bpitch, size_t Cpitch, bool orderColMajor, bool TransA, bool TransB, int ExecuteLinpackCallbacks)
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
	if ((signed) tmp_m != -1) Config->m = tmp_m;
	if ((signed) tmp_n != -1) Config->n = tmp_n;
	if ((signed) tmp_k != -1) Config->Width = tmp_k;

	A_pitch = ((signed) Apitch != -1) ? Apitch : Config->Width;
	B_pitch = ((signed) Bpitch != -1) ? Bpitch : Config->n;
	C_pitch = ((signed) Cpitch != -1) ? Cpitch : Config->n;
	ResetTimers();

	if (orderColMajor)
	{
		double* tmpd;
		size_t tmpi;
		bool tmpt;
		tmpd = A; A = B; B = tmpd;
		tmpi = Config->m; Config->m = Config->n; Config->n = tmpi;
		tmpi = A_pitch; A_pitch = B_pitch; B_pitch = tmpi;
		tmpt = TransA;TransA = TransB;TransB = tmpt;
	}

	if (!Config->Quiet) fprintf(STD_OUT, "Starting DGEMM Run m=%lld k=%lld n=%lld Alpha=%lf Beta=%lf LDA=0x%lx LDB=0x%lx LDC=0x%lx At=%d Bt=%d ColMajor=%d (A=0x%llx, B=0x%llx, C=0x%llx, (C-A=%lld, (C-B)/w=%lld))\n", (long long int) Config->m, (long long int) Config->Width, (long long int) Config->n, Alpha, Beta, A_pitch, B_pitch, C_pitch, (int) (TransA), (int) (TransB), (int) (orderColMajor), (long long int) A, (long long int) B, (long long int) C, (long long int) ((size_t) C - (size_t) A) / sizeof(double), (long long int) ((size_t) C - (size_t) B) / sizeof(double) / Config->Width);

	//Check for double == 1.0 is unsafe and causes compiler warning
	const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double
#if defined(CALDGEMM_44) && !defined(CALDGEMM_USE_MEMEXPORT)
	const unsigned long long int double_minus_one = 0xBFF0000000000000;
	const int kernel_num = ((Config->Width == BufferWidth && reinterpret_cast<unsigned long long int &>(reinterpret_cast<char &>(Beta)) == double_one && reinterpret_cast<unsigned long long int &>(reinterpret_cast<char &>(Alpha)) == double_minus_one) ? 2 : (reinterpret_cast<unsigned long long int &>(reinterpret_cast<char &>(Alpha)) == double_one));
#else
	const int kernel_num = (reinterpret_cast<unsigned long long int &>(Alpha) == double_one);
#endif
	if (Config->Debug && Config->UseGPU) fprintf(STD_OUT, "Using Kernel %d (alpha=0x%llX (%2.3lf), width = %lld)\n", kernel_num, (reinterpret_cast<long long int &>(Alpha)), Alpha, (long long int) Config->Width);

	TransposeA = TransA;
	TransposeB = TransB;    
	ExecLinpack = ExecuteLinpackCallbacks;
	orig_m = Config->m;
	orig_n = Config->n;
	orig_a = A;
	orig_b = B;
	orig_c = C;

	if (Config->Verify)
	{
		D = new double[(size_t) Config->m * (size_t) C_pitch];
		if (D == NULL)
		{
			fprintf(STD_OUT, "Memory allocation error\n");
			return(1);
		}
		memcpy(D, C, Config->m * C_pitch * sizeof(double));
	}

	if (Config->DumpMatrix) DumpMatrix(A, B, C, Alpha, Beta, Config->m, Config->Width, Config->n, A_pitch, B_pitch, C_pitch);

	Timers.System.Start();

	if (ExecuteLinpackCallbacks)
	{
		if (Config->m < Config->Width)
		{
			MaxGpuM = 0;
		}
		else
		{
			MaxGpuM = Config->m - Config->Width;
		}
	}
	else
	{
		MaxGpuM = Config->m;
	}
	MaxGpuN = Config->n;

#ifndef TESTMODE    
	//Check if the GPU can/shall process the required dgemm task
	if (Config->Iterations > 1 || !Config->UseCPU);
	else if (Config->Width % 8 || Config->Width < 256) forceCPU = true;
	else if (MaxGpuM < 512 || MaxGpuN < 512) forceCPU = true;
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
		if (ExecuteLinpackCallbacks >= 2 && !Config->SmallTiles)
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
		if (Config->Height > BufferHeight) Config->Height = BufferHeight;
		if ((Config->Height != BufferHeight && !Config->Quiet) || Config->Debug)  fprintf(STD_OUT, "Using Height %lld of max %lld\n", (long long int) Config->Height, (long long int) BufferHeight);
	}

	if (Config->Width > BufferWidth || Config->Height > BufferHeight) forceReinit = true;

	if (Config->UseCPU)
	    if (Config->UseGPU == false || Config->m < Config->Height || Config->n < Config->Height || (forceReinit && (long long int) MaxGpuM * (long long int) MaxGpuN * (long long int) Config->Width < (long long int) 24 * 1024 * 1024 * 1024) || (Config->Width < 1024 && Config->Height < 1024)) forceCPU = true;

	if (forceCPU)
	{
		if (Config->Debug) fprintf(STD_OUT, "Running CPU only DGEMM\n");
		if (Config->LinpackSwapN != NULL) Config->linpack_swap_function();
#ifndef NO_ASYNC_LINPACK
		if (ExecuteLinpackCallbacks)
		{
			Timers.CPUTimer.Start();
			cblas_dgemm(CblasRowMajor, TransposeA ? CblasTrans : CblasNoTrans, TransposeB ? CblasTrans : CblasNoTrans, Config->Width, Config->n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
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

			Config->m -= Config->Width;
			A += Config->Width * (TransposeA ? 1 : A_pitch);
			C += Config->Width * (C_pitch);
		}
#endif
		Timers.CPUTimer.Start();

		goto_set_num_threads(conf_numprocs);
		cblas_dgemm(CblasRowMajor, TransposeA ? CblasTrans : CblasNoTrans, TransposeB ? CblasTrans : CblasNoTrans, Config->m, Config->n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
		Timers.CPUTimer.Stop();
		CPUOnlyRun = true;
//		goto RunCALDGEMM_end;
	}
	else
	{
	CPUOnlyRun = false;

	if (ExecuteLinpackCallbacks)
	{
		outputthreads = mymin(CALDGEMM_OUTPUT_THREADS_SLOW, outputthreads + CALDGEMM_EXTRA_OUTPUT_THREADS_LINPACK);
	}

	cpu_set_t main_mask;
	CPU_ZERO(&main_mask);
	CPU_SET(Config->PinMainThread, &main_mask);
	if (Config->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&main_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &main_mask);

	if (forceReinit)
	{
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Reinit for increased buffer width / height\n");
		if (ReinitDevices()) return(1);
	}

	InitConstantData(alpha);

	if (Config->SlowCPU)
	{
		GPURatio = 1.0;
	}
	else if (Config->GPURatio <= -0.99)
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
		if (Config->Debug) fprintf(STD_OUT, "GPU Curve Ration: %1.3lf, CPUScale %1.3lf, GPUScale %1.3lf\n", GPURatio, CPUscale, GPUscale);
		GPURatio = GPUscale * GPURatio / (GPUscale * GPURatio + (1.0 - GPURatio) * CPUscale);
		
		if (Config->Debug) fprintf(STD_OUT, "GPURatio automatically set to %1.3lf\n", GPURatio);
		if (GPURatio > 1.) GPURatio = 1.0;
		if ((Config->n + 4) % 4096 < 8 && GPURatio > 0.5) GPURatio = 1. - 0.95 * (1. - GPURatio);
	}
	else
	{
		GPURatio = fabs(Config->GPURatio);
	}

	if (ExecuteLinpackCallbacks && (Config->GPURatio < 0 || GPURatio < 0.99) && !Config->SlowCPU)
	{
		if (Config->GPURatio <= -0.99)
		{
			if (ExecuteLinpackCallbacks > 1) GPURatio = 1.0 - (1.0 - GPURatio) * 0.80 * Config->Width / 1024;
			else GPURatio = 1.0 - (1.0 - GPURatio) * 0.90;
			if (GPURatio > 1.0) GPURatio = 1.0;
		}
		if (linpack_last_mn[ExecuteLinpackCallbacks] > 0 && (((double) MaxGpuM * (double) MaxGpuN) - linpack_last_mn[ExecuteLinpackCallbacks]) / linpack_last_mn[ExecuteLinpackCallbacks] < 0.3 && linpackGPURatios[ExecuteLinpackCallbacks] > 0.0001)
		{
			GPURatio = linpackGPURatios[ExecuteLinpackCallbacks];
			if (Config->Debug) fprintf(STD_OUT, "Taking GPU Ratio from table, entry %d, val %2.3lf\n", ExecuteLinpackCallbacks, 100 * GPURatio);
		}
		else
		{
			linpackGPURatios[ExecuteLinpackCallbacks] = GPURatio;
			if (Config->Debug) fprintf(STD_OUT, "Initializing ratio table entry %d with %2.3lf\n", ExecuteLinpackCallbacks, 100 * GPURatio);
		}
	}
	if (Config->GPURatio < 0 && Config->GPURatio > -0.99 && GPURatio < -Config->GPURatio) GPURatio = -Config->GPURatio;

	gpu_ratio_used = GPURatio;

	if (ExecuteLinpackCallbacks)
	{
		Config->m -= Config->Width;
		A += Config->Width * (TransposeA ? 1 : A_pitch);
		C += Config->Width * (C_pitch);
	}

	cParam.dynamic_run = 0;
	cParam.dynamic_run2 = 0;
	cParam.borders_done = false;
	SmallTileHeight = (Config->SmallTiles == 1 ? CALDGEMM_MIN_TILE_DIM : Config->Height);
recalculate_ratio:
	if (Config->UseCPU == true && Config->UseGPU == true)
	{
		if ((DGEMM_split_m = (Config->LinpackSwapN == NULL ? (Config->m >= Config->n) : 0)))
		{
			size_t virtualm = Config->m + (Config->n % SmallTileHeight) * Config->m / Config->n;
			if (ExecuteLinpackCallbacks) virtualm += Config->Width * (0.25 + (float) Config->m / Config->n);
			gpu_m = GPURatio * (float) virtualm + (SmallTileHeight - 1);
			if (gpu_m > Config->m)
			{
				if (Config->SmallTiles == 2 && SmallTileHeight > CALDGEMM_MIN_TILE_DIM)
				{
					if (SmallTileHeight > 1024) SmallTileHeight = 1024;
					else SmallTileHeight = CALDGEMM_MIN_TILE_DIM;
					goto recalculate_ratio;
				}
				gpu_m = Config->m;
			}
			gpu_m -= gpu_m % SmallTileHeight;
			cParam.cblas_size = Config->m - gpu_m;
			gpu_n = Config->n;
			gpu_n -= gpu_n % SmallTileHeight;
			if (Config->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld, Tilesize %lld\n", (long long int) gpu_m, (long long int) gpu_n, (long long int) Config->m - gpu_m, (long long int) gpu_n, (long long int) SmallTileHeight);
		}
		else
		{
			size_t virtualn = Config->n + (Config->m % SmallTileHeight) * Config->n / Config->m;
			if (ExecuteLinpackCallbacks) virtualn += Config->Width * (0.25 + (float) Config->n / Config->m);
			gpu_n = GPURatio * (float) virtualn + (SmallTileHeight - 1);
			if (gpu_n > Config->n)
			{
				if (Config->SmallTiles == 2 && SmallTileHeight > CALDGEMM_MIN_TILE_DIM)
				{
					if (SmallTileHeight > 1024) SmallTileHeight = 1024;
					else SmallTileHeight = CALDGEMM_MIN_TILE_DIM;
					goto recalculate_ratio;
				}
				gpu_n = Config->n;
			}
			gpu_n -= gpu_n % SmallTileHeight;
			cParam.cblas_size = Config->n - gpu_n;
			gpu_m = Config->m;
			gpu_m -= gpu_m % SmallTileHeight;
			if (Config->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld, Tilesize %lld\n", (long long int) gpu_m, (long long int) gpu_n, (long long int) Config->m, (long long int) Config->n - gpu_n, (long long int) SmallTileHeight);
		}
	}
	else
	{
		if (Config->n % Config->Height || Config->m % Config->Height)
		{
			fprintf(STD_OUT, "Invalid matrix size for GPU only (%lld %% %lld = %lld, %lld %% %lld = %lld)\n", (long long int) Config->n, (long long int) Config->Height, (long long int) Config->n % Config->Height, (long long int) Config->m, (long long int) Config->Height, (long long int) Config->m % Config->Height);
			return(1);
		}
		gpu_n = Config->n;
		gpu_m = Config->m;
	}
	DGEMM_favor_m = Config->LinpackSwapN == NULL ? (gpu_m >= gpu_n) : 1;
	
	if (!Config->Quiet) fprintf(STD_OUT, "Ratio %lf - gpu_m %lld gpu_n %lld - Split %c Favor %c - Tiling %lld\n", GPURatio, (long long int) gpu_m, (long long int) gpu_n, DGEMM_split_m ? 'm' : 'n', DGEMM_favor_m ? 'm' : 'n', (long long int) SmallTileHeight);
	
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
			cblas_wrapper((void*) &cParam);
		}
		if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}
	else if (Config->LinpackSwapN != NULL)
	{
	    Config->linpack_swap_function();
	}

	Timers.GPUTimer.Start();

	for (unsigned int i = 0; i < Config->Iterations; ++i)
	{
		for (int ii = 0;ii < nDevices;ii++)
		{
			buffersMajor[ii] = -1;
			for (int j = 0;j < bbuffers[ii];j++) buffersMinor[ii][j] = -1;
			next_buffer_A[ii] = 0;
			next_buffer_B[ii] = 0;
		}

		int oldj[max_devices];
		int j[max_devices];
		int iMergeThread[max_devices];
		memset(j, 0, nDevices * sizeof(int));
		memset(iMergeThread, 0, nDevices * sizeof(int));
		int use_device = 0;

		size_t blockm = 0, blockn = 0;
		unsigned long long int lastk[max_devices];
		for (int l = 0;l < nDevices;l++) lastk[l] = -1;
		
		size_t nextk = 0;
		size_t next_device_k[max_devices];
		memset(next_device_k, 0, nDevices * sizeof(size_t));

		if (Config->Debug)
		{
			if (DGEMM_favor_m)
			{
				fprintf(STD_OUT, "Favoring m direction, %lld blocks (%lld x %lld)\n", (long long int) nBlocks, (long long int) mb, (long long int) nb);
			}
			else
			{
				fprintf(STD_OUT, "Not favoring m direction, %lld blocks (%lld x %lld)\n", (long long int) nBlocks, (long long int) mb, (long long int) nb);
			}
		}

		if (!Config->NoPerformanceWarnings && (buffersSwitchable ? mymin(nb, mb) : nb) > (size_t) (bbuffers[use_device] * nDevices)) fprintf(STD_OUT, "WARNING: Insufficient buffers for Input Matrices, retransfer required\n");
		
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int l = 0;l < nDevices;l++)
			{
				if (pthread_mutex_unlock(&DGEMMTasks[l].mutex_finished)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			}
		}

		int* tileDistribution = NULL;
		int ImprovedSchedPhase1 = 0;
		int forcePreparation[max_devices];
		for (int l = 0;l < nDevices;l++) forcePreparation[l] = 0;
		if (Config->ImprovedScheduler)
		{
			tileDistribution = new int[nBlocks];
			ImprovedSchedPhase1 = 1;
			for (size_t l = 0;l < nBlocks;l++)
			{
				int k;
				if (DGEMM_favor_m)
				{
					blockn = l % nb;
					blockm = l / nb;
					k = blockn * mb + blockm;
				}
				else
				{
					blockm = l % mb;
					blockn = l / mb;
					k = blockn + blockm * nb;
				}
				tileDistribution[l] = nDevices * k / nBlocks;
				
				//if (Config->Debug) fprintf(STD_OUT, "Tile %lld processed by device %d\n", l, tileDistribution[l]);
			}
		}
		for (int l = 0;l < nDevices;l++)
		{
			buffer_pointers_A[l] = new int[mb];
			for (size_t ll = 0;ll < mb;ll++) buffer_pointers_A[l][ll] = -1;
			buffer_pointers_B[l] = new int[nb];
			for (size_t ll = 0;ll < nb;ll++) buffer_pointers_B[l][ll] = -1;
		}

		if (RunCALDGEMM_Init()) return(0);

		bool cpu_k_barrier_hit = false;
		if (gpu_n && gpu_m)
		{
			for (size_t k = 0;k < nBlocks + 2 * nDevices;k++)
			{
restartkloop:
				//fprintf(STD_OUT, "!!!!! k %lld nd k %lld nextk %lld\n", k, next_device_k[use_device], nextk);
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
							if (Config->Debug) fprintf(STD_OUT, "Skipping tile %lld (m=%lld n=%lld) for device %d\n", (long long int) k, (long long int) blockm, (long long int) blockn, use_device);
							k++;
						}
						if (k == nBlocks) goto endimprovedphase;
					}
					if (Config->ImprovedScheduler)
					{
						if (tileDistribution[k] < 0)
						{
							if (Config->Debug) fprintf(STD_OUT, "Tile %lld (m=%lld n=%lld) already processed, skipping\n", (long long int) k, (long long int) blockm, (long long int) blockn);
							continue;
						}
					}
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

				if (ImprovedSchedPhase1 && k >= nBlocks)
				{
endimprovedphase:			if (Config->Debug) fprintf(STD_OUT, "First improved scheduling phase ended\n");
					ImprovedSchedPhase1 = 0;
					k = nextk = 0;
					for (int l = 0;l < nDevices;l++)
					{
						next_device_k[l] = 0;
						forcePreparation[l] = 1;
					}
					goto restartkloop;
				}
				
				if (k < nBlocks)
				{
					if (Config->Debug) fprintf(STD_OUT, "Iteration k = %lld, m = %lld, n = %lld (device %d obuffer %d)\n", (long long int) k, (long long int) blockm, (long long int) blockn, use_device, j[use_device]);

					if (Config->MultiThreadDivide && Config->GPUMapping[use_device] != Config->PinMainThread && UseInputPthreads() && DGEMMTasks[use_device].thread_running)
					{
						DGEMMTasks[use_device].thread_running = 0;
						if (Config->Debug) fprintf(STD_OUT, "Waiting for divide thread for device %d (k=%lld lastk = %lld)\n", use_device, (long long int) k, lastk[use_device]);
						//if (pthread_mutex_lock(&DGEMMTasks[use_device].mutex_finished)) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);

						int tmpval = pthread_mutex_trylock(&DGEMMTasks[use_device].mutex_finished);
						if (tmpval == EBUSY)
						{
							int tmp_device = *(DGEMMTasks[use_device].next_device);
							if (tmp_device != use_device)
							{
								
								if (Config->Debug) fprintf(STD_OUT, "Divide thread waiting for wrong device, skipping device %d\n", *(DGEMMTasks[use_device].next_device));
								DGEMMTasks[*(DGEMMTasks[use_device].next_device)].skip_device_to = use_device;
								if (pthread_mutex_unlock(&DGEMMTasks[*(DGEMMTasks[use_device].next_device)].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
							}
							if (pthread_mutex_lock(&DGEMMTasks[use_device].mutex_finished)) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
						}
						else if (tmpval) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);


						if (Config->Debug) fprintf(STD_OUT, "Main thread: Divide thread for device %d finished\n", use_device);
					}

					DGEMMPrepareAndExecuteTask& Task = DGEMMTasks[use_device];
					Task.PrepareTasks[0].j = Task.PrepareTasks[1].j = -1;
					Task.device = use_device;
					Task.kernel_num = kernel_num;
					Task.k = k;
					Task.j = j[use_device];

					if (next_device_k[use_device] == 0 || obuffercount == 1 || Config->AsyncDMA == false || forcePreparation[use_device])
					{
						WaitForLASWP(blockm);
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
					if (obuffercount > 1 && (signed) lastk[use_device] != -1 && Config->AsyncDMA && k + (nDevices - use_device - 1) % nDevices + 1 < nBlocks && cpu_k_barrier_hit == false)
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
							WaitForLASWP(nextblockm);
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

					if (Config->MultiThreadDivide && Config->GPUMapping[use_device] != Config->PinMainThread && UseInputPthreads() && cpu_k_barrier_hit == false)
					{
						if (Config->Debug) fprintf(STD_OUT, "Starting PrepareAndExecute task on divide thread for device %d (k = %lld)\n", use_device, (long long int) k);
						if (pthread_mutex_unlock(&DGEMMTasks[use_device].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
						DGEMMTasks[use_device].thread_running = 1;
					}
					else
					{
						if (DGEMMPrepareAndExecute(Task)) return(1);
						//if (Config->MultiThreadDivide && UseInputPthreads() && pthread_mutex_unlock(&DGEMMTasks[use_device].mutex_finished)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
					}
				}
				if (obuffercount == 1)
				{
					oldj[use_device] = j[use_device];
					lastk[use_device] = k;
				}
				if ((obuffercount > 1) ? ((signed) lastk[use_device] != -1) : (k < nBlocks))
				{
					if (nBlocks <= k && (signed) lastk[use_device] < cpu_k_barrier && Config->MultiThreadDivide && Config->GPUMapping[use_device] != Config->PinMainThread && UseInputPthreads() && DGEMMTasks[use_device].thread_running)
					{
						DGEMMTasks[use_device].thread_running = 0;
						if (Config->Debug) fprintf(STD_OUT, "Waiting for divide thread for device %d (late phase, k=%lld lastk = %lld)\n", use_device, (long long int) k, lastk[use_device]);
						int tmpval = pthread_mutex_trylock(&DGEMMTasks[use_device].mutex_finished);
						if (tmpval == EBUSY)
						{
							int tmp_device = *(DGEMMTasks[use_device].next_device);
							if (tmp_device != use_device)
							{
								
								if (Config->Debug) fprintf(STD_OUT, "Divide thread waiting for wrong device (late phase), skipping device %d\n", *(DGEMMTasks[use_device].next_device));
								DGEMMTasks[*(DGEMMTasks[use_device].next_device)].skip_device_to = use_device;
								if (pthread_mutex_unlock(&DGEMMTasks[*(DGEMMTasks[use_device].next_device)].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
							}
							if (pthread_mutex_lock(&DGEMMTasks[use_device].mutex_finished)) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
						}
						else if (tmpval) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
					}
					size_t lastm, lastn;
					DGEMM_getblocks(lastk[use_device], lastm, lastn);
					int must_lock = 0;
					if (!Config->ThreadSaveDriver) for (int ii = 0;ii < nDevices;ii++) if (Config->GPUMapping[ii] != Config->PinMainThread)
					{
						must_lock = 1;
						break;
					}
					if ((signed long int) lastk[use_device] != -1 && lastk[use_device] < nBlocks)
					{
						if (WaitForEvent(oldj[use_device], use_device, must_lock)) return(1);
						if (Config->Debug) fprintf(STD_OUT, "Processing Output (Iteration %lld) for device %d tile %lld (m = %lld, n = %lld)\n", (long long int) k, use_device, (long long int) lastk[use_device], (long long int) lastm, (long long int) lastn);
						if (Config->ImplicitDriverSync == 0 && Config->DstMemory == 'g')
						{
							if (FetchResult(use_device, oldj[use_device], lastm, lastn)) {fprintf(STD_OUT, "Error copying from GPU\n");return(1);}
							if (WaitForEvent(oldj[use_device], use_device)) return(1);
						}
					}
					if (Config->VerboseTiming) Timers.CounterMerge.Start();

					if (k == nBlocks + 2 * nDevices - 1 || Config->MultiThread == false || UseOutputPthreads() == 0)
					{
						if (lastk[use_device] < nBlocks)
						{
							if (Config->Debug) fprintf(STD_OUT, "\tMerging buffer (device %d, obuffer %d, k = %lld, main thread)\n", use_device, oldj[use_device], (long long int) lastk[use_device]);
							if (RunMergeBuffers(C + lastn * Config->Height + lastm * C_pitch * Config->Height, use_device, oldj[use_device], (lastn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height, (lastm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height, BufferHeight, BufferHeight, C_pitch)) {fprintf(STD_OUT, "Error merging\n"); return(1);}
							if (Config->Debug) fprintf(STD_OUT, "Main thread unlocking obuffer mutex device %d obuffer %d\n", use_device, oldj[use_device]);
							if (Config->MultiThread && UseOutputPthreads() && pthread_mutex_unlock(&obufferMutex[use_device][oldj[use_device]])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
						}
						if (Config->MultiThread && UseOutputPthreads())
						{
							for (int l = 0;l < obuffercount;l++)
							{
								for (int ll = 0;ll < nDevices;ll++)
								{
									if ((ll != use_device || l != oldj[ll]) && (signed) lastk[ll] != -1)
									{
										if (Config->Debug) fprintf(STD_OUT, "Waiting to finish merge process for device %d obuffer %d\n", ll, l);
										if (pthread_mutex_lock(&obufferMutex[ll][l])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
										if (pthread_mutex_unlock(&obufferMutex[ll][l])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
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
						if (pthread_mutex_lock(&mParam[use_device][iMergeThread[use_device]].mergeThreadMutex[1])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
						if (Config->AsyncTiming)
						{
							Timers.ATime.Stop();
							if ((!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001) || Config->Debug) fprintf(STD_OUT, "\t\tWARNING: Wait Time for merge thread: %1.5lf\n", Timers.ATime.GetElapsedTime());
						}
						if (Config->Debug) fprintf(STD_OUT, "\t\tUnlocking outputthread mutex %d to process device %d obuffer %d\n", iMergeThread[use_device], use_device, oldj[use_device]);
						mParam[use_device][iMergeThread[use_device]].nContext = oldj[use_device];
						mParam[use_device][iMergeThread[use_device]].dst = C + (lastn * Config->Height + lastm * C_pitch * Config->Height);
						mParam[use_device][iMergeThread[use_device]].k = lastk[use_device];
						if (pthread_mutex_unlock(&mParam[use_device][iMergeThread[use_device]].mergeThreadMutex[0])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
						iMergeThread[use_device] = (iMergeThread[use_device] + 1) % outputthreads;
					}

					if (Config->VerboseTiming) Timers.CounterMerge.Stop();
				}
				oldj[use_device] = j[use_device];
				j[use_device] = (j[use_device] + 1) % obuffercount;
				lastk[use_device] = k;
				if (Config->MultiThread) use_device = (use_device + 1) % nDevices;
			}
			if(Config->Verify && i < Config->Iterations - 1) AnalyzeResults();
		}
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int l = 0;l < nDevices;l++)
			{
				int tmp = pthread_mutex_trylock(&DGEMMTasks[l].mutex_finished);
				if (tmp != 0 && tmp != EBUSY) fprintf(STD_OUT, "ERROR trylocking mutex (%d): %s - %d\n", l, __FILE__, __LINE__);
			}
			for (int l = 0;l < divideThreads;l++)
			{
				dParam[l].reset = 1;
				if (pthread_mutex_unlock(&DGEMMTasks[dParam[l].curDevice].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			}
		}

		if (Config->ImprovedScheduler)
		{
			delete[] tileDistribution;
		}
		for (int l = 0;l < nDevices;l++)
		{
			delete[] buffer_pointers_A[l];
			delete[] buffer_pointers_B[l];
		}
		if (RunCALDGEMM_Exit()) return(0);
	}
	Timers.GPUTimer.Stop();

	if (Config->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&oldcpumask));
	sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

	if (Config->UseCPU)
	{
		if (Config->MultiThread)
		{
			Timers.ATime.Reset();
			Timers.ATime.Start();
		}
		if (Config->Debug) fprintf(STD_OUT, "Waiting for CPU DGEMM to finish\n");
		if (pthread_mutex_lock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (Config->MultiThread)
		{
			Timers.ATime.Stop();
			cpu_wait_time = Timers.ATime.GetElapsedTime();
			if (!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() >= 0.15 && cParam.cblas_size > 0)
			{
			    fprintf(STD_OUT, "WARNING: CPU synchronisation took %2.4lf sec\n", Timers.ATime.GetElapsedTime());
			}
			else if (Config->Debug)
			{
			    fprintf(STD_OUT, "CPU synchronisation took %2.4lf sec\n", Timers.ATime.GetElapsedTime());
			}
		}
	}

//RunCALDGEMM_end:
	}
	if (Config->LinpackSwapN != NULL) *Config->LinpackSwapN = 0;
	outputthreads = old_outputthreads;

#ifndef NO_ASYNC_LINPACK
	if (!Config->UseCPU && ExecuteLinpackCallbacks)
#else
	if (ExecuteLinpackCallbacks)
#endif
	{
		if (!Config->Quiet) fprintf(STD_OUT, "No asynchronous processing of linpack functions possible, executing linpack callback functions\n");
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

	Timers.System.Stop();
	if (Config->Debug) fprintf(STD_OUT, "DGEMM Run Complete\n");

#ifdef TESTMODE
	print_submatrices(C, 12, 24, C_pitch, 1, 1, 1, 1);
#endif

	if (!Config->NoPerformanceWarnings && Config->UseCPU && Config->UseGPU && !CPUOnlyRun && fabs(Timers.TotalCPUTimer.GetElapsedTime() - Timers.GPUTimer.GetElapsedTime()) > 1.0)
	{
		fprintf(STD_OUT, "WARNING: Bad GPU / CPU Splitting: GPU Time: %2.4lf, CPU Time: %2.4lf (m = %lld, n = %lld)\n", Timers.GPUTimer.GetElapsedTime(), Timers.TotalCPUTimer.GetElapsedTime(), (long long int) Config->m, (long long int) Config->n);
	}
	displayMatrixTiming("caldgemm");
	A = orig_a;
	B = orig_b;
	C = orig_c;
	Config->m = orig_m;
	Config->n = orig_n;
	AnalyzeResults();
	if (Config->Verify) delete[] D;

	if (ExecuteLinpackCallbacks)
	{
		if (Timers.CPUTimer.GetElapsedTime() < 2.0)
		{
		    gpu_ratio_used = 1 - 0.6 * (1 - gpu_ratio_used);
		}
		if (ExecuteLinpackCallbacks >= 2 && Timers.GPUTimer.GetElapsedTime() - Timers.LinpackTimer1.GetElapsedTime() < 1.0)
		{
		    gpu_ratio_used = 1 - 0.6 * (1 - gpu_ratio_used);
		}
		const double tmpratio = cpu_wait_time > 0.15 ? 0.0 : 0.5;
		const double newratio = tmpratio * linpackGPURatios[ExecuteLinpackCallbacks] + (1.0 - tmpratio) * gpu_ratio_used;
		if (Config->Debug) fprintf(STD_OUT, "updating ratio table entry %d (old: %2.3lf, new: %2.3lf, factor: %2.3lf) => %2.3lf\n", ExecuteLinpackCallbacks, 100 * linpackGPURatios[ExecuteLinpackCallbacks], 100 * gpu_ratio_used, tmpratio, 100 * newratio);

		linpackGPURatios[ExecuteLinpackCallbacks] = newratio;
		linpackCPUDGEMMTime[ExecuteLinpackCallbacks] = Timers.CPUTimer.GetElapsedTime();
		linpackBcastTime[ExecuteLinpackCallbacks] = Timers.LinpackTimer2.GetElapsedTime();
		linpack_last_mn[ExecuteLinpackCallbacks] = (double) Config->m * (double) Config->n;
	}

	return(0);
}

int caldgemm::DGEMMPrepareAndExecute(caldgemm::DGEMMPrepareAndExecuteTask& Task)
{
	pthread_mutex_lock(&device_mutex[Task.device]);
	for (int l = 0;l < 2;l++)
	{
		if (Task.PrepareTasks[l].j != -1)
		{
			if (DGEMM_prepare(Task.PrepareTasks[l].k, Task.PrepareTasks[l].j, Task.device)) return(1);
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
		if (pthread_mutex_lock(&obufferMutex[Task.device][Task.j])) fprintf(STD_OUT, "ERROR locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (Config->AsyncTiming)
		{
			Timers.ATime.Stop();
			if ((!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001) || Config->Debug) fprintf(STD_OUT, "\t\tWait Time for output buffer: %1.5lf\n", Timers.ATime.GetElapsedTime());
		}
	}
	size_t blockm, blockn;
	DGEMM_getblocks(Task.k, blockm, blockn);

	if (buffer_pointers_A[Task.device][blockm] < 0 || buffer_pointers_B[Task.device][blockn] < 0)
	{
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING, Buffer falsified by previous iteration, need to retransfer\n");
		DGEMM_prepare(Task.k, Task.j, Task.device);
	}
	if (ExecuteKernels(Task, blockm, blockn)) return(1);
	pthread_mutex_unlock(&device_mutex[Task.device]);
	return(0);
}

int caldgemm::ExitCALDGEMM()
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "CALDGEMM not initialized, cannot uninitialize!\n");
		return(1);
	}
	if (Config->Debug) fprintf(STD_OUT, "Uninitializing CALDGEMM\n");
	if (ExitDevices()) return(1);
	if (Config->MultiThread && UseOutputPthreads())
	{
		for (int num_device = 0;num_device < nDevices;num_device++)
		{
			for (int i = 0;i < max_outputthreads;i++)
			{
				if (Config->Debug) fprintf(STD_OUT, "Trying to terminate merge slave %d\n", i);
				mParam[num_device][i].terminate = true;
				if (pthread_mutex_unlock(&mParam[num_device][i].mergeThreadMutex[0])) fprintf(STD_OUT, "ERROR unlocking mergemutex %d/1 to terminate slave\n", i);
			}
		}
	}

	ExitRuntime();

	if (Config->UseCPU && Config->UseGPU)
	{
		if (Config->Debug) fprintf(STD_OUT, "Trying to terminate blas slave\n");
		cParam.terminate = true;
		if (Config->MultiThread && pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "ERROR unlocking blas mutex 1 to terminate thread\n");
		if (pthread_mutex_unlock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "ERROR unlocking blas mutex 0 to terminate thread\n");
	}

	if (Config->MultiThread)
	{
		if (Config->Debug) fprintf(STD_OUT, "Trying to terminate linpack slave\n");
		linpackParameters.terminate = true;
		if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "ERROR unlocking blas mutex 0 to terminate thread\n");
		if (Config->Debug) fprintf(STD_OUT, "Waiting for linpack slave to terminate\n");
		while (pthread_mutex_trylock(&linpackParameters.linpackMutex[0]) != EBUSY) if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "ERROR unlocking blasMutex 1\n");

		if (UseOutputPthreads())
		{
			if (Config->Debug) fprintf(STD_OUT, "Waiting for merge threads to terminate\n");
			for (int i = 0;i < max_outputthreads;i++)
			{
				for (int num_device = 0;num_device < nDevices;num_device++)
				{
					while (pthread_mutex_trylock(&mParam[num_device][i].mergeThreadMutex[0]) != EBUSY) if (pthread_mutex_unlock(&mParam[num_device][i].mergeThreadMutex[0])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
					if (pthread_mutex_unlock(&mParam[num_device][i].mergeThreadMutex[0])) fprintf(STD_OUT, "ERROR unlocking mergeMutex %d/1\n", i);
				}
			}
		}
		if (Config->UseCPU && Config->UseGPU)
		{
			if (Config->Debug) fprintf(STD_OUT, "Waiting for blas threads to terminate\n");
			while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "ERROR unlocking blasMutex 1\n");
		}
		
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int i = 0;i < divideThreads;i++)
			{
				dParam[i].terminate = 1;
				pthread_mutex_unlock(&DGEMMTasks[dParam[i].curDevice].mutex_start);
				while (pthread_mutex_trylock(&DGEMMTasks[dParam[i].curDevice].mutex_start) != EBUSY) if (pthread_mutex_unlock(&DGEMMTasks[dParam[i].curDevice].mutex_start)) fprintf(STD_OUT, "ERROR unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			}
		}
	}
	

	for (int j = 0;j < 2;j++)
	{
		if (Config->UseCPU && Config->UseGPU) if (pthread_mutex_destroy(&cParam.cblasMutex[j])) fprintf(STD_OUT, "ERROR destroying blas mutex %d\n", j);
	}
	if (Config->MultiThread)
	{
		if (UseOutputPthreads())
		{
			for (int num_device = 0;num_device < nDevices;num_device++)
			{
				for (int i = 0;i < obuffercount;i++) if (pthread_mutex_destroy(&obufferMutex[num_device][i])) fprintf(STD_OUT, "ERROR destroying obuffermutex %d for device %d\n", i, num_device);
				for (int i = 0;i < max_outputthreads;i++) for (int j = 0;j < 2;j++) if (pthread_mutex_destroy(&mParam[num_device][i].mergeThreadMutex[j])) fprintf(STD_OUT, "ERROR destroying merge thread mutex %d/%d for device %d\n", i, j, num_device);
			}
		}
		if (pthread_mutex_destroy(&scheduleMutex)) fprintf(STD_OUT, "ERROR destroying schedule mutex\n");
		
		if (Config->MultiThreadDivide && UseInputPthreads())
		{
			for (int i = 0;i < nDevices;i++)
			{
				if (pthread_mutex_unlock(&DGEMMTasks[i].mutex_start)) fprintf(STD_OUT, "ERROR unlocking divide start mutex (%d)\n", i);
				if (pthread_mutex_unlock(&DGEMMTasks[i].mutex_finished)) fprintf(STD_OUT, "ERROR unlocking divide finished mutex (%d)\n", i);
				if (pthread_mutex_destroy(&DGEMMTasks[i].mutex_start)) fprintf(STD_OUT, "ERROR destroying divide start mutex (%d)\n", i);
				pthread_mutex_destroy(&DGEMMTasks[i].mutex_finished); //TODO, check why this fails
			}
		}
		if (Config->MultiThreadDivide && UseMutexPerDevice())
		{
			for (int i = 0;i < nDevices;i++)
			{
				pthread_mutex_destroy(&device_mutex[i]);
			}
		}
	}

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
}

#define MAX_HUGE_ADDRESSES 256
double* huge_page_addresses[MAX_HUGE_ADDRESSES];
int nHugeAddresses = 0;
#ifndef HUGE_PAGESIZE
#define HUGE_PAGESIZE (1024 * 2048)
#endif

double* caldgemm::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible, bool Cmatrix)
{
#ifndef USE_OLD_HUGE_MALLOC
	return((double*) qmalloc::qMalloc(nDoubles * sizeof(double), huge_pages, false, page_locked));
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

void caldgemm::FreeMemory(double* ptr, bool gpuaccessible)
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
}

void caldgemm::displayMatrixTiming(const char* name)
{
	double gflops_CPU = (double) 1e-09 * orig_m * orig_n * (2 * Config->Width + 2) * (double) Config->Iterations / Timers.System.GetElapsedTime();
	avggflops = ((double) avgngflops * avggflops + gflops_CPU) / (double) (avgngflops + 1);
	avgngflops++;
	if (!Config->Quiet || (Config->DisplayTiming /*&& Config->m * Config->n >= 16 * 24 * 1024 * 1024*/)) fprintf(STD_OUT, "%sProgram: %s Sizes - A: %lldx%lld B: %lldx%lld C:%lldx%lld (Host: %s) System Time %2.3lf System Gflops %2.3lf\n", Config->PreOut, name, 
		(long long int) orig_m, (long long int) Config->Width, (long long int) Config->Width, (long long int) orig_n, (long long int) orig_m, (long long int) orig_n, hostname, Timers.System.GetElapsedTime(), gflops_CPU);
	if (Config->UseCPU == true && Config->UseGPU == true)
	{
		double flopsc, flopsg;
		if (CPUOnlyRun)
		{
			flopsc = (double) 1e-09 * orig_m * orig_n * (2 * Config->Width + 2) * Config->Iterations / Timers.CPUTimer.GetElapsedTime();
			flopsg = 0.0;
		}
		else if (DGEMM_split_m)
		{
			flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Config->n + (Config->n % SmallTileHeight) * (Config->m - cParam.cblas_size) + cParam.dynamic_run2 * Config->Height * Config->Height + (ExecLinpack ? Config->Width * Config->n : 0)) * (2 * Config->Width + 2) * Config->Iterations / Timers.CPUTimer.GetElapsedTime();
			flopsg = (double) 1e-09 * ((Config->m - cParam.cblas_size) * (Config->n - Config->n % SmallTileHeight) - cParam.dynamic_run * cParam.dynamic_size - cParam.dynamic_run2 * Config->Height * Config->Height) * (2 * Config->Width + 2) * Config->Iterations / Timers.GPUTimer.GetElapsedTime();
		}
		else
		{
			flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Config->m + (Config->m % SmallTileHeight) * (Config->n - cParam.cblas_size) + cParam.dynamic_run2 * Config->Height * Config->Height + (ExecLinpack ? Config->Width * Config->n : 0)) * (2 * Config->Width + 2) * Config->Iterations / Timers.CPUTimer.GetElapsedTime();
			flopsg = (double) 1e-09 * ((Config->n - cParam.cblas_size) * (Config->m - Config->m % SmallTileHeight) - cParam.dynamic_run * cParam.dynamic_size - cParam.dynamic_run2 * Config->Height * Config->Height) * (2 * Config->Width + 2) * Config->Iterations / Timers.GPUTimer.GetElapsedTime();
		}
		
		if (Config->GPUClock && Config->m * Config->n >= 24 * 24 * 1024 * 1024 && flopsg <= (double) 460 * (double) Config->GPUClock / (double) 850 - (double) 20)
		{
			fprintf(STD_OUT, "%sThrottling: %s (%2.3lf GFlops)\n", Config->PreOut, hostname, flopsg);
		}

		const double gpu_ratio_used_new = flopsg / (flopsc * (Timers.System.GetElapsedTime() - Timers.LinpackTimer1.GetElapsedTime() - Timers.LinpackTimer2.GetElapsedTime() - Timers.LinpackTimer3.GetElapsedTime()) / Timers.System.GetElapsedTime() + flopsg);
		if (!Config->Quiet || (Config->DisplayTiming /*&& Config->m * Config->n >= 16 * 24 * 1024 * 1024*/))
		{
			char timingoutputbase[1024];
			char *timingoutput = timingoutputbase;
			timingoutput += sprintf(timingoutput, "%sGPU Time %2.4lf (%2.4lf Gflops)   CPU Time %2.4lf (%2.4lf Gflops)", Config->PreOut, Timers.GPUTimer.GetElapsedTime(), flopsg, Timers.CPUTimer.GetElapsedTime(), flopsc);
			if (ExecLinpack) timingoutput += sprintf(timingoutput, "   Linpack Time: %2.4lf (%d, %2.4lf, %2.4lf)  Total CPU Time: %2.4lf", Timers.LinpackTimer1.GetElapsedTime(), ExecLinpack, Timers.LinpackTimer2.GetElapsedTime(), Timers.LinpackTimer3.GetElapsedTime(), Timers.TotalCPUTimer.GetElapsedTime());
			if (Config->TabularTiming)
			{
				timingoutput += sprintf(timingoutput, " --- GPU Ratio - Real: %2.3lf Corrected: %2.3lf Guessed: %2.3lf , m*n: %.1E, CPU Wait Time: %2.3lf", (flopsg / (flopsc + flopsg)), gpu_ratio_used_new, gpu_ratio_used, (double) (Config->m * Config->n), cpu_wait_time);
			}
			sprintf(timingoutput, "\n");
			fwrite(timingoutputbase, 1, strlen(timingoutputbase), STD_OUT);
		}
		gpu_ratio_used = gpu_ratio_used_new;
	}
	if ((!Config->Quiet || (Config->DisplayTiming /*&& Config->n * Config->m >= 16 * 24 * 1024 * 1024*/)) && Config->VerboseTiming)
	{
		double gflops = (double)1e-09 * Config->m * Config->n * (2 * Config->Width - 1) * (double)Config->Iterations / Timers.Kernel.GetElapsedTime();
#ifdef CALDGEMM_BENCHMARK_KERNEL
		gflops *= (double) CALDGEMM_BENCHMARK_KERNEL;
#endif
		double copyto = Config->DivideToGPU ? 0 : ((double) 1e-09 * ((Config->Height * Timers.divideA + Config->Height * Timers.divideB) * Config->Width + Timers.divideC * Config->Height * Config->Height) * sizeof(double) * (double)Config->Iterations / Timers.CounterCopyTo.GetElapsedTime());
		double copyfrom = Config->DstMemory == 'g' ? ((double) 1e-09 * Config->m * Config->n * sizeof(double) * (double)Config->Iterations / Timers.CounterCopyFrom.GetElapsedTime()) : 0;
		double copyMerge = Config->MultiThread || UseOutputPthreads() == 0 ? 0 :((double) 1e-09 * Config->m * Config->n * sizeof(double) * (double)Config->Iterations / Timers.CounterMerge.GetElapsedTime());
		double copyDivide = UseInputPthreads() ? (double) 1e-09 * (Config->Height * Timers.divideA + Config->Height * Timers.divideB) * Config->Width * sizeof(double) * (double)Config->Iterations / Timers.CounterDivide.GetElapsedTime() : 0;
		fprintf(STD_OUT, "Times:  Kernel                    Divide (%d,%d)            Merge                   Copy To                 Copy From\n", Timers.divideA, Timers.divideB);
		fprintf(STD_OUT, "        %2.4lf (%2.4lf Gflops)  %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf Gb/s)\n", Timers.Kernel.GetElapsedTime(), gflops, Timers.CounterDivide.GetElapsedTime(), copyDivide, Timers.CounterMerge.GetElapsedTime(), copyMerge, Timers.CounterCopyTo.GetElapsedTime(), copyto, Timers.CounterCopyFrom.GetElapsedTime(), copyfrom);
		if (Config->TabularTiming)
		{
			fprintf(STD_OUT, "TIMES:\tw\t%lld\th\t%lld\tkernel\t%2.4lf\tdivide\t%2.4lf\tmerge\t%2.4lf\tcopyto\t%2.4lf\tcopyfr\t%2.4lf\n", (long long int) Config->Width, (long long int) Config->Height, gflops, copyDivide, copyMerge, copyto, copyfrom);
		}
	}
}

unsigned int caldgemm::AnalyzeResults()
{
	size_t errors = 0;
	size_t total = 0;

	if (Config->Verify)
	{
		if (!Config->Quiet) fprintf(STD_OUT, "Verifying results can take a long time on large matrices.\n");
		HighResTimer Timer;
		Timer.Reset();
		Timer.Start();
		cblas_dgemm(CblasRowMajor, TransposeA ? CblasTrans : CblasNoTrans, TransposeB ? CblasTrans : CblasNoTrans, Config->m, Config->n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, D, C_pitch);
		Timer.Stop();
		if (!Config->Quiet) fprintf(STD_OUT, "CPU Time: %lf Gflops: %lf\n", Timer.GetElapsedTime(), (double)1e-09 * 2 * Config->m * Config->n * Config->Width / Timer.GetElapsedTime());

		int nblocksm = Config->m / Config->Height + 1;
		int* errortiles = (int*) malloc((Config->n / Config->Height + 1) * nblocksm * sizeof(int));
		memset(errortiles, 0, (Config->n / Config->Height + 1) * nblocksm * sizeof(int));
		size_t errorsrel[3];
		memset(errorsrel, 0, 3 * sizeof(size_t));

		for (size_t i=0; i < Config->m; i++)
		{
			for (size_t j=0; j < Config->n; j++)
			{
				if (!isDoubleEqual(C[i * C_pitch + j],D[i * C_pitch + j]))
				{
					if (errors < 5) fprintf(STD_OUT, "Error found at row %lld, col %lld: Expected: %3.5le, Found: %3.5le, Diff: %3.5le, Relative: %3.5le\n", (long long int) i, (long long int) j, D[i * C_pitch + j], C[i * C_pitch + j], D[i * C_pitch + j] - C[i * C_pitch + j], (D[i * C_pitch + j] - C[i * C_pitch + j]) / D[i * C_pitch + j]);
					++errors;
					errortiles[j / Config->Height * nblocksm + i / Config->Height]++;
					if ((C[i * C_pitch + j] - D[i * C_pitch + j]) / D[i * C_pitch + j] > 0.05) errorsrel[0]++;
					else if ((C[i * C_pitch + j] - D[i * C_pitch + j]) / D[i * C_pitch + j] < 0.0001) errorsrel[2]++;
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
				fprintf(STD_OUT, "FAILED\n");
			}
		}
		else
		{
			fprintf(STD_OUT, "Passed!\n");
		}
		if (!Config->Quiet && (errors || Config->Debug))
		{
			fprintf(STD_OUT, "GPU output matrix\n");
			print_submatrices(C, Config->n, Config->m, C_pitch, 1, 1, Config->Height, Config->Height);
			fprintf(STD_OUT, "Reference matrix\n");
			print_submatrices(D, Config->n, Config->m, C_pitch, 1, 1, Config->Height, Config->Height, C);
		}

		if (!Config->Quiet && errors)
		{
			fprintf(STD_OUT, "Number of errors in tiles\n");
			for (size_t i = 0;i < Config->m;i += Config->Height)
			{
				for (size_t j = 0;j < Config->n;j += Config->Height)
				{
					fprintf(STD_OUT, "%8d\t", errortiles[j / Config->Height * nblocksm + i / Config->Height]);
				}
				fprintf(STD_OUT, "\n");
			}
		}

		free(errortiles);
	}

	return(errors == 0);
}

bool caldgemm::isDoubleEqual(double a, double b)
{
	double epsilon1 = 1e-6;
	double epsilon2 = 1e-4;

	if(fabs(b) <1e-13)
	{
		return (fabs(a-b) < epsilon1);
	}
	else
	{
		return (fabs((a-b)/b) < epsilon2);
	}
}

int caldgemm::DGEMM_prepare(size_t k, int j, unsigned int num_device)
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
		if (DGEMM_favor_m) buffersMajor[num_device] = blockm;
		else if (buffersSufficiant0)
		{
			const int buffer_pos = next_buffer_A[num_device] % (buffersSufficiant ? bbuffers[num_device] : 2);
			if (buffersMinor[num_device][next_buffer_A[num_device] % bbuffers[num_device]] != -1)
			{
				if (Config->Debug) fprintf(STD_OUT, "WARNING: Insufficient BBuffers, replacing blockm %d by %d in buffer %d\n", buffersMinor[num_device][buffer_pos], (int) blockm, buffer_pos);
				buffer_pointers_A[num_device][buffersMinor[num_device][buffer_pos]] = -1;
				
			}
			buffersMinor[num_device][buffer_pos] = blockm;
		}
		buffer_pointers_A[num_device][blockm] = next_buffer_A[num_device];
	}
	else if (Config->Debug) fprintf(STD_OUT, "\tSkipping preprocessing part of A (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);

	if (prepareN)
	{
		if (!DGEMM_favor_m) buffersMajor[num_device] = blockn;
		else if (buffersSufficiant0)
		{
			const int buffer_pos = next_buffer_B[num_device] % (buffersSufficiant ? bbuffers[num_device] : 2);
			if (buffersMinor[num_device][buffer_pos] != -1)
			{
				if (Config->Debug) fprintf(STD_OUT, "WARNING: Insufficient BBuffers, replacing blockn %d by %d in buffer %d\n", buffersMinor[num_device][buffer_pos], (int) blockn, buffer_pos);
				buffer_pointers_B[num_device][buffersMinor[num_device][buffer_pos]] = -1;
			}
			buffersMinor[num_device][buffer_pos] = blockn;
		}
		buffer_pointers_B[num_device][blockn] = next_buffer_B[num_device];
	}
	else if (Config->Debug) fprintf(STD_OUT, "\tSkipping preprocessing part of B (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);

	if(DGEMM_prepare_backend(k, j, num_device, prepareM, prepareN, buffersSufficiant, buffersSufficiant0)) return(1);

	if (prepareM) next_buffer_A[num_device]++;
	if (prepareN) next_buffer_B[num_device]++;

	return(0);
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
}

#endif

// vim: ts=4 sw=4 noet sts=4 tw=100
