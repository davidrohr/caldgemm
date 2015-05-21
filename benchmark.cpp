/**
 * Benchmark utility for CALDGEMM.
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

#ifdef CALDGEMM_OPENCL
#include "caldgemm_opencl.h"
#endif
#ifdef CALDGEMM_CAL
#include "caldgemm_cal.h"
#endif
#ifdef CALDGEMM_CUDA
#include "caldgemm_cuda.h"
#endif
#include "caldgemm_cpu.h"

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#else
#include "cmodules/pthread_mutex_win32_wrapper.h"
#endif
#include "cmodules/affinity.h"
#include "cmodules/qmath.h"

#define FASTRAND_THREADS_MAX 24

double *AA = NULL, *BB = NULL, *CC = NULL;
bool benchmark = false;
bool fastinit = false;
bool loadmatrix = false;
bool transa = false;
bool transb = false;
bool initialrun = true;
bool verifylarge = false;
bool quietbench = false;
bool alphaone = false;
bool betazero = false;
bool linpackmemory = false;
double* linpackmem = NULL;
int reduced_height = -1;
int reduced_width = -1;
int iterations = 1;
size_t pitch_a, pitch_b, pitch_c;
bool linpackpitch = false;
size_t height_a, height_b;
bool colmajor = false;

bool mem_page_lock = true;
bool mem_huge_table = false;
bool mem_gpu_access = false;
int linpack_callbacks = 0;

bool wait_key = false;

int use_opencl_not_cal = 0;

int random_seed = 0;

int torture = 0;

char* matrixfile;

long seedused;

caldgemm* dgemm_obj;

size_t matrix_m, matrix_n;

int MaxGPUTemperature = -1;

void PrintUsage()
{
	fprintf(STD_OUT,"Command Line Arguments\n");
	fprintf(STD_OUT, "\t-?        Display this help information\n");
	fprintf(STD_OUT, "\t-e        Verify Computational Correctness\n");
	fprintf(STD_OUT, "\t-q        Supress Display Output\n");
	fprintf(STD_OUT, "\t-a        Print the disassembled kernel image\n");
	fprintf(STD_OUT, "\t-i        Print IL Kernel used\n");
	fprintf(STD_OUT, "\t-if <int> Force DGEMM Kernel Variant\n");
	fprintf(STD_OUT, "\t-o  <c|g> Specify the output location, c = CPU, g = GPU, default GPU\n");
	fprintf(STD_OUT, "\t-I  <int> Set implicit driver sync\n");
	fprintf(STD_OUT, "\t-^  <int> Set DMA queue parameter\n");
	fprintf(STD_OUT, "\t-h  <int> block size for matrix multiply, default 4096\n");
	fprintf(STD_OUT, "\t-H  <int> Reduced block size for actual matrix multiply (buffer size given by -h)\n");
	fprintf(STD_OUT, "\t-w  <int> k for matrix multiply, default 1024\n");
	fprintf(STD_OUT, "\t-W  <int> reduced width, see H\n");
	fprintf(STD_OUT, "\t-l        Automatically select height for good performance\n");
	fprintf(STD_OUT, "\t-m  <int> m for matrix multiply, must be multiple of h, default 1024\n");
	fprintf(STD_OUT, "\t-n  <int> n for matrix multiply, must be multiple of h, default 1024\n");
	fprintf(STD_OUT, "\t-v        Verbose Synchronous Timing for Single Kernels / Transfers\n");
	fprintf(STD_OUT, "\t-k        Print Timing of Asynchronous DGEMM Operation\n");
	fprintf(STD_OUT, "\t-r  <int> Number of iterations to run the program (inside caldgemm)\n");
	fprintf(STD_OUT, "\t-R  <int> Number of iterations to run the program (seperate caldgemm calls)\n");
	fprintf(STD_OUT, "\t-y  <int> Force Device ID (-1 = all devices)\n");
	fprintf(STD_OUT, "\t-Y  <int> Use n devices\n");
	fprintf(STD_OUT, "\t-bb <int> Maxumum number of allowed bbuffers\n");
	fprintf(STD_OUT, "\t-d        Enable Debug Mode\n");
	fprintf(STD_OUT, "\t-z        Enable Multithreading\n");
	fprintf(STD_OUT, "\t-Z        Enable Multithreading for DivideBuffer\n");
	fprintf(STD_OUT, "\t-b        Enable Benchmarking\n");
	fprintf(STD_OUT, "\t-c        Use CPU\n");
	fprintf(STD_OUT, "\t-g        Use GPU\n");
	fprintf(STD_OUT, "\t-f        Fast Init (Empty Matrices)\n");
	fprintf(STD_OUT, "\t-j  <dbl> GPU to CPU ratio\n");
	fprintf(STD_OUT, "\t-jf <dbl> GPU to CPU ratio during factorization\n");
	fprintf(STD_OUT, "\t-jm <dbl> Max GPU to CPU ratio during autocalculation\n");
	fprintf(STD_OUT, "\t-jt <dbl> Margin time during auto calculation\n");
	fprintf(STD_OUT, "\t-js <dbl> Margin time during factorization\n");
	fprintf(STD_OUT, "\t-jl <dbl> Lookahead size modifier in ratio calculation\n");
	fprintf(STD_OUT, "\t-jp <int> Lookahead penalties\n");
	fprintf(STD_OUT, "\t-jq <dbl> Lookahead penalty factor\n");
	fprintf(STD_OUT, "\t-s        Dynamic CPU GPU scheduling\n");
	fprintf(STD_OUT, "\t-M        Disable third phase in dynamic scheduling\n");
	fprintf(STD_OUT, "\t-N        Disable second phase in dynamic scheduling\n");
	fprintf(STD_OUT, "\t-rr       Rereserve Linpack CPU after broadcast\n");
	fprintf(STD_OUT, "\t-p        Interleaving Memory Policy\n");
	fprintf(STD_OUT, "\t-u        Dump Test Matrix\n");
	fprintf(STD_OUT, "\t-1        Transpose A Matrix\n");
	fprintf(STD_OUT, "\t-2        Transpose B Matrix\n");
	fprintf(STD_OUT, "\t-3        Set alpha parameter to 1.0 to test optimized kernel\n");
	fprintf(STD_OUT, "\t-#        Set beta parameter to 0.0 to test optimized memcpy\n");
	fprintf(STD_OUT, "\t-5        Quiet Benchmark mode (different from quiet caldgemm mode)\n");
	fprintf(STD_OUT, "\t-6  <int> Set m/n to value * height\n");
	fprintf(STD_OUT, "\t-4  <int> Set m/n to the closest multiple of height to value\n");
	fprintf(STD_OUT, "\t-7        Verify Large Matrices\n");
	fprintf(STD_OUT, "\t-8        No initial run to negate cache effects\n");
	fprintf(STD_OUT, "\t-9        Output a table with timing information\n");
	fprintf(STD_OUT, "\t-0        Write the output of divideBuffers directly to GPU instead of a seperate DMA transfer\n");
	fprintf(STD_OUT, "\t-A        Do the DMA transfer to GPU asynchronously\n");
	fprintf(STD_OUT, "\t-Ap       Enable pipelined CALDGEMM operation\n");
	fprintf(STD_OUT, "\t-Aq <int> Position of middle marker of pipeline\n");
	fprintf(STD_OUT, "\t-L        Memory Organisation like in HPL (LINPACK)\n");
	fprintf(STD_OUT, "\t-C        Call fake LINPACK callback functions\n");
	fprintf(STD_OUT, "\t-Ca <int> Linpack Option: Set alternate lookahead threshold\n");
	fprintf(STD_OUT, "\t-Cm <int> Linpack Option: Minimize CPU part as soon as matrix size below threshold\n");
	fprintf(STD_OUT, "\t-P  <int> LDA=LDB=LDC = val for HPL like memory\n");
	fprintf(STD_OUT, "\t-T        Allocate Memory using Huge Tables\n");
	fprintf(STD_OUT, "\t-B        Keep DMA Buffers mapped during kernel execution\n");
	fprintf(STD_OUT, "\t-x <file> Load Matrix\n");
	fprintf(STD_OUT, "\t--  <int> Torture Test, n iterations\n");
	fprintf(STD_OUT, "\t-t  <int> Pin GPU thread to core n\n");
	fprintf(STD_OUT, "\t-ts       Show thread pinning\n");
	fprintf(STD_OUT, "\t-tc       Show CALDGEMM config\n");
	fprintf(STD_OUT, "\t-tr <int> Pin device runtime threads to code <int>, set -1 for all cores");
	fprintf(STD_OUT, "\t-K  <int> Pin GPU main thread to core n\n");
	fprintf(STD_OUT, "\t-Kb <int> Pin Broadcast thread to core n\n");
	fprintf(STD_OUT, "\t-Gx <int> Pin CPU threads of GPU x to same die as the CPU core id provided\n");
	fprintf(STD_OUT, "\t-Ux <int> Pin CPU postprocessing threads of GPU x to CPU core <int>, -1 = default mapping\n");
	fprintf(STD_OUT, "\t-UAx <int>Allocate memory for GPU x for die <int>, -1 = default mapping\n");
	fprintf(STD_OUT, "\t-UBx <int>Set DMA Mapping\n");
	fprintf(STD_OUT, "\t-V <int>  Thread save GPU driver (0: no (default), 1: yes, -1: use global lock)\n");
	fprintf(STD_OUT, "\t-S        Run on system with slow CPU\n");
	fprintf(STD_OUT, "\t-X        Advanced multi-GPU tiling scheduler\n");
	fprintf(STD_OUT, "\t-Xb <int> Balancing mode for improved scheduler\n");
	fprintf(STD_OUT, "\t-E <int>  Define random seed (0 for time)\n");
	fprintf(STD_OUT, "\t-O <int>  Backend to use: 0 = CAL, 1 = OpenCL, 2 = CUDA, 3 = CPUOnly\n");
	fprintf(STD_OUT, "\t-Oc <int> Set GPU_C parameter\n");
	fprintf(STD_OUT, "\t-Ol lib   Set library name used to obtain OpenCL DGEMM kernel\n");
	fprintf(STD_OUT, "\t-Oe       Do not allow multiple concurrent OpenCL kernels\n");
	fprintf(STD_OUT, "\t-Oq       Use simple GPU Queuing\n");
	fprintf(STD_OUT, "\t-Op <int> Preallocate buffers for at max <int> blocks (nb/mb)\n");
	fprintf(STD_OUT, "\t-Oa       Create async side queues and use such a queue to test a single-tile dgemm\n");
	fprintf(STD_OUT, "\t-Od       Use async side queue to offload DTRSM as well\n");
	fprintf(STD_OUT, "\t-Ox       Do not put the CPU in the OpenCL context\n");
	fprintf(STD_OUT, "\t-Ot       Use 3rdPartyTranspose kernel\n");
	fprintf(STD_OUT, "\t-F <int>  OpenCL Platform ID to use\n");
	fprintf(STD_OUT, "\t-Fc       Allow CPU device as OpenCL device\n");
	fprintf(STD_OUT, "\t-J <int>  Allow small tiles to process the remainder on GPU (0 disable, 1 enable, 2 auto)\n");
	fprintf(STD_OUT, "\t-Q        Wait for pressing a key before exiting\n");
	fprintf(STD_OUT, "\t-!        Do not use page locked memory\n");
	fprintf(STD_OUT, "\t-_        Allocate memory using the GPU runtime library (e.g. OpenCL)\n");
	fprintf(STD_OUT, "\t-= <int>  Define number of output threads\n");
	fprintf(STD_OUT, "\t-%%        Skip CPU Pre- and Postprocessing\n");
	fprintf(STD_OUT, "\t-@ <list> Comma or Semicolon separated list of CPU cores to exclude\n");
	fprintf(STD_OUT, "\t-.        Repin Main Thread During Active Wait for GPU Event\n");
	fprintf(STD_OUT, "\t-~        Always repin main thread\n");
	fprintf(STD_OUT, "\t-, <int>  Sleep for n usec during active wait\n");
	fprintf(STD_OUT, "\t-:        Enable NUMA Pinning\n");
	fprintf(STD_OUT, "\t-/ <list> Comma or Semicolon separated list of GPU devices to use (replaces -y for multiple devices)\n");
	fprintf(STD_OUT, "\t-* <int>  Enable Parallel DMA option if n >= <int>\n");
	fprintf(STD_OUT, "\t-[ <int>  Enable Grouped Parallel DMA option if n < <int>\n");
	fprintf(STD_OUT, "\t-] <int>  Maximum allowed GPU temperature (check applied after one caldgemm iteration, meaningfull in combination with -R)\n");
	//available: -D 
}

void linpack_fake1() {fprintf(STD_OUT, "Linpack fake 1 called\n");}
void linpack_fake2() {fprintf(STD_OUT, "Linpack fake 2 called\n");}
void linpack_fake3() {fprintf(STD_OUT, "Linpack fake 3 called\n");}

int ParseCommandLine(unsigned int argc, char* argv[], caldgemm::caldgemm_config* Config)
{
	Config->Quiet = false;
#ifndef TEST_PARAMETERS
	Config->Verify = false;
	Config->MemPolicy = false;
	Config->Disassemble = false;
	Config->PrintILKernel = false;
	Config->MultiThread = false;
	Config->MultiThreadDivide = false;
	//Config->DeviceNum = 0;
	//Config->Width = 1024;
	//Config->Height = 4096;
	Config->AutoHeight = false;
	Config->DynamicSched = false;
	Config->VerboseTiming = false;
	Config->TabularTiming = false;
	Config->Debug = false;
	matrix_m = matrix_n = 4096;
	Config->Iterations = 1;
	//Config->DstMemory = 'g';
	Config->UseCPU = Config->UseGPU = false;
	//Config->GPURatio = -1;
	Config->DumpMatrix = false;
	Config->DivideToGPU = false;
	Config->AsyncDMA = false;
	Config->KeepBuffersMapped = false;
#endif

	const int max_devices = caldgemm::max_devices;
#define CALDGEMM_PARAMETERS_BENCHMARK
#include "caldgemm_parse_parameters.h"
#undef CALDGEMM_PARAMETERS_BENCHMARK

	if (!quietbench) fprintf(STD_OUT, "Use -? for help\n");
	if (Config->UseCPU == false && Config->UseGPU == false) Config->UseGPU = true;

	return(0);
}

int fastrand_seed;
volatile int fastrand_done[FASTRAND_THREADS_MAX];
double* fastrand_A;
size_t fastrand_size;
int nfastmatthreads;

void* fastmatgen_slave(void* arg)
{
	int num = (int) (size_t) arg;

	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(num, &mask);
	sched_setaffinity(0, sizeof(cpu_set_t), &mask);

	size_t fastrand_num = fastrand_seed + 65537 * num;
	const size_t fastrand_mul = 84937482743;
	const size_t fastrand_add = 138493846343;
	const size_t fastrand_mod = 538948374763;
	
	size_t sizeperthread = fastrand_size / nfastmatthreads;

	double* A = fastrand_A + num * sizeperthread;
	size_t size = (num == nfastmatthreads - 1) ? (fastrand_size - (nfastmatthreads - 1) * sizeperthread) : sizeperthread;

	for (size_t i = 0;i < size;i++)
	{
		double randval = 0;
		for (int k = 0;k < 100;k++)
		{
			fastrand_num = (fastrand_num * fastrand_mul + fastrand_add) % fastrand_mod;
			randval += (double) -0.5 + (double) fastrand_num / (double)fastrand_mod;
		}
		A[i] = randval;
	}

	fastrand_done[num] = 1;
	return(NULL);
}

void fastmatgen(int SEED, double* A, size_t size)
{
	fastrand_seed = SEED;
	fastrand_A = A;
	fastrand_size = size;
	
#ifdef _WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );
	nfastmatthreads = sysinfo.dwNumberOfProcessors;
#else
	nfastmatthreads = sysconf(_SC_NPROCESSORS_CONF);
#endif
	if (nfastmatthreads < 1) nfastmatthreads = 1;
	if (nfastmatthreads > FASTRAND_THREADS_MAX) nfastmatthreads = FASTRAND_THREADS_MAX;
	
	memset((void*) fastrand_done, 0, nfastmatthreads * sizeof(int));

	cpu_set_t oldmask;
	sched_getaffinity(0, sizeof(cpu_set_t), &oldmask);

	for (int i = 0;i < nfastmatthreads - 1;i++)
	{
		pthread_t thr;
		pthread_create(&thr, NULL, fastmatgen_slave, (void*) (size_t) i);
	}
	fastmatgen_slave((void*) (size_t) (nfastmatthreads - 1));

	for (int i = 0;i < nfastmatthreads;i++)
	{
		while (fastrand_done[i] == 0) {}
	}
	sched_setaffinity(0, sizeof(cpu_set_t), &oldmask);
}

void SetupUserDataC(caldgemm::caldgemm_config &Config)
{
	if (fastinit || torture)
	{
		if (torture) memset(CC, 0, matrix_m * pitch_c * sizeof(double));
	}
	else
	{
		for (size_t i = 0;i < matrix_m;i++)
		{
			for (size_t j = 0;j < matrix_n;j++)
			{
#ifdef TESTMODE
				CC[i * pitch_c + j] = 0;
#else
				CC[i * pitch_c + j] = (double) ((i + j) % 16);
#endif
			}
		}
	}
}

int SetupUserData(caldgemm::caldgemm_config &Config)
{
#ifdef _WIN32
	LARGE_INTEGER randtime;
	QueryPerformanceCounter(&randtime);
	srand((int) randtime.LowPart);
#else
	timespec randtime;
	if (random_seed == 0)
	{
		clock_gettime(CLOCK_REALTIME, &randtime);
	}
	else
	{
		randtime.tv_nsec = random_seed;
	}
	srand((int) (seedused = randtime.tv_nsec));
#endif
	size_t width_a, width_b;

	if (linpackmemory)
	{
		if (transa || transb) fprintf(STD_OUT, "WARNING: Transposed not supported in linpackmem-mode, disabling !!!\n");
		transa = transb = false;
		if (linpackmem) delete[] linpackmem;

		if (linpackpitch)
		{
			if (pitch_c < matrix_m + Config.Width)
			{
				fprintf(STD_OUT, "Pitch too small\n");
				return(1);
			}
			pitch_a = pitch_b = pitch_c;
		}
		else
		{
			pitch_c = matrix_m + Config.Width;
			if (pitch_c % 8)
			{
				pitch_c += 8;
				pitch_c -= pitch_c % 8;
			}
			pitch_a = pitch_b = pitch_c;
		}
		size_t memsize = pitch_c * (matrix_n + Config.Width + 1) + 16;
		fprintf(stderr, "Allocating %lld KB...", (long long int) (memsize * 8 / 1024));
		linpackmem = dgemm_obj->AllocMemory(memsize, mem_page_lock, mem_huge_table, mem_gpu_access);
		if (linpackmem == NULL) {fprintf(STD_OUT, "Memory Allocation Error\n"); return(1);}

		char* linpackmem2 = (char*) linpackmem;
		if ((size_t) linpackmem2 % 64) linpackmem2 += 64 - ((size_t) linpackmem2) % 64;
		double* linpackmem3 = (double*) linpackmem2;
		
		colmajor = true;

		AA = linpackmem3 + Config.Width;
		BB = linpackmem3 + Config.Width * pitch_c;
		CC = linpackmem3 + Config.Width * (pitch_c + 1);
		
		width_a = Config.Width;
		height_a = matrix_m;
		width_b = matrix_n;
		height_b = Config.Width;
	}
	else
	{
		if (transa)
		{
			pitch_a = matrix_m;
			height_a = Config.Width;
			width_a = matrix_m;
		}
		else
		{
			pitch_a = Config.Width;
			height_a = matrix_m;
			width_a = Config.Width;
		}
		if (pitch_a % 8) pitch_a += (8 - pitch_a % 8);
		if (((pitch_a / 8) & 1) == 0)
		{
			pitch_a += 8;
		}
		if (transb)
		{
			pitch_b = Config.Width;
			height_b = matrix_n;
			width_b = Config.Width;
		}
		else
		{
			height_b = Config.Width;
			pitch_b = matrix_n;
			width_b = matrix_n;
		}
		if (pitch_b % 8) pitch_b += (8 - pitch_b % 8);
		if (((pitch_b / 8) & 1) == 0)
		{
			pitch_b += 8;
		}
		pitch_c = matrix_n;
		if (pitch_c % 8) pitch_c += (8 - pitch_c % 8);
		if (matrix_n % 8) fprintf(STD_OUT, "Padding 8 bytes for correct alignment of B, n = %lld, pitch = %lld\n", (long long int) matrix_n, (long long int) pitch_b);
		if (((pitch_c / 8) & 1) == 0)
		{
			pitch_c += 8;
		}

		if (AA) dgemm_obj->FreeMemory(AA, mem_gpu_access);
		//if (BB) dgemm_obj->FreeMemory(BB, mem_gpu_access);
		//if (CC) dgemm_obj->FreeMemory(CC, mem_gpu_access);
		if (!quietbench) fprintf(stderr, "...alloc A (%lld KB) B (%lld KB) C (%lld KB)...", (long long int) (height_a * pitch_a * sizeof(double) / 1024), (long long int) (height_b * pitch_b * sizeof(double)  / 1024), (long long int) (matrix_m * pitch_c * sizeof(double)  / 1024));
		AA = dgemm_obj->AllocMemory(height_a * pitch_a + height_b * pitch_b + matrix_m * pitch_c, mem_page_lock, mem_huge_table, mem_gpu_access);
		BB = AA + height_a * pitch_a;
		CC = BB + height_b * pitch_b;

		if (AA == NULL || BB == NULL || CC == NULL)
		{
			fprintf(STD_OUT, "Memory allocation error allocating matrices\n");
			return(1);
		}
	}

#ifdef TESTMODE
	for (unsigned int i = 0;i < height_a;i++)
	{
		for (unsigned int j = 0;j < width_a;j++)
		{
			AA[i * pitch_a + j] = i;
		}
	}
	for (unsigned int i = 0;i < height_b;i++)
	{
		for (unsigned int j = 0;j < width_b;j++)
		{
			BB[i * pitch_b + j] = j;
		}
	}
#else
	if (fastinit)
	{
		//memset(AA, 0, height_a * pitch_a * sizeof(double));
		//memset(BB, 0, height_b * pitch_b * sizeof(double));
	}
	else
	{
		if (!quietbench) fprintf(stderr, "...init A");
		fastmatgen(rand() * 100, AA, (colmajor ? width_a : height_a) * pitch_a);
		if (!quietbench) fprintf(stderr, "...init B");
		fastmatgen(rand() * 100, BB, (colmajor ? width_b : height_b) * pitch_b);
	}
#endif
	if (Config.Debug) fprintf(STD_OUT, "User Data Initialized\n");
	if (!quietbench) fprintf(stderr, "...");
	return(0);
}

bool isDoubleEqual(double a, double b)
{
	if (!qIsFinite(a) || !qIsFinite(b)) return(false);
	double valmax = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
	if (valmax < 1e-10)
	{
		return(fabs(a - b) < 1e10);
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

int main(int argc, char** argv)
{
	setUnknownNames("Unknown - Before Main");
	caldgemm::caldgemm_config Config;

	if (ParseCommandLine(argc, argv, &Config))
	{
		fprintf(STD_OUT, "Error parsing command line options\n");
		return 1;
	}

#ifdef CALDGEMM_CAL
	if (use_opencl_not_cal == 0)
	{
		dgemm_obj = new caldgemm_cal;
	} else
#endif
#ifdef CALDGEMM_OPENCL
	if (use_opencl_not_cal == 1)
	{
		dgemm_obj = new caldgemm_opencl;
	} else
#endif
#ifdef CALDGEMM_CUDA
	if (use_opencl_not_cal == 2)
	{
		dgemm_obj = new caldgemm_cuda;
	} else
#endif
	{
		dgemm_obj = new caldgemm_cpu;
	}

	if (dgemm_obj == NULL)
	{
		fprintf(STD_OUT, "Error creating caldgem object\n");
		return(1);
	}
	Config.config_backend = dgemm_obj->create_caldgemm_config_backend();
	Config.InitializeBackendOptions();

	if (dgemm_obj->InitCALDGEMM(&Config))
	{
		fprintf(STD_OUT, "Error initializing CALDGEMM\n");
		return(1);
	}

	if (reduced_height != -1)
	{
		fprintf(STD_OUT, "Using partial buffers %d / %lld\n", reduced_height, (long long int) Config.Height);
		Config.Height = reduced_height;
	}
	if (reduced_width != -1)
	{
		fprintf(STD_OUT, "Using partial buffer width %d / %lld\n", reduced_width, (long long int) Config.Width);
		Config.Width = reduced_width;
	}

#ifndef TEST_PARAMETERS
	if (loadmatrix)
	{
		FILE* fp;
		double* a, b, c;
		double alpha, beta;
		int tmp_m, tmp_k, tmp_n;
		int Apitch, Bpitch, Cpitch;
		size_t nread;

		if ((fp = fopen(matrixfile, "rb")) == NULL)
		{
			fprintf(STD_OUT, "Error opening matrix dump\n");
			return(1);
		}
		nread = fread(&a, sizeof(a), 1, fp);
		nread += fread(&b, sizeof(b), 1, fp);
		nread += fread(&c, sizeof(c), 1, fp);
		nread += fread(&alpha, sizeof(alpha), 1, fp);
		nread += fread(&beta, sizeof(beta), 1, fp);
		nread += fread(&tmp_m, sizeof(tmp_m), 1, fp);
		nread += fread(&tmp_k, sizeof(tmp_k), 1, fp);
		nread += fread(&tmp_n, sizeof(tmp_n), 1, fp);
		nread += fread(&Apitch, sizeof(Apitch), 1, fp);
		nread += fread(&Bpitch, sizeof(Bpitch), 1, fp);
		nread += fread(&Cpitch, sizeof(Cpitch), 1, fp);

		Apitch = 1536;

		AA = new double[(size_t) tmp_m * (size_t) Apitch];
		BB = new double[(size_t) tmp_k * (size_t) Bpitch];
		CC = new double[(size_t) tmp_m * (size_t) Cpitch];

		for (int i = 0;i < tmp_m;i++)
		{
			nread += fread(AA + i * Apitch, tmp_k, sizeof(double), fp);
		}
		for (int i = 0;i < tmp_k;i++)
		{
			nread += fread(BB + i * Bpitch, tmp_n, sizeof(double), fp);
		}
		fclose(fp);
		if (nread == 0)
		{
			fprintf(STD_OUT, "Error Reading matrix file");
			return(1);
		}
		memset(CC, 0, (size_t) tmp_m * (size_t) Cpitch * sizeof(double));

		fprintf(STD_OUT, "matrix loaded: m=%d k=%d n=%d lda=%d ldb=%d ldc=%d alpha=%2.4lf beta=%2.4lf\n", tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch, alpha, beta);

		dgemm_obj->RunCALDGEMM(AA, BB, CC, alpha, beta, tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch);
	}
	else
	{
		if (!quietbench)
		{
			fprintf(stderr, "Initializing Data... ");
		}
		if (SetupUserData(Config))
		{
			return(1);
		}
		if (!quietbench)
		{
			fprintf(stderr, "Done\n");
		}

		//Initial run to negate cache effects
#ifndef TESTMODE
#ifndef DEBUG_MSG_TIMED
		if (Config.Debug == false && Config.DumpMatrix == false && initialrun && !torture)
#endif
		{
			if (!quietbench)
			{
				fprintf(stderr, "Doing initial run... ");
			}
			bool tmpquiet = Config.Quiet, tmpverify = Config.Verify;
			unsigned int tmpiter = Config.Iterations;
			unsigned int tmpm = matrix_m, tmpn = matrix_n, tmpdebug = Config.Debug;
			unsigned int tmpshowpin = Config.ShowThreadPinning;
			unsigned int tmpautoheight = Config.AutoHeight;
			Config.ShowThreadPinning = 0;
			Config.Quiet = true;
			Config.Verify = false;
			Config.Iterations = 1;
			Config.Debug = false;
			Config.AutoHeight = 0;
			if (matrix_m > 2 * Config.Height) matrix_m = 2 * Config.Height;
			else if (matrix_m % Config.Height) matrix_m -= matrix_m % Config.Height;
			if (matrix_n > 2 * Config.Height) matrix_n = 2 * Config.Height;
			else if (matrix_n % Config.Height) matrix_n -= matrix_n % Config.Height;
			if (dgemm_obj->RunCALDGEMM(AA, BB, CC, alphaone ? 1.0 : 0.5, 1.0, matrix_m, Config.Width, matrix_n, pitch_a, pitch_b, pitch_c, colmajor, transa, transb))
			{
				fprintf(STD_OUT, "Error running CALDGEMM\nexiting\n");
				return(1);
			}
			if (Config.PipelinedOperation) dgemm_obj->FinishCALDGEMM();
			matrix_m = tmpm;
			matrix_n = tmpn;
			Config.AutoHeight = tmpautoheight;
			Config.Quiet = tmpquiet;
			Config.Verify = tmpverify;
			Config.Iterations = tmpiter;
			Config.Debug = tmpdebug;
			Config.ShowThreadPinning = tmpshowpin;
			if (!quietbench)
			{
				fprintf(STD_OUT, "Done\n");
			}
		}
#endif
		if (!quietbench)
		{
			fprintf(STD_OUT, "Initializing Matrix C\n");
		}
		SetupUserDataC(Config);
		dgemm_obj->ResetTimers();
		if (!quietbench)
		{
			fprintf(STD_OUT, "Running Benchmark\n");
		}
		do
		{
			double *org_AA = AA, *org_BB = BB, *org_CC = CC;
			size_t org_m = matrix_m, org_n = matrix_n;
			for (int iter = 0;iter < iterations;iter++)
			{
				if (iterations > 1 && !quietbench) fprintf(STD_OUT, "\nDGEMM Call Iteration %d\n\n", iter);
#ifdef TESTMODE
				double use_alpha = 1.0;
				double use_beta = 0.0;
#else
				double use_alpha = alphaone ? 1.0 : -1.0;
				double use_beta = betazero ? 0.0 : 1.0;
#endif
				size_t tmpn = matrix_m > matrix_n ? matrix_m : matrix_n;
				if (linpack_callbacks) Config.LinpackSwapN = &tmpn;
				if (Config.AsyncSideQueue)
				{
					if (dgemm_obj->RunAsyncSingleTileDGEMM(AA, BB, CC, use_alpha, use_beta, matrix_m, Config.Width, matrix_n, pitch_a, pitch_b, pitch_c, colmajor, transa, transb))
					{
						fprintf(STD_OUT, "Error running async CALDGEMM");
						return(1);
					}
				}
				else
				{
					if (dgemm_obj->RunCALDGEMM(AA, BB, CC, use_alpha, use_beta, matrix_m, Config.Width, matrix_n, pitch_a, pitch_b, pitch_c, colmajor, transa, transb, linpack_callbacks))
					{
						fprintf(STD_OUT, "Error running CALDGEMM\n");
						return(1);
					}
					
					if (Config.PipelinedOperation && iter == iterations - 1)
					{
						fprintf(STD_OUT, "Pipelined run issued, waiting for result\n");
						dgemm_obj->FinishCALDGEMM();
					}
				}
				if (MaxGPUTemperature > 0)
				{
					int tmpVal = (int) dgemm_obj->getMaxGPUTemperature();
					if (tmpVal > MaxGPUTemperature && tmpVal < 500)
					{
						fprintf(STD_OUT, "Maximum GPU Temperature of %d exceeded, temperature is %d\n", MaxGPUTemperature, tmpVal);
						return(1);
					}
				}
				fflush(STD_OUT);
				if (torture)
				{
					dgemm_obj->RunCALDGEMM(AA, BB, CC, 1.0, 1.0, matrix_m, Config.Width, matrix_n, pitch_a, pitch_b, pitch_c, colmajor, transa, transb, linpack_callbacks);
					if (Config.PipelinedOperation) dgemm_obj->FinishCALDGEMM();
				}
				
				if (linpackmemory && iterations > 1)
				{
					if (matrix_m > Config.Width && matrix_n > Config.Width)
					{
						AA += Config.Width;
						BB += Config.Width * pitch_c;
						CC += Config.Width * (pitch_c + 1);
						matrix_m -= Config.Width;
						matrix_n -= Config.Width;
					}
					else
					{
						AA = org_AA;
						BB = org_BB;
						CC = org_CC;
						matrix_m = org_m;
						matrix_n = org_n;
					}
				}
			}
		} while (benchmark && (matrix_n += Config.Height) < 70000 && (matrix_m += Config.Height) < 70000 && SetupUserData(Config) == 0);
		
	}
	
	if (torture)
	{
		for (size_t i = 0;i < matrix_m * pitch_c;i++)
		{
			if (CC[i] > 10E-10)
			{
				fprintf(STD_OUT, "Torture Test FAILED\n");
				if (!quietbench) fprintf(STD_OUT, "Entry %lld is %lf\n", (long long int) i, CC[i]);
				torture = 0;
				break;
			}
		}
		if (torture) fprintf(STD_OUT, "Torture Test PASSED (%2.3lf gflops)\n", dgemm_obj->avggflops);
	}

	if (verifylarge)
	{
		fprintf(STD_OUT, "Running verification for large matrices\n");
		srand((int) seedused);
		Config.UseGPU = false;
		Config.UseCPU = true;
		Config.Verify = false;
		Config.Quiet = true;
		dgemm_obj->RunCALDGEMM(AA, BB, CC, alphaone ? -1.0 : 1.0, 1.0, matrix_m, Config.Width, matrix_n, pitch_a, pitch_b, pitch_c, colmajor, transa, transb);
		if (Config.PipelinedOperation) dgemm_obj->FinishCALDGEMM();
		fprintf(STD_OUT, "CPU DGEMM Comparison run complete, comparing results\n");
		int verifyok = 1;
		for (size_t i = 0;i < matrix_m;i++)
		{
			for (size_t j = 0;j < matrix_n;j++)
			{
				if (!isDoubleEqual(CC[i * pitch_c + j] * 1.0, (double) ((i + j) % 16)))
				{
					fprintf(STD_OUT, "Verification failed at i = %lld, j = %lld (%e %e)\n", (long long int) i, (long long int) j, CC[i * pitch_c + j] * 1.0, (double) ((i + j) % 16));
					verifyok = 0;
					static int ii = 0;
					if (++ii > 1) break;
				}
			}
			if (!verifyok) break;
		}
		if (verifyok) fprintf(STD_OUT, "Verification succeeded\n");
	}
#else //TEST_PARAMETERS
	char* mem = new char[(size_t) 40 * 1024 * 1024 * 1024];
	{
		size_t tmpmem = (size_t) mem;
		fprintf(STD_OUT, "tmpmem = 0x%llx\n", tmpmem);
		tmpmem += (size_t) 1024 * 1024 * 1024;
		fprintf(STD_OUT, "tmpmem = 0x%llx\n", tmpmem);
		tmpmem -= ((size_t) tmpmem) % ((size_t) 1024 * 1024 * 1024);
		fprintf(STD_OUT, "tmpmem = 0x%llx\n", tmpmem);
		AA = (double*) tmpmem;
		tmpmem += (size_t) 10 * 1024 * 1024 * 1024;
		BB = (double*) tmpmem;
		tmpmem += (size_t) 10 * 1024 * 1024 * 1024;
		CC = (double*) tmpmem;

		AA = (double*) (((size_t) AA) | ((size_t) 0x6ea040));
		BB = (double*) (((size_t) BB) | ((size_t) 0xeec080));
		CC = (double*) (((size_t) CC) | ((size_t) 0x495040));
		double ALPHA = -1.0;
		double BETA = 1.0;
		size_t M = 3072, N = 3072, K = 1024;
		size_t APITCH = 4104, BPITCH = 3072, CPITCH = 4104;
		bool ORDER = true;
		bool TRANSA = false, TRANSB = true;
		fprintf(STD_OUT, "Filling Source Matrices with random data\n");
		for (int i = 0;i < APITCH * (M > K ? M : K);i++) AA[i] = i % 257;
		for (int i = 0;i < BPITCH * (N > K ? N : K);i++) BB[i] = i % 97;
		for (int i = 0;i < CPITCH * (M > N ? M : N);i++) CC[i] = i % 65537;

		fprintf(STD_OUT, "Running with caldgemm parameters: A=0x%llx, B=0x%llx, C=0x%llx, ALPHA=%2.4lf, BETA=%2.4lf, m=%lld, k=%lld, n=%lld, Apitch=0x%llx, Bpitch=0x%llx, Cpitch=0x%llx, ColMajor=%d, TransA=%d, TransB=%d\n", AA, BB, CC, ALPHA, BETA, M, K, N, APITCH, BPITCH, CPITCH, (int) (ORDER == CblasColMajor), (int) (TRANSA == CblasTrans), (int) (TRANSB == CblasTrans));
		dgemm_obj->RunCALDGEMM(AA, BB, CC, ALPHA, BETA, M, K, N, APITCH, BPITCH, CPITCH, ORDER, TRANSA, TRANSB);
		fprintf(STD_OUT, "Caldgemm run complete\n");

		delete[] mem;
	}
#endif //TEST_PARAMETERS

#ifndef TEST_PARAMETERS
	if (linpackmemory)
	{
		dgemm_obj->FreeMemory(linpackmem, mem_gpu_access);
	}
	else
	{
		dgemm_obj->FreeMemory(AA, mem_gpu_access);
		//dgemm_obj->FreeMemory(BB, mem_gpu_access);
		//dgemm_obj->FreeMemory(CC, mem_gpu_access);
	}
#endif

	dgemm_obj->ExitCALDGEMM();
	delete dgemm_obj;

	if (wait_key)
	{
		fprintf(STD_OUT, "Press return to exit!\n");
		getchar();
	}
	return 0;
}
