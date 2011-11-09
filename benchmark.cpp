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

#if !defined(CALDGEMM_CAL) & !defined(CALDGEMM_OPENCL)
#error Please enable either CAL or OpenCL!
#endif

#ifdef CALDGEMM_OPENCL
#include "caldgemm_opencl.h"
#endif
#ifdef CALDGEMM_CAL
#include "caldgemm_cal.h"
#endif


#ifndef _WIN32
#include <sys/mman.h>
#include <common.h>
#include <unistd.h>
#include <pthread.h>
#else
#include "cmodules/pthread_mutex_win32_wrapper.h"
#endif

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

bool mem_page_lock = true;
bool mem_huge_table = false;
bool linpack_callbacks = false;

bool wait_key = false;

bool use_opencl_not_cal = false;

int random_seed = 0;

int torture = 0;

char* matrixfile;

long seedused;

caldgemm* dgemm;

void PrintUsage()
{
	fprintf(STD_OUT,"Command Line Arguments\n");
	fprintf(STD_OUT, "\t-?        Display this help information\n" );
	fprintf(STD_OUT, "\t-e        Verify Computational Correctness\n" );
	fprintf(STD_OUT, "\t-q        Supress Display Output\n" );
	fprintf(STD_OUT, "\t-a        Print the disassembled kernel image\n" );
	fprintf(STD_OUT, "\t-i        Print IL Kernel used\n" );
	fprintf(STD_OUT, "\t-o  <c|g> Specify the output location, c = CPU, g = GPU, default GPU\n" );
	fprintf(STD_OUT, "\t-I  <int> Set implicit driver sync\n" );
	fprintf(STD_OUT, "\t-h  <int> block size for matrix multiply, default 4096\n" );
	fprintf(STD_OUT, "\t-H  <int> Reduced block size for actual matrix multiply (buffer size given by -h)\n" );
	fprintf(STD_OUT, "\t-w  <int> k for matrix multiply, default 1024\n" );
	fprintf(STD_OUT, "\t-W  <int> reduced width, see H\n" );
	fprintf(STD_OUT, "\t-l        Automatically select height for good performance\n" );
	fprintf(STD_OUT, "\t-m  <int> m for matrix multiply, must be multiple of h, default 1024\n" );
	fprintf(STD_OUT, "\t-n  <int> n for matrix multiply, must be multiple of h, default 1024\n" );
	fprintf(STD_OUT, "\t-v        Verbose Synchronous Timing for Single Kernels / Transfers\n" );
	fprintf(STD_OUT, "\t-k        Print Timing of Asynchronous DGEMM Operation\n" );
	fprintf(STD_OUT, "\t-r  <int> Number of iterations to run the program (inside caldgemm)\n" );
	fprintf(STD_OUT, "\t-R  <int> Number of iterations to run the program (seperate caldgemm calls)\n" );
	fprintf(STD_OUT, "\t-y  <int> Force Device ID (-1 = all devices)\n" );
	fprintf(STD_OUT, "\t-Y  <int> Use n devices\n" );
	fprintf(STD_OUT, "\t-d        Enable Debug Mode\n" );
	fprintf(STD_OUT, "\t-z        Enable Multithreading\n" );
	fprintf(STD_OUT, "\t-Z        Enable Multithreading for DivideBuffer\n" );
	fprintf(STD_OUT, "\t-b        Enable Benchmarking\n" );
	fprintf(STD_OUT, "\t-c        Use CPU\n" );
	fprintf(STD_OUT, "\t-g        Use GPU\n" );
	fprintf(STD_OUT, "\t-f        Fast Init (Empty Matrices)\n" );
	fprintf(STD_OUT, "\t-j  <dbl> GPU to CPU ratio\n" );
	fprintf(STD_OUT, "\t-s        Dynamic CPU GPU scheduling\n" );
	fprintf(STD_OUT, "\t-M        Disable third phase in dynamic scheduling\n" );
	fprintf(STD_OUT, "\t-N        Disable second phase in dynamic scheduling\n" );
	fprintf(STD_OUT, "\t-p        Interleaving Memory Policy\n" );
	fprintf(STD_OUT, "\t-u        Dump Test Matrix\n" );
	fprintf(STD_OUT, "\t-1        Transpose A Matrix\n" );
	fprintf(STD_OUT, "\t-2        Transpose B Matrix\n" );
	fprintf(STD_OUT, "\t-3        Set alpha parameter to 1.0 to test optimized kernel\n" );
	fprintf(STD_OUT, "\t-#        Set beta parameter to 0.0 to test optimized memcpy\n" );
	fprintf(STD_OUT, "\t-5        Quiet Benchmark mode (different from quiet caldgemm mode)\n" );
	fprintf(STD_OUT, "\t-6  <int> Set m/n to value * height\n" );
	fprintf(STD_OUT, "\t-4  <int> Set m/n to the closest multiple of height to value\n" );
	fprintf(STD_OUT, "\t-7        Verify Large Matrices\n" );
	fprintf(STD_OUT, "\t-8        No initial run to negate cache effects\n" );
	fprintf(STD_OUT, "\t-9        Output a table with timing information\n" );
	fprintf(STD_OUT, "\t-0        Write the output of divideBuffers directly to GPU instead of a seperate DMA transfer\n" );
	fprintf(STD_OUT, "\t-A        Do the DMA transfer to GPU asynchronously\n" );
	fprintf(STD_OUT, "\t-L        Memory Organisation like in HPL (LINPACK)\n" );
	fprintf(STD_OUT, "\t-C        Call fake LINPACK callback functions\n" );
	fprintf(STD_OUT, "\t-P  <int> LDA=LDB=LDC = val for HPL like memory\n" );
	fprintf(STD_OUT, "\t-T        Allocate Memory using Huge Tables\n" );
	fprintf(STD_OUT, "\t-B        Keep DMA Buffers mapped during kernel execution\n" );
	fprintf(STD_OUT, "\t-x <file> Load Matrix\n" );
	fprintf(STD_OUT, "\t--  <int> Torture Test, n iterations\n" );
	fprintf(STD_OUT, "\t-t  <int> Pin GPU thread to core n\n" );
	fprintf(STD_OUT, "\t-K  <int> Pin GPU main thread for DMA handling to core n\n" );
	fprintf(STD_OUT, "\t-Gx <int> Pin CPU threads of GPU x to same die as the CPU core id provided\n" );
	fprintf(STD_OUT, "\t-Ux <int> Pin CPU postprocessing threads of GPU x to CPU core <int>, -1 = default mapping\n" );
	fprintf(STD_OUT, "\t-UAx <int>Allocate memory for GPU x for die <int>, -1 = default mapping\n" );
	fprintf(STD_OUT, "\t-V        Thread save GPU driver\n" );
	fprintf(STD_OUT, "\t-S        Run on system with slow CPU\n" );
	fprintf(STD_OUT, "\t-X        Advanced multi-GPU tiling scheduler\n" );
	fprintf(STD_OUT, "\t-E <int>  Define random seed (0 for time)\n" );
	fprintf(STD_OUT, "\t-O        Backend to use: not set = CAL, set = OpenCL\n" );
	fprintf(STD_OUT, "\t-F <int>  OpenCL Platform ID to use\n" );
	fprintf(STD_OUT, "\t-J        Allow small tiles to process the remainder on GPU\n");
	fprintf(STD_OUT, "\t-Q        Wait for pressing a key before exiting\n");
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
	Config->m = Config->n = 4096;
	Config->Iterations = 1;
	//Config->DstMemory = 'g';
	Config->UseCPU = Config->UseGPU = false;
	//Config->GPURatio = -1;
	Config->DumpMatrix = false;
	Config->DivideToGPU = false;
	Config->AsyncDMA = false;
	Config->KeepBuffersMapped = false;

	Config->linpack_factorize_function = linpack_fake1;
	Config->linpack_broadcast_function = linpack_fake2;
	Config->linpack_swap_function = linpack_fake3;
#endif

	for (unsigned int x = 1; x < argc; ++x)
	{
		switch(argv[x][1])
		{
		default:
			fprintf(STD_OUT, "Invalid parameter: %s\n", argv[x]);
			PrintUsage();
			return(1);
		case 'q':
			Config->Quiet = true;
			break;
		case 'Q':
			wait_key = true;
			break;
		case '?':
			PrintUsage();
			return(1);
		case 'e':
			Config->Verify = true;
			break;
		case 'p':
			Config->MemPolicy = true;
			break;
		case 'b':
			benchmark = true;
			break;
		case 'u':
			Config->DumpMatrix = true;
			break;
		case 'a':
			Config->Disassemble = true;
			break;
		case '1':
			transa = true;
			break;
		case '2':
			transb = true;
			break;
		case '9':
			Config->TabularTiming = true;
			break;
		case '0':
			Config->DivideToGPU = true;
			break;
		case 'X':
			Config->ImprovedScheduler = true;
			break;
		case 'A':
			Config->AsyncDMA = true;
			break;
		case 'J':
			Config->SmallTiles = true;
			break;
		case 'B':
			Config->KeepBuffersMapped = true;
			break;
		case 'L':
			linpackmemory = true;
			break;
		case 'C':
			linpack_callbacks = true;
			break;
		case 'P':
			if (++x >= argc) return(1);
			linpackpitch = true;
			sscanf(argv[x], "%lld", (long long int*) &pitch_c);
			break;
		case '-':
			if (argv[x][2])
			{
				fprintf(STD_OUT, "Invalid parameter: %s\n", argv[x]);
				PrintUsage();
				return(1);
			}
			if (++x >= argc) return(1);
			Config->AsyncDMA = Config->KeepBuffersMapped = true;
			Config->m = Config->n = 86016;
			Config->MemPolicy = true;
			Config->MultiThread = true;
			Config->UseCPU = false;
			Config->UseGPU = true;
			sscanf(argv[x], "%d", &torture);
			iterations = torture;
			break;
		case 'T':
			mem_huge_table = true;
			break;
		case '8':
			initialrun = false;
			break;
		case '7':
			verifylarge = true;
			break;
		case '6':
			fprintf(STD_OUT, "Set m and n to %lld\n", (long long int) (Config->m = Config->n = Config->Height * atoi(argv[++x])));
			break;
		case '4':
			Config->m = atoi(argv[++x]);
			Config->m -= Config->m % Config->Height;
			fprintf(STD_OUT, "Set m and n to %lld\n", (long long int) (Config->n = Config->m));
			break;
		case '5':
			quietbench = true;
			break;
		case '3':
			alphaone = true;
			break;
		case '#':
			betazero = true;
			break;
		case 'i':
			Config->PrintILKernel = true;
			break;
		case 'c':
			Config->UseCPU = true;
			break;
		case 'l':
			Config->AutoHeight = true;
			break;
		case 's':
			Config->DynamicSched = true;
			break;
		case 'M':
			Config->ThirdPhaseDynamicRuns = false;
			break;
		case 'N':
			Config->SecondPhaseDynamicRuns = false;
			break;
		case 'S':
			Config->SlowCPU = true;
			break;
		case 'g':
			Config->UseGPU = true;
			break;
		case 'I':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", (int*) &Config->ImplicitDriverSync);
			break;
		case 'E':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &random_seed);
			break;
		case 'f':
			fastinit = true;
			break;
		case 'O':
			use_opencl_not_cal = true;
			break;
		case 'F':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->OpenCLPlatform);
			break;
		case 'o':
			if (++x >= argc) return(1);
			Config->DstMemory = argv[x][0];
			if (Config->DstMemory != 'c' && Config->DstMemory != 'g')
			{
				fprintf(STD_OUT, "Invalid destination memory type\n" );
				return(1);
			}
			break;
		case 'w':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", (long long int*) &Config->Width);
			break;
		case 'W':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &reduced_width);
			break;
		case 't':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PinCPU);
			break;
		case 'K':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PinMainThread);
			break;
		case 'G':
			if (x + 1 >= argc) return(1);
			int gpuid;
			sscanf(&argv[x++][2], "%d", &gpuid);
			if ((unsigned) gpuid >= sizeof(Config->GPUMapping) / sizeof(Config->GPUMapping[0]))
			{
			    fprintf(STD_OUT, "Invalid GPU ID (%d)\n", gpuid);
			    break;
			}
			sscanf(argv[x], "%d", &Config->GPUMapping[gpuid]);
			printf("Set CPU core for GPU %d to %d\n", gpuid, Config->GPUMapping[gpuid]);
			break;
		case 'U':
			if (x + 1 >= argc) return(1);
			if (argv[x][2] == 'A')
			{
				sscanf(&argv[x++][3], "%d", &gpuid);
				if ((unsigned) gpuid >= sizeof(Config->AllocMapping) / sizeof(Config->AllocMapping[0]))
				{
					fprintf(STD_OUT, "Invalid GPU ID (%d)\n", gpuid);
					break;
				}
				sscanf(argv[x], "%d", &Config->AllocMapping[gpuid]);
				printf("Allocating memory for GPU %d on core %d\n", gpuid, Config->PostprocessMapping[gpuid]);
			}
			else
			{
				sscanf(&argv[x++][2], "%d", &gpuid);
				if ((unsigned) gpuid >= sizeof(Config->PostprocessMapping) / sizeof(Config->PostprocessMapping[0]))
				{
					fprintf(STD_OUT, "Invalid GPU ID (%d)\n", gpuid);
					break;
				}
				sscanf(argv[x], "%d", &Config->PostprocessMapping[gpuid]);
				printf("Set CPU core for postprocessing of GPU %d to %d\n", gpuid, Config->PostprocessMapping[gpuid]);
			}
			break;
		case 'h':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", (long long int*) &Config->Height);
			break;
		case 'H':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &reduced_height);
			break;
		case 'm':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", (long long int*) &Config->m);
			break;
		case 'n':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", (long long int*) &Config->n);
			break;
		case 'x':
			if (++x >= argc) return(1);
			loadmatrix = true;
			matrixfile = argv[x];
			break;
		case 'v':
			Config->VerboseTiming = true;
			break;
		case 'V':
			Config->ThreadSaveDriver = true;
			break;
		case 'k':
			Config->AsyncTiming = true;
			break;
		case 'd':
			Config->Debug = true;
			break;
		case 'z':
			Config->MultiThread = true;
			break;
		case 'Z':
			Config->MultiThreadDivide = true;
			break;
		case 'r':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%u", &Config->Iterations);
			break;
		case 'R':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%u", &iterations);
			break;
		case 'y':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%u", &Config->DeviceNum);
			break;
		case 'Y':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%u", &Config->NumDevices);
			break;
		case 'j':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lf", &Config->GPURatio);
			fprintf(STD_OUT, "Using GPU Ratio %lf\n", Config->GPURatio);
			break;
		};
	}

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
		if (torture) memset(CC, 0, Config.m * pitch_c * sizeof(double));
	}
	else
	{
		for (size_t i = 0;i < Config.m;i++)
		{
			for (size_t j = 0;j < Config.n;j++)
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
			pitch_a = pitch_b = pitch_c;
		}
		else
		{
			pitch_a = pitch_b = pitch_c = Config.n + Config.Width + (Config.n + Config.Width) % 8;
		}
		linpackmem = dgemm->AllocMemory(pitch_c * (Config.m + Config.Width + 1) + 8, mem_page_lock, mem_huge_table);
		if (linpackmem == NULL) {fprintf(STD_OUT, "Memory Allocation Error\n"); return(1);}

		char* linpackmem2 = (char*) linpackmem;
		if ((size_t) linpackmem2 % 64) linpackmem2 += 64 - ((size_t) linpackmem2) % 64;
		double* linpackmem3 = (double*) linpackmem2;


		AA = linpackmem3 + Config.Width * pitch_c;
		BB = linpackmem3 + Config.Width;
		CC = linpackmem3 + Config.Width * (pitch_c + 1);
	}
	else
	{
		if (transa)
		{
			pitch_a = Config.m;
			height_a = Config.Width;
			width_a = Config.Width;
		}
		else
		{
			pitch_a = Config.Width;
			height_a = Config.m;
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
			height_b = Config.n;
			width_b = Config.Width;
		}
		else
		{
			height_b = Config.Width;
			pitch_b = Config.n;
			width_b = Config.n;
		}
		if (pitch_b % 8) pitch_b += (8 - pitch_b % 8);
		if (((pitch_b / 8) & 1) == 0)
		{
			pitch_b += 8;
		}
		pitch_c = Config.n;
		if (pitch_c % 8) pitch_c += (8 - pitch_c % 8);
		if (Config.n % 8) fprintf(STD_OUT, "Padding 8 bytes for correct alignment of B, n = %lld, pitch = %lld\n", (long long int) Config.n, (long long int) pitch_b);
		if (((pitch_c / 8) & 1) == 0)
		{
			pitch_c += 8;
		}

		if (AA) dgemm->FreeMemory(AA);
		if (BB) dgemm->FreeMemory(BB);
		if (CC) dgemm->FreeMemory(CC);
		if (!quietbench) fprintf(stderr, "...alloc A");
		AA = dgemm->AllocMemory(height_a * pitch_a, mem_page_lock, mem_huge_table);
		if (!quietbench) fprintf(stderr, "...alloc B");
		BB = dgemm->AllocMemory(height_b * pitch_b, mem_page_lock, mem_huge_table);
		if (!quietbench) fprintf(stderr, "...alloc C");
		CC = dgemm->AllocMemory(Config.m * pitch_c, mem_page_lock, mem_huge_table);

		if (AA == NULL || BB == NULL || CC == NULL)
		{
			fprintf(STD_OUT, "Memory allocation error allocating matrices\n");
			return(1);
		}
	}

#ifdef TESTMODE
	for (int i = 0;i < height_a;i++)
	{
		for (int j = 0;j < width_a;j++)
		{
			AA[i * pitch_a + j] = 1;
		}
	}
	for (int i = 0;i < height_b;i++)
	{
		for (int j = 0;j < width_b;j++)
		{
			BB[i * pitch_b + j] = 1;
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
		fastmatgen(rand() * 100, AA, height_a * pitch_a);
		if (!quietbench) fprintf(stderr, "...init B");
		fastmatgen(rand() * 100, BB, height_b * pitch_b);
	}
#endif
	if (Config.Debug) fprintf(STD_OUT, "User Data Initialized\n");
	if (!quietbench) fprintf(stderr, "...");
	return(0);
}

bool isDoubleEqual(double a, double b)
{
	double epsilon = 1e-6;

	if(fabs(b) <1e-13)
		return (fabs(a-b) < epsilon);
	else
		return (fabs((a-b)/b) < epsilon);
}

int main(int argc, char** argv)
{
	caldgemm::caldgemm_config Config;

	if (ParseCommandLine(argc, argv, &Config))
	{
		return 1;
	}

#ifndef CALDGEMM_CAL
	use_opencl_not_cal = true;
#endif
#ifndef CALDGEMM_OPENCL
	use_opencl_not_cal = false;
#endif

	if (use_opencl_not_cal)
	{
#ifdef CALDGEMM_CAL
		dgemm = new caldgemm_opencl;
#endif
	}
	else
	{

#ifdef CALDGEMM_OPENCL
		dgemm = new caldgemm_cal;
#endif
	}

	
	if (dgemm == NULL)
	{
		fprintf(STD_OUT, "Error creating caldgem object\n");
		return(1);
	}

	if (dgemm->InitCALDGEMM(&Config))
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
		nread = fread(&b, sizeof(b), 1, fp);
		nread = fread(&c, sizeof(c), 1, fp);
		nread = fread(&alpha, sizeof(alpha), 1, fp);
		nread = fread(&beta, sizeof(beta), 1, fp);
		nread = fread(&tmp_m, sizeof(tmp_m), 1, fp);
		nread = fread(&tmp_k, sizeof(tmp_k), 1, fp);
		nread = fread(&tmp_n, sizeof(tmp_n), 1, fp);
		nread = fread(&Apitch, sizeof(Apitch), 1, fp);
		nread = fread(&Bpitch, sizeof(Bpitch), 1, fp);
		nread = fread(&Cpitch, sizeof(Cpitch), 1, fp);

		Apitch = 1536;

		AA = new double[(size_t) tmp_m * (size_t) Apitch];
		BB = new double[(size_t) tmp_k * (size_t) Bpitch];
		CC = new double[(size_t) tmp_m * (size_t) Cpitch];

		for (int i = 0;i < tmp_m;i++)
		{
			nread = fread(AA + i * Apitch, tmp_k, sizeof(double), fp);
		}
		for (int i = 0;i < tmp_k;i++)
		{
			nread = fread(BB + i * Bpitch, tmp_n, sizeof(double), fp);
		}
		fclose(fp);
		memset(CC, 0, (size_t) tmp_m * (size_t) Cpitch * sizeof(double));

		fprintf(STD_OUT, "matrix loaded: m=%d k=%d n=%d lda=%d ldb=%d ldc=%d alpha=%2.4lf beta=%2.4lf\n", tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch, alpha, beta);

		dgemm->RunCALDGEMM(AA, BB, CC, alpha, beta, tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch);
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
			unsigned int tmpm = Config.m, tmpn = Config.n, tmpdebug = Config.Debug;
			Config.Quiet = true;
			Config.Verify = false;
			Config.Iterations = 1;
			Config.Debug = false;
			if (Config.m > 2 * Config.Height) Config.m = 2 * Config.Height;
			if (Config.n > 2 * Config.Height) Config.n = 2 * Config.Height;
			if (dgemm->RunCALDGEMM(AA, BB, CC, alphaone ? 1.0 : 0.5, 1.0, Config.m, Config.Width, Config.n, pitch_a, pitch_b, pitch_c, false, transa, transb))
			{
				fprintf(STD_OUT, "Error running CALDGEMM\nexiting\n");
				return(1);
			}
			Config.m = tmpm;
			Config.n = tmpn;
			Config.Quiet = tmpquiet;
			Config.Verify = tmpverify;
			Config.Iterations = tmpiter;
			Config.Debug = tmpdebug;
			if (!quietbench)
			{
				fprintf(stderr, "Done\n");
			}
		}
#endif
		if (!quietbench)
		{
			fprintf(stderr, "Initializing Matrix C\n");
		}
		SetupUserDataC(Config);
		dgemm->ResetTimers();
		if (!quietbench)
		{
			fprintf(stderr, "Running Benchmark\n");
		}
		do
		{
			for (int iter = 0;iter < iterations;iter++)
			{
				if (iterations > 1 && !quietbench) fprintf(STD_OUT, "\nDGEMM Call Iteration %d\n\n", iter);
#ifdef TESTMODE
				if (dgemm->RunCALDGEMM(AA, BB, CC, 1.0, 0.0, Config.m, Config.Width, Config.n, pitch_a, pitch_b, pitch_c, false, transa, transb))
#else
				size_t tmpn = Config.m > Config.n ? Config.m : Config.n;
				if (linpack_callbacks) Config.LinpackSwapN = &tmpn;
				if (dgemm->RunCALDGEMM(AA, BB, CC, alphaone ? 1.0 : -1.0, betazero ? 0.0 : 1.0, Config.m, Config.Width, Config.n, pitch_a, pitch_b, pitch_c, false, transa, transb, linpack_callbacks))
#endif
				{
					fprintf(STD_OUT, "Error running CALDGEMM\n");
					return(1);
				}
				if (torture)
				{
					dgemm->RunCALDGEMM(AA, BB, CC, 1.0, 1.0, Config.m, Config.Width, Config.n, pitch_a, pitch_b, pitch_c, false, transa, transb, linpack_callbacks);
				}
			}
		} while (benchmark && (Config.n += Config.Height) < 70000 && (Config.m += Config.Height) < 70000 && SetupUserData(Config) == 0);
		
	}
	
	if (torture)
	{
		for (size_t i = 0;i < Config.m * pitch_c;i++)
		{
			if (CC[i] > 10E-10)
			{
				fprintf(STD_OUT, "Torture Test FAILED\n");
				if (!quietbench) fprintf(STD_OUT, "Entry %lld is %lf\n", (long long int) i, CC[i]);
				torture = 0;
				break;
			}
		}
		if (torture) fprintf(STD_OUT, "Torture Test PASSED (%2.3lf gflops)\n", dgemm->avggflops);
	}

	if (verifylarge)
	{
		fprintf(STD_OUT, "Running verification for large matrices\n");
		srand((int) seedused);
		Config.UseGPU = false;
		Config.UseCPU = true;
		Config.Verify = false;
		Config.Quiet = true;
		dgemm->RunCALDGEMM(AA, BB, CC, alphaone ? -1.0 : 1.0, 1.0, Config.m, Config.Width, Config.n, pitch_a, pitch_b, pitch_c, false, transa, transb);
		fprintf(STD_OUT, "CPU DGEMM Comparison run complete, comparing results\n");
		int verifyok = 1;
		for (size_t i = 0;i < Config.m;i++)
		{
			for (size_t j = 0;j < Config.n;j++)
			{
				if (!isDoubleEqual(CC[i * pitch_c + j] * 1.0, (double) ((i + j) % 16)))
				{
					fprintf(STD_OUT, "Verification failed at i = %lld, m = %lld, n = %lld\n", (long long int) i, (long long int) i / pitch_c, (long long int) i % pitch_c);
					verifyok = 0;
					break;
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
		dgemm->RunCALDGEMM(AA, BB, CC, ALPHA, BETA, M, K, N, APITCH, BPITCH, CPITCH, ORDER, TRANSA, TRANSB);
		fprintf(STD_OUT, "Caldgemm run complete\n");

		delete[] mem;
	}
#endif //TEST_PARAMETERS

	dgemm->ExitCALDGEMM();

#ifndef TEST_PARAMETERS
	if (linpackmemory)
	{
		dgemm->FreeMemory(linpackmem);
	}
	else
	{
		dgemm->FreeMemory(AA);
		dgemm->FreeMemory(BB);
		dgemm->FreeMemory(CC);
	}
#endif

	delete dgemm;

	if (wait_key)
	{
		fprintf(STD_OUT, "Press return to exit!\n");
		getchar();
	}
	return 0;
}
