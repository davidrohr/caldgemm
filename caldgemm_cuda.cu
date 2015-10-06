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

#include "caldgemm_cuda.h"

#define CUDAKernelName CUDAKernel
#include "cudakernel.cu"
#undef CUDAKernelName
#define CUDAKernelName CUDAKernelALPHA1
#define CALDGEMM_ALPHA1
#include "cudakernel.cu"
#undef CUDAKernelName
#define CUDAKernelName CUDAKernelLinpack
#define CALDGEMM_LINPACK_KERNEL
#include "cudakernel.cu"
#undef CALDGEMM_LINPACK_KERNEL
#undef CALDGEMM_ALPHA1
#undef CUDAKernelName

__global__ void CUDAConversionKernel(const double* __restrict__ iBuffer, double* __restrict__ oBuffer, size_t width, size_t height)
{
	for (int j = blockIdx.y * blockDim.y + threadIdx.y;j < height;j += blockDim.y * gridDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < width;i += blockDim.x * gridDim.x)
		{
			oBuffer[i * height + j] = iBuffer[j * width + i];
		}
	}
}

#define ERRRET(...) {fprintf(STD_OUT, __VA_ARGS__);fprintf(STD_OUT, "\n");return(1);}

#define CHKRET(result, ...) \
	if (result != cudaSuccess) \
	{ \
		fprintf(STD_OUT, "CUDA Error %d: %s\n%s:%d\n", result, cudaGetErrorString(result), __FILE__, __LINE__); \
		ERRRET(__VA_ARGS__); \
	}

caldgemm_cuda::caldgemm_cuda() : caldgemm()
{
}

caldgemm_cuda::~caldgemm_cuda()
{
}

int caldgemm_cuda::WaitForEventAndRelease(cudaEvent_t* pEvent)
{
	CHKRET(cudaEventSynchronize(*pEvent), "Waiting for event");
	return(0);
}

int caldgemm_cuda::WaitForEvent(int a, int b, int)
{
	if (Config->Debug) fprintf(STD_OUT, "\tWaiting for event from device %d obuffer %d...\n", b, a);
	return(WaitForEventAndRelease(&cuda_events[b][a]));
}

int caldgemm_cuda::Initialize(bool nocalinit)
{
	int deviceNum = Config->DeviceNum;
	if (!Config->Quiet) fprintf(STD_OUT, "Initializing CALDGEMM (CUDA Runtime)\n");
	if (Config->Debug) fprintf(STD_OUT, "CUDA Initialice\n");

	if (nDevices > (signed) max_devices) nDevices = max_devices;
	int nGoodDevices = 0;
	int* goodDevices = new int[nDevices];
	for (int i = 0;i < nDevices;i++)
	{
		cudaDeviceProp deviceProp;
		CHKRET(cudaGetDeviceProperties(&deviceProp, i), "Getting CUDA Device Properties");
		int deviceOK = deviceProp.major < 9 && deviceProp.major >= 2;
		if (Config->Debug) fprintf(STD_OUT, "Device %2d (%d): %30s %s\n", deviceOK ? nGoodDevices : -1, i, deviceProp.name, deviceOK ? "OK" : "--");
		if (deviceOK) goodDevices[nGoodDevices++] = i;
	}

	nDevices = nGoodDevices;
	if (nDevices > Config->NumDevices) nDevices = Config->NumDevices;
	if (deviceNum >= nDevices) ERRRET("CUDA Device %d not available\n", deviceNum);
	if (deviceNum >= 0) nDevices = 1;
	gpu_available = (nDevices > 0);
	
	if (nDevices > 1 && !Config->MultiThread)
	{
		fprintf(STD_OUT, "Cannot use multiple devices without multithreading\n");
		nDevices = 1;
	}

	for (int i = 0;i < nDevices;i++)
	{
		if (deviceNum >= 0) cuda_devices[i] = goodDevices[deviceNum];
		else cuda_devices[i] = goodDevices[Config->DeviceNums[i]];
	}
	delete[] goodDevices;

	if (Config->DstMemory == 'c')
	{
		CHKRET(cudaSetDeviceFlags(cudaDeviceMapHost), "Setting CUDA Device flags");
	}
	
	for (int i = 0;i < nDevices;i++)
	{
		CHKRET(cudaSetDevice(cuda_devices[i]), "Settig CUDA Device");
		for (int j = 0;j < obuffercount;j++)
		{
			CHKRET(cudaStreamCreate(&cuda_command_queues[i][j]), "Creating CUDA Stream");
		}
#ifdef CALDGEMM_CUDA_CUBLAS		
		CHKRET((cudaError_t)cublasCreate(&cublas_handles[i]),"Initializing Cublas library");
#endif
	}

	return(0);
}

int caldgemm_cuda::ValidateRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA ValidateRuntime\n");
	Config->MultiThreadDivide = false;
	if (Config->ThreadSaveDriver != -1) Config->ThreadSaveDriver = 1;

	Config->GPU_C = true;
	CHKRET(cudaGetDeviceCount(&nDevices), "Getting Device Count");
	if (nDevices == 0) ERRRET("No CUDA device for this platform found\n");
	if (Config->Debug) fprintf(STD_OUT, "%d CUDA devices found for this platform\n", nDevices);

	SetDefaultKernelSettings();
#ifdef CALDGEMM_CUDA_CUBLAS
	KernelSettings.transposeA = false;
	KernelSettings.transposeB = true;
#else
	KernelSettings.transposeA = false;
	KernelSettings.transposeB = true;
#endif
	KernelSettings.texture_buffers = false;
	KernelSettings.tiling_x = 1;
	KernelSettings.tiling_y = 1;
	KernelSettings.group_size_x = GROUP_SIZE_X;
	KernelSettings.group_size_y = GROUP_SIZE_Y;
	KernelSettings.min_tile_size = 32;
	KernelSettings.min_k = 4;

	if (!(KernelSettings.transposeA ^ KernelSettings.transposeB))
	{
		fprintf(STD_OUT, "Must set either transposed A or transposed B\n");
		return(1);
	}

	return(0);
}

int caldgemm_cuda::CheckDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA CheckDevices\n");
	return(0);
}

int caldgemm_cuda::InitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA InitDevices\n");

	cudaError_t cuda_error;

	int num_bbuffers;
	if (Config->max_bbuffers) num_bbuffers = Config->max_bbuffers;
	else if (Config->DstMemory == 'g') num_bbuffers =  max_bbuffers_g;
	else num_bbuffers = max_bbuffers;
	
	if (num_bbuffers > max_bbuffers)
	{
		fprintf(STD_OUT, "Requested number of bbuffers (%d) larger than max_bbuffers constant (%d)!", num_bbuffers, max_bbuffers);
		return(1);
	}

	SetupBufferSizes();

	for (int i = 0;i < nDevices;i++)
	{
		CHKRET(cudaSetDevice(cuda_devices[i]), "Setting CUDA Device");
		for (int j = 0;j < ibuffercount;j++)
		{
			CHKRET(cudaMalloc(&cuda_abuffers[i][j], BufferWidth * BufferHeight * sizeof(double)), "Error allocating device memory (A)");
		}

		for (int j = 0;j < obuffercount;j++)
		{
			if (Config->DstMemory == 'g')
			{
				CHKRET(cudaMalloc(&cuda_cbuffers[i][j], BufferHeight * BufferHeight * sizeof(double)), "Error allocating device memory (C)");
			}
			CHKRET(cudaMalloc(&cuda_tmp_abuffers[i][j], BufferWidth * BufferHeight * sizeof(double)), "Error allocating device memory (A) (tmp)");
			CHKRET(cudaMalloc(&cuda_tmp_bbuffers[i][j], BufferWidth * BufferHeight * sizeof(double)), "Error allocating device memory (B) (tmp)");
		}

		for (int j = 0;j < num_bbuffers;j++)
		{
			cuda_error = cudaMalloc(&cuda_bbuffers[i][j], BufferWidth * BufferHeight * sizeof(double));
			if (cuda_error != cudaSuccess)
			{
				if (j < obuffercount)
				{
					CHKRET(cuda_error, "Error allocating device memory (B)");
				}
				else break;
			}

			bbuffers[i] = j + 1;
		}
		if (Config->Debug) fprintf(STD_OUT, "Allocated %d BBuffers on Device %d\n", bbuffers[i], i);

		for (int j = 0;j < obuffercount;j++)
		{
			CHKRET(cudaEventCreateWithFlags(&cuda_events[i][j], cudaEventDisableTiming), "Creating Event 1 %d %d\n", i, j);
		}
		for (int j = 0;j < 2;j++)
		{
			CHKRET(cudaEventCreateWithFlags(&cuda_conversion_events[i][j], cudaEventDisableTiming), "Creating Event 1 %d %d\n", i, j);
		}
	}

	return(0);
}

int caldgemm_cuda::ReinitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA ReinitDevices\n");
	fprintf(STD_OUT, "Reinit of CUDA devices not supported yet\n");
	return(1);
}

int caldgemm_cuda::InitConstantData(double alpha)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA InitConstantData\n");
	return(0);
}

int caldgemm_cuda::ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA ExecuteKernels\n");

	if (Config->Debug) fprintf(STD_OUT, "\tExecuting MM kernel (device %d obuffer %d, k=%lld m=%lld n=%lld)\n", Task.device, Task.j, (long long int) Task.k, (long long int) blockm, (long long int) blockn);

	double *abuffer, *bbuffer;
#ifdef REUSE_BBUFFERS
	if (!DGEMM_favor_m && buffersSwitchable)
	{
		const int buffer_pos = buffer_pointers_A[Task.device][blockm] % (buffer_pointers_A[Task.device][blockm] < bbuffers[Task.device] ? bbuffers[Task.device] : ibuffercount);
		abuffer = (double*) cuda_bbuffers[Task.device][buffer_pos];
		bbuffer = (double*) cuda_abuffers[Task.device][buffer_pointers_B[Task.device][blockn] % ibuffercount];
	}
	else
#endif
	{
#ifdef REUSE_BBUFFERS
		const bool buffersSufficiant = buffer_pointers_B[Task.device][blockn] < bbuffers[Task.device];
#else
		const bool buffersSufficiant = false;
#endif
		abuffer = (double*) cuda_abuffers[Task.device][buffer_pointers_A[Task.device][blockm] % ibuffercount];
		bbuffer = (double*) cuda_bbuffers[Task.device][!buffersSufficiant ? (buffer_pointers_B[Task.device][blockn] % ibuffercount) : (buffer_pointers_B[Task.device][blockn] % bbuffers[Task.device])];
	}

	double* cbuffer;
	size_t height1 = (((size_t) blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height);
	size_t height2 = (((size_t) blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height);
	size_t width = Config->Width;

	size_t pitch;
	if (Config->DstMemory == 'g')
	{
		cbuffer = (double*) cuda_cbuffers[Task.device][Task.j];
		pitch = height1;
	}
	else
	{
		cbuffer = C + blockn * Config->Height + blockm * Config->Height * C_pitch;
		pitch = C_pitch;
	}

	CHKRET(cudaSetDevice(cuda_devices[Task.device]), "Setting CUDA Device");
	if (Config->VerboseTiming)
	{
		CHKRET(cudaStreamSynchronize(cuda_command_queues[Task.device][Task.j]), "Synchronizing CUDA Stream");
		Timers.Kernel.Start();
	}
	if (Config->Debug) fprintf(STD_OUT, "MM Kernel: height1 %d height2 %d width %d alpha %f beta %f pitch %d\n", (int) height1, (int) height2, (int) width, Alpha, Beta, (int) pitch);
	dim3 threads(GROUP_SIZE_X, GROUP_SIZE_Y), blocks(GROUP_COUNT_X, GROUP_COUNT_Y);
#ifndef CALDGEMM_CUDA_CUBLAS
	CUDAKernel <<<blocks, threads, 0, cuda_command_queues[Task.device][Task.j]>>> (cbuffer, abuffer, bbuffer, height1, height2, width, Alpha, Beta, pitch);
#else
	cublasSetStream(cublas_handles[Task.device], cuda_command_queues[Task.device][Task.j]);
	cublasDgemm(cublas_handles[Task.device],CUBLAS_OP_T,CUBLAS_OP_N,height1,height2,width,&Alpha,bbuffer,width,abuffer,width,&Beta,cbuffer,pitch);
	cublasDgemm(cublas_handles[Task.device],CUBLAS_OP_N,CUBLAS_OP_T,height1,height2,width,&Alpha,bbuffer,height1,abuffer,height2,&Beta,cbuffer,pitch);
#endif
	CHKRET(cudaGetLastError(), "CUDA Kernel Execution");

	if (Config->VerboseTiming)
	{
		CHKRET(cudaStreamSynchronize(cuda_command_queues[Task.device][Task.j]), "Synchronizing CUDA Stream");
		Timers.Kernel.Stop();
		Timers.CounterCopyFrom.Start();
	}

	if (Config->DstMemory == 'g')
	{
		if (Config->Debug) fprintf(STD_OUT, "Transfer C from GPU: region %d x %d\n", (int) (height1 * sizeof(double)), (int) height2);
		CHKRET(cudaMemcpy2DAsync(C + blockn * Config->Height + blockm * Config->Height * C_pitch, C_pitch * sizeof(double), cuda_cbuffers[Task.device][Task.j], height1 * sizeof(double), height1 * sizeof(double), height2, cudaMemcpyDeviceToHost, cuda_command_queues[Task.device][Task.j]), "Fetching result");
	}

	if (Config->VerboseTiming)
	{
		CHKRET(cudaStreamSynchronize(cuda_command_queues[Task.device][Task.j]), "Synchronizing CUDA Stream");
		Timers.CounterCopyFrom.Stop();
	}

	return(0);
}

int caldgemm_cuda::ExitRuntime()
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA ExitRuntime\n");

	for (int i = 0;i < nDevices;i++)
	{
		CHKRET(cudaSetDevice(cuda_devices[i]), "Setting CUDA Device");
		for (int j = 0;j < obuffercount;j++)
		{
			CHKRET(cudaStreamDestroy(cuda_command_queues[i][j]), "Destroying CUDA Stream");
		}
	}

	return(0);
}

int caldgemm_cuda::FetchResult(int device, int j, int m, int n, int mustlock)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA FetchResult\n");
	return(0);
}

int caldgemm_cuda::CheckDMAQueue(int device, int forcej)
{
	return(0);
}

int caldgemm_cuda::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA RunMergeBuffers\n");
	return(0);
}

int caldgemm_cuda::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	CALDGEMM_PREPARE_BACKEND_VARS1;

	size_t width, height;

	CHKRET(cudaSetDevice(cuda_devices[num_device]), "Setting CUDA Device");
	if (Config->VerboseTiming) Timers.CounterCopyTo.Start();

	for (int iMat = 0;iMat < 2;iMat++)
	{
		if (cuda_conversion_events_use[num_device][iMat])
		{
			WaitForEventAndRelease(&cuda_conversion_events[num_device][iMat]);
			cuda_conversion_events_use[num_device][iMat] = 0;
		}
		if (iMat ? prepareM : prepareN)
		{
			CALDGEMM_PREPARE_BACKEND_VARS2;
			
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of %c to GPU (k = %lld, m = %lld, n = %lld)\n", myMat, (long long int) k, (long long int) blockm, (long long int) blockn);

			void* dest_buffer_tmp = iMat ? cuda_tmp_bbuffers[num_device][j] : cuda_tmp_abuffers[num_device][j];
			width = (((bool) iMat ^ myTranspose) ? myHeight : Config->Width) * sizeof(double);
			height = ((bool) iMat ^ myTranspose) ? Config->Width : myHeight;
			int arg_transpose = myTranspose ^ myKernelTranspose;
			void* dest_image = access_bbuffers ? cuda_bbuffers[num_device][destbuffer] : cuda_abuffers[num_device][destbuffer];
			if (arg_transpose == 0) dest_buffer_tmp = dest_image;

			if (Config->Debug) fprintf(STD_OUT, "Transfer %c to GPU: region %d x %d\n", myMat, (int) width, (int) height);
			CHKRET(cudaMemcpy2DAsync(dest_buffer_tmp, width, src_ptr, pitch * sizeof(double), width, height, cudaMemcpyHostToDevice, cuda_command_queues[num_device][j]), "Copying %c to device", myMat);
			if (Config->Debug && Config->VerboseTiming) cudaStreamSynchronize(cuda_command_queues[num_device][j]);

			if (arg_transpose)
			{
				dim3 threads(GROUP_SIZE_X, GROUP_SIZE_Y), blocks(GROUP_COUNT_X, GROUP_COUNT_Y);
				size_t arg_width = width / sizeof(double), arg_height = height;
				if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel %c: x %d y %d\n", myMat, (int) arg_width, (int) arg_height);
				CUDAConversionKernel <<<blocks, threads, 0, cuda_command_queues[num_device][j]>>> ((double*) dest_buffer_tmp, (double*) dest_image, arg_width, arg_height);
				CHKRET(cudaGetLastError(), "CUDA Conversion Kernel Execution");
				if (Config->Debug && Config->VerboseTiming) cudaStreamSynchronize(cuda_command_queues[num_device][j]);
			}
			CHKRET(cudaEventRecord(cuda_conversion_events[num_device][iMat], cuda_command_queues[num_device][j]), "Recording event %d %d", num_device, iMat);
			cuda_conversion_events_use[num_device][iMat] = 1;
		}
	}

	if (Config->DstMemory == 'g')
	{
		width = HeightN * sizeof(double);
		height = HeightM;
		Timers.divideC++;
		if (Config->Debug) fprintf(STD_OUT, "Transfer C to GPU: region %d x %d\n", (int) width, (int) height);
		CHKRET(cudaMemcpy2DAsync(cuda_cbuffers[num_device][j], width, C + blockn * Config->Height + blockm * Config->Height * C_pitch, C_pitch * sizeof(double), width, height, cudaMemcpyHostToDevice, cuda_command_queues[num_device][j]), "Copying C to device");
	}

	if (Config->VerboseTiming)
	{
		CHKRET(cudaStreamSynchronize(cuda_command_queues[num_device][j]), "Synchronizing CUDA Stream");
		Timers.CounterCopyTo.Stop();
	}

	return(0);
}

int caldgemm_cuda::ExitDevices()
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA ExitDevices\n");

	for (int i = 0;i < nDevices;i++)
	{
		CHKRET(cudaSetDevice(cuda_devices[i]), "Setting CUDA Device");
		for (int j = 0;j < ibuffercount;j++)
		{
			CHKRET(cudaFree(cuda_abuffers[i][j]), "Freeing memory A %d %d\n", i, j);
		}
		for (int j = 0;j < obuffercount;j++)
		{
			if (Config->DstMemory == 'g')
			{
				CHKRET(cudaFree(cuda_cbuffers[i][j]), "Freeing memory C %d %d\n", i, j);
			}
			CHKRET(cudaFree(cuda_tmp_abuffers[i][j]), "Freeing memory A tmp %d %d\n", i, j);
			CHKRET(cudaFree(cuda_tmp_bbuffers[i][j]), "Freeing memory B tmp %d %d\n", i, j);
		}
		for (int j = 0;j < bbuffers[i];j++)
		{
			CHKRET(cudaFree(cuda_bbuffers[i][j]), "Freeing memory B %d %d\n", i, j);
		}
		for (int j = 0;j < obuffercount;j++)
		{
			CHKRET(cudaEventDestroy(cuda_events[i][j]), "Destroying Event 1 %d %d\n", i, j);
		}
		for (int j = 0;j < 2;j++)
		{
			CHKRET(cudaEventDestroy(cuda_conversion_events[i][j]), "Destroying Event 1 %d %d\n", i, j);
		}
	}
	return(0);
}

int caldgemm_cuda::UseOutputPthreads() {return(0);}
int caldgemm_cuda::UseInputPthreads() {return(0);}
int caldgemm_cuda::UseMutexPerDevice() {return(0);}

int caldgemm_cuda::RunCALDGEMM_Init()
{
	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			cuda_conversion_events_use[i][j] = 0;
		}
	}
	return(0);
}

int caldgemm_cuda::RunCALDGEMM_Exit()
{
	CHKRET(cudaThreadSynchronize(), "Synchronizing CUDA Thread");
	return(0);
}

#define MAX_GPU_MEM_COUNT 64
struct gpu_mem_struct_cuda
{
	void* ptr;
};
static gpu_mem_struct_cuda gpu_mem[MAX_GPU_MEM_COUNT];
static int nGPUMEM = 0;

double* caldgemm_cuda::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages, bool gpuaccessible, bool interleave)
{
	if (gpuaccessible)
	{
		void* ptr;
		unsigned int flags = cudaHostAllocPortable;
		if (Config->DstMemory == 'c') flags |= cudaHostAllocMapped;
		cudaError_t cuda_error = cudaHostAlloc(&ptr, nDoubles * sizeof(double), flags);
		if (cuda_error != cudaSuccess)
		{
			fprintf(STD_OUT, "cudaHostAlloc: CUDA Error %d: %s\n", cuda_error, cudaGetErrorString(cuda_error));
			return(NULL);
		}
		gpu_mem[nGPUMEM++].ptr = ptr;
		if (Config->DstMemory == 'c')
		{
			void* C_device;
			cuda_error = cudaHostGetDevicePointer(&C_device, ptr, 0);
			if (cuda_error != cudaSuccess || ptr != C_device)
			{
				if (cuda_error != cudaSuccess) fprintf(STD_OUT, "cudaHostGetDevicePtr: CUDA Error %d: %s\n", cuda_error, cudaGetErrorString(cuda_error));
				else fprintf(STD_OUT, "Host pointer does not match device pointer\n");
				cudaFreeHost(ptr);
				return(NULL);
			}
		}
		return((double*) ptr);
	}
	double* ptr = caldgemm::AllocMemory(nDoubles, page_locked, huge_pages, gpuaccessible, interleave);
	return(ptr);
}

int caldgemm_cuda::FreeMemory(double* ptr, bool gpuaccessible)
{
	if (gpuaccessible)
	{
		for (int i = 0;i < nGPUMEM;i++)
		{
			if (gpu_mem[i].ptr == (void*) ptr)
			{
				cudaFreeHost(ptr);
				gpu_mem[i] = gpu_mem[--nGPUMEM];
				return(0);
			}
		}
	}
	return(caldgemm::FreeMemory(ptr));
}
