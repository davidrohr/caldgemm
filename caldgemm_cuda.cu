/*
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
	if (Config->SimpleGPUQueuing) return(0);
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
	if (nDevices > (signed) max_devices) nDevices = max_devices;
	if (nDevices > Config->NumDevices) nDevices = Config->NumDevices;
	if (deviceNum >= nDevices) ERRRET("CUDA Device %d not available\n", deviceNum);
	if (deviceNum >= 0) nDevices = 1;
	if (!Config->UseGPU) nDevices = 0;
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

	AlternateLookaheadDoneMutexSQ.Lock();
	
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
	KernelSettings.transposeA = true;
	KernelSettings.transposeB = false;
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
			CHKRET(cudaEventCreateWithFlags(&cuda_events[i][j], cudaEventDisableTiming), "Creating Event 0 %d %d\n", i, j);
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
	return(0);
}

int caldgemm_cuda::ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA ExecuteKernels\n");

	if (Config->Debug) fprintf(STD_OUT, "\tExecuting MM kernel (device %d obuffer %d, k=%lld m=%lld n=%lld)\n", Task.device, Task.j, (long long int) Task.k, (long long int) blockm, (long long int) blockn);
	
	int use_queue = Task.j;
	
	if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g')
	{
		CHKRET(cudaStreamWaitEvent(cuda_command_queues[Task.device][Task.j], alternateSimpleQueueCopyCEvent[Task.device][Task.j], 0), "StreamWaitEvent");
	}
	else
	{
		if ((Config->AlternateSimpleQueuing || Task.j != simple_queue_events[Task.device][0][blockm].num_queue) && simple_queue_event_requested[Task.device][use_queue][0][blockm] == 0)
		{
			CHKRET(cudaStreamWaitEvent(cuda_command_queues[Task.device][Task.j], simple_queue_events[Task.device][0][blockm].event, 0), "StreamWaitEvent");
			simple_queue_event_requested[Task.device][use_queue][0][blockm] = 1;
		}
		if ((Config->AlternateSimpleQueuing || Task.j != simple_queue_events[Task.device][1][blockn].num_queue) && simple_queue_event_requested[Task.device][use_queue][1][blockn] == 0)
		{
			CHKRET(cudaStreamWaitEvent(cuda_command_queues[Task.device][Task.j], simple_queue_events[Task.device][1][blockn].event, 0), "StreamWaitEvent");
			simple_queue_event_requested[Task.device][use_queue][1][blockn] = 1;
		}
	}
	

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
	cublasDgemm(cublas_handles[Task.device], CUBLAS_OP_N, CUBLAS_OP_T, height1, height2, width, &Alpha, bbuffer, height1, abuffer, height2, &Beta, cbuffer, pitch);
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
	
	if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n)
	{
		if (!Config->SimpleGPUQueuing)
		{
			CHKRET(cudaEventRecord(cuda_events[Task.device][Task.j], cuda_command_queues[Task.device][Task.j]), "Recording event %d %d", Task.device, Task.j); //This is only needed to check whether alternate lookahead has finished
		}
		else if(AlternateLookaheadTilesRemaining && blockm < AlternateLookaheadBlocksM)
		{
			CHKRET(cudaEventRecord(AlternateLookaheadTilesRemainingSQ_events[AlternateLookaheadTilesRemaining - 1], cuda_command_queues[Task.device][Task.j]), "Recording alternateLookeadSQevent %d", AlternateLookaheadTilesRemaining); //This is only needed to check whether alternate lookahead has finished
		}
	}
	if (NeedSimpleQueueKernelEvent(blockm, blockn, Task.k, Task.device))
	{
		const int buf = (DGEMM_favor_m ? buffer_pointers_A[Task.device][blockm] : buffer_pointers_B[Task.device][blockn]) % ibuffercount;
		CHKRET(cudaEventRecord(simple_queue_event_kernels[Task.device][buf][Task.j], cuda_command_queues[Task.device][Task.j]), "Recording simple queue kernel buffer event  %d %d", Task.device, Task.j);
		simple_queue_event_kernels_used[Task.device][buf][use_queue] = true;
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
	AlternateLookaheadDoneMutexSQ.Unlock();

	return(0);
}

int caldgemm_cuda::FetchResult(int device, int j, int m, int n, int mustlock)
{
	return(0);
}

int caldgemm_cuda::CheckDMAQueue(int device, int forcej)
{
	return(0);
}

int caldgemm_cuda::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)
{
	return(0);
}

int caldgemm_cuda::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0 CALDGEMM_DIVBUFA)
{
	if (Config->Debug) fprintf(STD_OUT, "CUDA DGEMM_prepare k=%lld j=%d device=%d\n", (long long int) k, j, num_device);
	
	CALDGEMM_PREPARE_BACKEND_VARS1;

	size_t width, height;

	CHKRET(cudaSetDevice(cuda_devices[num_device]), "Setting CUDA Device");
	if (Config->VerboseTiming) Timers.CounterCopyTo.Start();

	conversionKernelTaskStruct convTasks[2];
	int nConvTasks = 0;

	for (int iMat = 0;iMat < 2;iMat++)
	{
		if (!Config->SimpleGPUQueuing && cuda_conversion_events_use[num_device][iMat])
		{
			WaitForEventAndRelease(&cuda_conversion_events[num_device][iMat]);
			cuda_conversion_events_use[num_device][iMat] = 0;
		}
		if (iMat ? prepareN : prepareM)
		{
			CALDGEMM_PREPARE_BACKEND_VARS2;
			
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of %c to GPU (k = %lld, m = %lld, n = %lld)\n", myMat, (long long int) k, (long long int) blockm, (long long int) blockn);
			
			if (!access_bbuffers && Config->SimpleGPUQueuing)
			{
				for (int i = 0;i < obuffercount;i++)
				{
					if (Config->AlternateSimpleQueuing) i = 2;
					if ((Config->AlternateSimpleQueuing || i != j) && simple_queue_event_kernels_used[num_device][destbuffer][i])
					{
						CHKRET(cudaStreamWaitEvent(cuda_command_queues[num_device][j], simple_queue_event_kernels[num_device][destbuffer][i], 0), "StreamWaitEvent");
					}
					simple_queue_event_kernels_used[num_device][destbuffer][i] = false;
					if (Config->AlternateSimpleQueuing) break;
				}
			}

			width = (((bool) iMat ^ myTranspose) ? myHeight : Config->Width) * sizeof(double);
			height = ((bool) iMat ^ myTranspose) ? Config->Width : myHeight;
			int arg_transpose = myTranspose ^ myKernelTranspose;
			void* dest_image = access_bbuffers ? cuda_bbuffers[num_device][destbuffer] : cuda_abuffers[num_device][destbuffer];
			void*& dest_buffer_tmp = arg_transpose ? (iMat ? cuda_tmp_bbuffers[num_device][j] : cuda_tmp_abuffers[num_device][j]) : dest_image;

			if (Config->Debug) fprintf(STD_OUT, "Transfer %c to GPU: region %d x %d\n", myMat, (int) width, (int) height);
			CHKRET(cudaMemcpy2DAsync(dest_buffer_tmp, width, src_ptr, pitch * sizeof(double), width, height, cudaMemcpyHostToDevice, cuda_command_queues[num_device][j]), "Copying %c to device", myMat);
			if (Config->Debug && Config->VerboseTiming) cudaStreamSynchronize(cuda_command_queues[num_device][j]);

			if (arg_transpose)
			{
				size_t arg_width = width / sizeof(double), arg_height = height;
				convTasks[nConvTasks++] = conversionKernelTaskStruct(dest_buffer_tmp, dest_image, arg_width, arg_height, myMat);
			}
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
	
	for (int i = 0;i < nConvTasks;i++)
	{
		dim3 threads(GROUP_SIZE_X, GROUP_SIZE_Y), blocks(GROUP_COUNT_X, GROUP_COUNT_Y);
		if (Config->Debug) fprintf(STD_OUT, "Conversion Kernel %c: x %d y %d\n", convTasks[i].myMat, (int) convTasks[i].arg_width, (int) convTasks[i].arg_height);
		CUDAConversionKernel <<<blocks, threads, 0, cuda_command_queues[num_device][j]>>> ((double*) convTasks[i].dest_buffer_tmp, (double*) convTasks[i].dest_image, convTasks[i].arg_width, convTasks[i].arg_height);
		CHKRET(cudaGetLastError(), "CUDA Conversion Kernel Execution");
		if (Config->Debug && Config->VerboseTiming) cudaStreamSynchronize(cuda_command_queues[num_device][j]);
	}
	
	for (int iMat = 0;iMat < 2;iMat++)
	{
		if (iMat ? prepareN : prepareM)
		{
			if (Config->SimpleGPUQueuing)
			{
				if (!(Config->AlternateSimpleQueuing && Config->DstMemory == 'g'))
				{
					size_t& myblock = iMat ? blockn : blockm;
					simple_queue_events[num_device][iMat][myblock].num_queue = j;
					CHKRET(cudaEventRecord(simple_queue_events[num_device][iMat][myblock].event, cuda_command_queues[num_device][j]), "Recording simpleQueueEvent event %d %d", num_device, iMat);
				}
			}
			else
			{
				CHKRET(cudaEventRecord(cuda_conversion_events[num_device][iMat], cuda_command_queues[num_device][j]), "Recording conversion event %d %d", num_device, iMat);
				cuda_conversion_events_use[num_device][iMat] = 1;
			}
		}
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
			CHKRET(cudaEventDestroy(cuda_events[i][j]), "Destroying Event 0 %d %d\n", i, j);
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

int caldgemm_cuda::Preallocate()
{
	if (caldgemm::Preallocate()) return(1);
	simple_queue_events[0][0] = new caldgemm_cuda_simple_queue_event[nDevices * (Config->PreallocData + Config->PreallocData)];
	for (int i = 0;i < nDevices * (Config->PreallocData + Config->PreallocData);i++) CHKRET(cudaEventCreateWithFlags(&simple_queue_events[0][0][i].event, cudaEventDisableTiming), "Creating Prealocate Event 0 %d\n", i);
	simple_queue_event_requested[0][0][0] = new bool[nDevices * obuffercount * (Config->PreallocData + Config->PreallocData)];
	memset(simple_queue_event_requested[0][0][0], 0, nDevices * obuffercount * (Config->PreallocData + Config->PreallocData) * sizeof(bool));
	AlternateLookaheadTilesRemainingSQ_events = new cudaEvent_t[Config->PreallocData * PREALLOC_ALTERNATE_LOOKAHEAD];
	for (int i = 0;i < Config->PreallocData * PREALLOC_ALTERNATE_LOOKAHEAD;i++) CHKRET(cudaEventCreateWithFlags(&AlternateLookaheadTilesRemainingSQ_events[i], cudaEventDisableTiming), "Creating Prealocate Event 1 %d\n", i);
	return(0);
}

int caldgemm_cuda::PreallocateFree()
{
	if (caldgemm::PreallocateFree()) return(1);
	for (int i = 0;i < nDevices * (Config->PreallocData + Config->PreallocData);i++) CHKRET(cudaEventDestroy(simple_queue_events[0][0][i].event), "Destroying Prealocate Event 0 %d\n", i);
	delete[] simple_queue_events[0][0];
	delete[] simple_queue_event_requested[0][0][0];
	for (int i = 0;i < Config->PreallocData * PREALLOC_ALTERNATE_LOOKAHEAD;i++) CHKRET(cudaEventDestroy(AlternateLookaheadTilesRemainingSQ_events[i]), "Destroying Prealocate Event 1 %d\n", i);
	delete[] AlternateLookaheadTilesRemainingSQ_events;
	return(0);
}

int caldgemm_cuda::RunCALDGEMM_Init()
{
	for (int i = 0;i < nDevices;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			cuda_conversion_events_use[i][j] = 0;
		}
	}
	if (Config->SimpleGPUQueuing)
	{
		const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
		const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;

		if (((int) (DGEMM_favor_m ? nb : mb) + nDevices - 1) / nDevices > min_bbuffers)
		{
			fprintf(STD_OUT, "SimpleGPUQueuing can only work if [Number of BBuffers] * [Number of GPUs] > [Number of Blocks in one dimension] (bbuffers %d, devices %d, blocks %d)\n", min_bbuffers, nDevices, (int) (DGEMM_favor_m ? nb : mb));
			return(1);
		}

		if (!Config->PreallocData)
		{
			simple_queue_events[0][0] = new caldgemm_cuda_simple_queue_event[nDevices * (mb + nb)];
			for (unsigned int i = 0;i < nDevices * (mb + nb);i++) CHKRET(cudaEventCreateWithFlags(&simple_queue_events[0][0][i].event, cudaEventDisableTiming), "Creating Event 0 %d\n", i);
			simple_queue_event_requested[0][0][0] = new bool[nDevices * obuffercount * (mb + nb)];
			if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && AlternateLookaheadTilesFull)
			{
				AlternateLookaheadTilesRemainingSQ_events = new cudaEvent_t[AlternateLookaheadTilesFull];
				for (int i = 0;i < AlternateLookaheadTilesFull;i++) CHKRET(cudaEventCreateWithFlags(&AlternateLookaheadTilesRemainingSQ_events[i], cudaEventDisableTiming), "Creating Event 1 %d\n", i);
			}
		}
		else if (AlternateLookaheadTilesFull > Config->PreallocData * PREALLOC_ALTERNATE_LOOKAHEAD)
		{
			fprintf(STD_OUT, "Insufficient preallocation for alternate lookahead, increase PreallocData or increase constant!");
			return(1);
		}

		SetupSimpleQueue(mb, nb);
		if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g')
		{
			for (int i = 0;i < nDevices;i++) for (int j = 0;j < obuffercount;j++) alternateSimpleQueueCBuffferEvent[i][j].must_release = alternateSimpleQueueCBuffferEvent[i][j].used = false;
		}
		memset(simple_queue_event_requested[0][0][0], 0, nDevices * obuffercount * (mb + nb) * sizeof(bool));

		if (Config->AlternateSimpleQueuing)
		{
			memset(alternateSimpleQueueEvent_tmp_abuffers_used, 0, nDevices * obuffercount * sizeof(bool));
			memset(alternateSimpleQueueEvent_tmp_bbuffers_used, 0, nDevices * obuffercount * sizeof(bool));
		}
		memset(&simple_queue_event_kernels_used[0][0][0], 0, nDevices * ibuffercount * obuffercount * sizeof(bool));
	}
	return(0);
}

int caldgemm_cuda::RunCALDGEMM_Exit()
{
	CHKRET(cudaThreadSynchronize(), "Synchronizing CUDA Thread");
	
	if (Config->SimpleGPUQueuing && !Config->PreallocData)
	{
		const size_t mb = (gpu_m + Config->Height - 1) / Config->Height;
		const size_t nb = (gpu_n + Config->Height - 1) / Config->Height;

		for (unsigned int i = 0;i < nDevices * (mb + nb);i++) CHKRET(cudaEventDestroy(simple_queue_events[0][0][i].event), "Destroying Event 0 %d\n", i);
		delete[] simple_queue_events[0][0];
		delete[] simple_queue_event_requested[0][0][0];
		if (ExecLinpack >= 2 && Config->AlternateLookahead > matrix_n && AlternateLookaheadTilesFull)
		{
			for (int i = 0;i < AlternateLookaheadTilesFull;i++) CHKRET(cudaEventDestroy(AlternateLookaheadTilesRemainingSQ_events[i]), "Destroying Event 1 %d\n", i);
			delete[] AlternateLookaheadTilesRemainingSQ_events;
		}
	}

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
		warn_wrong_memory_allocation = false;
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
				warn_wrong_memory_allocation = (nGPUMEM == 0);
				return(0);
			}
		}
	}
	return(caldgemm::FreeMemory(ptr));
}

int caldgemm_cuda::CheckAlternateTilesRemainingSQ()
{
	for (int i = 0;i < AlternateLookaheadTilesFull;i++)
	{
		CHKRET(cudaEventSynchronize(AlternateLookaheadTilesRemainingSQ_events[i]), "Error waiting for alternate lookahead tiles events %d", i);
	}
	AlternateLookaheadDoneMutexSQ.Unlock();
	return(0);
}

void caldgemm_cuda::SetupSimpleQueue(size_t mb, size_t nb)
{
	if (Config->AlternateSimpleQueuing && Config->DstMemory == 'g') return;
	for (int i = 0;i < nDevices;i++) for (int j = 0;j < obuffercount;j++) for (int k = 0;k < 2;k++)
		if (i || j || k)
		{
			simple_queue_event_requested[i][j][k] = &simple_queue_event_requested[0][0][0][(k ? mb : 0) + j * (mb + nb) + i * obuffercount * (mb + nb)];
			if (j == 0)
			{
				simple_queue_events[i][k] = &simple_queue_events[0][0][(k ? mb : 0) + i * (mb + nb)];
			}
		}
}
