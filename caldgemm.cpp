/* ============================================================

The source code is property of the Frankfurt Institute for Advanced Studies (FIAS).
None of the material may be copied, reproduced, distributed, republished, downloaded,
displayed, posted or transmitted in any form or by any means, including, but not
limited to, electronic, mechanical, photocopying, recording, or otherwise,
without the prior written permission of FIAS.

Authors:
David Rohr (drohr@jwdt.org)
Matthias Bach (bach@compeng.uni-frankfurt.de)
Matthias Kretz (kretz@compeng.uni-frankfurt.de)

============================================================ */

#include "caldgemm.h"

#define ILKernelName ILKernel
#include "caldgemm.il"
#undef ILKernelName
#define ILKernelName ILKernelALPHA1
#define CALDGEMM_ALPHA1
#include "caldgemm.il"
#undef ILKernelName
#define ILKernelName ILKernelLinpack
#define CALDGEMM_LINPACK_KERNEL
#include "caldgemm.il"
#undef CALDGEMM_LINPACK_KERNEL
#undef CALDGEMM_ALPHA1
#undef ILKernelName

const char* caldgemm::ILFakeKernel =
"il_ps_2_0\n"
"dcl_input_position_interp(linear_noperspective) vWinCoord0.xy__\n"
"end\n"
;

#include <syscall.h>
#include <errno.h>
extern "C" {
#include <common.h>
}
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3

#ifndef SHM_HUGETLB
#define SHM_HUGETLB 04000
#endif

template <class T> T mymin(const T a, const T b) {return(a < b ? a : b);}
template <class T> T mymax(const T a, const T b) {return(a > b ? a : b);}

#define CHKERR(cmd, text) if (cmd != CAL_RESULT_OK) {fprintf(STD_OUT, "Error '%s' while " text "\n", calGetErrorString());return(1);}
#define WAITFOREVENT(ctx, eventnr) { CALresult r; if (Config->Debug) fprintf(STD_OUT, "\tWaiting for event from context %d...\n", eventnr); do { r = calCtxIsEventDone(ctx, events[eventnr]); if (r == CAL_RESULT_ERROR) { fprintf(STD_OUT, "Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}
#define WAITFOREVENTA(ctx, event) { CALresult r; do { r = calCtxIsEventDone(ctx, event); if (r == CAL_RESULT_ERROR) { fprintf(STD_OUT, "Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}

caldgemm::caldgemm()
{
	caldgemm_initialized = false;
	memset(linpack_last_mn, 0, max_linpack_callback_types * sizeof(double));
	memset(linpackGPURatios, 0, 3 * sizeof(double));
	memset(linpackCPUDGEMMTime, 0, 3 * sizeof(double));
	memset(linpackBcastTime, 0, 3 * sizeof(double));

	avggflops = 0;
	avgngflops = 0;
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
	DeviceNum = 0;
	Width = 1024;
	Height = 4096;
	AutoHeight = true;
	Iterations = 1;
	DstMemory = 'c';
	VerboseTiming = false;
	AsyncTiming = false;
	TabularTiming = false;
	Debug = false;
	MultiThread = true;
	UseGPU = true;
	UseCPU = true;
	GPURatio = -1.0;
	DynamicSched = true;
	MemPolicy = true;
	DumpMatrix = false;
	DivideToGPU = false;
	AsyncDMA = true;
	KeepBuffersMapped = true;
	NoPerformanceWarnings = false;
	PinCPU = 0;
	m = 0;
	n = 0;
	LinpackNodes = 0;
	LinpackSwapN = NULL;
	MPIRank = -1;
	PreOut = EmptyOut;
	GPUClock = 0;
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
	for (int j = 0;j < height;j += stridey)
	{
		for (int jj = j;jj < j + suby && jj < height;jj++)
		{
			for (int i = 0;i < width;i += stridex)
			{
				for (int ii = i;ii < i + subx && ii < width;ii++)
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
							if (jj >= Config->m - Config->m & Config->Height || ii >= Config->n - cParam.cblas_size) sprintf(tmpcolor, "01;34");
						}

						size_t k = gpu_m / Config->Height * gpu_n / Config->Height;
						for (int l = 0;l < cParam.dynamic_run2;l++)
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
						fprintf(STD_OUT, "\33[%sm%d\33[%sm%+10.3lf\t", ok ? "01;32" : "01;31", ok , tmpcolor, M[jj * pitch + ii]);
					}
					else
					{
						fprintf(STD_OUT, " %+10.3lf\t", M[jj * pitch + ii]);
					}
				}
			}
			fprintf(STD_OUT, "\33[0m\n");
		}
	}
	fprintf(STD_OUT, "Done\n");
}

#ifdef CALDGEMM_UNALIGNED_ADDRESSES
#define _mm_load_pd_use _mm_loadu_pd
#else
#define _mm_load_pd_use _mm_load_pd
#endif

#define _mm_store_pd_use _mm_stream_pd
#define CALDGEMM_USE_VEC_MEMCPY_PREFETCH

int caldgemm::divideBuffer(BufferProperties* dst, double* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers, bool transpose)
{
	if (Config->Debug) fprintf(STD_OUT, "\t\tSRC=0x%llx, w: %d, h: %d, pitch: %d (gpuw: %d, gpuh: %d, transpose: %d)\n", (long long int) src, width, height, pitch, gpu_width, gpu_height, (int) transpose);

	if (Config->DivideToGPU)
		for (unsigned int i = 0;i < numBuffers;i++)
		{
			CHKERR(calResMap(&dst[i].ptr_void, &dst[i].pitch, dst[i].res, 0), "mapping input buffer for buffer division");
			if (((size_t) dst[i].ptr_void) & (vcpysize - 1))
			{
				fprintf(STD_OUT, "Invalid alignment\n");
				return(1);
			}
		}

		if (transpose)
		{
#if !defined(CALDGEMM_44)
			if (numBuffers <= 4)
			{
				for (int y = 0;y < width;y += 4)
				{
					double* saddr = src + (y * pitch);
					double* saddr2 = src + ((y + 1) * pitch);
					double* saddr3 = src + ((y + 2) * pitch);
					double* saddr4 = src + ((y + 3) * pitch);

					double* daddr = dst[0].ptr_double + y;
					double* daddr2 = dst[1 % numBuffers].ptr_double + (1 / numBuffers) * gpu_width + y;
					double* daddr3 = dst[2 % numBuffers].ptr_double + (2 / numBuffers) * gpu_width + y;
					double* daddr4 = dst[3 % numBuffers].ptr_double + (3 / numBuffers) * gpu_width + y;

					const int dpitch = 4 / numBuffers * gpu_width;

					for (int i = 0;i < height;i += 4)
					{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
						//Prefetching disabled as it currently has a negative performance impact
						/*_mm_prefetch(saddr + 100, _MM_HINT_NTA);
						_mm_prefetch(saddr2 + 100, _MM_HINT_NTA);
						_mm_prefetch(saddr3 + 100, _MM_HINT_NTA);
						_mm_prefetch(saddr4 + 100, _MM_HINT_NTA);*/
#endif
						__m128d x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;
						x1 = _mm_load_pd_use(saddr);
						x3 = _mm_load_pd_use(saddr + 2);
						x2 = _mm_load_pd_use(saddr2);
						x4 = _mm_load_pd_use(saddr2 + 2);
						x5 = _mm_load_pd_use(saddr3);
						x7 = _mm_load_pd_use(saddr3 + 2);
						x6 = _mm_load_pd_use(saddr4);
						x8 = _mm_load_pd_use(saddr4 + 2);

						x9 = _mm_unpacklo_pd(x1, x2);
						x10 = _mm_unpackhi_pd(x1, x2);
						x1 = _mm_unpacklo_pd(x3, x4);
						x2 = _mm_unpackhi_pd(x3, x4);
						x3 = _mm_unpacklo_pd(x5, x6);
						x4 = _mm_unpackhi_pd(x5, x6);
						x5 = _mm_unpacklo_pd(x7, x8);
						x6 = _mm_unpackhi_pd(x7, x8);

						_mm_store_pd_use(daddr, x9);
						_mm_store_pd_use(daddr2, x10);
						_mm_store_pd_use(daddr + 2, x3);
						_mm_store_pd_use(daddr2 + 2, x4);
						_mm_store_pd_use(daddr3, x1);
						_mm_store_pd_use(daddr4, x2);
						_mm_store_pd_use(daddr3 + 2, x5);
						_mm_store_pd_use(daddr4 + 2, x6);

						saddr += 4;
						saddr2 += 4;
						saddr3 += 4;
						saddr4 += 4;

						daddr += dpitch;
						daddr2 += dpitch;
						daddr3 += dpitch;
						daddr4 += dpitch;
					}
				}
			}
			else
#endif
#if (defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B))
				assert((height & 3) == 0);
				const int height_4 = height / 4;

				for (int y=0; y < width; y += 4)
				{
					const double *__restrict__ saddr0 = &src[(y + 0) * pitch];
					const double *__restrict__ saddr2 = &src[(y + 2) * pitch];

					double *__restrict__ dstBank0 = &dst[0].ptr_double[y * 2];
					double *__restrict__ dstBank1 = &dst[1].ptr_double[y * 2];

					for (int i = 0; i < height_4; ++i)
					{
						double *__restrict__ daddr0 = &dstBank0[i * gpu_width * 2];
						double *__restrict__ daddr1 = &dstBank1[i * gpu_width * 2];

						const __m128d x0 = _mm_load_pd_use(&saddr0[0]);
						const __m128d x1 = _mm_load_pd_use(&saddr0[pitch]);
						const __m128d x2 = _mm_load_pd_use(&saddr2[0]);
						const __m128d x3 = _mm_load_pd_use(&saddr2[pitch]);
						saddr0 += 2;
						saddr2 += 2;

						const __m128d x4 = _mm_load_pd_use(&saddr0[0]);
						const __m128d x5 = _mm_load_pd_use(&saddr0[pitch]);
						const __m128d x6 = _mm_load_pd_use(&saddr2[0]);
						const __m128d x7 = _mm_load_pd_use(&saddr2[pitch]);
						saddr0 += 2;
						saddr2 += 2;

						_mm_stream_pd(&daddr0[0], _mm_unpacklo_pd(x0, x1));
						_mm_stream_pd(&daddr0[2], _mm_unpackhi_pd(x0, x1));
						_mm_stream_pd(&daddr0[4], _mm_unpacklo_pd(x2, x3));
						_mm_stream_pd(&daddr0[6], _mm_unpackhi_pd(x2, x3));

						_mm_stream_pd(&daddr1[0], _mm_unpacklo_pd(x4, x5));
						_mm_stream_pd(&daddr1[2], _mm_unpackhi_pd(x4, x5));
						_mm_stream_pd(&daddr1[4], _mm_unpacklo_pd(x6, x7));
						_mm_stream_pd(&daddr1[6], _mm_unpackhi_pd(x6, x7));
					}
				}
#else
				for (int y=0; y < width; y += 2)
				{
					double* saddr = src + (y * pitch);
					double* saddr2 = src + ((y + 1) * pitch);

					for (int i = 0;i < height;i += 2)
					{
#if defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
						int bank = (i / 2) % 2;
						double* daddr = dst[bank].ptr_double + (i / 4) * gpu_width * 2 + y * 2;
						double* daddr2 = dst[bank].ptr_double + (i / 4) * gpu_width * 2 + y * 2 + 2;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
						//Col Interleaved Storage, Numbuffers is either 2 or 4, might be optimized in 2 branches
						int bank = (y / 2) % numBuffers;
#ifdef CALDGEMM_DIAGONAL_TEXTURE
						double* daddr = dst[bank].ptr_double + i * gpu_width / 2 + (((y / 2) & 0xFFFFFFFE) + 2 * i) % (gpu_width / 2);
						double* daddr2 = dst[bank].ptr_double + (i + 1) * gpu_width / 2 + (((y / 2) & 0xFFFFFFFE) + 2 * i + 2) % (gpu_width / 2);
#else
						double* daddr = dst[bank].ptr_double + (i * gpu_width / numBuffers + ((y / numBuffers) & 0xFFFFFFFE));
						double* daddr2 = dst[bank].ptr_double + ((i + 1) * gpu_width / numBuffers + ((y / numBuffers) & 0xFFFFFFFE));
#endif
#else
						//Standard Storage
						int bank = (i) % numBuffers;
						int bank2 = (i + 1) % numBuffers;
						double* daddr = dst[bank].ptr_double + (i / numBuffers) * gpu_width + y;
						double* daddr2 = dst[bank2].ptr_double + (i / numBuffers) * gpu_width + y;
#endif

#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
						_mm_prefetch(saddr + 100, _MM_HINT_NTA);
						_mm_prefetch(saddr2 + 100, _MM_HINT_NTA);
#endif
						__m128d x1, x2, x3, x4;
						x1 = _mm_load_pd_use(saddr);
						x2 = _mm_load_pd_use(saddr2);
						x3 = _mm_unpacklo_pd(x1, x2);
						x4 = _mm_unpackhi_pd(x1, x2);
						_mm_store_pd_use(daddr, x3);
						_mm_store_pd_use(daddr2, x4);
						saddr += 2;
						saddr2 += 2;
					}
				}
#endif

		}
		else
		{
#if defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
			//Row / Col Interleaved Storage with 2 rows stored in one col
			for (int y = 0;y < height / 2;y++)
			{
				double* daddr = dst[y % 2].ptr_double + y / 2 * gpu_width * 2;
				double* saddr = src + 2 * y * pitch;
				double* saddr2 = src + (2 * y + 1) * pitch;
				for (int i = 0;i < width;i += 4)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(saddr + 60, _MM_HINT_NTA);
					_mm_prefetch(saddr2 + 60, _MM_HINT_NTA);
#endif
					_mm_store_pd_use(daddr + 0, _mm_load_pd_use(saddr));
					_mm_store_pd_use(daddr + 2, _mm_load_pd_use(saddr2));
					_mm_store_pd_use(daddr + 4, _mm_load_pd_use(saddr + 2));
					_mm_store_pd_use(daddr + 6, _mm_load_pd_use(saddr2 + 2));
					daddr += 8;
					saddr += 4;
					saddr2 += 4;
				}
			}

#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
			//Col Interleaved Storage for transposed A with 4x4, 8x4 and 8x8 tiling
			if (numBuffers == 4)
			{
				double* daddr = dst[0].ptr_double;
				double* daddr2 = dst[1].ptr_double;
				double* daddr3 = dst[2].ptr_double;
				double* daddr4 = dst[3].ptr_double;
				for (int y=0; y < height; y++)
				{
					int count = dst[0].DataSize * width;
					double* saddr = src + (y * pitch);

					for (int i = 0;i < count;i += 256)
					{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
						_mm_prefetch(saddr + 30, _MM_HINT_NTA);
#endif
						_mm_store_pd_use(daddr, _mm_load_pd_use(saddr));
						_mm_store_pd_use(daddr2, _mm_load_pd_use(saddr + 2));
						_mm_store_pd_use(daddr3, _mm_load_pd_use(saddr + 4));
						_mm_store_pd_use(daddr4, _mm_load_pd_use(saddr + 6));
						_mm_store_pd_use(daddr + 2, _mm_load_pd_use(saddr + 8));
						_mm_store_pd_use(daddr2 + 2, _mm_load_pd_use(saddr + 10));
						_mm_store_pd_use(daddr3 + 2, _mm_load_pd_use(saddr + 12));
						_mm_store_pd_use(daddr4 + 2, _mm_load_pd_use(saddr + 14));
						_mm_store_pd_use(daddr + 4, _mm_load_pd_use(saddr + 16));
						_mm_store_pd_use(daddr2 + 4, _mm_load_pd_use(saddr + 18));
						_mm_store_pd_use(daddr3 + 4, _mm_load_pd_use(saddr + 20));
						_mm_store_pd_use(daddr4 + 4, _mm_load_pd_use(saddr + 22));
						_mm_store_pd_use(daddr + 6, _mm_load_pd_use(saddr + 24));
						_mm_store_pd_use(daddr2 + 6, _mm_load_pd_use(saddr + 26));
						_mm_store_pd_use(daddr3 + 6, _mm_load_pd_use(saddr + 28));
						_mm_store_pd_use(daddr4 + 6, _mm_load_pd_use(saddr + 30));
						saddr += 32;
						daddr += 8;
						daddr2+= 8;
						daddr3 += 8;
						daddr4 += 8;
					}
					daddr += (gpu_width - width) / numBuffers;
					daddr2 += (gpu_width - width) / numBuffers;
					daddr3 += (gpu_width - width) / numBuffers;
					daddr4 += (gpu_width - width) / numBuffers;
				}
			}
			else
			{
				double* daddr = dst[0].ptr_double;
				double* daddr2 = dst[1].ptr_double;
				for (int y=0; y < height; y++)
				{
					int count = dst[0].DataSize * width;
					double* saddr = src + (y * pitch);

					for (int i = 0;i < count;i += 64)
					{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
						_mm_prefetch(saddr + 76, _MM_HINT_NTA);
#endif
						_mm_store_pd_use(daddr, _mm_load_pd_use(saddr));
						_mm_store_pd_use(daddr2, _mm_load_pd_use(saddr + 2));
						_mm_store_pd_use(daddr + 2, _mm_load_pd_use(saddr + 4));
						_mm_store_pd_use(daddr2 + 2, _mm_load_pd_use(saddr + 6));
						saddr += 8;
						daddr += 4;
						daddr2+= 4;
					}
					daddr += (gpu_width - width) / numBuffers;
					daddr2 += (gpu_width - width) / numBuffers;
				}
			}
#else
			// Array to store the position from which data will be filled in the various output buffers.
			int* position = new int[numBuffers];
			memset((void*) position, 0, numBuffers * sizeof(int));
			for (int y=0; y < height; y++)
			{
				int bank = y % numBuffers;
				double* daddr = dst[bank].ptr_double + position[bank];
				double* saddr = src + (y * pitch);
				int count = dst[bank].DataSize * width;

				for (int i = 0;i < count;i += 64)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(saddr + 100, _MM_HINT_NTA);
#endif
					_mm_store_pd_use(daddr, _mm_load_pd_use(saddr));
					_mm_store_pd_use(daddr + 2, _mm_load_pd_use(saddr + 2));
					_mm_store_pd_use(daddr + 4, _mm_load_pd_use(saddr + 4));
					_mm_store_pd_use(daddr + 6, _mm_load_pd_use(saddr + 6));
					saddr += 8;
					daddr += 8;
				}

				position[bank] += gpu_width;
			}
			delete[] position;
#endif
		}

		if (Config->DivideToGPU)
			for (unsigned int i = 0;i < numBuffers;i++)
			{
				CHKERR(calResUnmap(dst[i].res), "unmapping input buffer for buffer division");
			}
			return(0);
}

int caldgemm::mergeBuffers(double* dst, BufferProperties* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers)
{
#ifdef CALDGEMM_BENCHMARK_KERNEL
	return(0);
#endif
	if (Config->DstMemory == 'c' && !Config->KeepBuffersMapped)
	{
		for (unsigned int i = 0;i < dwBuffersC;i++)
		{
			CHKERR(calResMap(&src[i].ptr_void, &src[i].pitch, src[i].res, 0), "mapping output buffer for merging");
			if (((size_t) src[i].ptr_void) & (vcpysize - 1))
			{
				fprintf(STD_OUT, "Invalid alignment\n");
				return(1);
			}
		}
	}
		
#if defined(CALDGEMM_44) && !defined(CALDGEMM_USE_MEMEXPORT)
	const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double
	const unsigned long long int double_minus_one = 0xBFF0000000000000;

	if (Config->Width == 1024 && reinterpret_cast<long long int &>(Beta) == double_one && reinterpret_cast<long long int &>(Alpha) == double_minus_one)
	{
		//Special Linpack Function
		for (int y=0; y < height; y++)
		{
			int bank = y % 4;
			double* saddr = src[bank].ptr_double + (y / 4) * (gpu_width / 2);
			double* saddr2 = src[bank + 4].ptr_double + (y / 4) * (gpu_width / 2);
			double* daddr = dst + (y * pitch);
			int count = src[bank].DataSize * width;


#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
				for (int i = 0;i < width;i += 8)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(saddr + 50, _MM_HINT_NTA);
					_mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
#ifndef _NO_AMD_CPU
					_m_prefetchw(daddr + 50);
#else
					_mm_prefetch(daddr + 50, _MM_HINT_NTA);
#endif
#endif
					_mm_store_pd_use(daddr, _mm_sub_pd(_mm_load_pd(daddr), _mm_load_pd(saddr)));
					_mm_store_pd_use(daddr + 4, _mm_sub_pd(_mm_load_pd(daddr + 4), _mm_load_pd(saddr + 2)));
					_mm_store_pd_use(daddr + 2, _mm_sub_pd(_mm_load_pd(daddr + 2), _mm_load_pd(saddr2)));
					_mm_store_pd_use(daddr + 6, _mm_sub_pd(_mm_load_pd(daddr + 6), _mm_load_pd(saddr2 + 2)));

					saddr += 4;
					saddr2 += 4;
					daddr += 8;
				}
			
				
		}
	}
	else
#endif
	{
	// Array to store the position from which data will be pulled in from the input buffers
	int* position = new int[numBuffers];
	memset((void*) position, 0, numBuffers * sizeof(int));

	for (int y=0; y < height; y++)
	{
		//CALDGEMM_44 Init
#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)
		int bank = y % 4;
		double* saddr2 = src[bank + 4].ptr_double + position[bank];
		double* paddr2 = src[(y + 1) % 4 + 4].ptr_double + position[(y + 1) % 4];
#else
		int bank = y % numBuffers;
#endif

		double* daddr = dst + (y * pitch);
		double* saddr = src[bank].ptr_double + position[bank];
		double* paddr = src[(y + 1) % 4].ptr_double + position[(y + 1) % 4];
		int count = src[bank].DataSize * width;

#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)

		if (Config->KeepBuffersMapped)
		{
			if (__fpclassify(Beta) == FP_ZERO)
			{
				//CALDGEMM_44 BETA=ZERO HACKED LIB
				for (int i = 0;i < count;i += 64)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(saddr + 50, _MM_HINT_NTA);
					_mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
#endif
					_mm_store_pd_use(daddr, _mm_load_pd(saddr));
					_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr2));
					_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 2));
					_mm_store_pd_use(daddr + 6, _mm_load_pd(saddr2 + 2));
					saddr += 4;
					saddr2 += 4;
					daddr += 8;
				}
			}
			else
			{
				//CALDGEMM_44 GENERAL CASE ORIGINAL LIB
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
				__m128d beta = _mm_set1_pd(Beta);
				for (int i = 0;i < count;i += 64)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(saddr + 50, _MM_HINT_NTA);
					_mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
#ifndef _NO_AMD_CPU
					_m_prefetchw(daddr + 50);
#else
					_mm_prefetch(daddr + 50, _MM_HINT_NTA);
#endif
#endif
					_mm_store_pd_use(daddr, _mm_add_pd(_mm_load_pd(saddr), _mm_mul_pd(beta, _mm_load_pd(daddr))));
					_mm_store_pd_use(daddr + 4, _mm_add_pd(_mm_load_pd(saddr + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 4))));
					_mm_store_pd_use(daddr + 2, _mm_add_pd(_mm_load_pd(saddr2), _mm_mul_pd(beta, _mm_load_pd(daddr + 2))));
					_mm_store_pd_use(daddr + 6, _mm_add_pd(_mm_load_pd(saddr2 + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 6))));

					saddr += 4;
					saddr2 += 4;
					paddr += 4;
					paddr2 += 4;
					daddr += 8;
				}
			}
		}
		else
		{
			if (__fpclassify(Beta) == FP_ZERO)
			{
				//CALDGEMM_44 BETA=ZERO ORIGINAL LIB
				for (int i = 0;i < count;i += 128)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(saddr + 50, _MM_HINT_NTA);
					_mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
#endif
					_mm_store_pd_use(daddr, _mm_load_pd(saddr));
					_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr2));
					_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 2));
					_mm_store_pd_use(daddr + 6, _mm_load_pd(saddr2 + 2));
					_mm_store_pd_use(daddr + 8, _mm_load_pd(saddr + 4));
					_mm_store_pd_use(daddr + 10, _mm_load_pd(saddr2 + 4));
					_mm_store_pd_use(daddr + 12, _mm_load_pd(saddr + 6));
					_mm_store_pd_use(daddr + 14, _mm_load_pd(saddr2 + 6));
					saddr += 8;
					saddr2 += 8;
					daddr += 16;
				}
			}
			else
			{
				//CALDGEMM_44 GENERAL CASE ORIGINAL LIB
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
				__m128d beta = _mm_set1_pd(Beta);
				for (int i = 0;i < count;i += 128)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					//    		    _mm_prefetch(saddr + 50, _MM_HINT_NTA);
					//    		    _mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
#ifndef _NO_AMD_CPU
					_m_prefetchw(daddr + 50);
#else
					_mm_prefetch(daddr + 50, _MM_HINT_NTA);
#endif
#endif
					_mm_store_pd_use(daddr, _mm_add_pd(_mm_load_pd(saddr), _mm_mul_pd(beta, _mm_load_pd(daddr))));
					_mm_store_pd_use(daddr + 4, _mm_add_pd(_mm_load_pd(saddr + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 4))));
					_mm_store_pd_use(daddr + 8, _mm_add_pd(_mm_load_pd(saddr + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 8))));
					_mm_store_pd_use(daddr + 12, _mm_add_pd(_mm_load_pd(saddr + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 12))));
					_mm_store_pd_use(daddr + 2, _mm_add_pd(_mm_load_pd(saddr2), _mm_mul_pd(beta, _mm_load_pd(daddr + 2))));
					_mm_store_pd_use(daddr + 6, _mm_add_pd(_mm_load_pd(saddr2 + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 6))));
					_mm_store_pd_use(daddr + 10, _mm_add_pd(_mm_load_pd(saddr2 + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 10))));
					_mm_store_pd_use(daddr + 14, _mm_add_pd(_mm_load_pd(saddr2 + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 14))));
					saddr += 8;
					saddr2 += 8;
					/*    		    paddr += 8;
									paddr2 += 8;*/
					daddr += 16;
				}
			}
		}

		position[bank] += gpu_width / 2;
#else        
		if (__fpclassify(Beta) == FP_ZERO)
		{
			//CALDGEMM_84 BETA=0
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_stream_pd
			for (int i = 0;i < count;i += 64)
			{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
				_mm_prefetch(saddr + 100, _MM_HINT_NTA);
#endif
				_mm_store_pd_use(daddr, _mm_load_pd(saddr));
				_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr + 2));
				_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 4));
				_mm_store_pd_use(daddr + 6, _mm_load_pd(saddr + 6));
				saddr += 8;
				daddr += 8;
			}
		}
		else
		{
			//CALDGEMM_82 General Case
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
			__m128d beta = _mm_set1_pd(Beta);
			for (int i = 0;i < count;i += 64)
			{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
				_mm_prefetch(saddr + 100, _MM_HINT_NTA);
#ifndef _NO_AMD_CPU
				_m_prefetchw(daddr + 100);
#endif
#endif
				_mm_store_pd_use(daddr, _mm_add_pd(_mm_load_pd(saddr), _mm_mul_pd(beta, _mm_load_pd(daddr))));
				_mm_store_pd_use(daddr + 2, _mm_add_pd(_mm_load_pd(saddr + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 2))));
				_mm_store_pd_use(daddr + 4, _mm_add_pd(_mm_load_pd(saddr + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 4))));
				_mm_store_pd_use(daddr + 6, _mm_add_pd(_mm_load_pd(saddr + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 6))));
				saddr += 8;
				daddr += 8;
			}
		}

		position[bank] += gpu_width;
#endif //CALDGEMM_44
	}

	delete[] position;
	}
	if (Config->DstMemory == 'c' && !Config->KeepBuffersMapped)
	{
		for (unsigned int i = 0;i < dwBuffersC;i++)
		{
			CHKERR(calResUnmap(src[i].res), "unmapping output buffer for merging");
		}
	}
	return(0);
}

void caldgemm::checkCalPatch()
{
	unsigned char *RunProgPTL = (unsigned char *)(&calCtxRunProgram);
	unsigned char **RunProgWrapperFunc = *(unsigned char ***)((size_t)(*(unsigned int *)(RunProgPTL + 2)) + RunProgPTL + 6);
	//fprintf(STD_OUT, "RunProgWrapperFunc = %p, ddi_interface[?] = %p\n", RunProgWrapperFunc, RunProgWrapperFunc + (0x10f588 - 0x4220)/sizeof(void*));
	
	//10.9 ATI Driver
	unsigned char *RunProgFunc9 = *(RunProgWrapperFunc + (0x10f588 - 0x4220) / sizeof(void*));
	unsigned char *patchpos9 = RunProgFunc9 + 0x7fffe591b631 - 0x7fffe591b560;
	
	//10.10 ATI Driver
	unsigned char *RunProgFunc10 = *(RunProgWrapperFunc + (0x7ffff7fcc588 - 0x7ffff7ec1220) / sizeof(void*));
	unsigned char *patchpos10 = RunProgFunc10 + 0x7ffff5933cdf - 0x7ffff5933bd0;
		
	if (*patchpos9 == 0x74 || *patchpos10 == 0x74)
	{
		if (Config->KeepBuffersMapped && !Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: CAL library not patched, KeepBuffersMapped unavailable\n");
		Config->KeepBuffersMapped = false;
	}
	else if (*patchpos9 != 0xEB && *patchpos10 != 0xEB)
	{
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Unknown CAL Library found, KeepBuffersMapped unavailable\n");
		Config->KeepBuffersMapped = false;
	}
	else if (Config->Debug)
	{
		fprintf(STD_OUT, "Patched CAL library found, KeepBuffersMapped available\n");
	}
}

int caldgemm::InitCALDGEMM(caldgemm_config* pInfo)
{
	Config = pInfo;

	if (Config->Iterations > 1 && Config->UseCPU)
	{
		fprintf(STD_OUT, "ERROR: Multiple Iterations not supported with CPU enabled\n");
		return(1);
	}

	gethostname(hostname, 255);
	sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);

	CPU_ZERO(&gpumask);
	CPU_SET(Config->PinCPU, &gpumask);

	if (Config->Debug) fprintf(STD_OUT, "Init Caldgemm, setting CPU mask %X\n", getcpumask(&gpumask));
	if (0 != sched_setaffinity(0, sizeof(gpumask), &gpumask))
	{
		fprintf(STD_OUT, "Error setting CPU affinity\n");
		return(1);
	}

	if(!ValidateCALRuntime())
	{
		fprintf(STD_OUT, "Error. Could not find a compatible CAL runtime.\n");
		return 0;
	}

#ifdef CALDGEMM_44
	if (Config->Width % 8)
	{
		fprintf(STD_OUT, "Only width of multiples of 8 are computable.\n");
		return(0);
	}
	else if (Config->Width % 64)
	{
		Config->Width += 64 - Config->Width % 64;
		fprintf(STD_OUT, "Cannot allocate buffers of size that is not multiple of 64, increasing buffer size to %lld\n", (long long int) Config->Width);
	}
#else
	if (Config->Width % 64)
	{
		fprintf(STD_OUT, "Only width of size 64 are computable.\n");
		return(0);
	}
#endif
	if (Config->Height & 0x7)
	{
		fprintf(STD_OUT, "Only heights with multiple of 8 are computable.\n" );
		return(0);
	}

	numInputs = dwBuffersA + dwBuffersB;
	numOutputs = dwBuffersC;
	numConstantBuffers = 1;
	device = 0;

	if (Config->Debug) fprintf(STD_OUT, "Initializing CAL\n");
	if (Initialize(&device, &ctx_main, Config->DeviceNum))
	{
		return 1;
	}

	CALdeviceattribs attribs;
	attribs.struct_size = sizeof(CALdeviceattribs);
	if (calDeviceGetAttribs(&attribs, Config->DeviceNum) != CAL_RESULT_OK)
	{
		fprintf(STD_OUT, "Error getting device attributes\n");
		return 1;
	}
	if (!attribs.doublePrecision)
	{
		fprintf(STD_OUT, "The device does not support double precision\n");
		return(1);
	}

	if (Config->KeepBuffersMapped)
	{
		if (SetupKernel(ILFakeKernel, &fakeModule, &ctx_main, false)) return(1);
		if (RunProgram(&ctx_main, &fakeModule, 0, 0, events)) {fprintf(STD_OUT, "Error running test kernel on GPU\n"); return(1);}
		if (Config->KeepBuffersMapped) checkCalPatch();
		if (calModuleUnload(ctx_main, fakeModule) != CAL_RESULT_OK )
		{
			fprintf(STD_OUT, "Error unloading test module\n");
			fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
		}
	}
	outputthreads = Config->KeepBuffersMapped ? CALDGEMM_OUTPUT_THREADS : CALDGEMM_OUTPUT_THREADS_SLOW;

	for (int i = 0;i < max_bbuffers;i++)
	{
		if (i < 1)
		{
			if (SetupKernel(ILKernel, &modules[i][0], &ctx_main, (bool) (Config->Disassemble && i == 0)) ||
				SetupKernel(ILKernelALPHA1, &modules[i][1], &ctx_main, (bool) (Config->Disassemble && i == 0)) ||
				SetupKernel(ILKernelLinpack, &modules[i][2], &ctx_main, (bool) (Config->Disassemble && i == 0)))
			{
				return 1;
			}
			for (int j = 0;j < kernel_count;j++) progNames[i][j] = new CALname[numInputs + numOutputs + numConstantBuffers];
		}

		datas[i] = new BufferProperties[numInputs + numOutputs + numConstantBuffers];
		resourceHandlers[i] = new CALresource[numInputs + numOutputs + numConstantBuffers];
		memset(datas[i], 0, (numInputs + numOutputs + numConstantBuffers) * sizeof(BufferProperties));
		memset(resourceHandlers[i], 0, (numInputs + numOutputs + numConstantBuffers) * sizeof(CALresource));
		if (SetupData(modules[i], resourceHandlers[i], datas[i], &device, &ctx_main, numInputs, numOutputs, numConstantBuffers, progNames[i], i))
		{
			if (i < ctxcount) return 1;
			else break;
		}
		bbuffers = i + 1;


		if (i < ctxcount && Config->MultiThread)
		{
			pthread_mutex_init(&obufferMutex[i], NULL);
		}

		if (i < max_outputthreads && Config->MultiThread)
		{
			mParam[i].cls = this;
			mParam[i].terminate = false;
			mParam[i].nMergeThread = i;
			for (int j = 0;j < 2;j++) pthread_mutex_init(&mParam[i].mergeThreadMutex[j], NULL);
			pthread_t thr;
			pthread_create(&thr, NULL, merge_wrapper, &mParam[i]);

			while (pthread_mutex_trylock(&mParam[i].mergeThreadMutex[0]) != EBUSY) if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		}
	}
	if (Config->Debug) fprintf(STD_OUT, "Was able to allocate %d bbuffers\n", bbuffers);
	if (Config->UseCPU)
	{
		cParam.cls = this;
		cParam.terminate = false;
		for (int j = 0;j < 2;j++) pthread_mutex_init(&cParam.cblasMutex[j], NULL);
		if (pthread_mutex_lock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (Config->MultiThread)
		{
			pthread_t thr;
			pthread_create(&thr, NULL, cblas_wrapper, &cParam);
			if (Config->Debug) fprintf(STD_OUT, "Waiting for cblas slave to start\n");
			while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		}
	}
	if (Config->MultiThread)
	{
		linpackParameters.terminate = false;
		for (int j = 0;j < 2;j++) pthread_mutex_init(&linpackParameters.linpackMutex[j], NULL);
		if (pthread_mutex_lock(&linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
		pthread_t thr;
		pthread_create(&thr, NULL, linpack_wrapper, this);
		if (Config->Debug) fprintf(STD_OUT, "Waiting for linpack slave to start\n");
		while (pthread_mutex_trylock(&linpackParameters.linpackMutex[1]) != EBUSY) if (pthread_mutex_unlock(&linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		pthread_mutex_init(&scheduleMutex, NULL);
	}

	if (Config->MemPolicy)
	{
		unsigned long nodemask = 0xffffff;
		syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
	}

	/*fprintf(STD_OUT, "Setting FIFO scheduler\n");
	sched_param param;
	sched_getparam( 0, &param );
	param.sched_priority = 1;
	if ( 0 != sched_setscheduler( 0, SCHED_FIFO, &param ) )
	{
	fprintf(STD_OUT, "Error setting scheduler\n");
	return(1);
	}*/
	//setpriority(PRIO_PROCESS, 0, -20);

	if (Config->Debug) fprintf(STD_OUT, "Caldgemm Init complete, setting CPU mask %X\n", getcpumask(&oldcpumask));
	sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

	caldgemm_initialized = true;

	return(0);
}

void caldgemm::cal_init_constant_data(BufferProperties* &data, double alpha)
{
	// Setup the constants for the kernel
	data[dwBuffersA + dwBuffersB].ptr_float[0] = (float) TILING_Y / Config->Height;			//Scale factor for normalized y pos
	data[dwBuffersA + dwBuffersB].ptr_float[2] = (float) TILING_X / Config->Height;			//Scale factor for normalized x pos
#ifdef CALDGEMM_44
	data[dwBuffersA + dwBuffersB].ptr_float[1] = 1.f / Config->Width;							//Step in K direction
	data[dwBuffersA + dwBuffersB].ptr_float[4] = static_cast<float>(Config->Width);			//Iterations of loop in IL Kernel
#else //CALDGEMM_44
	data[dwBuffersA + dwBuffersB].ptr_float[1] = 2.f / Config->Width;							//Step in K direction
	data[dwBuffersA + dwBuffersB].ptr_float[4] = static_cast<float>(Config->Width / (dwBuffersB << 2));	//Iterations of loop in IL Kernel
#endif //CALDGEMM_44
	data[dwBuffersA + dwBuffersB].ptr_float[3] = 0.f;
	data[dwBuffersA + dwBuffersB].ptr_float[5] = (float) dwBuffersA / Config->Height;			//For transposed matrix finer y resolution is needed
	data[dwBuffersA + dwBuffersB].ptr_float[8] = 0.5f - 0.5f / (float) (TILING_Y / dwBuffersA);

	//Constants for Memexport
	data[dwBuffersA + dwBuffersB].ptr_int[9] = TILING_Y * Config->Height / 2;				//2 for double2
	data[dwBuffersA + dwBuffersB].ptr_int[10] = TILING_X / 2;								//x tiling in double2
#if defined(CALDGEMM_84)
	data[dwBuffersA + dwBuffersB].ptr_int[12] = 0 + 0 * Config->Height / 2;					//8 consecutive entries in x
	data[dwBuffersA + dwBuffersB].ptr_int[13] = 1 + 0 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[14] = 2 + 0 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[15] = 3 + 0 * Config->Height / 2;

	data[dwBuffersA + dwBuffersB].ptr_int[16] = 0 + 1 * Config->Height / 2;					//Next row
	data[dwBuffersA + dwBuffersB].ptr_int[17] = 0 + 1 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[18] = 0 + 1 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[19] = 0 + 1 * Config->Height / 2;

	data[dwBuffersA + dwBuffersB].ptr_int[20] = 0 + 2 * Config->Height / 2;					//Proceed by two rows
	data[dwBuffersA + dwBuffersB].ptr_int[21] = 0 + 2 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[22] = 0 + 2 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[23] = 0 + 2 * Config->Height / 2;
#elif defined(CALDGEMM_44)
	data[dwBuffersA + dwBuffersB].ptr_int[12] = 0 + 0 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[13] = 1 + 0 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[14] = 0 + 1 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[15] = 1 + 1 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[16] = 0 + 2 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[17] = 1 + 2 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[18] = 0 + 3 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[19] = 1 + 3 * Config->Height / 2;
#ifdef CALDGEMM_48
	data[dwBuffersA + dwBuffersB].ptr_int[20] = 0 + 4 * Config->Height / 2;					//Proceed by 4 rows
	data[dwBuffersA + dwBuffersB].ptr_int[21] = 0 + 4 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[22] = 0 + 4 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[23] = 0 + 4 * Config->Height / 2;
#endif
#else
	data[dwBuffersA + dwBuffersB].ptr_int[12] = 0 + 0 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[13] = 0 + 4 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[14] = 0 + 1 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[15] = 0 + 5 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[16] = 0 + 2 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[17] = 0 + 6 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[18] = 0 + 3 * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[19] = 0 + 7 * Config->Height / 2;
#endif
#ifdef CALDGEMM_DIAGONAL_TEXTURE
	data[dwBuffersA + dwBuffersB].ptr_float[11] = 8.f / Config->Height;						//Offset for diagonal texture read
#endif
	data[dwBuffersA + dwBuffersB].ptr_double[3] = alpha;
}

int caldgemm::broadcastcore()
{
	return(outputthreads + 1);
}

void* linpack_wrapper(void* arg)
{
	caldgemm* cls = (caldgemm*) arg;
	volatile caldgemm::caldgemm_config* Config = cls->Config;
	if (Config->Debug) fprintf(STD_OUT, "Linpack helper thread started\n");

	cpu_set_t linpack_mask;
	CPU_ZERO(&linpack_mask);
	//CPU_SET(0, &linpack_mask);
	CPU_SET(Config->PinCPU + cls->outputthreads + 1, &linpack_mask);
	if (Config->Debug) fprintf(STD_OUT, "Linpack Thread, setting CPU mask %X\n", cls->getcpumask(&linpack_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &linpack_mask);

	if (pthread_mutex_lock(&cls->linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&cls->linpackParameters.linpackMutex[0]) == 0 && cls->linpackParameters.terminate == false)
	{
		cls->Timers.LinpackTimer2.Start();
		Config->linpack_broadcast_function();
		cls->Timers.LinpackTimer2.Stop();
		cls->Timers.BcastTimer.Start();

		if (pthread_mutex_unlock(&cls->linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
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
		const size_t mb = gpu_m / Config->Height;
		const size_t nb = gpu_n / Config->Height;
		size_t nBlocks = mb * nb;

		pthread_mutex_lock(&scheduleMutex);
		const size_t k = gpu_k_barrier == -1 ? 0 : gpu_k_barrier;

		if (gpu_k_barrier < nBlocks - 1)
		{
			size_t blockm, blockn;
			DGEMM_getblocks(k, blockm, blockn);

			if (cParam.dynamic_run == 0)
			{
				cParam.dynamic_size = ((1.0f - gpu_ratio_used) * (float) (nBlocks - k - 1) + 0.5) * Config->Height;
				if (cParam.dynamic_size > (nBlocks - k - 1) * Config->Height) cParam.dynamic_size = (nBlocks - k - 1) * Config->Height;
				if (cParam.dynamic_size > Config->Height)
				{
					cParam.dynamic_run = 1 + cParam.dynamic_size / mymin(gpu_m, gpu_n);
					cParam.dynamic_size /= cParam.dynamic_run;
					cParam.dynamic_size -= cParam.dynamic_size % Config->Height;
					cParam.dynamic_run *= Config->Height;

					while (DGEMM_favor_m ? (blockm * Config->Height >= gpu_m - cParam.dynamic_run && blockn * Config->Height >= gpu_n - cParam.dynamic_size) :
						(blockn * Config->Height >= gpu_n - cParam.dynamic_run && blockm * Config->Height >= gpu_m - cParam.dynamic_size))
					{
						cParam.dynamic_run -= Config->Height;
						cParam.dynamic_size = mymin(gpu_m, gpu_n);
						if (Config->Debug) fprintf(STD_OUT, "cParam dynamic size reduced to: %lld blockrows, %lld blocks\n", (long long int) cParam.dynamic_run / Config->Height, (long long int) cParam.dynamic_size / Config->Height);
					}

					if (nBlocks >= 256 && nBlocks - k - 1 > 16 && cParam.dynamic_run == Config->Height && cParam.dynamic_size < mymin(gpu_m, gpu_n)) cParam.dynamic_size += Config->Height;

					if (!Config->Quiet) fprintf(STD_OUT, "Scheduling Additional CPU DGEMM Run over %lld blockrows, %lld blocks\n", (long long int) cParam.dynamic_run / Config->Height, (long long int) cParam.dynamic_size / Config->Height);
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
				size_t test_cpu_k = cpu_k_barrier - 1;
				size_t cpublockm, cpublockn;
				DGEMM_getblocks(test_cpu_k, cpublockm, cpublockn);
				while (test_cpu_k > k && (DGEMM_favor_m ? (cpublockm * Config->Height >= gpu_m - cParam.dynamic_run && cpublockn * Config->Height >= gpu_n - cParam.dynamic_size) :
					(cpublockn * Config->Height >= gpu_n - cParam.dynamic_run && cpublockm * Config->Height >= gpu_m - cParam.dynamic_size)))
				{
					test_cpu_k--;
					DGEMM_getblocks(test_cpu_k, cpublockm, cpublockn);
				}
				if (test_cpu_k && k < test_cpu_k - 1)
				{
					if (!Config->Quiet) fprintf(STD_OUT, "Scheduling dynamic 3rd phase run, CPU taking tile %lld (m=%lld,n=%lld) from GPU\n", (long long int) test_cpu_k, (long long int) cpublockm, (long long int) cpublockn);
					cParam.dynamic_run2++;
					cParam.cpu_k = test_cpu_k;
					cpu_k_barrier = test_cpu_k;
					retVal = 1;
				}
			}
		}
		pthread_mutex_unlock(&scheduleMutex);
	}
	return(retVal);
}

void* cblas_wrapper(void* arg)
{
	volatile caldgemm::cblasParameters* par = (caldgemm::cblasParameters*) arg;
	volatile caldgemm::caldgemm_config* Config = par->cls->Config;

	if (Config->Debug) fprintf(STD_OUT, "Cblas helper thread started\n");

	if (Config->Debug) fprintf(STD_OUT, "Cblas thread Thread, setting CPU mask %X\n", par->cls->getcpumask(&par->cls->oldcpumask));
	sched_setaffinity(0, sizeof(par->cls->oldcpumask), &par->cls->oldcpumask);
	
	if (Config->MultiThread) if (pthread_mutex_lock(&par->cls->cParam.cblasMutex[1])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
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
		const size_t A_pitch_use = (par->cls->TransposeA == CblasTrans ? 1 : A_pitch);
		const size_t B_pitch_use = (par->cls->TransposeB == CblasTrans ? B_pitch : 1);
		const CBLAS_TRANSPOSE TransposeA = par->cls->TransposeA;
		const CBLAS_TRANSPOSE TransposeB = par->cls->TransposeB;
		if (!Config->Quiet) fprintf(STD_OUT, "\t\tSlave thread starting cblas (m: %lld, n: %lld, cblas_size: %lld (%lld), dynamic: %lld/%lld, cpu_k: %lld)\n", (long long int) Config->m, (long long int) Config->n, (long long int) par->cblas_size, (long long int) Config->Height, (long long int) par->dynamic_run, (long long int) par->dynamic_size, (long long int) par->cpu_k);


		int old_goto_threads = get_num_procs();

		int require_threads = par->cls->outputthreads + 1;
		if (par->cls->ExecLinpack && par->cls->Config->LinpackNodes > 1)
		{
			require_threads++;
		}
		if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu (/ Linpack)\n", require_threads);
		goto_set_num_threads(old_goto_threads - require_threads);
		caldgemm_goto_reserve_cpus(require_threads);

		par->cls->Timers.TotalCPUTimer.Start();
		par->cls->Timers.LinpackTimer3.Start();
		if (Config->LinpackSwapN != NULL)
		{
			goto_set_num_threads(8);
			Config->linpack_swap_function();
			goto_set_num_threads(old_goto_threads - require_threads);
		}
		par->cls->Timers.LinpackTimer3.Stop();

		if (par->cls->ExecLinpack)
		{
			if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tDoint initial cblas runs to prepare Linpack factorization\n");
			par->cls->Timers.CPUTimer.Start();
			cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->Width, Config->n, Config->Width, Alpha, A - Config->Width * A_pitch_use, A_pitch, B, B_pitch, Beta, C - Config->Width * C_pitch, C_pitch);
			par->cls->Timers.CPUTimer.Stop();
#ifndef NO_ASYNC_LINPACK
			if (!Config->Quiet) fprintf(STD_OUT, "\t\t\tStarting Linpack factorization\n");
			goto_set_num_threads(8);
			par->cls->Timers.LinpackTimer1.Start();
			Config->linpack_factorize_function();
			par->cls->Timers.LinpackTimer1.Stop();
			goto_set_num_threads(old_goto_threads - require_threads);

			if (par->cls->Config->LinpackNodes > 1)
			{
				if (Config->MultiThread)
				{
					pthread_mutex_unlock(&par->cls->linpackParameters.linpackMutex[0]);
				}
				else
				{
					par->cls->Timers.LinpackTimer1.Start();
					Config->linpack_broadcast_function();
					par->cls->Timers.LinpackTimer1.Stop();
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
				cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->Height, Config->Height, Config->Width, Alpha, A + blockm * Config->Height * A_pitch_use, A_pitch, B + blockn * Config->Height * B_pitch_use, B_pitch, Beta, C + blockm * Config->Height * C_pitch + blockn * Config->Height, C_pitch);
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
								if (!Config->NoPerformanceWarnings) printf("WARNING: Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
							}
							else
							{
								par->cls->Timers.BcastTimer.Stop();
								if (!Config->NoPerformanceWarnings && par->cls->Timers.BcastTimer.GetElapsedTime() > 1.0) fprintf(STD_OUT, "Bcast core idle for %2.4lf seconds\n", par->cls->Timers.BcastTimer.GetElapsedTime());

								int require_threads_new = par->cls->outputthreads + 1;
								if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
								goto_set_num_threads(old_goto_threads - require_threads_new);
								caldgemm_goto_reserve_cpus(require_threads_new);
								linpackfinished = true;
							}
						}
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->cblas_size, Config->n - cblas2, Config->Width, Alpha, A + (Config->m - par->cblas_size) * A_pitch_use, A_pitch, B + cblas2 * B_pitch_use, B_pitch, Beta, C + (Config->m - par->cblas_size) * C_pitch + cblas2, C_pitch);
					}

					if (Config->n % Config->Height && par->borders_done == false)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m - par->cblas_size, Config->n % Config->Height, Config->Width, Alpha, A, A_pitch, B + (Config->n - Config->n % Config->Height) * B_pitch_use, B_pitch, Beta, C + Config->n - Config->n % Config->Height, C_pitch);
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
								if (!Config->NoPerformanceWarnings) printf("Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
							}
							else
							{
								int require_threads_new = par->cls->outputthreads + 1;
								if (Config->Debug) fprintf(STD_OUT, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
								goto_set_num_threads(old_goto_threads - require_threads_new);
								caldgemm_goto_reserve_cpus(require_threads_new);
								linpackfinished = true;
							}
						}
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m - cblas2, par->cblas_size, Config->Width, Alpha, A + cblas2 * A_pitch_use, A_pitch, B + (Config->n - par->cblas_size) * B_pitch_use, B_pitch, Beta, C + cblas2 * C_pitch + Config->n - par->cblas_size, C_pitch);
					}

					if (Config->m % Config->Height && par->borders_done == false)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m % Config->Height, Config->n - par->cblas_size, Config->Width, Alpha, A + (Config->m - Config->m % Config->Height) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Config->m - Config->m % Config->Height) * C_pitch, C_pitch);
					}
				}
			}
			par->borders_done = true;
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
		if (pthread_mutex_unlock(&par->cls->cParam.cblasMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (!Config->MultiThread) break;
	}
	if (Config->Debug) fprintf(STD_OUT, "blas slave terminating\n");
	if (Config->MultiThread)
	{
		pthread_exit(NULL);
	}
	return(NULL);
}

void* merge_wrapper(void* arg)
{
	caldgemm::mergeParameters* par = (caldgemm::mergeParameters*) arg;

	if (par->cls->Config->Debug) fprintf(STD_OUT, "Merger Thread %d started\n", par->nMergeThread);

	cpu_set_t merge_mask;
	CPU_ZERO(&merge_mask);
	CPU_SET(par->cls->Config->PinCPU + par->nMergeThread + 1, &merge_mask);
	if (par->cls->Config->Debug) fprintf(STD_OUT, "Merge Thread %d, setting CPU mask %X\n", par->nMergeThread, par->cls->getcpumask(&merge_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &merge_mask);

	HighResTimer mergeTimer;

	if (pthread_mutex_lock(&par->mergeThreadMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&par->mergeThreadMutex[0]) == 0 && par->terminate == false)
	{
		if (par->cls->Config->Debug) fprintf(STD_OUT, "\t\tSlave thread %d starting merge process for context %d\n", par->nMergeThread, par->nContext);
		if (par->cls->Config->Debug)
		{
		    mergeTimer.Reset();
		    mergeTimer.Start();
		}
		par->cls->mergeBuffers(par->dst, par->src, par->cls->Config->Height, par->cls->Config->Height, par->cls->BufferHeight, par->cls->BufferHeight, par->cls->C_pitch, par->cls->dwBuffersC);
		if (par->cls->Config->Debug)
		{
		    mergeTimer.Stop();
		    fprintf(STD_OUT, "Merge time: %2.3lf\n", mergeTimer.GetElapsedTime());
		}
		if (par->cls->Config->Debug) fprintf(STD_OUT, "\t\tUnlocking mutex obuffer %d (Slavethread %d)\n", par->nContext, par->nMergeThread);
		if (pthread_mutex_unlock(&par->cls->obufferMutex[par->nContext])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (pthread_mutex_unlock(&par->mergeThreadMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}
	if (par->cls->Config->Debug) fprintf(STD_OUT, "merge slave %d terminating\n", par->nMergeThread);
	pthread_exit(NULL);
	return(NULL);
}

int caldgemm::DumpMatrix(double* a, double* b, double* c, double alpha, double beta, int tmp_m, int tmp_k, int tmp_n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB)
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
	fwrite(&a, sizeof(a), 1, fp);
	fwrite(&b, sizeof(b), 1, fp);
	fwrite(&c, sizeof(c), 1, fp);
	fwrite(&alpha, sizeof(alpha), 1, fp);
	fwrite(&beta, sizeof(beta), 1, fp);
	fwrite(&tmp_m, sizeof(tmp_m), 1, fp);
	fwrite(&tmp_k, sizeof(tmp_k), 1, fp);
	fwrite(&tmp_n, sizeof(tmp_n), 1, fp);
	fwrite(&Apitch, sizeof(Apitch), 1, fp);
	fwrite(&Bpitch, sizeof(Bpitch), 1, fp);
	fwrite(&Cpitch, sizeof(Cpitch), 1, fp);
	for (i = 0;i < tmp_m;i++)
	{
		fwrite(a + i * Apitch, sizeof(double), tmp_k, fp);
	}
	for (i = 0;i < tmp_k;i++)
	{
		fwrite(b + i * Bpitch, sizeof(double), tmp_n, fp);
	}
	fclose(fp);
	return(0);
}

void caldgemm::WaitForLASWP(size_t n)
{
	if (Config->LinpackSwapN != NULL)
	{
		while (*Config->LinpackSwapN < (n + 1) * Config->Height + (ExecLinpack ? Config->Width : 0))
		{
			if (Config->Debug) fprintf(STD_OUT, "Waiting for LASWP / DTRSM... %lld of %lld\n", (long long int) *Config->LinpackSwapN, (long long int) (n + 1) * Config->Height);
		}
	}
}

int caldgemm::RunCALDGEMM(double* a, double* b, double* c, double alpha, double beta, size_t tmp_m, size_t tmp_k, size_t tmp_n, size_t Apitch, size_t Bpitch, size_t Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int ExecuteLinpackCallbacks)
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "Caldgemm not initialized, aborting DGEMM run\n");
		return(1);
	}

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
			if (Config->LinpackNodes > 1) Config->linpack_broadcast_function();
			Timers.LinpackTimer1.Stop();
		}
		return(0);		//Do Nothing
	}

	/*  //Disable output for all but one host in MPI rin
	if (strcmp(hostname, "gpu-dev05") != 0)
	{
	Config->Debug = false;
	Config->Quiet = true;
	Config->Verify = false;
	}*/

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
	if (tmp_m != -1) Config->m = tmp_m;
	if (tmp_n != -1) Config->n = tmp_n;
	if (tmp_k != -1) Config->Width = tmp_k;

	A_pitch = Apitch != -1 ? Apitch : Config->Width;
	B_pitch = Bpitch != -1 ? Bpitch : Config->n;
	C_pitch = Cpitch != -1 ? Cpitch : Config->n;
	ResetTimers();

	if (order == CblasColMajor)
	{
		double* tmpd;
		size_t tmpi;
		CBLAS_TRANSPOSE tmpt;
		tmpd = A; A = B; B = tmpd;
		tmpi = Config->m; Config->m = Config->n; Config->n = tmpi;
		tmpi = A_pitch; A_pitch = B_pitch; B_pitch = tmpi;
		tmpt = TransA;TransA = TransB;TransB = tmpt;
	}

	if (!Config->Quiet) fprintf(STD_OUT, "Starting DGEMM Run m=%lld k=%lld n=%lld Alpha=%lf Beta=%lf LDA=0x%lx LDB=0x%lx LDC=0x%lx At=%d Bt=%d ColMajor=%d (A=0x%llx, B=0x%llx, C=0x%llx, (C-A=%lld, (C-B)/w=%lld))\n", (long long int) Config->m, (long long int) Config->Width, (long long int) Config->n, Alpha, Beta, A_pitch, B_pitch, C_pitch, (int) (TransA == CblasTrans), (int) (TransB == CblasTrans), (int) (order == CblasColMajor), (long long int) A, (long long int) B, (long long int) C, (long long int) ((size_t) C - (size_t) A) / sizeof(double), (long long int) ((size_t) C - (size_t) B) / sizeof(double) / Config->Width);

	//Check for double == 1.0 is unsafe and causes compiler warning
	const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double
	const unsigned long long int double_minus_one = 0xBFF0000000000000;
#if defined(CALDGEMM_44) && !defined(CALDGEMM_USE_MEMEXPORT)
	const int kernel_num = ((Config->Width == 1024 && reinterpret_cast<long long int &>(Beta) == double_one && reinterpret_cast<long long int &>(Alpha) == double_minus_one) ? 2 : (reinterpret_cast<long long int &>(Alpha) == double_one));
#else
	const int kernel_num = (reinterpret_cast<long long int &>(Alpha) == double_one);
#endif
	if (Config->Debug) fprintf(STD_OUT, "Using Kernel %d (alpha=0x%llX (%2.3lf), width = %lld)\n", kernel_num, (reinterpret_cast<long long int &>(Alpha)), Alpha, (long long int) Config->Width);

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

	if (Config->DumpMatrix) DumpMatrix(A, B, C, Alpha, Beta, Config->m, Config->Width, Config->n, A_pitch, B_pitch, C_pitch, CblasRowMajor, TransposeA, TransposeB);

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
	if (Config->Iterations > 1);
	else if (Config->Width % 8 || Config->Width < 256) forceCPU = true;
	else if (MaxGpuM < 512 || MaxGpuN < 512) forceCPU = true;
	else if (__fpclassify(Alpha) == FP_ZERO) forceCPU = true;
	else if (((size_t) A) & (vcpysize - 1) || ((size_t) B) & (vcpysize - 1) || ((size_t) C) & (vcpysize - 1) ||
		A_pitch & (vcpysize / sizeof(double) - 1) || B_pitch & (vcpysize / sizeof(double) - 1) || C_pitch & (vcpysize / sizeof(double) - 1))
	{
		fprintf(STD_OUT, "Input addresses not aligned correctly: A 0x%llX B 0x%llX C 0x%llX Pitch 0x%llX 0x%llX 0x%llX\n", (long long int) A, (long long int) B, (long long int) C, (long long int) A_pitch, (long long int) B_pitch, (long long int) C_pitch);
		forceCPU = true;
	}
#endif

	if (Config->AutoHeight)
	{
		if (ExecuteLinpackCallbacks >= 2)
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
			else if (MaxGpuM < 2048 || MaxGpuN < 2048 || MaxGpuM * MaxGpuN < 16 * 1024 * 1024)
			{
				Config->Height = 1024;
			}
			else if (MaxGpuM < 3072 || MaxGpuN < 3072 || MaxGpuM * MaxGpuN < 120 * 1024 * 1024)
			{
				Config->Height = 2048;
			}
			else if (MaxGpuM < 4096 || MaxGpuN < 4096 || MaxGpuM * MaxGpuN < (size_t) 60 * 60 * 1024 * 1024)
			{
				Config->Height = 3072;
			}
			else
			{
				Config->Height = 4096;
			}
		}
		if ((Config->Height != BufferHeight && !Config->Quiet) || Config->Debug)  fprintf(STD_OUT, "Using Height %lld of max %lld\n", (long long int) Config->Height, (long long int) BufferHeight);
	}

	if (Config->Width > BufferWidth || Config->Height > BufferHeight) forceReinit = true;

	if (Config->UseGPU == false || Config->m < Config->Height || Config->n < Config->Height || (forceReinit && (long long int) MaxGpuM * (long long int) MaxGpuN * (long long int) Config->Width < (long long int) 24 * 1024 * 1024 * 1024) || (Config->Width < 1024 && Config->Height < 1024)) forceCPU = true;

	/*  //Run on CPU on all but one node in MPIRUN
	if (strcmp(hostname, "gpu-dev05") != 0)
	{
	fprintf(STD_OUT, "Hostname not 5 but %s\n", hostname);
	forceCPU = true;
	}*/

	if (forceCPU)
	{
		if (Config->Debug) fprintf(STD_OUT, "Running CPU only DGEMM\n");
		if (Config->LinpackSwapN != NULL) Config->linpack_swap_function();
		Timers.CPUTimer.Start();
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m, Config->n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
		Timers.CPUTimer.Stop();
		CPUOnlyRun = true;
#ifndef NO_ASYNC_LINPACK
		if (ExecuteLinpackCallbacks)
		{
			if (Config->Debug) fprintf(STD_OUT, "DGEMM was running on CPU only, executing linpack callback functions\n");
			Timers.LinpackTimer1.Start();
			Config->linpack_factorize_function();
			if (Config->LinpackNodes > 1) Config->linpack_broadcast_function();
			Timers.LinpackTimer1.Stop();
		}
#endif
		goto RunCALDGEMM_end;
	}
	CPUOnlyRun = false;

	if (ExecuteLinpackCallbacks)
	{
		outputthreads = mymin(CALDGEMM_OUTPUT_THREADS_SLOW, outputthreads + CALDGEMM_EXTRA_OUTPUT_THREADS_LINPACK);
	}

	cpu_set_t divide_mask;
	CPU_ZERO(&divide_mask);
	if (ExecuteLinpackCallbacks)
	{
		//CPU_SET(outputthreads + 1, &divide_mask);
		CPU_SET(Config->PinCPU, &divide_mask);
	}
	else
	{
		CPU_SET(Config->PinCPU, &divide_mask);
	}
	if (Config->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&divide_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &divide_mask);

	if (forceReinit)
	{
		if (!Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Reinit for increased buffer width / height\n");
		for (int i = 0;i < bbuffers;i++)
		{
			CleanupData(&ctx_main, resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers, i);
			SetupData(modules[i], resourceHandlers[i], datas[i], &device, &ctx_main, numInputs, numOutputs, numConstantBuffers, progNames[i], i);
		}
	}

	if (Config->Debug) fprintf(STD_OUT, "Initiliazing GPU Constant Buffers...");
	for (int i = 0;i < 1;i++)
	{
		if (Config->Debug) fprintf(STD_OUT, "%d", i);
		cal_init_constant_data(datas[i], alpha);
		if (CopyDataToGPU(&ctx_main, resourceHandlers[i] + numInputs, datas[i] + numInputs, numConstantBuffers, true, &events[i])) return(1);
	}
	if (Config->Debug) fprintf(STD_OUT, "   Done\n");

	if (Config->GPURatio < 0)
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
		GPURatio *= (double) Config->Width / (double) 1024;
		if (Config->Height < 1024) GPURatio *= (double) Config->Height / (double) 1024 * (double) Config->Height / (double) 1024;
		
		const int require_threads = outputthreads + 1 + (ExecLinpack && Config->LinpackNodes > 1);
		GPURatio = GPURatio / (GPURatio + (1.0 - GPURatio) / (double) (24 - require_threads) * (double) (get_num_procs() - require_threads));
		
		if (Config->Debug) fprintf(STD_OUT, "GPURatio automatically set to %1.2lf\n", GPURatio);
		if ((Config->n + 4) % 4096 < 8) GPURatio = 1. - 0.95 * (1. - GPURatio);
	}
	else
	{
		GPURatio = Config->GPURatio;
	}

	if (ExecuteLinpackCallbacks)
	{
		if (ExecuteLinpackCallbacks > 1) GPURatio = 1.0 - (1.0 - GPURatio) * 0.80;
		else GPURatio = 1.0 - (1.0 - GPURatio) * 0.90;
		if ((((double) MaxGpuM * (double) MaxGpuN) - linpack_last_mn[ExecuteLinpackCallbacks]) / linpack_last_mn[ExecuteLinpackCallbacks] < 0.3 && linpackGPURatios[ExecuteLinpackCallbacks] > 0.0001)
		{
			GPURatio = linpackGPURatios[ExecuteLinpackCallbacks];
			if (Config->Debug) fprintf(STD_OUT, "Taking GPU Ratio from table, entry %d, val %2.1lf\n", ExecuteLinpackCallbacks, 100 * GPURatio);
		}
		else
		{
			linpackGPURatios[ExecuteLinpackCallbacks] = GPURatio;
			if (Config->Debug) fprintf(STD_OUT, "Initializing ratio table entry %d with %2.1lf\n", ExecuteLinpackCallbacks, 100 * GPURatio);
		}
	}

	gpu_ratio_used = GPURatio;

	if (ExecuteLinpackCallbacks)
	{
		Config->m -= Config->Width;
		A += Config->Width * (TransposeA == CblasTrans ? 1 : A_pitch);
		C += Config->Width * (C_pitch);
	}

	cParam.dynamic_run = 0;
	cParam.dynamic_run2 = 0;
	cParam.borders_done = false;
	if (Config->UseCPU == true && Config->UseGPU == true)
	{
		if (DGEMM_split_m = (Config->m >= Config->n))
		{
			size_t virtualm = Config->m + (Config->n % Config->Height) * Config->m / Config->n;
			if (ExecuteLinpackCallbacks) virtualm += Config->Width * (1.0 + (float) Config->m / Config->n);
			gpu_m = GPURatio * (float) virtualm + (Config->Height - 1);
			if (gpu_m > Config->m) gpu_m = Config->m;
			gpu_m -= gpu_m % Config->Height;
			cParam.cblas_size = Config->m - gpu_m;
			gpu_n = Config->n;
			gpu_n -= gpu_n % Config->Height;
			if (Config->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld\n", (long long int) gpu_m, (long long int) gpu_n, (long long int) Config->m - gpu_m, (long long int) gpu_n);
		}
		else
		{
			size_t virtualn = Config->n + (Config->m % Config->Height) * Config->n / Config->m;
			if (ExecuteLinpackCallbacks) virtualn += Config->Width * (1.0 + (float) Config->n / Config->m);
			gpu_n = GPURatio * (float) virtualn + (Config->Height - 1);
			if (gpu_n > Config->n) gpu_n = Config->n;
			gpu_n -= gpu_n % Config->Height;
			cParam.cblas_size = Config->n - gpu_n;
			gpu_m = Config->m;
			gpu_m -= gpu_m % Config->Height;
			if (Config->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld\n", (long long int) gpu_m, (long long int) gpu_n, (long long int) Config->m, (long long int) Config->n - gpu_n);
		}
		/*if (cParam.cblas_size == 0 && Config->DynamicSched == true)
		{
		cParam.dynamic_size = Config->Height;
		cParam.dynamic_run = (1.0f - GPURatio) * (float) mymax(gpu_m, gpu_n);
		cParam.dynamic_run -= cParam.dynamic_run % Config->Height;
		if (!Config->Quiet) fprintf(STD_OUT, "Scheduling initial dynamic run over %lldx%lld blocks\n", cParam.dynamic_run, cParam.dynamic_size);
		}*/
	}
	else
	{
		if (Config->n % Config->Height || Config->m % Config->Height)
		{
			fprintf(STD_OUT, "Invalid matrix size for GPU only\n");
			return(1);
		}
		gpu_n = Config->n;
		gpu_m = Config->m;
	}
	DGEMM_favor_m = (gpu_m >= gpu_n);

	if (Config->UseCPU)
	{
		if (!Config->MultiThread)
		{
			cblas_wrapper((void*) &cParam);
		}
		if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}
	else if (Config->LinpackSwapN != NULL)
	{
	    Config->linpack_swap_function();
	}

	Timers.GPUTimer.Start();

	for (unsigned int i = 0; i < Config->Iterations; ++i)
	{
		int oldj;
		int j = 0;
		int iMergeThread = 0;

		const size_t mb = gpu_m / Config->Height;
		const size_t nb = gpu_n / Config->Height;
		size_t blockm, blockn, lastm, lastn;
		size_t nBlocks = mb * nb;

		if (!Config->NoPerformanceWarnings && (buffersSwitchable ? mymin(nb, mb) : nb) > bbuffers) fprintf(STD_OUT, "WARNING: Insufficient buffers for Input Matrices, retransfer required\n");

		cParam.cpu_k = nBlocks;
		gpu_k_barrier = -1;
		cpu_k_barrier = nBlocks;
		if (gpu_n && gpu_m)
			for (size_t k = 0;k <= nBlocks;k++)
			{
				size_t newblockm, newblockn;
				if (k < nBlocks)
				{
					DGEMM_getblocks(k, newblockm, newblockn);

					if (cParam.dynamic_run)
					{
						if (DGEMM_favor_m)
						{
							if (newblockm * Config->Height >= gpu_m - cParam.dynamic_run && newblockn * Config->Height >= gpu_n - cParam.dynamic_size)
							{
								if (Config->Debug) fprintf(STD_OUT, "GPU skipping k = %lld (Dynamic Run 2nd Phase)\n", (long long int) k);
								continue;
							}
						}
						else
						{
							if (newblockn * Config->Height >= gpu_n - cParam.dynamic_run && newblockm * Config->Height >= gpu_m - cParam.dynamic_size)
							{
								if (Config->Debug) fprintf(STD_OUT, "GPU skipping k = %lld (Dynamic Run 2nd Phase)\n", (long long int) k);
								continue;
							}
						}
					}

					pthread_mutex_lock(&scheduleMutex);
					if (k < cpu_k_barrier)
					{
						gpu_k_barrier = k;
					}
					else
					{
						if (Config->Debug) fprintf(STD_OUT, "gpu_k %lld reached cpu_k_barrier %lld, skipping remaining k (Dynamic Run 3rd Phase", (long long int) k, (long long int) cpu_k_barrier);
						k = nBlocks;
					}
					pthread_mutex_unlock(&scheduleMutex);
				}

				lastm = blockm;
				lastn = blockn;
				if (k < nBlocks)
				{
					blockn = newblockn;
					blockm = newblockm;
					if (Config->Debug) fprintf(STD_OUT, "Iteration k = %lld, m = %lld, n = %lld (Context %d)\n", (long long int) k, (long long int) blockm, (long long int) blockn, j);

					if (k <= 1 || ctxcount == 1 || Config->AsyncDMA == false)
					{
						WaitForLASWP(blockm);
						DGEMM_prepare(k, j);
					}
					if (ctxcount > 1 && k >= 1 && Config->AsyncDMA)
					{
						size_t nextk = k + 1;
						size_t nextblockm, nextblockn;
						DGEMM_getblocks(nextk, nextblockm, nextblockn);
						if (cParam.dynamic_run)
						{
							while (DGEMM_favor_m ? (nextk < nBlocks && nextblockm * Config->Height >= gpu_m - cParam.dynamic_run && nextblockn * Config->Height >= gpu_n - cParam.dynamic_size) :
								(nextk < nBlocks && nextblockn * Config->Height >= gpu_n - cParam.dynamic_run && nextblockm * Config->Height >= gpu_m - cParam.dynamic_size))
							{
								nextk++;
								DGEMM_getblocks(nextk, nextblockm, nextblockn);
							}
						}
						if (nextk < nBlocks)
						{
							WaitForLASWP(nextblockm);
							DGEMM_prepare(nextk, (j + 1) % ctxcount);
						}
					}

					if (Config->MultiThread)
					{
						if (Config->Debug) fprintf(STD_OUT, "\tLocking mutex %d\n", j);
						if (Config->AsyncTiming)
						{
							Timers.ATime.Reset();
							Timers.ATime.Start();
						}
						if (pthread_mutex_lock(&obufferMutex[j])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
						if (Config->AsyncTiming)
						{
							Timers.ATime.Stop();
							if (!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001 || Config->Debug) fprintf(STD_OUT, "\t\tWait Time for output buffer: %1.5lf\n", Timers.ATime.GetElapsedTime());
						}
					}
					WAITFOREVENT(ctx_main, j);
					if (Config->Debug) fprintf(STD_OUT, "\tExecuting MM kernel\n");
					if (!DGEMM_favor_m && buffersSwitchable && bbuffers >= mb)
					{
						for (int l = 0;l < dwBuffersA;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][l], datas[blockm][dwBuffersA + l].dstMem), "setting kernel memory A");
						for (int l = 0;l < dwBuffersB;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][dwBuffersA + l], datas[blockn % 2][l].dstMem), "setting kernel memory B");
					}
					else
					{
						for (int l = 0;l < dwBuffersA;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][l], datas[blockm % 2][l].dstMem), "setting kernel memory A");
						for (int l = dwBuffersA;l < dwBuffersA + dwBuffersB;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][l], datas[nb > bbuffers ? (blockn % 2) : blockn][l].dstMem), "setting kernel memory B");
					}
					for (int l = 0;l < dwBuffersC;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][numInputs + numConstantBuffers + l], datas[j][numInputs + numConstantBuffers + l].dstMem), "setting kernel output memroy");
					if (RunProgram(&ctx_main, &modules[0][kernel_num], Config->Height / TILING_X, Config->Height / TILING_Y, &events[j])) {fprintf(STD_OUT, "Error running program\n"); return 1;}
					calCtxFlush(ctx_main);
				}
				if (ctxcount == 1)
				{
					oldj = j;
					lastm = blockm;
					lastn = blockn;
				}
				if ((ctxcount > 1) ? (k > 0) : (k < nBlocks))
				{
					WAITFOREVENT(ctx_main, oldj);
					if (Config->DstMemory == 'g')
					{
						if (Config->VerboseTiming) Timers.CounterCopyFrom.Start();
						if (Config->Debug == true) fprintf(STD_OUT, "\tFething part of C from GPU (m = %lld, n = %lld)\n", (long long int) lastm, (long long int) lastn);
						if (CopyDataFromGPU(&ctx_main, resourceHandlers[oldj] + numInputs + numConstantBuffers, datas[oldj] + numInputs + numConstantBuffers, numOutputs, &events[oldj])) {fprintf(STD_OUT, "Error copying from GPU\n"); return(1);}
						if (Config->VerboseTiming) Timers.CounterCopyFrom.Stop();
						WAITFOREVENT(ctx_main, oldj);
					}
					if (Config->VerboseTiming) Timers.CounterMerge.Start();

					if (k == nBlocks || Config->MultiThread == false)
					{
						if (Config->Debug) fprintf(STD_OUT, "\tMerging buffer (context %d, main thread)\n", oldj);
						if (mergeBuffers(C + lastn * Config->Height + lastm * C_pitch * Config->Height, datas[oldj] + numInputs + numConstantBuffers, Config->Height, Config->Height, BufferHeight, BufferHeight, C_pitch, dwBuffersC)) {fprintf(STD_OUT, "Error merging\n"); return(1);}
						if (Config->MultiThread)
						{
							if (pthread_mutex_unlock(&obufferMutex[oldj])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
							for (int l = 1;l < ctxcount;l++)
							{
								if (Config->Debug) fprintf(STD_OUT, "Waiting for context %d to finish merge process\n", (oldj + l) % ctxcount);
								if (pthread_mutex_lock(&obufferMutex[(oldj + l) % ctxcount])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
								if (pthread_mutex_unlock(&obufferMutex[(oldj + l) % ctxcount])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
							}
						}
					}
					else
					{
						if (Config->AsyncTiming)
						{
							Timers.ATime.Reset();
							Timers.ATime.Start();
						}
						if (pthread_mutex_lock(&mParam[iMergeThread].mergeThreadMutex[1])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
						if (Config->AsyncTiming)
						{
							Timers.ATime.Stop();
							if (!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001 || Config->Debug) fprintf(STD_OUT, "\t\tWARNING: Wait Time for merge thread: %1.5lf\n", Timers.ATime.GetElapsedTime());
						}
						if (Config->Debug) fprintf(STD_OUT, "\t\tUnlocking outputthread mutex %d to process context %d\n", iMergeThread, oldj);
						mParam[iMergeThread].nContext = oldj;
						mParam[iMergeThread].dst = C + (lastn * Config->Height + lastm * C_pitch * Config->Height);
						mParam[iMergeThread].src = datas[oldj] + numInputs + numConstantBuffers;
						if (pthread_mutex_unlock(&mParam[iMergeThread].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
						iMergeThread = (iMergeThread + 1) % outputthreads;
					}

					if (Config->VerboseTiming) Timers.CounterMerge.Stop();
				}
				oldj = j;
				j = (j + 1) % ctxcount;
			}
			if(Config->Verify && i < Config->Iterations - 1) AnalyzeResults();
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
		if (pthread_mutex_lock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (Config->MultiThread)
		{
			Timers.ATime.Stop();
			cpu_wait_time = Timers.ATime.GetElapsedTime();
			if (!Config->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() >= 0.15 && cParam.cblas_size > 0) fprintf(STD_OUT, "WARNING: CPU synchronisation took %2.4lf sec\n", Timers.ATime.GetElapsedTime());
			else if (Config->Debug) fprintf(STD_OUT, "CPU synchronisation took %2.4lf sec\n", Timers.ATime.GetElapsedTime());
		}
	}

RunCALDGEMM_end:
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
		if (Config->LinpackNodes > 1) Config->linpack_broadcast_function();
		Timers.LinpackTimer1.Stop();
	}

	Timers.System.Stop();
	if (Config->Debug) fprintf(STD_OUT, "DGEMM Run Complete\n");

#ifdef TESTMODE
	print_submatrices(C, 12, 24, Config->n, 1, 1, 1, 1);
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
		if (Config->Debug) fprintf(STD_OUT, "updating ratio table entry %d (old: %2.1lf, new: %2.1lf, factor: %2.1lf) => %2.1lf\n", ExecuteLinpackCallbacks, 100 * linpackGPURatios[ExecuteLinpackCallbacks], 100 * gpu_ratio_used, tmpratio, 100 * newratio);

		linpackGPURatios[ExecuteLinpackCallbacks] = newratio;
		linpackCPUDGEMMTime[ExecuteLinpackCallbacks] = Timers.CPUTimer.GetElapsedTime();
		linpackBcastTime[ExecuteLinpackCallbacks] = Timers.LinpackTimer2.GetElapsedTime();
		linpack_last_mn[ExecuteLinpackCallbacks] = (double) Config->m * (double) Config->n;
	}

	return(0);
}

inline void caldgemm::DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn)
{
	if (DGEMM_favor_m)
	{
		const int nb = gpu_n / Config->Height;
		blockn = k % nb;
		blockm = k / nb;
	}
	else
	{
		const int mb = gpu_m / Config->Height;
		blockm = k % mb;
		blockn = k / mb;
	}
}

int caldgemm::DGEMM_prepare(size_t k, int j)
{
#ifdef CALDGEMM_BENCHMARK_KERNEL
	return(0);
#endif
	const size_t nb = gpu_n / Config->Height;
	const size_t mb = gpu_m / Config->Height;
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	bool buffersSufficiant;
#ifdef REUSE_BBUFFERS
	if (DGEMM_favor_m)
	{
		buffersSufficiant = (bbuffers >= nb);
	}
	else
	{
		buffersSufficiant = (bbuffers >= mb && buffersSwitchable);
	}
#else
	buffersSufficiant = false;
#endif

	if (Config->VerboseTiming) Timers.CounterDivide.Start();
	if (blockn == 0 || (!DGEMM_favor_m && !buffersSufficiant)) 
	{
		if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer A (k = %lld)\n", (long long int) k);
		Timers.divideA++;
#ifdef CALDGEMM_TRANSPOSED_A
		if (divideBuffer(Config->DivideToGPU && !DGEMM_favor_m && buffersSufficiant ? (datas[blockm] + dwBuffersA) : datas[blockm % 2], A + blockm * Config->Height * (TransposeA == CblasTrans ? 1 : A_pitch), Config->Height, Config->Width, BufferHeight, BufferWidth, A_pitch, dwBuffersA, TransposeA == CblasNoTrans)) return(1);
#else
		if (divideBuffer(Config->DivideToGPU && !DGEMM_favor_m && buffersSufficiant ? (datas[blockm] + dwBuffersA) : datas[blockm % 2], A + blockm * Config->Height * (TransposeA == CblasTrans ? 1 : A_pitch), Config->Width, Config->Height, BufferWidth, BufferHeight, A_pitch, dwBuffersA, TransposeA == CblasTrans)) return(1);
#endif
	}
	if (blockm == 0 || (DGEMM_favor_m && !buffersSufficiant))
	{
		if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer B (k = %lld)\n", (long long int) k);
		Timers.divideB++;
#ifdef CALDGEMM_TRANSPOSED_B
		divideBuffer(Config->DivideToGPU && buffersSufficiant ? (datas[blockn] + (DGEMM_favor_m ? dwBuffersA : 0)) : (datas[blockn % 2] + dwBuffersA), B + blockn * Config->Height * (TransposeB == CblasTrans ? B_pitch : 1), Config->Width, Config->Height, BufferWidth, BufferHeight, B_pitch, dwBuffersB, TransposeB == CblasNoTrans);
#else
		divideBuffer(Config->DivideToGPU && buffersSufficiant ? (datas[blockn] + (DGEMM_favor_m ? dwBuffersA : 0)) : (datas[blockn % 2] + dwBuffersA), B + blockn * Config->Height * (TransposeB == CblasTrans ? B_pitch : 1), Config->Height, Config->Width, BufferHeight, BufferWidth, B_pitch, dwBuffersB, TransposeB == CblasTrans);
#endif
	}
	if (Config->VerboseTiming) Timers.CounterDivide.Stop();

	if (Config->VerboseTiming) Timers.CounterCopyTo.Start();
	if (Config->DivideToGPU == false)
	{
		if (blockn == 0 || (!DGEMM_favor_m && !buffersSufficiant))
		{
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (k = %lld)\n", (long long int) k);
			if (!DGEMM_favor_m && buffersSufficiant)
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j], datas[blockm % 2], dwBuffersA, false, &events[j], datas[blockm] + dwBuffersA)) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
			else
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j], datas[blockm % 2], dwBuffersA, false, &events[j])) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
		}

		if (blockm == 0 || (DGEMM_favor_m && !buffersSufficiant))
		{
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of B to GPU (k = %lld)\n", (long long int) k);
			if (!DGEMM_favor_m && buffersSufficiant)
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j] + dwBuffersA, datas[blockn % 2] + dwBuffersA, dwBuffersB, false, &events[j], datas[blockn % 2])) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
			else
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j] + dwBuffersA, datas[blockn % 2] + dwBuffersA, dwBuffersB, false, &events[j], datas[buffersSufficiant ? blockn : (blockn % 2)] + dwBuffersA)) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
		}
	}
	if (Config->VerboseTiming) Timers.CounterCopyTo.Stop();
	calCtxFlush(ctx_main);
}

int caldgemm::ExitCALDGEMM()
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "CALDGEMM not initialized, cannot uninitialize!\n");
		return(1);
	}
	if (Config->Debug) fprintf(STD_OUT, "Uninitializing CALDGEMM\n");
	for (int i = 0;i < bbuffers;i++)
	{
		if (Cleanup(&device, &ctx_main, modules[i], resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers, i))
		{
			return 1;
		}
		if (i < max_outputthreads && Config->MultiThread)
		{
			if (Config->Debug) fprintf(STD_OUT, "Trying to terminate merge slave %d\n", i);
			mParam[i].terminate = true;
			if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mergemutex %d/1 to terminate slave\n", i);
		}
	}

	// Close the device
	if (ctx_main) calCtxDestroy(ctx_main);
	if (device)
	{
		if (calDeviceClose(device) != CAL_RESULT_OK )
		{
			fprintf(STD_OUT, "There was an error closing the device.\n");
			fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
		}
	}

	// Shutdown cal device
	if (calShutdown() != CAL_RESULT_OK )
	{
		fprintf(STD_OUT, "There was an error during cal shutdown.\n");
		fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
	}

	for (int j = 0;j < kernel_count;j++) delete[] progNames[0][j];

	if (Config->UseCPU && Config->UseGPU)
	{
		if (Config->Debug) fprintf(STD_OUT, "Trying to terminate blas slave\n");
		cParam.terminate = true;
		if (Config->MultiThread && pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking blas mutex 1 to terminate thread\n");
		if (pthread_mutex_unlock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "Error unlocking blas mutex 0 to terminate thread\n");
	}

	if (Config->MultiThread)
	{
		if (Config->Debug) fprintf(STD_OUT, "Trying to terminate linpack slave\n");
		linpackParameters.terminate = true;
		if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error unlocking blas mutex 0 to terminate thread\n");
		if (Config->Debug) fprintf(STD_OUT, "Waiting for linpack slave to terminate\n");
		while (pthread_mutex_trylock(&linpackParameters.linpackMutex[0]) != EBUSY) if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error unlocking blasMutex 1\n");

		if (Config->Debug) fprintf(STD_OUT, "Waiting for merge threads to terminate\n");
		for (int i = 0;i < max_outputthreads;i++)
		{
			while (pthread_mutex_trylock(&mParam[i].mergeThreadMutex[0]) != EBUSY) if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mergeMutex %d/1\n", i);
		}
		if (Config->UseCPU && Config->UseGPU)
		{
			if (Config->Debug) fprintf(STD_OUT, "Waiting for blas threads to terminate\n");
			while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking blasMutex 1\n");
		}
	}

	for (int j = 0;j < 2;j++)
	{
		if (Config->UseCPU && Config->UseGPU) if (pthread_mutex_destroy(&cParam.cblasMutex[j])) fprintf(STD_OUT, "Error destroying blas mutex %d\n", j);
	}
	if (Config->MultiThread)
	{
		for (int i = 0;i < ctxcount;i++) if (pthread_mutex_destroy(&obufferMutex[i])) fprintf(STD_OUT, "Error destroying obuffermutex %d\n", i);
		for (int i = 0;i < max_outputthreads;i++) for (int j = 0;j < 2;j++) if (pthread_mutex_destroy(&mParam[i].mergeThreadMutex[j])) fprintf(STD_OUT, "Error destroying merge thread mutex %d/%d\n", i, j);
		if (pthread_mutex_destroy(&scheduleMutex)) fprintf(STD_OUT, "Error destroying schedule mutex\n");
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
	Timers.divideA = Timers.divideB = 0;
	Timers.LinpackTimer1.Reset();
	Timers.LinpackTimer2.Reset();
	Timers.LinpackTimer3.Reset();
	Timers.BcastTimer.Reset();
}

#define MAX_HUGE_ADDRESSES 256
double* huge_page_addresses[MAX_HUGE_ADDRESSES];
int nHugeAddresses = 0;

double* caldgemm::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages)
{
#ifdef WASTE_MEMORY
	nDoubles += 40 * 1024 * 1024;
#endif
	double* ptr;
	if (huge_pages)
	{
		if (nHugeAddresses >= MAX_HUGE_ADDRESSES - 1)
		{
			fprintf(STD_OUT, "No more huge_page memory available, increase MAX_HUGE_ADDRESSES\n");
			return(NULL);
		}
		int shmid;
		void *address;

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
			fprintf(STD_OUT, "Error Locking HugePage Memory\n");
			shmdt((void*) ptr);
			return(NULL);
		}

		huge_page_addresses[nHugeAddresses++] = ptr;

	}
	else
	{
		ptr = new double[nDoubles];
	}
	if (ptr == NULL) return(NULL);
#ifdef WASTE_MEMORY
	nDoubles -= 40 * 1024 * 1024;
	ptr += 20 * 1024 * 1024;
#endif
	if (!huge_pages && page_locked && mlock(ptr, nDoubles * sizeof(double)))
	{
		fprintf(STD_OUT, "Error locking Pages\n");
		if (!huge_pages) delete[] ptr;
		return(NULL);
	}
	return(ptr);
}

void caldgemm::FreeMemory(double* ptr)
{
#ifdef WASTE_MEMORY
	ptr -= 20 * 1024 * 1024;
#endif
	for (int i = 0;i < nHugeAddresses;i++)
	{
		if (huge_page_addresses[i] == ptr)
		{
			shmdt((void*) ptr);
			huge_page_addresses[i] = huge_page_addresses[--nHugeAddresses];
			return;
		}
	}
	delete[] ptr;
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
			flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Config->n + (Config->n % Config->Height) * (Config->m - cParam.cblas_size) + cParam.dynamic_run2 * Config->Height * Config->Height + (ExecLinpack ? Config->Width * Config->n : 0)) * (2 * Config->Width + 2) * Config->Iterations / Timers.CPUTimer.GetElapsedTime();
			flopsg = (double) 1e-09 * ((Config->m - cParam.cblas_size) * (Config->n - Config->n % Config->Height) - cParam.dynamic_run * cParam.dynamic_size - cParam.dynamic_run2 * Config->Height * Config->Height) * (2 * Config->Width + 2) * Config->Iterations / Timers.GPUTimer.GetElapsedTime();
		}
		else
		{
			flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Config->m + (Config->m % Config->Height) * (Config->n - cParam.cblas_size) + cParam.dynamic_run2 * Config->Height * Config->Height + (ExecLinpack ? Config->Width * Config->n : 0)) * (2 * Config->Width + 2) * Config->Iterations / Timers.CPUTimer.GetElapsedTime();
			flopsg = (double) 1e-09 * ((Config->n - cParam.cblas_size) * (Config->m - Config->m % Config->Height) - cParam.dynamic_run * cParam.dynamic_size - cParam.dynamic_run2 * Config->Height * Config->Height) * (2 * Config->Width + 2) * Config->Iterations / Timers.GPUTimer.GetElapsedTime();
		}
		
		if (Config->GPUClock && Config->m * Config->n >= 24 * 24 * 1024 * 1024 && flopsg <= (double) 460 * (double) Config->GPUClock / (double) 850 - (double) 20)
		{
			fprintf(STD_OUT, "%sThrottling: %s (%2.3lf GFlops)\n", Config->PreOut, hostname, flopsg);
		}

		if (!Config->Quiet || (Config->DisplayTiming /*&& Config->m * Config->n >= 16 * 24 * 1024 * 1024*/))
		{
			char timingoutputbase[1024];
			char *timingoutput = timingoutputbase;
			timingoutput += sprintf(timingoutput, "%sGPU Time %2.4lf (%2.4lf Gflops)   CPU Time %2.4lf (%2.4lf Gflops)", Config->PreOut, Timers.GPUTimer.GetElapsedTime(), flopsg, Timers.CPUTimer.GetElapsedTime(), flopsc);
			if (ExecLinpack) timingoutput += sprintf(timingoutput, "   Linpack Time: %2.4lf (%d, %2.4lf, %2.4lf)  Total CPU Time: %2.4lf", Timers.LinpackTimer1.GetElapsedTime(), ExecLinpack, Timers.LinpackTimer2.GetElapsedTime(), Timers.LinpackTimer3.GetElapsedTime(), Timers.TotalCPUTimer.GetElapsedTime());
			if (Config->TabularTiming)
			{
				timingoutput += sprintf(timingoutput, " --- GPU Ratio - Real: %2.2lf Guessed: %2.2lf , m*n: %.1E, CPU Wait Time: %2.3lf", (flopsg / (flopsc + flopsg)), gpu_ratio_used, (double) (Config->m * Config->n), cpu_wait_time);
			}
			sprintf(timingoutput, "\n");
			fwrite(timingoutputbase, 1, strlen(timingoutputbase), STD_OUT);
		}
		gpu_ratio_used = flopsg / (flopsc * Timers.CPUTimer.GetElapsedTime() / Timers.System.GetElapsedTime() + flopsg);
	}
	if ((!Config->Quiet || (Config->DisplayTiming /*&& Config->n * Config->m >= 16 * 24 * 1024 * 1024*/)) && Config->VerboseTiming)
	{
		double gflops = (double)1e-09 * Config->m * Config->n * (2 * Config->Width - 1) * (double)Config->Iterations / Timers.Kernel.GetElapsedTime();
#ifdef CALDGEMM_BENCHMARK_KERNEL
		gflops *= (double) CALDGEMM_BENCHMARK_KERNEL;
#endif
		double copyto = Config->DivideToGPU ? 0 : ((double) 1e-09 * (Config->Height * Timers.divideA + Config->Height * Timers.divideB) * Config->Width * sizeof(double) * (double)Config->Iterations / Timers.CounterCopyTo.GetElapsedTime());
		double copyfrom = Config->DstMemory == 'g' ? ((double) 1e-09 * Config->m * Config->n * sizeof(double) * (double)Config->Iterations / Timers.CounterCopyFrom.GetElapsedTime()) : 0;
		double copyMerge = Config->MultiThread ? 0 :((double) 1e-09 * Config->m * Config->n * sizeof(double) * (double)Config->Iterations / Timers.CounterMerge.GetElapsedTime());
		double copyDivide = (double) 1e-09 * (Config->Height * Timers.divideA + Config->Height * Timers.divideB) * Config->Width * sizeof(double) * (double)Config->Iterations / Timers.CounterDivide.GetElapsedTime();
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
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Config->m, Config->n, Config->Width, Alpha, A, A_pitch, B, B_pitch, Beta, D, C_pitch);
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
					if (errors < 1000) fprintf(STD_OUT, "Error found at row %lld, col %lld: Expected: %3.5le, Found: %3.5le, Diff: %3.5le\n", (long long int) i, (long long int) j, D[i * C_pitch + j], C[i * C_pitch + j], D[i * C_pitch + j] - C[i * C_pitch + j]);
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
			for (int i = 0;i < Config->m;i += Config->Height)
			{
				for (int j = 0;j < Config->n;j += Config->Height)
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

int caldgemm::SetupData ( CALmodule *module, CALresource* &_Res, BufferProperties* &data, CALdevice *device, CALcontext *ctx, unsigned int numInputs, unsigned int numOutputs, unsigned int numConstantBuffers, CALname** ctxProgNames, int nContext )
{
	BufferHeight = Config->Height;
	BufferWidth = Config->Width;
	// Fill in the dimensions
	const unsigned int bStop = dwBuffersA + dwBuffersB;
	const unsigned int fStop = bStop + numConstantBuffers;
	const unsigned int cStop = fStop + dwBuffersC;
	CALresult r = CAL_RESULT_OK;

	for (unsigned int i = 0; i < cStop; ++i)
	{
		if (nContext >= 1 && i == dwBuffersA + dwBuffersB) continue;
		if (nContext >= 2 && i < dwBuffersA) continue;
		if (nContext >= ctxcount && (i < dwBuffersA || i >= bStop)) continue;
		unsigned int tWidth = 0;
		unsigned int tHeight = 0;
		CALresallocflags flag = static_cast<CALresallocflags>(0);
		char mem = 'g';
		unsigned int mComponents = 2;
		if (i < dwBuffersA)
		{
#if defined(CALDGEMM_48) & defined(CALDGEMM_TRANSPOSED_A)
			tWidth = Config->Height / 8;
			tHeight = Config->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
			tWidth = Config->Height / 4;
			tHeight = Config->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
			tHeight = Config->Height / 4;
			tWidth = Config->Width;
#elif defined(CALDGEMM_TRANSPOSED_A)
			/* A matrix sizes are shrunk by 2 (double2) in the width and 8 (8 resources) in the height */
			tWidth = Config->Height / 2;
			tHeight = Config->Width / dwBuffersA;
#else
			tWidth = Config->Width / 2;
			tHeight = Config->Height / dwBuffersA;
#endif
			mem = 'g';
		}
		else if (i >= dwBuffersA && i < bStop)
		{
#if defined(CALDGEMM_84) & defined(CALDGEMM_TRANSPOSED_A)
			tWidth = Config->Height / 8;
			tHeight = Config->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
			tWidth = Config->Height / 4;
			tHeight = Config->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
			tHeight = Config->Height / 4;
			tWidth = Config->Width;
#elif defined (CALDGEMM_TRANSPOSED_B)
			tWidth = Config->Width / 2;
			tHeight = Config->Height / dwBuffersB;
#else
			/* B matrix sizes are shrunk by 2 (double2) in the width and 2 (2 resources) in the height */
			tWidth = Config->Height / 2;
			tHeight = Config->Width / dwBuffersB;
#endif
			mem = 'g';
		}
		else if (i >= bStop && i < fStop)
		{
			tWidth = 8;
			tHeight = 1;
			flag = static_cast<CALresallocflags>(0);
		}
		else if (i >= fStop && i < cStop)
		{
#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)
			tWidth = tHeight = Config->Height / 4;
#else
			tWidth = Config->Height / 2;
			tHeight = Config->Height / dwBuffersC;
#endif
			mem = Config->DstMemory;
			flag = (CALresallocflags) (flag | CAL_RESALLOC_CACHEABLE);
		}

		data[i].DataSize = sizeof(double);
		data[i].Width = tWidth;
		data[i].Height = tHeight;
		data[i].VectorSize = mComponents;
		bool allocated = false;
		if (tHeight > 1)
		{
			data[i].CALMemory = true;
			if ((Config->DstMemory == 'g' || i < dwBuffersA + dwBuffersB) && (Config->DivideToGPU == false || i >= dwBuffersA + dwBuffersB) && (nContext < 2 || (Config->DstMemory == 'g' && i >= dwBuffersA + dwBuffersB + numConstantBuffers)))
			{
				allocated = true;
				CHKERR(calResAllocRemote2D(&data[i].res, device, 1, tWidth, tHeight, getFormat(mComponents, sizeof(double), true), flag), "allocattion of remote memory");
				CHKERR(calCtxGetMem(&data[i].mem, *ctx, data[i].res), "getting remote memory for context");
				CHKERR(calResMap(&data[i].ptr_void, &data[i].pitch, data[i].res, NULL), "mapping of remote memory");
				if (((size_t) data[i].ptr_void) & (vcpysize - 1))
				{
					fprintf(STD_OUT, "Memory not aligned correctly\n");
					return(1);
				}
			}
		}
		else
		{
			if (nContext == 0)
			{
				data[i].ptr_char = new char[tWidth * sizeof(double) * mComponents * tHeight];
				allocated = true;
			}
			data[i].CALMemory = false;
		}
		if (allocated)
		{
			memset((void*)data[i].ptr_char, 0, tWidth * sizeof(double) * mComponents * tHeight);
		}

		if (nContext >= 2 && i < dwBuffersA) continue;
		if (nContext >= ctxcount && (i < dwBuffersA || i >= dwBuffersA + dwBuffersB)) continue;
		if (nContext >= 1 && i == dwBuffersA + dwBuffersB) continue;
		flag = (CALresallocflags) NULL;
		mem = 'g';
		if (i >= fStop && i < cStop)
		{
			mem = Config->DstMemory;
			if (mem == 'c') flag = static_cast<CALresallocflags>(flag | CAL_RESALLOC_CACHEABLE);
		}
		else if (i >= bStop && i < fStop)
		{
			continue;
		}

#ifdef CALDGEMM_USE_MEMEXPORT
		if (i >= fStop)
		{
			flag = (CALresallocflags) (flag | CAL_RESALLOC_GLOBAL_BUFFER);
		}
#endif
		switch(mem)
		{
		case 'g':
			r = calResAllocLocal2D(&_Res[i], *device, tWidth, tHeight, getFormat(mComponents, data[i].DataSize, true), flag);

			break;
		case 'c':
			r = calResAllocRemote2D(&_Res[i], device, 1, tWidth, tHeight, getFormat(mComponents, data[i].DataSize, true), flag);
			break;
		}
		if (r != CAL_RESULT_OK)
		{
			for (unsigned int j = dwBuffersA;j < i;j++)
			{
				calCtxReleaseMem(*ctx, data[j].dstMem);
				calResFree(_Res[j]);
			}

			if (nContext < ctxcount)
			{
				fprintf(STD_OUT, "There was an error in allocating resources and binding them to memory\n");
				return(1);
			}
			else if (Config->Debug) fprintf(STD_OUT, "No more memory available for bbuffers\n");
		}
		CHKERR(calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]), "binding memory to context");
		if ((Config->DstMemory == 'c' && i >= fStop) || (Config->DivideToGPU && i < bStop))
		{
			data[i].mem = data[i].dstMem;
			data[i].res = _Res[i];
		}
		if (Config->DstMemory == 'c' && i >= fStop && Config->KeepBuffersMapped)
		{
			CHKERR(calResMap(&data[i].ptr_void, &data[i].pitch, data[i].res, NULL), "mapping of remote output memory");
		}
	}

	if (nContext >= 1) return(0);

	for (unsigned int i = bStop; i < fStop; ++i)
	{
		int cWidth = data[i].Width * data[i].Height;
		CHKERR(calResAllocRemote1D(&_Res[i], device, 1, cWidth, getFormat(data[i].VectorSize,data[i].DataSize), 0), "allocating constant memory");
		CHKERR(calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]), "binding constant memory to context");
	}

	for (int i = 0;i < kernel_count;i++)
	{
		for (unsigned int j = 0; j < cStop; ++j)
		{
			char buffer[10];
			if (j < bStop)
			{
				sprintf(buffer,"i%d", j);
			}
			else if (j >= fStop && j < cStop)
			{
#ifdef CALDGEMM_USE_MEMEXPORT
				sprintf(buffer, "g[]", j - fStop);
#else
				sprintf(buffer,"o%d", j - fStop);
#endif
			}
			else if (j >= bStop && j < fStop)
			{
				sprintf(buffer,"cb%d", j - bStop);
			}
			printf("Getting module buffer name kernel %d buffer %d name %s\n", i, j, buffer);
			CHKERR(calModuleGetName(&ctxProgNames[i][j], *ctx, *module, buffer), "getting buffer name");
			if (j >= bStop && j < fStop)
			{
				CHKERR(calCtxSetMem(*ctx, ctxProgNames[i][j], data[j].dstMem), "setting memory buffer to context");
			}
		}
	}

	return(0);
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

HighResTimer::HighResTimer()
{
	ElapsedTime = 0;
#ifdef ATI_OS_WIN
	__int64 ifreq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&ifreq);
	Frequency = (double) ifreq;
#endif

#ifdef ATI_OS_LINUX
	Frequency = 1.0E9;
#endif
}

HighResTimer::~HighResTimer() {}

void HighResTimer::Start()
{
#ifdef ATI_OS_WIN
	i64 istart;
	QueryPerformanceCounter((LARGE_INTEGER*)&istart);
	_start = (double) istart;
#endif

#ifdef ATI_OS_LINUX
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	StartTime = (double )tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
}

void HighResTimer::Stop()
{
	double EndTime = 0;
#ifdef ATI_OS_WIN
	__int64 iend;
	QueryPerformanceCounter((LARGE_INTEGER*) &iend);
	EndTime = (double) iend;
#endif

#ifdef ATI_OS_LINUX
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	EndTime = (double )tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
	ElapsedTime += EndTime - StartTime;
}

void HighResTimer::Reset()
{
	ElapsedTime = 0;
	StartTime = 0;
}

double HighResTimer::GetElapsedTime()
{
	return ElapsedTime / Frequency;
}

static void log_callback(const char *msg)
{
	fwrite(msg, 1, strlen(msg), STD_OUT);
}

int caldgemm::Initialize(CALdevice *device, CALcontext *ctx, unsigned int deviceNum )
{
	CHKERR(calInit(), "initializing CAL");
	CHKERR(calDeviceOpen(device, deviceNum), "opening CAL device");
	CHKERR(calCtxCreate(&ctx_main, *device), "creating CAL context");
	return(0);
}

int caldgemm::SetupKernel(const char* ILKernel, CALmodule* module, CALcontext* ctx, bool disassemble)
{
	CALimage image = NULL;
	bool success = false;

	CALdeviceattribs attribs;
	attribs.struct_size = sizeof(CALdeviceattribs);
	CHKERR(calDeviceGetAttribs(&attribs, Config->DeviceNum), "getting device attributes");

	CALobject obj;
	if (Config->PrintILKernel) fprintf(STD_OUT, "Kernel:\n%s\n", ILKernel);
#ifdef CALDGEMM_BENCHMARK_KERNEL
	char* ILKernelUse = (char*) malloc(strlen(ILKernel) + 1024);
	sprintf(ILKernelUse, ILKernel, Config->Width);
#else
	const char* ILKernelUse = ILKernel;
#endif
	CHKERR(calclCompile(&obj, CAL_LANGUAGE_IL, ILKernelUse, attribs.target), "compiling the kernel");
#ifdef CALDGEMM_BENCHMARK_KERNEL
	free(ILKernelUse);
#endif

	CHKERR(calclLink(&image, &obj, 1), "linking the kernel");
	CHKERR(calclFreeObject(obj), "freeing the object file");
	if (disassemble == true)
	{
		calclDisassembleImage(image, log_callback);
	}

	CHKERR(calModuleLoad(module, *ctx, image), "loading the module to the context");
	CHKERR(calclFreeImage(image), "freeing kernel image");

	return(0);
}

int caldgemm::RunProgram(CALcontext *ctx, CALmodule *module, unsigned int Width, unsigned int Height, CALevent* event)
{
	CALfunc func;
	CALresult r = CAL_RESULT_ERROR;
	CHKERR(calModuleGetEntry(&func, *ctx, *module, "main"), "finding module entry point");

	// Setup a computation domain
	CALdomain rect;
	rect.x = 0;
	rect.y = 0;
	rect.width = Width;
	rect.height = Height;

	// Execute the program iterations number of times
	if (Config->VerboseTiming) Timers.Kernel.Start();
#ifdef CALDGEMM_BENCHMARK_KERNEL
	for (int i = 0;i < CALDGEMM_BENCHMARK_KERNEL;i++)
#endif
	CHKERR(calCtxRunProgram(event, *ctx, func, &rect), "executing kernel");

	if (Config->VerboseTiming)
	{
		// Wait for the kernel to complete.
		if (event) WAITFOREVENTA(*ctx, *event);
		Timers.Kernel.Stop();
		if (Config->Debug) fprintf(STD_OUT, "\tTotal Kernel Time: %2.4lf\n", Timers.Kernel.GetElapsedTime());
	}

	return(0);
}

int caldgemm::CleanupData(CALcontext* ctx, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext)
{
	if (data)
	{
		for (unsigned int i = 0; i < numHandles;++i)
		{
			if ((nContext == 0 || i != dwBuffersA + dwBuffersB) && (nContext < 2 || i >= dwBuffersA) && (nContext < ctxcount || i < dwBuffersA + dwBuffersB) && data[i].ptr_char)
			{
				if (data[i].CALMemory )
				{
					if ((Config->DstMemory == 'g' || i <= dwBuffersA + dwBuffersB) && (Config->DivideToGPU == false || i >= dwBuffersA + dwBuffersB + numConstantBuffers) && nContext < 2)
					{
						calResUnmap(data[i].res);
						calCtxReleaseMem(*ctx, data[i].mem);
						calResFree(data[i].res);
					}
				}
				else
				{
					if (nContext == 0) delete [] data[i].ptr_char;
				}
				data[i].ptr_char = NULL;
			}
		}
	}

	// Free up the CALresource
	if (resourceHandler)
	{
		for (unsigned int i = 0; i < numHandles; i++ )
		{
			if ((nContext == 0 || i != dwBuffersA + dwBuffersB) && (nContext < 2 || i >= dwBuffersA) && (nContext < ctxcount || i < dwBuffersA + dwBuffersB) && resourceHandler[i])
			{
				if (Config->DstMemory == 'c' && i >= dwBuffersA + dwBuffersB + numConstantBuffers && Config->KeepBuffersMapped)
				{
					CHKERR(calResUnmap(data[i].res), "mapping of remote output memory");
				}
				CHKERR(calCtxReleaseMem(*ctx, data[i].dstMem), "releasing CAL memory");
				CHKERR(calResFree(resourceHandler[i]), "releasing CAL resources");
			}
		}
	}
	return(0);
}

int caldgemm::Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext)
{
	CleanupData(ctx, resourceHandler, data, numHandles, nContext);

	// Unload the module from the context

	if (nContext < 1)
	{
		for (int i = 0;i < kernel_count;i++)
		{
			if (module[i])
			{
				CHKERR(calModuleUnload(*ctx, module[i]), "unloading module");
			}
		}
	}
	delete[] resourceHandler;
	delete[] data;

	return(0);
}

int caldgemm::CopyDataFromGPU(CALcontext* ctx, CALresource* _Res, BufferProperties* data, unsigned int num, CALevent* event)
{
	if (Config->DstMemory == 'c') return 0;
	unsigned int pitch;
	CALresult r;
	char* ptr;
	WAITFOREVENTA(*ctx, *event);
	for (unsigned int i = 0; i < num; ++i)
	{
		if (data[i].CALMemory)
		{
			//if (Config->Debug) fprintf(STD_OUT, "GPUHandle: %d, CPUHandle: %d\n", data[i].dstMem, data[i].mem);
			CHKERR(calMemCopy(event, *ctx, data[i].dstMem, data[i].mem, NULL), "copying data from gpu");
			continue;
		}
		CHKERR(calResMap((void**)&ptr, &pitch, _Res[i], 0), "mapping buffer");
		memcpy(data[i].ptr_char, ptr, data[i].DataSize * data[i].VectorSize * data[i].Width * data[i].Height);
		CHKERR(calResUnmap(_Res[i]), "unmapping buffer");
	}
	if (Config->VerboseTiming) WAITFOREVENTA(*ctx, *event);
	return 0;
}

int caldgemm::CopyDataToGPU(CALcontext* ctx, CALresource* _Res, BufferProperties* data, unsigned int num, bool constants, CALevent* event, BufferProperties* dest_data)
{
	if (dest_data == NULL) dest_data = data;
	unsigned int pitch;
	CALresult r;
	char* ptr;
	for (unsigned int i = 0; i < num; ++i)
	{
		if (data[i].CALMemory == constants) continue;
		if (data[i].CALMemory)
		{
			CHKERR(calMemCopy(event, *ctx, data[i].mem, dest_data[i].dstMem, NULL), "copying data to gpu");
			continue;
		}
		CHKERR(calResMap((void**)&ptr, &pitch, _Res[i], 0), "Mapping Buffer");
		memcpy(ptr, data[i].ptr_char, data[i].DataSize * data[i].VectorSize * data[i].Width * data[i].Height);
		CHKERR(calResUnmap(_Res[i]), "unmapping buffer");
	}
	if (Config->VerboseTiming && constants == false) WAITFOREVENTA(*ctx, *event);

	return 0;
}

//MUST REWRITE!!!

CALformat caldgemm::getFormat(unsigned int formatSize, unsigned int dataSize, bool isInt)
{
	CALformat format; // = CAL_FORMAT_FLOAT_1;

	format = (isInt? CAL_FORMAT_UNSIGNED_INT32_1 : CAL_FORMAT_FLOAT_1);

	switch(dataSize)
	{
	case 4:

		switch(formatSize)
		{
		case 1:
			format = (isInt? CAL_FORMAT_UNSIGNED_INT32_1 : CAL_FORMAT_FLOAT_1);
			break;
		case 2:
			format = (isInt? CAL_FORMAT_UNSIGNED_INT32_2 : CAL_FORMAT_FLOAT_2);
			//format = CAL_FORMAT_FLOAT_2;
			break;
		case 4:
			format = (isInt? CAL_FORMAT_UNSIGNED_INT32_4 : CAL_FORMAT_FLOAT_4);
			//format = CAL_FORMAT_FLOAT_4;
			break;
		default:
			assert(!"attempted to use invalid format!" );
			break;
		};
		break;
	case 8:
		switch(formatSize)
		{
		case 1:
			format = (isInt? CAL_FORMAT_UNSIGNED_INT32_2 : CAL_FORMAT_FLOAT_2);
			//format = CAL_FORMAT_FLOAT_2;
			break;
		case 2:
			format = (isInt? CAL_FORMAT_UNSIGNED_INT32_4 : CAL_FORMAT_FLOAT_4);
			//format = CAL_FORMAT_FLOAT_4;
			break;
		default:
			assert(!"attempted to use invalid format!" );
			break;
		};

		break;
	case 16:
		switch(formatSize)
		{
		case 1:
			format = (isInt? CAL_FORMAT_UNSIGNED_INT32_4 : CAL_FORMAT_FLOAT_4);
			//format = CAL_FORMAT_FLOAT_4;
			break;
		default:
			assert(!"attempted to use invalid format!" );
			break;
		};
		break;
	}

	return format;
}

int caldgemm::QueryCALVersion(CALVersion required, const char* comparison, bool silent)
{
	CALVersion available;
	calGetVersion(&available.major, &available.minor, &available.imp);
	if (Config->Debug && !silent) fprintf(STD_OUT, "Found CAL Runtime Version: %d.%d.%d\n", available.major, available.minor, available.imp);

	if( strcmp(comparison,">") == 0 )
	{
		if( (available.major > required.major) ||
			(available.major == required.major && available.minor > required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			available.imp > required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,">=") == 0 )
	{
		if( (available.major > required.major) ||
			(available.major == required.major && available.minor > required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			available.imp >= required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,"<") == 0 )
	{
		if( (available.major < required.major) ||
			(available.major == required.major && available.minor < required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			available.imp < required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,"<=") == 0 )
	{
		if( (available.major < required.major) ||
			(available.major == required.major && available.minor < required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			available.imp <= required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,"==") == 0 )
	{
		if( available.major == required.major && available.minor == required.minor &&
			available.imp == required.imp )
		{
			return 1;
		}
	}
	else 
	{
		fprintf(STD_OUT, "Error. Invalid comparison operator: %s (QueryCALVersion)\n", comparison);
	}

	return 0;
}

void caldgemm::SupportedCALVersion(CALVersion *calVersion)
{
	calVersion->major = 1;
	calVersion->minor = 3;
	calVersion->imp = 185;
	if (Config->Debug) fprintf(STD_OUT, "Supported CAL Runtime Version: %d.%d.%d\n", calVersion->major, calVersion->minor, calVersion->imp);
}

int caldgemm::ValidateCALRuntime()
{
	CALVersion supportedCALRuntime;

	supportedCALRuntime.major = 1;
	supportedCALRuntime.minor = 4;
	supportedCALRuntime.imp = 815;
	if (QueryCALVersion(supportedCALRuntime, ">=", true) == 0)
	{
		if (Config->AsyncDMA && !Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Asynchronous DMA not supported by CAL Runtime Version\n");
		Config->AsyncDMA = false;
	}

	// Get the CAL runtime currently supported by the SDK 
	SupportedCALVersion( &supportedCALRuntime );

	// Check if this runtime is available 
	return QueryCALVersion( supportedCALRuntime, ">=" );
}

// vim: ts=4 sw=4 noet sts=4 tw=100
