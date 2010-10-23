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
#undef CALDGEMM_ALPHA1
#undef ILKernelName

const char* calutil::ILFakeKernel =
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
#define WAITFOREVENT(ctx, eventnr) { CALresult r; if (Info->Debug) fprintf(STD_OUT, "\tWaiting for event from context %d...\n", eventnr); do { r = calCtxIsEventDone(ctx, events[eventnr]); if (r == CAL_RESULT_ERROR) { fprintf(STD_OUT, "Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}

caldgemm::caldgemm()
{
	caldgemm_initialized = false;
	memset(linpack_last_mn, 0, max_linpack_callback_types * sizeof(double));
	memset(linpackGPURatios, 0, 3 * sizeof(double));
	memset(linpackCPUDGEMMTime, 0, 3 * sizeof(double));
	memset(linpackBcastTime, 0, 3 * sizeof(double));
}

caldgemm::~caldgemm()
{
	if (caldgemm_initialized) ExitCALDGEMM();
}

calutil::SampleInfo::SampleInfo()
{
	Verify = CAL_FALSE;
	Disassemble = CAL_FALSE;
	PrintILKernel = CAL_FALSE;
	Quiet = CAL_TRUE;
	DisplayTiming = CAL_FALSE;
	DeviceNum = 0;
	Width = 1024;
	Height = 4096;
	AutoHeight = CAL_TRUE;
	Iterations = 1;
	DstMemory = 'c';
	VerboseTiming = CAL_FALSE;
	AsyncTiming = CAL_FALSE;
	TabularTiming = CAL_FALSE;
	Debug = CAL_FALSE;
	MultiThread = CAL_TRUE;
	UseGPU = CAL_TRUE;
	UseCPU = CAL_TRUE;
	GPURatio = -1.0;
	DynamicSched = CAL_TRUE;
	MemPolicy = CAL_TRUE;
	DumpMatrix = CAL_FALSE;
	DivideToGPU = CAL_FALSE;
	AsyncDMA = CAL_TRUE;
	KeepBuffersMapped = CAL_TRUE;
	NoPerformanceWarnings = CAL_FALSE;
	m = 0;
	n = 0;
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
	fprintf(STD_OUT, "Matrix %lld x %lld, Subblocks %lld x %lld, Strides: %lld / %lld\n", width, height, subx, suby, stridex, stridey);
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
							if (jj >= Info->m - cParam.cblas_size || ii >= Info->n - Info->n % Info->Height) sprintf(tmpcolor, "01;34");
						}
						else
						{
							if (jj >= Info->m - Info->m & Info->Height || ii >= Info->n - cParam.cblas_size) sprintf(tmpcolor, "01;34");
						}

						size_t k = gpu_m / Info->Height * gpu_n / Info->Height;
						for (int l = 0;l < cParam.dynamic_run2;l++)
						{
							k--;
							size_t cpublockm, cpublockn;
							DGEMM_getblocks(k, cpublockm, cpublockn);
							while ((DGEMM_favor_m ? (cpublockm * Info->Height >= gpu_m - cParam.dynamic_run && cpublockn * Info->Height >= gpu_n - cParam.dynamic_size) :
								(cpublockn * Info->Height >= gpu_n - cParam.dynamic_run && cpublockm * Info->Height >= gpu_m - cParam.dynamic_size)))
							{
								k--;
								DGEMM_getblocks(k, cpublockm, cpublockn);
							}
							if (jj / Info->Height == cpublockm && ii / Info->Height == cpublockn)
							{
								sprintf(tmpcolor, "01;35");
							}
						}

						int ok = isDoubleEqual(M[jj * pitch + ii], M2[jj * pitch + ii]);
						fprintf(STD_OUT, "\33[%sm%d\33[%sm%+ 10.3lf\t", ok ? "01;32" : "01;31", ok , tmpcolor, M[jj * pitch + ii]);
					}
					else
					{
						fprintf(STD_OUT, " %+ 10.3lf\t", M[jj * pitch + ii]);
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

int caldgemm::divideBuffer(Data* dst, CALdouble* src, CALint width, CALint height, CALint gpu_width, CALint gpu_height, CALint pitch, CALint numBuffers, bool transpose)
{
	if (Info->Debug) fprintf(STD_OUT, "\t\tSRC=0x%llx, w: %d, h: %d, pitch: %d (gpuw: %d, gpuh: %d, transpose: %d)\n", src, width, height, pitch, gpu_width, gpu_height, (int) transpose);

	if (Info->DivideToGPU)
		for (CALuint i = 0;i < numBuffers;i++)
		{
			CHKERR(calResMap(&dst[i].v_data, &dst[i].pitch, dst[i].res, 0), "mapping input buffer for buffer division");
			if (((size_t) dst[i].v_data) & (vcpysize - 1))
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
				for (CALint y = 0;y < width;y += 4)
				{
					double* saddr = src + (y * pitch);
					double* saddr2 = src + ((y + 1) * pitch);
					double* saddr3 = src + ((y + 2) * pitch);
					double* saddr4 = src + ((y + 3) * pitch);

					double* daddr = dst[0].d_data + y;
					double* daddr2 = dst[1 % numBuffers].d_data + (1 / numBuffers) * gpu_width + y;
					double* daddr3 = dst[2 % numBuffers].d_data + (2 / numBuffers) * gpu_width + y;
					double* daddr4 = dst[3 % numBuffers].d_data + (3 / numBuffers) * gpu_width + y;

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

				for (CALint y=0; y < width; y += 4)
				{
					const double *__restrict__ saddr0 = &src[(y + 0) * pitch];
					const double *__restrict__ saddr2 = &src[(y + 2) * pitch];

					double *__restrict__ dstBank0 = &dst[0].d_data[y * 2];
					double *__restrict__ dstBank1 = &dst[1].d_data[y * 2];

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
				for (CALint y=0; y < width; y += 2)
				{
					double* saddr = src + (y * pitch);
					double* saddr2 = src + ((y + 1) * pitch);

					for (int i = 0;i < height;i += 2)
					{
#if defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
						CALint bank = (i / 2) % 2;
						double* daddr = dst[bank].d_data + (i / 4) * gpu_width * 2 + y * 2;
						double* daddr2 = dst[bank].d_data + (i / 4) * gpu_width * 2 + y * 2 + 2;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
						//Col Interleaved Storage, Numbuffers is either 2 or 4, might be optimized in 2 branches
						CALint bank = (y / 2) % numBuffers;
#ifdef CALDGEMM_DIAGONAL_TEXTURE
						double* daddr = dst[bank].d_data + i * gpu_width / 2 + (((y / 2) & 0xFFFFFFFE) + 2 * i) % (gpu_width / 2);
						double* daddr2 = dst[bank].d_data + (i + 1) * gpu_width / 2 + (((y / 2) & 0xFFFFFFFE) + 2 * i + 2) % (gpu_width / 2);
#else
						double* daddr = dst[bank].d_data + (i * gpu_width / numBuffers + ((y / numBuffers) & 0xFFFFFFFE));
						double* daddr2 = dst[bank].d_data + ((i + 1) * gpu_width / numBuffers + ((y / numBuffers) & 0xFFFFFFFE));
#endif
#else
						//Standard Storage
						CALint bank = (i) % numBuffers;
						CALint bank2 = (i + 1) % numBuffers;
						double* daddr = dst[bank].d_data + (i / numBuffers) * gpu_width + y;
						double* daddr2 = dst[bank2].d_data + (i / numBuffers) * gpu_width + y;
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
			for (CALint y = 0;y < height / 2;y++)
			{
				double* daddr = dst[y % 2].d_data + y / 2 * gpu_width * 2;
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
				double* daddr = dst[0].d_data;
				double* daddr2 = dst[1].d_data;
				double* daddr3 = dst[2].d_data;
				double* daddr4 = dst[3].d_data;
				for (CALint y=0; y < height; y++)
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
				double* daddr = dst[0].d_data;
				double* daddr2 = dst[1].d_data;
				for (CALint y=0; y < height; y++)
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
			CALint* position = new CALint[numBuffers];
			memset((CALvoid*) position, 0, numBuffers * sizeof(CALint));
			for (CALint y=0; y < height; y++)
			{
				CALint bank = y % numBuffers;
				double* daddr = dst[bank].d_data + position[bank];
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

		if (Info->DivideToGPU)
			for (CALuint i = 0;i < numBuffers;i++)
			{
				CHKERR(calResUnmap(dst[i].res), "unmapping input buffer for buffer division");
			}
			return(0);
}

int caldgemm::mergeBuffers(CALdouble* dst, Data* src, CALint width, CALint height, CALint gpu_width, CALint gpu_height, CALint pitch, CALint numBuffers)
{
	// Array to store the position from which data will be pulled in from the input buffers
	CALint* position = new CALint[numBuffers];
	memset((CALvoid*) position, 0, numBuffers * sizeof(CALint));

	if (Info->DstMemory == 'c' && !Info->KeepBuffersMapped)
		for (CALuint i = 0;i < cPartsNum;i++)
		{
			CHKERR(calResMap(&src[i].v_data, &src[i].pitch, src[i].res, 0), "mapping output buffer for merging");
			if (((size_t) src[i].v_data) & (vcpysize - 1))
			{
				fprintf(STD_OUT, "Invalid alignment\n");
				return(1);
			}
		}

	for (CALint y=0; y < height; y++)
	{
		//CALDGEMM_44 Init
#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)
		CALint bank = y % 4;
		double* saddr2 = src[bank + 4].d_data + position[bank];
		double* paddr2 = src[(y + 1) % 4 + 4].d_data + position[(y + 1) % 4];
#else
		CALint bank = y % numBuffers;
#endif

		double* daddr = dst + (y * pitch);
		double* saddr = src[bank].d_data + position[bank];
		double* paddr = src[(y + 1) % 4].d_data + position[(y + 1) % 4];
		int count = src[bank].DataSize * width;

#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)

		if (Info->KeepBuffersMapped)
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

	if (Info->DstMemory == 'c' && !Info->KeepBuffersMapped)
		for (CALuint i = 0;i < cPartsNum;i++)
		{
			CHKERR(calResUnmap(src[i].res), "unmapping output buffer for merging");
		}
	delete[] position;
	return(0);
}

void caldgemm::checkCalPatch()
{
	unsigned char *RunProgPTL = (unsigned char *)(&calCtxRunProgram);
	unsigned char **RunProgWrapperFunc = *(unsigned char ***)((size_t)(*(unsigned int *)(RunProgPTL + 2)) + RunProgPTL + 6);
	//fprintf(STD_OUT, "RunProgWrapperFunc = %p, ddi_interface[?] = %p\n", RunProgWrapperFunc, RunProgWrapperFunc + (0x10f588 - 0x4220)/sizeof(void*));
	unsigned char *RunProgFunc = *(RunProgWrapperFunc + (0x10f588 - 0x4220) / sizeof(void*));
	unsigned char *patchpos = RunProgFunc + 0x7fffe591b631 - 0x7fffe591b560;
	if (*patchpos == 0x74)
	{
		if (Info->KeepBuffersMapped && !Info->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: CAL library not patched, KeepBuffersMapped unavailable\n");
		Info->KeepBuffersMapped = CAL_FALSE;
	}
	else if (*patchpos != 0xEB)
	{
		if (!Info->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Unknown CAL Library found, KeepBuffersMapped unavailable\n");
		Info->KeepBuffersMapped = CAL_FALSE;
	}
	else if (Info->Debug)
	{
		fprintf(STD_OUT, "Patched CAL library found, KeepBuffersMapped available\n");
	}
}

int caldgemm::InitCALDGEMM(SampleInfo* pInfo)
{
	Info = pInfo;

	if (Info->Iterations > 1 && Info->UseCPU)
	{
		fprintf(STD_OUT, "ERROR: Multiple Iterations not supported with CPU enabled\n");
		return(1);
	}

	gethostname(hostname, 255);
	sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);

	CPU_ZERO(&gpumask);
	CPU_SET(0, &gpumask);

	if (Info->Debug) fprintf(STD_OUT, "Init Caldgemm, setting CPU mask %X\n", getcpumask(&gpumask));
	if (0 != sched_setaffinity(0, sizeof(gpumask), &gpumask))
	{
		fprintf(STD_OUT, "Error setting CPU affinity\n");
		return(1);
	}

	if(!ValidateCALRuntime())
	{
		fprintf(stderr, "Error. Could not find a compatible CAL runtime.\n");
		return 0;
	}

#ifdef CALDGEMM_44
	if (Info->Width % 8)
	{
		fprintf(STD_OUT, "Only width of multiples of 8 are computable.\n");
		return(0);
	}
	else if (Info->Width % 64)
	{
		Info->Width += 64 - Info->Width % 64;
		fprintf(STD_OUT, "Cannot allocate buffers of size that is not multiple of 64, increasing buffer size to %lld\n", Info->Width);
	}
#else
	if (Info->Width % 64)
	{
		fprintf(STD_OUT, "Only width of size 64 are computable.\n");
		return(0);
	}
#endif
	if (Info->Height & 0x7)
	{
		fprintf(STD_OUT, "Only heights with multiple of 8 are computable.\n" );
		return(0);
	}

	numInputs = aPartsNum + bPartsNum;
	numOutputs = cPartsNum;
	numConstantBuffers = 1;
	device = 0;

	if (Info->Debug) fprintf(STD_OUT, "Initializing CAL\n");
	if (!Initialize(&device, &ctx_main, Info->DeviceNum))
	{
		return 1;
	}

	CALdeviceattribs attribs;
	attribs.struct_size = sizeof(CALdeviceattribs);
	if (calDeviceGetAttribs(&attribs, Info->DeviceNum) != CAL_RESULT_OK)
	{
		fprintf(STD_OUT, "Error getting device attributes\n");
		return 1;
	}

	Features.DoublePrecision = CAL_TRUE;
	if(QueryDeviceCaps(Info->DeviceNum, &Features) != CAL_TRUE)
	{
		fprintf(STD_OUT, "The Device Number is invalid or this device is not capable of running this sample.");
		return 1;
	}

	if (Info->KeepBuffersMapped)
	{
		if (!SetupKernel(ILFakeKernel, &fakeModule, &ctx_main, CAL_FALSE)) return(1);
		if (!RunProgram(&ctx_main, &fakeModule, 0, 0, events)) {fprintf(STD_OUT, "Error running test kernel on GPU\n"); return(1);}
		if (Info->KeepBuffersMapped) checkCalPatch();
		if (calModuleUnload(ctx_main, fakeModule) != CAL_RESULT_OK )
		{
			fprintf(STD_OUT, "Error unloading test module\n");
			fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
		}
	}
	outputthreads = Info->KeepBuffersMapped ? CALDGEMM_OUTPUT_THREADS : CALDGEMM_OUTPUT_THREADS_SLOW;

	for (int i = 0;i < max_bbuffers;i++)
	{
		if (i < 1)
		{
			if (!SetupKernel(ILKernel, &modules[i][0], &ctx_main, (CALboolean) (Info->Disassemble && i == 0)) ||
				!SetupKernel(ILKernelALPHA1, &modules[i][1], &ctx_main, (CALboolean) (Info->Disassemble && i == 0)))
			{
				return 1;
			}
			for (int j = 0;j < kernel_count;j++) progNames[i][j] = new CALname[numInputs + numOutputs + numConstantBuffers];
		}

		datas[i] = new Data[numInputs + numOutputs + numConstantBuffers];
		resourceHandlers[i] = new CALresource[numInputs + numOutputs + numConstantBuffers];
		memset(datas[i], 0, (numInputs + numOutputs + numConstantBuffers) * sizeof(Data));
		memset(resourceHandlers[i], 0, (numInputs + numOutputs + numConstantBuffers) * sizeof(CALresource));
		if (!SetupData(modules[i], resourceHandlers[i], datas[i], &device, &ctx_main, numInputs, numOutputs, numConstantBuffers, progNames[i], i))
		{
			if (i < ctxcount)
				return 1;
			else
				break;
		}
		bbuffers = i + 1;


		if (i < ctxcount && Info->MultiThread)
		{
			pthread_mutex_init(&obufferMutex[i], NULL);
		}

		if (i < max_outputthreads && Info->MultiThread)
		{
			mParam[i].cls = this;
			mParam[i].terminate = CAL_FALSE;
			mParam[i].nMergeThread = i;
			for (int j = 0;j < 2;j++) pthread_mutex_init(&mParam[i].mergeThreadMutex[j], NULL);
			pthread_t thr;
			pthread_create(&thr, NULL, merge_wrapper, &mParam[i]);

			while (pthread_mutex_trylock(&mParam[i].mergeThreadMutex[0]) != EBUSY) if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		}
	}
	if (Info->Debug) fprintf(STD_OUT, "Was able to allocate %d bbuffers\n", bbuffers);
	if (Info->UseCPU)
	{
		cParam.cls = this;
		cParam.terminate = CAL_FALSE;
		for (int j = 0;j < 2;j++) pthread_mutex_init(&cParam.cblasMutex[j], NULL);
		if (pthread_mutex_lock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (Info->MultiThread)
		{
			pthread_t thr;
			pthread_create(&thr, NULL, cblas_wrapper, &cParam);
			if (Info->Debug) fprintf(STD_OUT, "Waiting for cblas slave to start\n");
			while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		}
	}
	if (Info->MultiThread)
	{
		linpackParameters.terminate = CAL_FALSE;
		for (int j = 0;j < 2;j++) pthread_mutex_init(&linpackParameters.linpackMutex[j], NULL);
		if (pthread_mutex_lock(&linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
		pthread_t thr;
		pthread_create(&thr, NULL, linpack_wrapper, this);
		if (Info->Debug) fprintf(STD_OUT, "Waiting for linpack slave to start\n");
		while (pthread_mutex_trylock(&linpackParameters.linpackMutex[1]) != EBUSY) if (pthread_mutex_unlock(&linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		pthread_mutex_init(&scheduleMutex, NULL);
	}

	if (Info->MemPolicy)
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

	if (Info->Debug) fprintf(STD_OUT, "Caldgemm Init complete, setting CPU mask %X\n", getcpumask(&oldcpumask));
	sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

	caldgemm_initialized = true;

	return(0);
}

void caldgemm::cal_init_constant_data(Data* &data, double alpha)
{
	// Setup the constants for the kernel
	data[aPartsNum + bPartsNum].f_data[0] = (float) TILING_Y / Info->Height;			//Scale factor for normalized y pos
	data[aPartsNum + bPartsNum].f_data[2] = (float) TILING_X / Info->Height;			//Scale factor for normalized x pos
#ifdef CALDGEMM_44
	data[aPartsNum + bPartsNum].f_data[1] = 1.f / Info->Width;							//Step in K direction
	data[aPartsNum + bPartsNum].f_data[4] = static_cast<CALfloat>(Info->Width);			//Iterations of loop in IL Kernel
#else //CALDGEMM_44
	data[aPartsNum + bPartsNum].f_data[1] = 2.f / Info->Width;							//Step in K direction
	data[aPartsNum + bPartsNum].f_data[4] = static_cast<CALfloat>(Info->Width / (bPartsNum << 2));	//Iterations of loop in IL Kernel
#endif //CALDGEMM_44
	data[aPartsNum + bPartsNum].f_data[3] = 0.f;
	data[aPartsNum + bPartsNum].f_data[5] = (float) aPartsNum / Info->Height;			//For transposed matrix finer y resolution is needed
	data[aPartsNum + bPartsNum].f_data[8] = 0.5f - 0.5f / (float) (TILING_Y / aPartsNum);

	//Constants for Memexport
	data[aPartsNum + bPartsNum].i_data[9] = TILING_Y * Info->Height / 2;				//2 for double2
	data[aPartsNum + bPartsNum].i_data[10] = TILING_X / 2;								//x tiling in double2
#if defined(CALDGEMM_84)
	data[aPartsNum + bPartsNum].i_data[12] = 0 + 0 * Info->Height / 2;					//8 consecutive entries in x
	data[aPartsNum + bPartsNum].i_data[13] = 1 + 0 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[14] = 2 + 0 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[15] = 3 + 0 * Info->Height / 2;

	data[aPartsNum + bPartsNum].i_data[16] = 0 + 1 * Info->Height / 2;					//Next row
	data[aPartsNum + bPartsNum].i_data[17] = 0 + 1 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[18] = 0 + 1 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[19] = 0 + 1 * Info->Height / 2;

	data[aPartsNum + bPartsNum].i_data[20] = 0 + 2 * Info->Height / 2;					//Proceed by two rows
	data[aPartsNum + bPartsNum].i_data[21] = 0 + 2 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[22] = 0 + 2 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[23] = 0 + 2 * Info->Height / 2;
#elif defined(CALDGEMM_44)
	data[aPartsNum + bPartsNum].i_data[12] = 0 + 0 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[13] = 1 + 0 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[14] = 0 + 1 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[15] = 1 + 1 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[16] = 0 + 2 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[17] = 1 + 2 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[18] = 0 + 3 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[19] = 1 + 3 * Info->Height / 2;
#ifdef CALDGEMM_48
	data[aPartsNum + bPartsNum].i_data[20] = 0 + 4 * Info->Height / 2;					//Proceed by 4 rows
	data[aPartsNum + bPartsNum].i_data[21] = 0 + 4 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[22] = 0 + 4 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[23] = 0 + 4 * Info->Height / 2;
#endif
#else
	data[aPartsNum + bPartsNum].i_data[12] = 0 + 0 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[13] = 0 + 4 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[14] = 0 + 1 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[15] = 0 + 5 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[16] = 0 + 2 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[17] = 0 + 6 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[18] = 0 + 3 * Info->Height / 2;
	data[aPartsNum + bPartsNum].i_data[19] = 0 + 7 * Info->Height / 2;
#endif
#ifdef CALDGEMM_DIAGONAL_TEXTURE
	data[aPartsNum + bPartsNum].f_data[11] = 8.f / Info->Height;						//Offset for diagonal texture read
#endif
	data[aPartsNum + bPartsNum].d_data[3] = alpha;
}

void* linpack_wrapper(void* arg)
{
	caldgemm* cls = (caldgemm*) arg;
	volatile caldgemm::SampleInfo* Info = cls->Info;
	if (Info->Debug) fprintf(STD_OUT, "Linpack helper thread started\n");

	cpu_set_t linpack_mask;
	CPU_ZERO(&linpack_mask);
	CPU_SET(0, &linpack_mask);
	if (Info->Debug) fprintf(STD_OUT, "Linpack Thread, setting CPU mask %X\n", cls->getcpumask(&linpack_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &linpack_mask);

	if (pthread_mutex_lock(&cls->linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&cls->linpackParameters.linpackMutex[0]) == 0 && cls->linpackParameters.terminate == CAL_FALSE)
	{
		cls->Timers.LinpackTimer2.Start();
		Info->linpack_broadcast_function();
		cls->Timers.LinpackTimer2.Stop();
		if (pthread_mutex_unlock(&cls->linpackParameters.linpackMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}

	if (Info->Debug) fprintf(STD_OUT, "linpack slave terminating\n");
	pthread_exit(NULL);
	return(NULL);
}

int caldgemm::cpuScheduler()
{
	int retVal = 0;
	if (Info->UseCPU && Info->MultiThread && Info->DynamicSched)
	{
		const size_t mb = gpu_m / Info->Height;
		const size_t nb = gpu_n / Info->Height;
		size_t nBlocks = mb * nb;

		pthread_mutex_lock(&scheduleMutex);
		const size_t k = gpu_k_barrier == -1 ? 0 : gpu_k_barrier;

		if (gpu_k_barrier < nBlocks - 1)
		{
			size_t blockm, blockn;
			DGEMM_getblocks(k, blockm, blockn);

			if (cParam.dynamic_run == 0)
			{
				cParam.dynamic_size = ((1.0f - gpu_ratio_used) * (float) (nBlocks - k - 1) + 0.5) * Info->Height;
				if (cParam.dynamic_size > (nBlocks - k - 1) * Info->Height) cParam.dynamic_size = (nBlocks - k - 1) * Info->Height;
				if (cParam.dynamic_size > Info->Height)
				{
					cParam.dynamic_run = 1 + cParam.dynamic_size / mymin(gpu_m, gpu_n);
					cParam.dynamic_size /= cParam.dynamic_run;
					cParam.dynamic_size -= cParam.dynamic_size % Info->Height;
					cParam.dynamic_run *= Info->Height;

					while (DGEMM_favor_m ? (blockm * Info->Height >= gpu_m - cParam.dynamic_run && blockn * Info->Height >= gpu_n - cParam.dynamic_size) :
						(blockn * Info->Height >= gpu_n - cParam.dynamic_run && blockm * Info->Height >= gpu_m - cParam.dynamic_size))
					{
						cParam.dynamic_run -= Info->Height;
						cParam.dynamic_size = mymin(gpu_m, gpu_n);
						if (Info->Debug) fprintf(STD_OUT, "cParam dynamic size reduced to: %lld blockrows, %lld blocks\n", cParam.dynamic_run / Info->Height, cParam.dynamic_size / Info->Height);
					}

					if (nBlocks >= 256 && nBlocks - k - 1 > 16 && cParam.dynamic_run == Info->Height && cParam.dynamic_size < mymin(gpu_m, gpu_n)) cParam.dynamic_size += Info->Height;

					if (!Info->Quiet) fprintf(STD_OUT, "Scheduling Additional CPU DGEMM Run over %lld blockrows, %lld blocks\n", cParam.dynamic_run / Info->Height, cParam.dynamic_size / Info->Height);
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
				while (test_cpu_k > k && (DGEMM_favor_m ? (cpublockm * Info->Height >= gpu_m - cParam.dynamic_run && cpublockn * Info->Height >= gpu_n - cParam.dynamic_size) :
					(cpublockn * Info->Height >= gpu_n - cParam.dynamic_run && cpublockm * Info->Height >= gpu_m - cParam.dynamic_size)))
				{
					test_cpu_k--;
					DGEMM_getblocks(test_cpu_k, cpublockm, cpublockn);
				}
				if (test_cpu_k && k < test_cpu_k - 1)
				{
					if (!Info->Quiet) fprintf(STD_OUT, "Scheduling dynamic 3rd phase run, CPU taking tile %lld (m=%lld,n=%lld) from GPU\n", test_cpu_k, cpublockm, cpublockn);
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
	volatile caldgemm::SampleInfo* Info = par->cls->Info;

	if (Info->Debug) fprintf(STD_OUT, "Cblas helper thread started\n");

	if (Info->Debug) fprintf(STD_OUT, "Cblas thread Thread, setting CPU mask %X\n", par->cls->getcpumask(&par->cls->oldcpumask));
	sched_setaffinity(0, sizeof(par->cls->oldcpumask), &par->cls->oldcpumask);

	if (Info->MultiThread) if (pthread_mutex_lock(&par->cls->cParam.cblasMutex[1])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&par->cls->cParam.cblasMutex[1]) == 0 && par->terminate == CAL_FALSE)
	{
		const CALdouble Alpha = par->cls->Alpha;
		const CALdouble Beta = par->cls->Beta;
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
		if (!Info->Quiet) fprintf(STD_OUT, "\t\tSlave thread starting cblas (m: %lld, n: %lld, cblas_size: %lld, dynamic: %lld/%lld, cpu_k: %lld)\n", Info->m, Info->n, par->cblas_size, par->dynamic_run, par->dynamic_size, par->cpu_k);


		int old_goto_threads = get_num_procs();

		int require_threads = par->cls->outputthreads + 1;
		if (par->cls->ExecLinpack)
		{
			require_threads++;
		}
		if (Info->Debug) fprintf(stderr, "Reserving %d threads for gpu (/ Linpack)\n", require_threads);
		goto_set_num_threads(old_goto_threads - require_threads);
		caldgemm_goto_reserve_cpus(require_threads);

		par->cls->Timers.TotalCPUTimer.Start();
		if (par->cls->ExecLinpack)
		{
			if (!Info->Quiet) fprintf(STD_OUT, "\t\t\tDoint initial cblas runs to prepare Linpack factorization\n");
			par->cls->Timers.CPUTimer.Start();
			cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->Width, Info->n, Info->Width, Alpha, A - Info->Width * A_pitch_use, A_pitch, B, B_pitch, Beta, C - Info->Width * C_pitch, C_pitch);
			par->cls->Timers.CPUTimer.Stop();
#ifndef NO_ASYNC_LINPACK
			if (!Info->Quiet) fprintf(STD_OUT, "\t\t\tStarting Linpack factorization\n");
			par->cls->Timers.LinpackTimer1.Start();
			Info->linpack_factorize_function();
			par->cls->Timers.LinpackTimer1.Stop();
			if (Info->MultiThread)
			{
				pthread_mutex_unlock(&par->cls->linpackParameters.linpackMutex[0]);
			}
			else
			{
				par->cls->Timers.LinpackTimer1.Start();
				Info->linpack_broadcast_function();
				par->cls->Timers.LinpackTimer1.Stop();
			}
#endif
		}

		par->cls->Timers.CPUTimer.Start();
		do
		{
			if (par->dynamic_run2)
			{
				size_t blockm, blockn;
				par->cls->DGEMM_getblocks(par->cpu_k, blockm, blockn);
				cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->Height, Info->Height, Info->Width, Alpha, A + blockm * Info->Height * A_pitch_use, A_pitch, B + blockn * Info->Height * B_pitch_use, B_pitch, Beta, C + blockm * Info->Height * C_pitch + blockn * Info->Height, C_pitch);
			}
			else
			{
				if (par->dynamic_run)
				{
					if (par->cls->gpu_m >= par->cls->gpu_n)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->dynamic_run, par->dynamic_size, Info->Width, Alpha, A + (par->cls->gpu_m - par->dynamic_run) * A_pitch_use, A_pitch, B + (par->cls->gpu_n - par->dynamic_size) * B_pitch_use, B_pitch, Beta, C + (par->cls->gpu_m - par->dynamic_run) * C_pitch + par->cls->gpu_n - par->dynamic_size, C_pitch);
					}
					else
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->dynamic_size, par->dynamic_run, Info->Width, Alpha, A + (par->cls->gpu_m - par->dynamic_size) * A_pitch_use, A_pitch, B + (par->cls->gpu_n - par->dynamic_run) * B_pitch_use, B_pitch, Beta, C + (par->cls->gpu_m - par->dynamic_size) * C_pitch + par->cls->gpu_n - par->dynamic_run, C_pitch);
					}
				}

				size_t cblas2;
				if (par->cls->ExecLinpack && Info->MultiThread (((double) Info->m * (double) Info->n) - par->cls->linpack_last_mn[ExecuteLinpackCallbacks]) / par->cls->linpack_last_mn[ExecuteLinpackCallbacks] < 0.3 && par->cls->linpackCPUDGEMMTime[ExecuteLinpackCallbacks] - par->cls->linpackBcastTime[ExecuteLinpackCallbacks] > 4.0)
				{
					cblas2 = (double) (DGEMM_split_m ? Info->n : Info->m) * (par->cls->linpackBcastTime[ExecuteLinpackCallbacks] + 2.0) / par->cls->linpackCPUDGEMMTime[ExecuteLinpackCallbacks];
					fprintf(STD_OUT, "Splitting CPU DGEMM for later enabling additional cores, cblas2=%lld\n", cblas2);
				}
				else
				{
					cblas2 = 0;
				}

				if (DGEMM_split_m)	//favor splitting m because of consecutive memory
				{
					if (par->dynamic_run == 0)
					{
						if (cblas2)
						{
							cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->cblas_size, cblas2, Info->Width, Alpha, A + (Info->m - par->cblas_size) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Info->m - par->cblas_size) * C_pitch, C_pitch);

							if (pthread_mutex_trylock(&par->cls->linpackParameters.linpackMutex[1]) == EBUSY)
							{
								if (!Info->NoPerformanceWarnings) printf("Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
							}
							else
							{
								int require_threads_new = par->cls->outputthreads + 1;
								if (Info->Debug) fprintf(stderr, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
								goto_set_num_threads(old_goto_threads - require_threads_new);
								caldgemm_goto_reserve_cpus(require_threads_new);
								pthread_mutex_unlock(&par->cls->linpackParameters.linpackMutex[1]);
							}
						}
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->cblas_size, Info->n - cblas2, Info->Width, Alpha, A + (Info->m - par->cblas_size) * A_pitch_use, A_pitch, B + cblas2 * B_pitch_use, B_pitch, Beta, C + (Info->m - par->cblas_size) * C_pitch + cblas2, C_pitch);
					}

					if (Info->n % Info->Height && par->borders_done == CAL_FALSE)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m - par->cblas_size, Info->n % Info->Height, Info->Width, Alpha, A, A_pitch, B + (Info->n - Info->n % Info->Height) * B_pitch_use, B_pitch, Beta, C + Info->n - Info->n % Info->Height, C_pitch);
					}
				}
				else
				{
					if (par->dynamic_run == 0)
					{
						if (cblas2)
						{
							cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, cblas2, par->cblas_size, Info->Width, Alpha, A, A_pitch, B + (Info->n - par->cblas_size) * B_pitch_use, B_pitch, Beta, C + Info->n - par->cblas_size, C_pitch);
							
							if (pthread_mutex_trylock(&par->cls->linpackParameters.linpackMutex[1]) == EBUSY)
							{
								if (!Info->NoPerformanceWarnings) printf("Linpack broadcast was not finished at predicted time, running CPU DGEMM with reduced core count\n");
							}
							else
							{
								int require_threads_new = par->cls->outputthreads + 1;
								if (Info->Debug) fprintf(stderr, "Reserving %d threads for gpu during second cpu run\n", require_threads_new);
								goto_set_num_threads(old_goto_threads - require_threads_new);
								caldgemm_goto_reserve_cpus(require_threads_new);
								pthread_mutex_unlock(&par->cls->linpackParameters.linpackMutex[1]);
							}
						}
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m - cblas2, par->cblas_size, Info->Width, Alpha, A + cblas2 * A_pitch_use, A_pitch, B + (Info->n - par->cblas_size) * B_pitch_use, B_pitch, Beta, C + cblas2 * C_pitch + Info->n - par->cblas_size, C_pitch);
					}

					if (Info->m % Info->Height && par->borders_done == CAL_FALSE)
					{
						cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m % Info->Height, Info->n - par->cblas_size, Info->Width, Alpha, A + (Info->m - Info->m % Info->Height) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Info->m - Info->m % Info->Height) * C_pitch, C_pitch);
					}
				}
			}
			par->borders_done = CAL_TRUE;
		} while (par->cls->cpuScheduler());
		par->cls->Timers.CPUTimer.Stop();

#ifndef NO_ASYNC_LINPACK
		if (par->cls->ExecLinpack && Info->MultiThread)
		{
			pthread_mutex_lock(&par->cls->linpackParameters.linpackMutex[1]);
		}
#endif
		par->cls->Timers.TotalCPUTimer.Stop();
		goto_set_num_threads(old_goto_threads);
		caldgemm_goto_reserve_cpus(0);

		if (Info->Debug) fprintf(STD_OUT, "\t\tUnlocking cblasmutex 0\n");
		if (pthread_mutex_unlock(&par->cls->cParam.cblasMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (!Info->MultiThread) break;
	}
	if (Info->Debug) fprintf(STD_OUT, "blas slave terminating\n");
	if (Info->MultiThread)
	{
		pthread_exit(NULL);
	}
	return(NULL);
}

void* merge_wrapper(void* arg)
{
	caldgemm::mergeParameters* par = (caldgemm::mergeParameters*) arg;

	if (par->cls->Info->Debug) fprintf(STD_OUT, "Merger Thread %d started\n", par->nMergeThread);

	cpu_set_t merge_mask;
	CPU_ZERO(&merge_mask);
	CPU_SET(par->nMergeThread + 1, &merge_mask);
	if (par->cls->Info->Debug) fprintf(STD_OUT, "Merge Thread %d, setting CPU mask %X\n", par->nMergeThread, par->cls->getcpumask(&merge_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &merge_mask);

	if (pthread_mutex_lock(&par->mergeThreadMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
	while (pthread_mutex_lock(&par->mergeThreadMutex[0]) == 0 && par->terminate == CAL_FALSE)
	{
		if (par->cls->Info->Debug) fprintf(STD_OUT, "\t\tSlave thread %d starting merge process for context %d\n", par->nMergeThread, par->nContext);
		par->cls->mergeBuffers(par->dst, par->src, par->cls->Info->Height, par->cls->Info->Height, par->cls->BufferHeight, par->cls->BufferHeight, par->cls->C_pitch, par->cls->cPartsNum);
		if (par->cls->Info->Debug) fprintf(STD_OUT, "\t\tUnlocking mutex obuffer %d (Slavethread %d)\n", par->nContext, par->nMergeThread);
		if (pthread_mutex_unlock(&par->cls->obufferMutex[par->nContext])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (pthread_mutex_unlock(&par->mergeThreadMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}
	if (par->cls->Info->Debug) fprintf(STD_OUT, "merge slave %d terminating\n", par->nMergeThread);
	pthread_exit(NULL);
	return(NULL);
}

int calutil::DumpMatrix(double* a, double* b, double* c, double alpha, double beta, int tmp_m, int tmp_k, int tmp_n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB)
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

int caldgemm::RunCALDGEMM(double* a, double* b, double* c, double alpha, double beta, size_t tmp_m, size_t tmp_k, size_t tmp_n, size_t Apitch, size_t Bpitch, size_t Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int ExecuteLinpackCallbacks)
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "Caldgemm not initialized, aborting DGEMM run\n");
		return(1);
	}

	if (tmp_m == 0 || tmp_k == 0 || tmp_n == 0)
	{
		if (ExecuteLinpackCallbacks)
		{
			Timers.LinpackTimer1.Start();
			Info->linpack_factorize_function();
			Info->linpack_broadcast_function();
			Timers.LinpackTimer1.Stop();
		}
		return(0);		//Do Nothing
	}

	/*  //Disable output for all but one host in MPI rin
	if (strcmp(hostname, "gpu-dev05") != 0)
	{
	Info->Debug = CAL_FALSE;
	Info->Quiet = CAL_TRUE;
	Info->Verify = CAL_FALSE;
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
	if (tmp_m != -1) Info->m = tmp_m;
	if (tmp_n != -1) Info->n = tmp_n;
	if (tmp_k != -1) Info->Width = tmp_k;

	A_pitch = Apitch != -1 ? Apitch : Info->Width;
	B_pitch = Bpitch != -1 ? Bpitch : Info->n;
	C_pitch = Cpitch != -1 ? Cpitch : Info->n;
	ResetTimers();

	if (order == CblasColMajor)
	{
		double* tmpd;
		size_t tmpi;
		CBLAS_TRANSPOSE tmpt;
		tmpd = A; A = B; B = tmpd;
		tmpi = Info->m; Info->m = Info->n; Info->n = tmpi;
		tmpi = A_pitch; A_pitch = B_pitch; B_pitch = tmpi;
		tmpt = TransA;TransA = TransB;TransB = tmpt;
	}

	if (!Info->Quiet) fprintf(STD_OUT, "Starting DGEMM Run m=%lld k=%lld n=%lld Alpha=%lf Beta=%lf LDA=0x%lx LDB=0x%lx LDC=0x%lx At=%d Bt=%d ColMajor=%d (A=0x%llx, B=0x%llx, C=0x%llx, (C-A=%lld, (C-B)/w=%lld))\n", Info->m, Info->Width, Info->n, Alpha, Beta, A_pitch, B_pitch, C_pitch, (int) (TransA == CblasTrans), (int) (TransB == CblasTrans), (int) (order == CblasColMajor), A, B, C, ((size_t) C - (size_t) A) / sizeof(double), ((size_t) C - (size_t) B) / sizeof(double) / Info->Width);

	//Check for double == 1.0 is unsafe and causes compiler warning
	const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double
	const int kernel_num = (reinterpret_cast<long long int &>(Alpha) == double_one);
	if (kernel_num && Info->Debug) fprintf(STD_OUT, "Using Kernel for ALPHA = 1\n");

	TransposeA = TransA;
	TransposeB = TransB;    
	ExecLinpack = ExecuteLinpackCallbacks;

	if (Info->Verify)
	{
		D = new CALdouble[(size_t) Info->m * (size_t) C_pitch];
		if (D == NULL)
		{
			fprintf(STD_OUT, "Memory allocation error\n");
			return(1);
		}
		memcpy(D, C, Info->m * C_pitch * sizeof(CALdouble));
	}

	if (Info->DumpMatrix) DumpMatrix(A, B, C, Alpha, Beta, Info->m, Info->Width, Info->n, A_pitch, B_pitch, C_pitch, CblasRowMajor, TransposeA, TransposeB);

	Timers.System.Start();

	if (ExecuteLinpackCallbacks)
	{
		if (Info->m < Info->Width)
		{
			MaxGpuM = 0;
		}
		else
		{
			MaxGpuM = Info->m - Info->Width;
		}
	}
	else
	{
		MaxGpuM = Info->m;
	}
	MaxGpuN = Info->n;

#ifndef TESTMODE    
	//Check if the GPU can/shall process the required dgemm task
	if (Info->Iterations > 1);
	else if (Info->Width % 8 || Info->Width < 256) forceCPU = true;
	else if (MaxGpuM < 512 || MaxGpuN < 512) forceCPU = true;
	else if (__fpclassify(Alpha) == FP_ZERO) forceCPU = true;
	else if (((size_t) A) & (vcpysize - 1) || ((size_t) B) & (vcpysize - 1) || ((size_t) C) & (vcpysize - 1) ||
		A_pitch & (vcpysize / sizeof(CALdouble) - 1) || B_pitch & (vcpysize / sizeof(CALdouble) - 1) || C_pitch & (vcpysize / sizeof(CALdouble) - 1))
	{
		fprintf(STD_OUT, "Input addresses not aligned correctly: A 0x%llX B 0x%llX C 0x%llX Pitch 0x%llX 0x%llX 0x%llX\n", A, B, C, A_pitch, B_pitch, C_pitch);
		forceCPU = true;
	}
#endif

	if (Info->AutoHeight)
	{
		if (MaxGpuM < 1024 || MaxGpuN < 1024)
		{
			Info->Height = 512;
		}
		else if (MaxGpuM < 2048 || MaxGpuN < 2048 || MaxGpuM * MaxGpuN < 16 * 1024 * 1024)
		{
			Info->Height = 1024;
		}
		else if (MaxGpuM < 3072 || MaxGpuN < 3072 || MaxGpuM * MaxGpuN < 120 * 1024 * 1024)
		{
			Info->Height = 2048;
		}
		else if (MaxGpuM < 4096 || MaxGpuN < 4096 || MaxGpuM * MaxGpuN < 400 * 1024 * 1024)
		{
			Info->Height = 3072;
		}
		else
		{
			Info->Height = 4096;
		}
		if ((Info->Height != BufferHeight && !Info->Quiet) || Info->Debug)  fprintf(STD_OUT, "Using Height %lld of max %lld\n", Info->Height, BufferHeight);
	}

	if (Info->Width > BufferWidth || Info->Height > BufferHeight) forceReinit = true;

	if (Info->UseGPU == CAL_FALSE || Info->m < Info->Height || Info->n < Info->Height || (forceReinit && (long long int) MaxGpuM * (long long int) MaxGpuN * (long long int) Info->Width < (long long int) 24 * 1024 * 1024 * 1024) || (Info->Width < 1024 && Info->Height < 1024)) forceCPU = true;

	/*  //Run on CPU on all but one node in MPIRUN
	if (strcmp(hostname, "gpu-dev05") != 0)
	{
	fprintf(STD_OUT, "Hostname not 5 but %s\n", hostname);
	forceCPU = true;
	}*/

	if (forceCPU)
	{
		if (Info->Debug) fprintf(STD_OUT, "Running CPU only DGEMM\n");
		Timers.CPUTimer.Start();
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m, Info->n, Info->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
		Timers.CPUTimer.Stop();
		CPUOnlyRun = true;
#ifndef NO_ASYNC_LINPACK
		if (ExecuteLinpackCallbacks)
		{
			if (Info->Debug) fprintf(STD_OUT, "DGEMM was running on CPU only, executing linpack callback functions\n");
			Timers.LinpackTimer1.Start();
			Info->linpack_factorize_function();
			Info->linpack_broadcast_function();
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
		CPU_SET(outputthreads + 1, &divide_mask);
	}
	else
	{
		CPU_SET(0, &divide_mask);
	}
	if (Info->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&divide_mask));
	sched_setaffinity(0, sizeof(cpu_set_t), &divide_mask);

	if (forceReinit)
	{
		if (!Info->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Reinit for increased buffer width / height\n");
		for (int i = 0;i < bbuffers;i++)
		{
			CleanupData(&ctx_main, resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers, i);
			SetupData(modules[i], resourceHandlers[i], datas[i], &device, &ctx_main, numInputs, numOutputs, numConstantBuffers, progNames[i], i);
		}
	}

	if (Info->Debug) fprintf(STD_OUT, "Initiliazing GPU Constant Buffers...");
	for (int i = 0;i < 1;i++)
	{
		if (Info->Debug) fprintf(STD_OUT, "%d", i);
		cal_init_constant_data(datas[i], alpha);
		if (CopyDataToGPU(&ctx_main, resourceHandlers[i] + numInputs, datas[i] + numInputs, numConstantBuffers, CAL_TRUE, &events[i])) return(1);
	}
	if (Info->Debug) fprintf(STD_OUT, "   Done\n");

	if (Info->GPURatio < 0)
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
		GPURatio *= (double) Info->Width / (double) 1024;
		if (Info->Height < 1024) GPURatio *= (double) Info->Height / (double) 1024 * (double) Info->Height / (double) 1024;
		if (Info->Debug) fprintf(STD_OUT, "GPURatio automatically set to %1.2lf\n", GPURatio);
	}
	else
	{
		GPURatio = Info->GPURatio;
	}

	if (ExecuteLinpackCallbacks)
	{
		if (ExecuteLinpackCallbacks > 1) GPURatio = 1.0 - (1.0 - GPURatio) * 0.70;
		else GPURatio = 1.0 - (1.0 - GPURatio) * 0.90;
		if ((((double) MaxGpuM * (double) MaxGpuN) - linpack_last_mn[ExecuteLinpackCallbacks]) / linpack_last_mn[ExecuteLinpackCallbacks] < 0.3 && linpackGPURatios[ExecuteLinpackCallbacks] > 0.0001)
		{
			GPURatio = linpackGPURatios[ExecuteLinpackCallbacks];
			if (Info->Debug) fprintf(STD_OUT, "Taking GPU Ratio from table, entry %d, val %2.1lf\n", ExecuteLinpackCallbacks, 100 * GPURatio);
		}
		else
		{
			linpackGPURatios[ExecuteLinpackCallbacks] = GPURatio;
			if (Info->Debug) fprintf(STD_OUT, "Initializing ratio table entry %d with %2.1lf\n", ExecuteLinpackCallbacks, 100 * GPURatio);
		}
	}

	gpu_ratio_used = GPURatio;

	if (ExecuteLinpackCallbacks)
	{
		Info->m -= Info->Width;
		A += Info->Width * (TransposeA == CblasTrans ? 1 : A_pitch);
		C += Info->Width * (C_pitch);
	}

	cParam.dynamic_run = 0;
	cParam.dynamic_run2 = 0;
	cParam.borders_done = CAL_FALSE;
	if (Info->UseCPU == CAL_TRUE && Info->UseGPU == CAL_TRUE)
	{
		if (DGEMM_split_m = (Info->m >= Info->n))
		{
			size_t virtualm = Info->m + (Info->n % Info->Height) * Info->m / Info->n;
			if (ExecuteLinpackCallbacks) virtualm += Info->Width * (1.0 + (float) Info->m / Info->n);
			gpu_m = GPURatio * (float) virtualm + (Info->Height - 1);
			if (gpu_m > Info->m) gpu_m = Info->m;
			gpu_m -= gpu_m % Info->Height;
			cParam.cblas_size = Info->m - gpu_m;
			gpu_n = Info->n;
			gpu_n -= gpu_n % Info->Height;
			if (Info->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld\n", gpu_m, gpu_n, Info->m - gpu_m, gpu_n);
		}
		else
		{
			size_t virtualn = Info->n + (Info->m % Info->Height) * Info->n / Info->m;
			if (ExecuteLinpackCallbacks) virtualn += Info->Width * (1.0 + (float) Info->n / Info->m);
			gpu_n = GPURatio * (float) virtualn + (Info->Height - 1);
			if (gpu_n > Info->n) gpu_n = Info->n;
			gpu_n -= gpu_n % Info->Height;
			cParam.cblas_size = Info->n - gpu_n;
			gpu_m = Info->m;
			gpu_m -= gpu_m % Info->Height;
			if (Info->Debug) fprintf(STD_OUT, "Splitting: GPU: %lld x %lld, CPU: %lld x %lld\n", gpu_m, gpu_n, Info->m, Info->n - gpu_n);
		}
		/*if (cParam.cblas_size == 0 && Info->DynamicSched == CAL_TRUE)
		{
		cParam.dynamic_size = Info->Height;
		cParam.dynamic_run = (1.0f - GPURatio) * (float) mymax(gpu_m, gpu_n);
		cParam.dynamic_run -= cParam.dynamic_run % Info->Height;
		if (!Info->Quiet) fprintf(STD_OUT, "Scheduling initial dynamic run over %lldx%lld blocks\n", cParam.dynamic_run, cParam.dynamic_size);
		}*/
	}
	else
	{
		if (Info->n % Info->Height || Info->m % Info->Height)
		{
			fprintf(STD_OUT, "Invalid matrix size for GPU only\n");
			return(1);
		}
		gpu_n = Info->n;
		gpu_m = Info->m;
	}
	DGEMM_favor_m = gpu_m >= gpu_n;

	if (Info->UseCPU)
	{
		if (!Info->MultiThread)
		{
			cblas_wrapper((void*) &cParam);
		}
		if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
	}

	Timers.GPUTimer.Start();

	for (CALuint i = 0; i < Info->Iterations; ++i)
	{
		int oldj;
		int j = 0;
		int iMergeThread = 0;

		const size_t mb = gpu_m / Info->Height;
		const size_t nb = gpu_n / Info->Height;
		size_t blockm, blockn, lastm, lastn;
		size_t nBlocks = mb * nb;

		if (!Info->NoPerformanceWarnings && (buffersSwitchable ? mymin(nb, mb) : nb) > bbuffers) fprintf(STD_OUT, "WARNING: Insufficient buffers for Input Matrices, retransfer required\n");

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
							if (newblockm * Info->Height >= gpu_m - cParam.dynamic_run && newblockn * Info->Height >= gpu_n - cParam.dynamic_size)
							{
								if (Info->Debug) fprintf(STD_OUT, "GPU skipping k = %lld (Dynamic Run 2nd Phase)\n", k);
								continue;
							}
						}
						else
						{
							if (newblockn * Info->Height >= gpu_n - cParam.dynamic_run && newblockm * Info->Height >= gpu_m - cParam.dynamic_size)
							{
								if (Info->Debug) fprintf(STD_OUT, "GPU skipping k = %lld (Dynamic Run 2nd Phase)\n", k);
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
						if (Info->Debug) fprintf(STD_OUT, "gpu_k %lld reached cpu_k_barrier %lld, skipping remaining k (Dynamic Run 3rd Phase", k, cpu_k_barrier);
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
					if (Info->Debug) fprintf(STD_OUT, "Iteration k = %lld, m = %lld, n = %lld (Context %d)\n", k, blockm, blockn, j);


					if (k <= 1 || ctxcount == 1 || Info->AsyncDMA == CAL_FALSE) DGEMM_prepare(k, j);
					if (ctxcount > 1 && k >= 1 && Info->AsyncDMA)
					{
						size_t nextk = k + 1;
						size_t nextblockm, nextblockn;
						DGEMM_getblocks(nextk, nextblockm, nextblockn);
						if (cParam.dynamic_run)
						{
							while (DGEMM_favor_m ? (nextk < nBlocks && nextblockm * Info->Height >= gpu_m - cParam.dynamic_run && nextblockn * Info->Height >= gpu_n - cParam.dynamic_size) :
								(nextk < nBlocks && nextblockn * Info->Height >= gpu_n - cParam.dynamic_run && nextblockm * Info->Height >= gpu_m - cParam.dynamic_size))
							{
								nextk++;
								DGEMM_getblocks(nextk, nextblockm, nextblockn);
							}
						}
						if (nextk < nBlocks) DGEMM_prepare(nextk, (j + 1) % ctxcount);
					}

					if (Info->MultiThread)
					{
						if (Info->Debug) fprintf(STD_OUT, "\tLocking mutex %d\n", j);
						if (Info->AsyncTiming)
						{
							Timers.ATime.Reset();
							Timers.ATime.Start();
						}
						if (pthread_mutex_lock(&obufferMutex[j])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
						if (Info->AsyncTiming)
						{
							Timers.ATime.Stop();
							if (!Info->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001 || Info->Debug) fprintf(STD_OUT, "\t\tWait Time for output buffer: %1.5lf\n", Timers.ATime.GetElapsedTime());
						}
					}
					WAITFOREVENT(ctx_main, j);
					if (Info->Debug) fprintf(STD_OUT, "\tExecuting MM kernel\n");
					if (gpu_m < gpu_n && buffersSwitchable && bbuffers >= mb)
					{
						for (int l = 0;l < aPartsNum;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][l], datas[blockm][aPartsNum + l].dstMem), "setting kernel memory A");
						for (int l = 0;l < bPartsNum;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][aPartsNum + l], datas[blockn % 2][l].dstMem), "setting kernel memory B");
					}
					else
					{
						for (int l = 0;l < aPartsNum;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][l], datas[blockm % 2][l].dstMem), "setting kernel memory A");
						for (int l = aPartsNum;l < aPartsNum + bPartsNum;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][l], datas[nb > bbuffers ? (blockn % 2) : blockn][l].dstMem), "setting kernel memory B");
					}
					for (int l = 0;l < cPartsNum;l++) CHKERR(calCtxSetMem(ctx_main, progNames[0][kernel_num][numInputs + numConstantBuffers + l], datas[j][numInputs + numConstantBuffers + l].dstMem), "setting kernel output memroy");
					if (!RunProgram(&ctx_main, &modules[0][kernel_num], Info->Height / TILING_X, Info->Height / TILING_Y, &events[j])) {fprintf(STD_OUT, "Error running program\n"); return 1;}
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
					if (Info->DstMemory == 'g')
					{
						if (Info->VerboseTiming) Timers.CounterCopyFrom.Start();
						if (Info->Debug == CAL_TRUE) fprintf(STD_OUT, "\tFething part of C from GPU (m = %lld, n = %lld)\n", lastm, lastn);
						if (CopyDataFromGPU(&ctx_main, resourceHandlers[oldj] + numInputs + numConstantBuffers, datas[oldj] + numInputs + numConstantBuffers, numOutputs, &events[oldj])) {fprintf(STD_OUT, "Error copying from GPU\n"); return(1);}
						if (Info->VerboseTiming) Timers.CounterCopyFrom.Stop();
						WAITFOREVENT(ctx_main, oldj);
					}
					if (Info->VerboseTiming) Timers.CounterMerge.Start();

					if (k == nBlocks || Info->MultiThread == CAL_FALSE)
					{
						if (Info->Debug) fprintf(STD_OUT, "\tMerging buffer (context %d, main thread)\n", oldj);
						if (mergeBuffers(C + lastn * Info->Height + lastm * C_pitch * Info->Height, datas[oldj] + numInputs + numConstantBuffers, Info->Height, Info->Height, BufferHeight, BufferHeight, C_pitch, cPartsNum)) {fprintf(STD_OUT, "Error merging\n"); return(1);}
						if (Info->MultiThread)
						{
							if (pthread_mutex_unlock(&obufferMutex[oldj])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
							for (int l = 1;l < ctxcount;l++)
							{
								if (Info->Debug) fprintf(STD_OUT, "Waiting for context %d to finish merge process\n", (oldj + l) % ctxcount);
								if (pthread_mutex_lock(&obufferMutex[(oldj + l) % ctxcount])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
								if (pthread_mutex_unlock(&obufferMutex[(oldj + l) % ctxcount])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
							}
						}
					}
					else
					{
						if (Info->AsyncTiming)
						{
							Timers.ATime.Reset();
							Timers.ATime.Start();
						}
						if (pthread_mutex_lock(&mParam[iMergeThread].mergeThreadMutex[1])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
						if (Info->AsyncTiming)
						{
							Timers.ATime.Stop();
							if (!Info->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() > 0.001 || Info->Debug) fprintf(STD_OUT, "\t\tWARNING: Wait Time for merge thread: %1.5lf\n", Timers.ATime.GetElapsedTime());
						}
						if (Info->Debug) fprintf(STD_OUT, "\t\tUnlocking outputthread mutex %d to process context %d\n", iMergeThread, oldj);
						mParam[iMergeThread].nContext = oldj;
						mParam[iMergeThread].dst = C + (lastn * Info->Height + lastm * C_pitch * Info->Height);
						mParam[iMergeThread].src = datas[oldj] + numInputs + numConstantBuffers;
						if (pthread_mutex_unlock(&mParam[iMergeThread].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
						iMergeThread = (iMergeThread + 1) % outputthreads;
					}

					if (Info->VerboseTiming) Timers.CounterMerge.Stop();
				}
				oldj = j;
				j = (j + 1) % ctxcount;
			}
			if(Info->Verify && i < Info->Iterations - 1) AnalyzeResults(datas[0]);
	}
	Timers.GPUTimer.Stop();

	if (Info->Debug) fprintf(STD_OUT, "Caldgemm Main Thread, setting CPU mask %X\n", getcpumask(&oldcpumask));
	sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

	if (Info->UseCPU)
	{
		if (!Info->NoPerformanceWarnings && Info->MultiThread)
		{
			Timers.ATime.Reset();
			Timers.ATime.Start();
		}
		if (Info->Debug) fprintf(STD_OUT, "Waiting for CPU DGEMM to finish\n");
		if (pthread_mutex_lock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "Error locking mutex: %s - %d\n", __FILE__, __LINE__);
		if (!Info->NoPerformanceWarnings && Info->MultiThread)
		{
			Timers.ATime.Stop();
			cpu_wait_time = Timers.ATime.GetElapsedTime();
			if (!Info->NoPerformanceWarnings && Timers.ATime.GetElapsedTime() >= 0.15) fprintf(STD_OUT, "WARNING: CPU synchronisation took %2.4lf sec\n", Timers.ATime.GetElapsedTime());
			else if (Info->Debug) fprintf(STD_OUT, "CPU synchronisation took %2.4lf sec\n", Timers.ATime.GetElapsedTime());
		}
	}

RunCALDGEMM_end:
	Timers.System.Stop();
	outputthreads = old_outputthreads;

#ifndef NO_ASYNC_LINPACK
	if (!Info->UseCPU && ExecuteLinpackCallbacks)
#else
	if (ExecuteLinpackCallbacks)
#endif
	{
		if (!Info->Quiet) fprintf(STD_OUT, "No asynchronous processing of linpack functions possible, executing linpack callback functions\n");
		Timers.LinpackTimer1.Start();
		Info->linpack_factorize_function();
		Info->linpack_broadcast_function();
		Timers.LinpackTimer1.Stop();
	}

	if (Info->Debug) fprintf(STD_OUT, "DGEMM Run Complete\n");

#ifdef TESTMODE
	print_submatrices(C, 12, 24, Info->n, 1, 1, 1, 1);
#endif

	if (!Info->NoPerformanceWarnings && Info->UseCPU && Info->UseGPU && !CPUOnlyRun && fabs(Timers.TotalCPUTimer.GetElapsedTime() - Timers.GPUTimer.GetElapsedTime()) > 1.0)
	{
		fprintf(STD_OUT, "WARNING: Bad GPU / CPU Splitting: GPU Time: %2.4lf, CPU Time: %2.4lf (m = %lld, n = %lld)\n", Timers.GPUTimer.GetElapsedTime(), Timers.TotalCPUTimer.GetElapsedTime(), Info->m, Info->n);
	}
	displayMatrixTiming("caldgemm");
	AnalyzeResults(datas[0]);
	if (Info->Verify) delete[] D;

	if (ExecuteLinpackCallbacks)
	{
		const double tmpratio = 0.5;
		const double newratio = tmpratio * linpackGPURatios[ExecuteLinpackCallbacks] + (1.0 - tmpratio) * gpu_ratio_used;
		if (Info->Debug) fprintf(STD_OUT, "updating ratio table entry %d (old: %2.1lf, new: %2.1lf, factor: %2.1lf) => %2.1lf\n", ExecuteLinpackCallbacks, 100 * linpackGPURatios[ExecuteLinpackCallbacks], 100 * gpu_ratio_used, tmpratio, 100 * newratio);

		linpackGPURatios[ExecuteLinpackCallbacks] = newratio;
		linpackCPUDGEMMTime[ExecuteLinpackCallbacks] = Timers.CPUTimer.GetElapsedTime();
		linpackBcaseTime[ExecuteLinpackCallbacks] = Timers.LinpackTimer2.GetElapsedTime();
		linpack_last_mn[ExecuteLinpackCallbacks] = (double) Info->m * (double) Info->n;
	}

	return(0);
}

inline void caldgemm::DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn)
{
	if (DGEMM_favor_m)
	{
		const int nb = gpu_n / Info->Height;
		blockn = k % nb;
		blockm = k / nb;
	}
	else
	{
		const int mb = gpu_m / Info->Height;
		blockm = k % mb;
		blockn = k / mb;
	}
}

int caldgemm::DGEMM_prepare(size_t k, int j)
{
	const size_t nb = gpu_n / Info->Height;
	const size_t mb = gpu_m / Info->Height;
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	bool buffersSufficiant;
	if (DGEMM_favor_m)
	{
		buffersSufficiant = (bbuffers >= nb);
	}
	else
	{
		buffersSufficiant = (bbuffers >= mb && buffersSwitchable);
	}

	if (Info->VerboseTiming) Timers.CounterDivide.Start();
	if (blockn == 0 || (gpu_m < gpu_n && !buffersSufficiant)) 
	{
		if (Info->Debug) fprintf(STD_OUT, "\tDividing Buffer A (k = %lld)\n", k);
		Timers.divideA++;
#ifdef CALDGEMM_TRANSPOSED_A
		if (divideBuffer(Info->DivideToGPU && gpu_m < gpu_n && buffersSufficiant ? (datas[blockm] + aPartsNum) : datas[blockm % 2], A + blockm * Info->Height * (TransposeA == CblasTrans ? 1 : A_pitch), Info->Height, Info->Width, BufferHeight, BufferWidth, A_pitch, aPartsNum, TransposeA == CblasNoTrans)) return(1);
#else
		if (divideBuffer(Info->DivideToGPU && gpu_m < gpu_n && buffersSufficiant ? (datas[blockm] + aPartsNum) : datas[blockm % 2], A + blockm * Info->Height * (TransposeA == CblasTrans ? 1 : A_pitch), Info->Width, Info->Height, BufferWidth, BufferHeight, A_pitch, aPartsNum, TransposeA == CblasTrans)) return(1);
#endif
	}
	if (blockm == 0 || (DGEMM_favor_m && !buffersSufficiant))
	{
		if (Info->Debug) fprintf(STD_OUT, "\tDividing Buffer B (k = %lld)\n", k);
		Timers.divideB++;
#ifdef CALDGEMM_TRANSPOSED_B
		divideBuffer(Info->DivideToGPU && buffersSufficiant ? (datas[blockn] + (DGEMM_favor_m ? aPartsNum : 0)) : (datas[blockn % 2] + aPartsNum), B + blockn * Info->Height * (TransposeB == CblasTrans ? B_pitch : 1), Info->Width, Info->Height, BufferWidth, BufferHeight, B_pitch, bPartsNum, TransposeB == CblasNoTrans);
#else
		divideBuffer(Info->DivideToGPU && buffersSufficiant ? (datas[blockn] + (DGEMM_favor_m ? aPartsNum : 0)) : (datas[blockn % 2] + aPartsNum), B + blockn * Info->Height * (TransposeB == CblasTrans ? B_pitch : 1), Info->Height, Info->Width, BufferHeight, BufferWidth, B_pitch, bPartsNum, TransposeB == CblasTrans);
#endif
	}
	if (Info->VerboseTiming) Timers.CounterDivide.Stop();

	if (Info->VerboseTiming) Timers.CounterCopyTo.Start();
	if (Info->DivideToGPU == CAL_FALSE)
	{
		if (blockn == 0 || (gpu_m < gpu_n && !buffersSufficiant))
		{
			if (Info->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (k = %lld)\n", k);
			if (gpu_m < gpu_n && buffersSufficiant)
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j], datas[blockm % 2], aPartsNum, CAL_FALSE, &events[j], datas[blockm] + aPartsNum)) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
			else
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j], datas[blockm % 2], aPartsNum, CAL_FALSE, &events[j])) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
		}

		if (blockm == 0 || (DGEMM_favor_m && !buffersSufficiant))
		{
			if (Info->Debug) fprintf(STD_OUT, "\tCopying part of B to GPU (k = %lld)\n", k);
			if (gpu_m < gpu_n && buffersSufficiant)
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j] + aPartsNum, datas[blockn % 2] + aPartsNum, bPartsNum, CAL_FALSE, &events[j], datas[blockn % 2])) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
			else
			{
				if (CopyDataToGPU(&ctx_main, resourceHandlers[j] + aPartsNum, datas[blockn % 2] + aPartsNum, bPartsNum, CAL_FALSE, &events[j], datas[buffersSufficiant ? blockn : (blockn % 2)] + aPartsNum)) {fprintf(STD_OUT, "Error copying to GPU\n"); return(1);}
			}
		}
	}
	if (Info->VerboseTiming) Timers.CounterCopyTo.Stop();
	calCtxFlush(ctx_main);
}

int caldgemm::ExitCALDGEMM()
{
	if (!caldgemm_initialized)
	{
		fprintf(STD_OUT, "CALDGEMM not initialized, cannot uninitialize!\n");
		return(1);
	}
	if (Info->Debug) fprintf(STD_OUT, "Uninitializing CALDGEMM\n");
	for (int i = 0;i < bbuffers;i++)
	{
		//if (Info->Debug) fprintf(STD_OUT, "Running Cleanup for Context/bbuffer %d\n", i);
		if (!Cleanup(&device, &ctx_main, modules[i], resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers, i))
		{
			return 1;
		}
		if (i < max_outputthreads && Info->MultiThread)
		{
			if (Info->Debug) fprintf(STD_OUT, "Trying to terminate merge slave %d\n", i);
			mParam[i].terminate = CAL_TRUE;
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

	if (Info->UseCPU && Info->UseGPU)
	{
		if (Info->Debug) fprintf(STD_OUT, "Trying to terminate blas slave\n");
		cParam.terminate = CAL_TRUE;
		if (Info->MultiThread && pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking blas mutex 1 to terminate thread\n");
		if (pthread_mutex_unlock(&cParam.cblasMutex[0])) fprintf(STD_OUT, "Error unlocking blas mutex 0 to terminate thread\n");
	}

	if (Info->MultiThread)
	{
		if (Info->Debug) fprintf(STD_OUT, "Trying to terminate linpack slave\n");
		linpackParameters.terminate = CAL_TRUE;
		if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error unlocking blas mutex 0 to terminate thread\n");
		if (Info->Debug) fprintf(STD_OUT, "Waiting for linpack slave to terminate\n");
		while (pthread_mutex_trylock(&linpackParameters.linpackMutex[0]) != EBUSY) if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
		if (pthread_mutex_unlock(&linpackParameters.linpackMutex[0])) fprintf(STD_OUT, "Error unlocking blasMutex 1\n");

		if (Info->Debug) fprintf(STD_OUT, "Waiting for merge threads to terminate\n");
		for (int i = 0;i < max_outputthreads;i++)
		{
			while (pthread_mutex_trylock(&mParam[i].mergeThreadMutex[0]) != EBUSY) if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			if (pthread_mutex_unlock(&mParam[i].mergeThreadMutex[0])) fprintf(STD_OUT, "Error unlocking mergeMutex %d/1\n", i);
		}
		if (Info->UseCPU && Info->UseGPU)
		{
			if (Info->Debug) fprintf(STD_OUT, "Waiting for blas threads to terminate\n");
			while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking mutex: %s - %d\n", __FILE__, __LINE__);
			if (pthread_mutex_unlock(&cParam.cblasMutex[1])) fprintf(STD_OUT, "Error unlocking blasMutex 1\n");
		}
	}

	for (int j = 0;j < 2;j++)
	{
		if (Info->UseCPU && Info->UseGPU) if (pthread_mutex_destroy(&cParam.cblasMutex[j])) fprintf(STD_OUT, "Error destroying blas mutex %d\n", j);
	}
	if (Info->MultiThread)
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
}

#define MAX_HUGE_ADDRESSES 256
double* huge_page_addresses[MAX_HUGE_ADDRESSES];
int nHugeAddresses = 0;

double* caldgemm::AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages)
{
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

		if (Info->Debug)  fprintf(STD_OUT, "Running Huge Maloc\n");

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

// vim: ts=4 sw=4 noet sts=4 tw=100
