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

#include "caldgemm_cal.h"
#include "caldgemm_common.h"

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

const char* caldgemm_cal::ILFakeKernel =
"il_ps_2_0\n"
"dcl_input_position_interp(linear_noperspective) vWinCoord0.xy__\n"
"dcl_output_generic o0\n"
"mov o0, vWinCoord0.0000\n"
"end\n"
;

const char* caldgemm_cal::ILConvertKernel =
"il_ps_2_0\n"
"dcl_input_position_interp(linear_noperspective) vWinCoord0.xy__\n"
"dcl_resource_id(0)_type(2d,unnorm)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(1)_type(2d,unnorm)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_output_generic o0\n"
"dcl_output_generic o1\n"
"dcl_literal l0, 2.0, 1.0, 0.5, 0.0\n"
"sub r0.xy__, vWinCoord0.xy00, l0.zz00\n"
"mul r0.xy__, r0.xy00, l0.xy00\n"
"add r0.xy__, r0.xy00, l0.zz00\n"
"add r1.xy__, r0.xy00, l0.yw00\n"
"sample_resource(0)_sampler(0) r13, r0.xy\n"
"sample_resource(0)_sampler(0) r14, r1.xy\n"
"sample_resource(1)_sampler(1) r15, r0.xy\n"
"sample_resource(1)_sampler(1) r16, r1.xy\n"
"mov o0.xy, r13.xy\n"
"mov o0.z, r14.x\n"
"mov o0.w, r14.y\n"
"mov o1.xy, r15.xy\n"
"mov o1.z, r16.x\n"
"mov o1.w, r16.y\n"
"end\n"
;

#define CHKERR(cmd, text) if (cmd != CAL_RESULT_OK) {fprintf(STD_OUT, "Error '%s' while " text "\n", calGetErrorString());return(1);}
#define WAITFOREVENTA(ctx, event) { CALresult r; do { r = calCtxIsEventDone(ctx, event); if (r == CAL_RESULT_ERROR) { fprintf(STD_OUT, "Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}

int caldgemm_cal::WaitForEvent(int eventnr, int devicenr, int lock)
{
	CALresult r;
	if (Config->Debug) fprintf(STD_OUT, "\tWaiting for event from device %d obuffer %d...\n", devicenr, eventnr);
	do
	{
		if (lock) pthread_mutex_lock(&device_mutex[devicenr]);
		r = calCtxIsEventDone(ctxs[devicenr], events[devicenr][eventnr]);
		if (lock) pthread_mutex_unlock(&device_mutex[devicenr]);
		if (r == CAL_RESULT_ERROR) { fprintf(STD_OUT, "Error while waiting for event\nError String: %s\n", calGetErrorString());
		return(1);}
	} while (r == CAL_RESULT_PENDING);
	return(0);
}

#ifdef CALDGEMM_UNALIGNED_ADDRESSES
#define _mm_load_pd_use _mm_loadu_pd
#else
#define _mm_load_pd_use _mm_load_pd
#endif

#define _mm_store_pd_use _mm_stream_pd
#define CALDGEMM_USE_VEC_MEMCPY_PREFETCH

caldgemm_cal::caldgemm_cal() : caldgemm()
{
}

caldgemm_cal::~caldgemm_cal()
{
}

int caldgemm_cal::divideBuffer(BufferProperties* dst, double* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers, bool transpose)
{
	if (Config->Debug) fprintf(STD_OUT, "\t\tSRC=0x%llx, w: %d, h: %d, pitch: %d (gpuw: %d, gpuh: %d, transpose: %d)\n", (long long int) src, width, height, pitch, gpu_width, gpu_height, (int) transpose);

	if (Config->DivideToGPU)
	{
		for (int i = 0;i < numBuffers;i++)
		{
			CHKERR(calResMap(&dst[i].ptr_void, &dst[i].pitch, dst[i].res, 0), "mapping input buffer for buffer division");
			if (((size_t) dst[i].ptr_void) & (vcpysize - 1))
			{
				fprintf(STD_OUT, "Invalid alignment\n");
				return(1);
			}
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
			const double* __restrict__ saddr0 = &src[(y + 0) * pitch];
			const double* __restrict__ saddr2 = &src[(y + 2) * pitch];

			double* __restrict__ dstBank0 = &dst[0].ptr_double[y * 2];
			double* __restrict__ dstBank1 = &dst[1].ptr_double[y * 2];

			for (int i = 0; i < height_4; ++i)
			{
				double* __restrict__ daddr0 = &dstBank0[i * gpu_width * 2];
				double* __restrict__ daddr1 = &dstBank1[i * gpu_width * 2];

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
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 100), _MM_HINT_NTA);
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr2 + 100), _MM_HINT_NTA);
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
				_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 60), _MM_HINT_NTA);
				_mm_prefetch(CAST_FOR_MMPREFETCH (saddr2 + 60), _MM_HINT_NTA);
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
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 30), _MM_HINT_NTA);
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
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 76), _MM_HINT_NTA);
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
				_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 100), _MM_HINT_NTA);
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
	{
		for (int i = 0;i < numBuffers;i++)
		{
			CHKERR(calResUnmap(dst[i].res), "unmapping input buffer for buffer division");
		}
	}
	return(0);
}

int caldgemm_cal::mergeBuffers(double* dst, BufferProperties* src, int width, int height, int gpu_width, int gpu_height, int pitch, int numBuffers)
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

	if (Config->Width == BufferWidth && reinterpret_cast<unsigned long long int &>(Beta) == double_one && reinterpret_cast<unsigned long long int &>(Alpha) == double_minus_one)
	{
		//Special Linpack Function
		for (int y=0; y < height; y++)
		{
			const int bank = y % 4;
			double* saddr = src[bank].ptr_double + (y / 4) * (gpu_width / 2);
			double* saddr2 = src[bank + 4].ptr_double + (y / 4) * (gpu_width / 2);
			double* daddr = dst + (y * pitch);
			//int count = src[bank].DataSize * width;


#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
				for (int i = 0;i < width;i += 8)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 50), _MM_HINT_NTA);
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr2 + 50), _MM_HINT_NTA);
#ifndef _NO_AMD_CPU
					_m_prefetchw(daddr + 50);
#else
					_mm_prefetch(CAST_FOR_MMPREFETCH (daddr + 50), _MM_HINT_NTA);
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
#ifdef _WIN32
			if (Beta == 0.)
#else
			if (__fpclassify(Beta) == FP_ZERO)
#endif
			{
				//CALDGEMM_44 BETA=ZERO HACKED LIB
				for (int i = 0;i < count;i += 64)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 50), _MM_HINT_NTA);
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr2 + 50), _MM_HINT_NTA);
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
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 50), _MM_HINT_NTA);
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr2 + 50), _MM_HINT_NTA);
#ifndef _NO_AMD_CPU
					_m_prefetchw(daddr + 50);
#else
					_mm_prefetch(CAST_FOR_MMPREFETCH (daddr + 50), _MM_HINT_NTA);
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
#ifdef _WIN32
			if (Beta == 0.)
#else
			if (__fpclassify(Beta) == FP_ZERO)
#endif
			{
				//CALDGEMM_44 BETA=ZERO ORIGINAL LIB
				for (int i = 0;i < count;i += 128)
				{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 50), _MM_HINT_NTA);
					_mm_prefetch(CAST_FOR_MMPREFETCH (saddr2 + 50), _MM_HINT_NTA);
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
					_mm_prefetch(CAST_FOR_MMPREFETCH (daddr + 50), _MM_HINT_NTA);
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
#ifdef _WIN32
		if (Beta == 0.)
#else
		if (__fpclassify(Beta) == FP_ZERO)
#endif
		{
			//CALDGEMM_84 BETA=0
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_stream_pd
			for (int i = 0;i < count;i += 64)
			{
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
				_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 100), _MM_HINT_NTA);
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
				_mm_prefetch(CAST_FOR_MMPREFETCH (saddr + 100), _MM_HINT_NTA);
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

void caldgemm_cal::checkCalPatch()
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

void caldgemm_cal::cal_init_constant_data(BufferProperties* &data, double alpha)
{
	data[dwBuffersA + dwBuffersB].ptr_float[0] = (float) TILING_Y / Config->Height;			//Scale factor for normalized y pos
	data[dwBuffersA + dwBuffersB].ptr_float[2] = (float) TILING_X / Config->Height;			//Scale factor for normalized x pos
#ifdef CALDGEMM_44
	data[dwBuffersA + dwBuffersB].ptr_float[1] = 1.f / Config->Width;							//Step in K direction
#ifdef CALDGEMM_44_BT_64_KERNEL
	data[dwBuffersA + dwBuffersB].ptr_float[4] = static_cast<float>(Config->Width * 2);			//Iterations of loop in IL Kernel
#else
	data[dwBuffersA + dwBuffersB].ptr_float[4] = static_cast<float>(Config->Width);			//Iterations of loop in IL Kernel
#endif
#else //CALDGEMM_44
	data[dwBuffersA + dwBuffersB].ptr_float[1] = 2.f / Config->Width;							//Step in K direction
	data[dwBuffersA + dwBuffersB].ptr_float[4] = static_cast<float>(Config->Width / (dwBuffersB << 2));	//Iterations of loop in IL Kernel
#endif //CALDGEMM_44
	data[dwBuffersA + dwBuffersB].ptr_float[3] = 0.f;
	data[dwBuffersA + dwBuffersB].ptr_float[5] = (float) dwBuffersA / Config->Height;			//For transposed matrix finer y resolution is needed
	data[dwBuffersA + dwBuffersB].ptr_float[8] = 0.5f - 0.5f / (float) (TILING_Y / dwBuffersA);

	//Constants for Memexport
	data[dwBuffersA + dwBuffersB].ptr_int[9] = TILING_Y * Config->Height / 2;				//2 for double2
	data[dwBuffersA + dwBuffersB].ptr_int[10] = TILING_X / 2;						//x tiling in double2
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
#ifdef CALDGEMM_SGEMM
	data[dwBuffersA + dwBuffersB].ptr_int[20] = Config->Height * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[21] = Config->Height * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[22] = Config->Height * Config->Height / 2;
	data[dwBuffersA + dwBuffersB].ptr_int[23] = Config->Height * Config->Height / 2;
#endif
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

int caldgemm_cal::Initialize(int deviceNum, bool nocalinit)
{
	if (!Config->Quiet) fprintf(STD_OUT, "Initializing CALDGEMM (CAL Runtime)\n");

	if (!nocalinit) CHKERR(calInit(), "initializing CAL");

	numInputs = dwBuffersA + dwBuffersB;
	numOutputs = dwBuffersC;
	numConstantBuffers = 1;

	if (deviceNum == -1 && obuffercount > 1 && Config->MultiThread)
	{
		CALuint tmp;
		calDeviceGetCount(&tmp);
		nDevices = tmp;
		if (nDevices > (signed) max_devices) nDevices = max_devices;
		if (nDevices > Config->NumDevices) nDevices = Config->NumDevices;
		for (int i = 0;i < nDevices;i++)
		{
			device_nums[i] = i;
		}
	}
	else
	{
		CALuint tmp;
		calDeviceGetCount(&tmp);
		if (tmp == 0)
		{
			nDevices = 0;
		}
		else
		{
			if (deviceNum == -1)
			{
				if (obuffercount == 1)
				{
					fprintf(STD_OUT, "Cannot use multiple devices with obuffercount = 1\n");
				}
				if (!Config->MultiThread)
				{
					fprintf(STD_OUT, "Cannot use multiple devices without multithreading\n");
				}
				deviceNum = 0;
			}
			nDevices = 1;
			device_nums[0] = deviceNum;
		}
	}
	if (Config->Debug) fprintf(STD_OUT, "Initializing CALDGEMM for %d devices\n", nDevices);
	gpu_available = (nDevices > 0);
	for (int i = 0;i < nDevices;i++)
	{
	    devices[i] = 0;
	    CHKERR(calDeviceOpen(&devices[i], device_nums[i]), "opening CAL device");
	    CHKERR(calCtxCreate(&ctxs[i], devices[i]), "creating CAL context");
	}
	return(0);
}

static void log_callback(const char *msg)
{
	fprintf(STD_OUT, "%s", msg);
}

int caldgemm_cal::SetupKernel(const char* ILKernel, CALmodule* module, CALcontext* ctx, unsigned int device_num, bool disassemble)
{
	CALimage image = NULL;

	CALdeviceattribs attribs;
	attribs.struct_size = sizeof(CALdeviceattribs);
	CHKERR(calDeviceGetAttribs(&attribs, device_num), "getting device attributes");

	CALobject obj;
	char* ILKernelUse = (char*) malloc(strlen(ILKernel) + 1024);
#ifdef CALDGEMM_44_BT_64_KERNEL
	sprintf(ILKernelUse, ILKernel, Config->Width * 2);
#else
	sprintf(ILKernelUse, ILKernel, Config->Width);
#endif
	if (Config->PrintILKernel) fprintf(STD_OUT, "Kernel:\n%s\n", ILKernelUse);
	CHKERR(calclCompile(&obj, CAL_LANGUAGE_IL, ILKernelUse, attribs.target), "compiling the kernel");
	free(ILKernelUse);

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

int caldgemm_cal::RunProgram(CALcontext *ctx, CALmodule *module, unsigned int Width, unsigned int Height, CALevent* event)
{
	CALfunc func;
	CHKERR(calModuleGetEntry(&func, *ctx, *module, "main"), "finding module entry point");

	CALdomain rect;
	rect.x = 0;
	rect.y = 0;
	rect.width = Width;
	rect.height = Height;

	if (Config->VerboseTiming) Timers.Kernel.Start();
#ifdef CALDGEMM_BENCHMARK_KERNEL
	for (int i = 0;i < CALDGEMM_BENCHMARK_KERNEL;i++)
#endif
	CHKERR(calCtxRunProgram(event, *ctx, func, &rect), "executing kernel");

	if (Config->VerboseTiming)
	{
		if (event) WAITFOREVENTA(*ctx, *event);
		Timers.Kernel.Stop();
		if (Config->Debug) fprintf(STD_OUT, "\tTotal Kernel Time: %2.4lf\n", Timers.Kernel.GetElapsedTime());
	}

	return(0);
}

int caldgemm_cal::CleanupData(CALcontext* ctx, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext, unsigned int num_device)
{
	if (data)
	{
		for (unsigned int i = 0; i < numHandles;++i)
		{
			if ((nContext == 0 || i != dwBuffersA + dwBuffersB) && (nContext < 2 || i >= dwBuffersA) && (nContext < obuffercount || i < dwBuffersA + dwBuffersB) && data[i].ptr_char)
			{
#ifdef DEBUG_MSG_ALLOCATION
				if (Config->Debug) fprintf(STD_OUT, "Freeing CAL Host memory, device %d context %d buffer %d\n", num_device, nContext, i);
#endif
				if (data[i].CALMemory)
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
#ifdef CALDGEMM_44_BT_64_CONVERT
			if (nContext < 2 && i < dwBuffersA + dwBuffersB)
			{
#ifdef DEBUG_MSG_ALLOCATION
				if (Config->Debug) fprintf(STD_OUT, "Freeing temporary CAL memory, device %d context %d buffer %d\n", num_device, nContext, i);
#endif
				CHKERR(calCtxReleaseMem(*ctx, data[i].tmpmem), "releasing temporary CAL memory");
				CHKERR(calResFree(data[i].tmpres), "releasing temporary CAL resources");
			}
#endif
		}
	}

	if (resourceHandler)
	{
		for (unsigned int i = 0; i < numHandles; i++)
		{
			if ((nContext == 0 || i != dwBuffersA + dwBuffersB) && (nContext < 2 || i >= dwBuffersA) && (nContext < obuffercount || i < dwBuffersA + dwBuffersB) && resourceHandler[i])
			{
#ifdef DEBUG_MSG_ALLOCATION
				if (Config->Debug) fprintf(STD_OUT, "Freeing CAL GPU memory, device %d context %d buffer %d\n", num_device, nContext, i);
#endif
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

int caldgemm_cal::Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, BufferProperties* &data, unsigned int numHandles, int nContext, unsigned int num_device)
{
	CleanupData(ctx, resourceHandler, data, numHandles, nContext, num_device);

	if (nContext < 1)
	{
		for (int i = 0;i < kernel_count;i++)
		{
			if (module[i])
			{
				CHKERR(calModuleUnload(*ctx, module[i]), "unloading module");
			}
		}
		CHKERR(calModuleUnload(*ctx, modulesConvert[num_device]), "unloading module");
	}
	delete[] resourceHandler;
	delete[] data;

	return(0);
}

int caldgemm_cal::CopyDataFromGPU(int nDevice, CALresource* _Res, BufferProperties* data, unsigned int num, int nContext, size_t lastm, size_t lastn)
{
	if (Config->DstMemory == 'c') return 0;
	CALcontext* ctx = &ctxs[nDevice];
	if (Config->VerboseTiming) Timers.CounterCopyFrom.Start();
	if (Config->Debug == true) fprintf(STD_OUT, "\tFetching part of C from GPU (m = %lld, n = %lld)\n", (long long int) lastm, (long long int) lastn);
	unsigned int pitch;
	char* ptr;
	if (Config->ImplicitDriverSync == 0) WaitForEvent(nContext, nDevice);
	for (unsigned int i = 0; i < num; ++i)
	{
		if (data[i].CALMemory)
		{
			//if (Config->Debug) fprintf(STD_OUT, "GPUHandle: %d, CPUHandle: %d\n", data[i].dstMem, data[i].mem);
			CHKERR(calMemCopy(&events[nDevice][nContext], *ctx, data[i].dstMem, data[i].mem, 0), "copying data from gpu");
			continue;
		}
		CHKERR(calResMap((void**)&ptr, &pitch, _Res[i], 0), "mapping buffer");
		memcpy(data[i].ptr_char, ptr, data[i].DataSize * data[i].VectorSize * data[i].Width * data[i].Height);
		CHKERR(calResUnmap(_Res[i]), "unmapping buffer");
	}
	if (Config->VerboseTiming)
	{
		WaitForEvent(nContext, nDevice);
		Timers.CounterCopyFrom.Stop();
	}
	return 0;
}

int caldgemm_cal::CopyDataToGPU(CALcontext* ctx, CALresource* _Res, BufferProperties* data, unsigned int num, bool constants, CALevent* event, int num_device, BufferProperties* dest_data)
{
	if (dest_data == NULL) dest_data = data;
	unsigned int pitch;
	char* ptr;
	for (unsigned int i = 0; i < num; ++i)
	{
		if (data[i].CALMemory == constants) continue;
		if (data[i].CALMemory)
		{
#ifdef CALDGEMM_44_BT_64_CONVERT
			CHKERR(calMemCopy(event, *ctx, data[i].mem, data[i].tmpmem, 0), "copying to gpu");
#else
			CHKERR(calMemCopy(event, *ctx, data[i].mem, dest_data[i].dstMem, 0), "copying data to gpu");
#endif
			continue;
		}
		CHKERR(calResMap((void**)&ptr, &pitch, _Res[i], 0), "Mapping Buffer");
		memcpy(ptr, data[i].ptr_char, data[i].DataSize * data[i].VectorSize * data[i].Width * data[i].Height);
		CHKERR(calResUnmap(_Res[i]), "unmapping buffer");
	}
	if (Config->VerboseTiming && constants == false) WAITFOREVENTA(*ctx, *event);
#ifdef CALDGEMM_44_BT_64_CONVERT
	if (!constants)
	{
		dest_data[0].conversionBuffer = data;
	}
#endif
	return 0;
}

int caldgemm_cal::ValidateCALRuntime()
{
	CALVersion available;
	calGetVersion(&available.major, &available.minor, &available.imp);
	
	if (Config->ImplicitDriverSync == -1)
	{
		if (available.major > 2 || available.minor > 4 || (available.minor == 4 && available.imp > 900)) Config->ImplicitDriverSync = 0;
		else Config->ImplicitDriverSync = 1;
		if (Config->DstMemory == 'g' && !Config->Quiet) fprintf(STD_OUT, "Implicit driver sync automatically set to %d\n", Config->ImplicitDriverSync);
	}
	
	if (available.major < 1) return(1);
	if (available.major > 2) return(0);
	if (available.minor < 3 || (available.minor == 3 && available.imp < 185)) return(1);
	if (available.minor < 4 || (available.minor == 4 && available.imp < 815))
	{
		if (Config->AsyncDMA && !Config->NoPerformanceWarnings) fprintf(STD_OUT, "WARNING: Asynchronous DMA not supported by CAL Runtime Version\n");
		Config->AsyncDMA = false;
	}
	if (Config->Debug)
	{
		fprintf(STD_OUT, "CAL Runtime Version:%d.%d.%d\n", available.major, available.minor, available.imp);
	}
	return(0);
}

int caldgemm_cal::ValidateRuntime()
{
	if(ValidateCALRuntime())
	{
		fprintf(STD_OUT, "Error. Could not find a compatible CAL runtime.\n");
		return(1);
	}

#ifdef CALDGEMM_44
	if (Config->Width % 8)
	{
		fprintf(STD_OUT, "Only width of multiples of 8 are computable.\n");
		return(1);
	}
	else if (Config->Width % 64)
	{
		fprintf(STD_OUT, "Cannot allocate buffer corresponding to Config->Width, increasing buffer size from %lld to %lld\n", (long long int) Config->Width, (long long int) (Config->Width + 64 - Config->Width % 64));
		Config->Width += 64 - Config->Width % 64;
	}
#else
	if (Config->Width % 64)
	{
		fprintf(STD_OUT, "Only width of size 64 are computable.\n");
		return(1);
	}
#endif
	if (Config->Height & 0x7)
	{
		fprintf(STD_OUT, "Only heights with multiple of 8 are computable.\n" );
		return(1);
	}
	else if (Config->Height % 256)
	{
		fprintf(STD_OUT, "Cannot allocate buffer corresponding to Config->Height, increasing buffer size from %lld to %lld\n", (long long int) Config->Height, (long long int) (Config->Height + 256 - Config->Height % 256));
		Config->Height += 256 - Config->Height % 256;
	}

	return(0);
}

int caldgemm_cal::CheckDevices()
{
	for (int i = 0;i < nDevices;i++)
	{
		CALdeviceattribs attribs;
		attribs.struct_size = sizeof(CALdeviceattribs);
		if (calDeviceGetAttribs(&attribs, device_nums[i]) != CAL_RESULT_OK)
		{
			fprintf(STD_OUT, "Error getting device attributes\n");
			return 1;
		}
		if (!attribs.doublePrecision)
		{
			fprintf(STD_OUT, "The device does not support double precision\n");
			return(1);
		}
		conf_gpufreq = attribs.engineClock ? attribs.engineClock : 850;
		conf_gpushaders = attribs.numberOfSIMD * attribs.wavefrontSize;
	}

	if (Config->KeepBuffersMapped)
	{
		if (SetupKernel(ILFakeKernel, &fakeModule, &ctxs[0], device_nums[0], false)) return(1);
		CALresource tmpres = 0;
		CHKERR(calResAllocLocal2D(&tmpres, devices[0], 128, 128, CAL_FORMAT_FLOAT_4, 0), "checking for CAL patch");
		void* tmpptr;
		unsigned int tmppitch;
		CHKERR(calResMap((CALvoid**) &tmpptr, &tmppitch, tmpres, 0), "checking for CAL patch");
		CALname tmpname;
		CHKERR(calModuleGetName(&tmpname, ctxs[0], fakeModule, "o0"), "checking for CAL patch");
		CALmem tmpmem;
		CHKERR(calCtxGetMem(&tmpmem, ctxs[0], tmpres), "checking for CAL patch");
		CHKERR(calCtxSetMem(ctxs[0], tmpname, tmpmem), "checking for CAL patch");

		if (RunProgram(&ctxs[0], &fakeModule, 0, 0, &events[0][0]))
		{
			fprintf(STD_OUT, "Error running test kernel on GPU\nKeepBuffersMapped disabled\n");
			Config->KeepBuffersMapped = false;
		}
		//if (Config->KeepBuffersMapped) checkCalPatch();

		calCtxReleaseMem(ctxs[0], tmpmem);
		calResFree(tmpres);
		if (calModuleUnload(ctxs[0], fakeModule) != CAL_RESULT_OK )
		{
			fprintf(STD_OUT, "Error unloading test module\n");
			fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
		}
	}

	return(0);
}

int caldgemm_cal::InitDevices()
{
	int min_bbuffers = max_bbuffers;
	for (int device_num = 0;device_num < nDevices;device_num++)
	{
		cpu_set_t tmpmask;
		CPU_ZERO(&tmpmask);
		CPU_SET(Config->GPUMapping[device_num], &tmpmask);
		sched_setaffinity(0, sizeof(tmpmask), &tmpmask);
		
		int num_bbuffers;
		if (Config->DstMemory == 'g') num_bbuffers =  max_bbuffers_g;
		else num_bbuffers = max_bbuffers;
		for (int i = 0;i < num_bbuffers;i++)
		{
			if (i < 1)
			{
				if (SetupKernel(ILKernel, &modules[device_num][0], &ctxs[device_num], device_nums[device_num], (bool) (Config->Disassemble && i == 0)) ||
					SetupKernel(ILKernelALPHA1, &modules[device_num][1], &ctxs[device_num], device_nums[device_num], (bool) (Config->Disassemble && i == 0)) ||
					SetupKernel(ILKernelLinpack, &modules[device_num][2], &ctxs[device_num], device_nums[device_num], (bool) (Config->Disassemble && i == 0)) ||
					SetupKernel(ILConvertKernel, &modulesConvert[device_num], &ctxs[device_num], device_nums[device_num], (bool) (Config->Disassemble && i == 0)))
				{
					return 1;
				}
				for (int j = 0;j < kernel_count;j++) progNames[device_num][j] = new CALname[numInputs + numOutputs + numConstantBuffers];
			}

			datas[device_num][i] = new BufferProperties[numInputs + numOutputs + numConstantBuffers];
			resourceHandlers[device_num][i] = new CALresource[numInputs + numOutputs + numConstantBuffers];
			memset(datas[device_num][i], 0, (numInputs + numOutputs + numConstantBuffers) * sizeof(BufferProperties));
			memset(resourceHandlers[device_num][i], 0, (numInputs + numOutputs + numConstantBuffers) * sizeof(CALresource));
			if (SetupData(modules[device_num], resourceHandlers[device_num][i], datas[device_num][i], &devices[device_num], &ctxs[device_num], numInputs, numOutputs, numConstantBuffers, progNames[device_num], i, device_num))
			{
				if (i < obuffercount) return 1;
				else break;
			}
			bbuffers[device_num] = i + 1;
		}
		if (Config->Debug) fprintf(STD_OUT, "Was able to allocate %d bbuffers on device %d\n", bbuffers[device_num], device_num);
		if (bbuffers[device_num] < min_bbuffers) min_bbuffers = bbuffers[device_num];
	}
	if (!Config->Quiet) fprintf(STD_OUT, "Running on %d devices with %d bbuffers\n", nDevices, min_bbuffers);

	return(0);
}

int caldgemm_cal::ReinitDevices()
{
	for (int num_device = 0;num_device < nDevices;num_device++)
	{
		for (int i = 0;i < bbuffers[num_device];i++)
		{
			CleanupData(&ctxs[num_device], resourceHandlers[num_device][i], datas[num_device][i], numInputs + numOutputs + numConstantBuffers, i, num_device);
			SetupData(modules[num_device], resourceHandlers[num_device][i], datas[num_device][i], &devices[num_device], &ctxs[num_device], numInputs, numOutputs, numConstantBuffers, progNames[num_device], i, num_device);
		}
	}
	return(0);
}

int caldgemm_cal::InitConstantData(double alpha)
{
	if (Config->Debug) fprintf(STD_OUT, "Initiliazing GPU Constant Buffers...");
	for (int i = 0;i < nDevices;i++)
	{
		if (Config->Debug) fprintf(STD_OUT, "%d", i);
		cal_init_constant_data(datas[i][0], alpha);
		if (CopyDataToGPU(&ctxs[i], resourceHandlers[i][0] + numInputs, datas[i][0] + numInputs, numConstantBuffers, true, &events[i][0], i)) return(1);
	}
	if (Config->Debug) fprintf(STD_OUT, "   Done\n");
	return(0);
}

int caldgemm_cal::ExecuteKernels(caldgemm::DGEMMPrepareAndExecuteTask& Task, int blockm, int blockn)
{
	if (WaitForEvent(Task.j, Task.device)) return(1);

	if (Config->Debug) fprintf(STD_OUT, "\tExecuting MM kernel (device %d obuffer %d, k=%lld m=%lld n=%lld)\n", Task.device, Task.j, (long long int) Task.k, (long long int) blockm, (long long int) blockn);
#ifdef REUSE_BBUFFERS
	if (!DGEMM_favor_m && buffersSwitchable)
	{
		const int buffer_pos = buffer_pointers_A[Task.device][blockm] % (buffer_pointers_A[Task.device][blockm] < bbuffers[Task.device] ? bbuffers[Task.device] : 2);
		for (unsigned int l = 0;l < dwBuffersA;l++) CHKERR(calCtxSetMem(ctxs[Task.device], progNames[Task.device][Task.kernel_num][l], datas[Task.device][buffer_pos][dwBuffersA + l].dstMem), "setting kernel memory A");
		for (unsigned int l = 0;l < dwBuffersB;l++) CHKERR(calCtxSetMem(ctxs[Task.device], progNames[Task.device][Task.kernel_num][dwBuffersA + l], datas[Task.device][buffer_pointers_B[Task.device][blockn] % 2][l].dstMem), "setting kernel memory B");
#ifdef CALDGEMM_44_BT_64_CONVERT
		for (int ll = 0;ll < 2;ll++)
		{
			BufferProperties* const chkBuffer = ((ll == 1) ? datas[Task.device][buffer_pointers_B[Task.device][blockn] % 2] : &datas[Task.device][buffer_pos][dwBuffersA]);
			if (chkBuffer[0].conversionBuffer != NULL)
			{
				if (Config->Debug) fprintf(STD_OUT, "\tStarting conversion kernel for device %d input matrix %d (path 1)\n", Task.device, ll);
				for (unsigned int i = 0; i < ((ll == 1) ? dwBuffersB : dwBuffersA); ++i)
				{
					CHKERR(calCtxSetMem(ctxs[Task.device], progNamesConvert[Task.device][i], chkBuffer[0].conversionBuffer[i].tmpmem), "setting convert kernel memory in");
					CHKERR(calCtxSetMem(ctxs[Task.device], progNamesConvert[Task.device][i + dwBuffersA], chkBuffer[i].dstMem), "setting convert kernel memory out");
				}
				if (RunProgram(&ctxs[Task.device], &modulesConvert[Task.device], chkBuffer[0].Width, chkBuffer[0].Height, &events[Task.device][Task.j]))
				{
					fprintf(STD_OUT, "Error running conversion kernel\n");
					return(1);
				}
				chkBuffer[0].conversionBuffer = NULL;
			}
		}
#endif
	}
	else
#endif
	{
#ifdef REUSE_BBUFFERS
		const bool buffersSufficiant = buffer_pointers_B[Task.device][blockn] < bbuffers[Task.device];
#else
		const bool buffersSufficiant = false;
#endif
		for (unsigned int l = 0;l < dwBuffersA;l++) CHKERR(calCtxSetMem(ctxs[Task.device], progNames[Task.device][Task.kernel_num][l], datas[Task.device][buffer_pointers_A[Task.device][blockm] % 2][l].dstMem), "setting kernel memory A");
		for (unsigned int l = dwBuffersA;l < dwBuffersA + dwBuffersB;l++) CHKERR(calCtxSetMem(ctxs[Task.device], progNames[Task.device][Task.kernel_num][l], datas[Task.device][!buffersSufficiant ? (buffer_pointers_B[Task.device][blockn] % 2) : (buffer_pointers_B[Task.device][blockn] % bbuffers[Task.device])][l].dstMem), "setting kernel memory B");
#ifdef CALDGEMM_44_BT_64_CONVERT
		for (int ll = 0;ll < 2;ll++)
		{
			BufferProperties* const chkBuffer = ((ll == 1) ? &datas[Task.device][!buffersSufficiant ? (buffer_pointers_B[Task.device][blockn] % 2) : (buffer_pointers_B[Task.device][blockn] % bbuffers[Task.device])][dwBuffersA] : datas[Task.device][buffer_pointers_A[Task.device][blockm] % 2]);
			if (chkBuffer[0].conversionBuffer != NULL)
			{
				if (Config->Debug) fprintf(STD_OUT, "\tStarting conversion kernel for device %d input matrix %d (path 2)\n", Task.device, ll);
				for (unsigned int i = 0; i < ((ll == 1) ? dwBuffersB : dwBuffersA); ++i)
				{
					CHKERR(calCtxSetMem(ctxs[Task.device], progNamesConvert[Task.device][i], chkBuffer[0].conversionBuffer[i].tmpmem), "setting convert kernel memory in");
					CHKERR(calCtxSetMem(ctxs[Task.device], progNamesConvert[Task.device][i + dwBuffersA], chkBuffer[i].dstMem), "setting convert kernel memory out");
				}
				if (RunProgram(&ctxs[Task.device], &modulesConvert[Task.device], chkBuffer[0].Width, chkBuffer[0].Height, &events[Task.device][Task.j]))
				{
					fprintf(STD_OUT, "Error running conversion kernel\n");
					return(1);
				}
				chkBuffer[0].conversionBuffer = NULL;
			}
		}
#endif
	}
	for (unsigned int l = 0;l < dwBuffersC;l++) CHKERR(calCtxSetMem(ctxs[Task.device], progNames[Task.device][Task.kernel_num][numInputs + numConstantBuffers + l], datas[Task.device][Task.j][numInputs + numConstantBuffers + l].dstMem), "setting kernel output memroy");
	if (RunProgram(&ctxs[Task.device], &modules[Task.device][Task.kernel_num], (((size_t) blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height) / TILING_X, (((size_t) blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height) / TILING_Y, &events[Task.device][Task.j])) {fprintf(STD_OUT, "Error running program\n"); return 1;}
	if (Config->ImplicitDriverSync && Config->DstMemory == 'g' && CopyDataFromGPU(Task.device, resourceHandlers[Task.device][Task.j] + numInputs + numConstantBuffers, datas[Task.device][Task.j] + numInputs + numConstantBuffers, numOutputs, Task.j, blockm, blockn)) {fprintf(STD_OUT, "Error copying from GPU\n"); return(1);}
	calCtxFlush(ctxs[Task.device]);
	return(0);
}

int caldgemm_cal::SetupData(CALmodule *module, CALresource* &_Res, BufferProperties* &data, CALdevice *device, CALcontext *ctx, unsigned int numInputs, unsigned int numOutputs, unsigned int numConstantBuffers, CALname** ctxProgNames, int nContext, unsigned int num_device)
{
	BufferHeight = Config->Height;
	BufferWidth = Config->Width;
	const unsigned int bStop = dwBuffersA + dwBuffersB;
	const unsigned int fStop = bStop + numConstantBuffers;
	const unsigned int cStop = fStop + dwBuffersC;
	CALresult r = CAL_RESULT_OK;

	for (unsigned int i = 0; i < cStop; ++i)
	{
		if (nContext >= 1 && i == dwBuffersA + dwBuffersB) continue;
		if (nContext >= 2 && i < dwBuffersA) continue;
		if (nContext >= obuffercount && (i < dwBuffersA || i >= bStop)) continue;

		cpu_set_t tmpmask;
		CPU_ZERO(&tmpmask);
		CPU_SET(Config->AllocMapping[num_device] == -1 ? (i >= fStop && Config->PostprocessMapping[num_device] != -1 ? Config->PostprocessMapping[num_device] : Config->GPUMapping[num_device]) : Config->AllocMapping[num_device], &tmpmask);
		sched_setaffinity(0, sizeof(tmpmask), &tmpmask);

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
#ifdef CALDGEMM_SGEMM
			tHeight *= 2;
#endif
			mem = Config->DstMemory;
			flag = (CALresallocflags) (flag | CAL_RESALLOC_CACHEABLE);
		}
		

		CALformat bufferformat;

		data[i].DataSize = sizeof(double);
		data[i].Width = tWidth;
		data[i].Height = tHeight;
		data[i].VectorSize = mComponents;
		bool allocated = false;

#ifdef CALDGEMM_44_BT_64
#ifdef CALDGEMM_44_BT_64_CONVERT
		if (i < dwBuffersA + dwBuffersB && nContext < 2)
#else
		if (i < dwBuffersA + dwBuffersB)
#endif
		{
			tWidth *= 2;					//Change size after storing to data[i] to make divide/mergebuffer run on original size
			bufferformat = CAL_FORMAT_UNSIGNED_INT32_2;
			mComponents = 1;
		}
		else
#endif
		{
			bufferformat = CAL_FORMAT_UNSIGNED_INT32_4;
		}

		if (tHeight > 1)
		{
			data[i].CALMemory = true;
			if ((Config->DstMemory == 'g' || i < dwBuffersA + dwBuffersB) && (Config->DivideToGPU == false || i >= dwBuffersA + dwBuffersB) && (nContext < 2 || (Config->DstMemory == 'g' && i >= dwBuffersA + dwBuffersB + numConstantBuffers)))
			{
				allocated = true;
#ifdef DEBUG_MSG_ALLOCATION
				if (Config->Debug) fprintf(STD_OUT, "Allocating Host buffer for device %d obuffer %d buffer %d\n", num_device, nContext, i);
#endif
				CHKERR(calResAllocRemote2D(&data[i].res, device, 1, tWidth, tHeight, bufferformat, flag), "allocattion of remote memory");
				CHKERR(calCtxGetMem(&data[i].mem, *ctx, data[i].res), "getting remote memory for context");
				CHKERR(calResMap(&data[i].ptr_void, &data[i].pitch, data[i].res, 0), "mapping of remote memory");
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
#ifdef DEBUG_MSG_ALLOCATION
				if (Config->Debug) fprintf(STD_OUT, "Allocating Host memory for device %d obuffer %d buffer %d\n", num_device, nContext, i);
#endif
				data[i].ptr_char = new char[tWidth * sizeof(double) * mComponents * tHeight];
				allocated = true;
			}
			data[i].CALMemory = false;
		}
		if (allocated)
		{
#ifdef DEBUG_MSG_ALLOCATION
			if (Config->Debug) fprintf(STD_OUT, "Clearing Memory at %p, Width = %d, Height = %d, components = %d, type=double\n", (void*) data[i].ptr_char, (int) tWidth, (int) tHeight, (int) mComponents);
#endif
			memset((void*)data[i].ptr_char, 0, tWidth * sizeof(double) * mComponents * tHeight);
		}

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
#ifdef DEBUG_MSG_ALLOCATION
		if (Config->Debug) fprintf(STD_OUT, "Allocating device buffer for device %d obuffer %d buffer %d\n", num_device, nContext, i);
#endif

		CALresource* resuse;
		CALmem* memuse;
#ifdef CALDGEMM_44_BT_64_CONVERT
		if (i <= dwBuffersA + dwBuffersB && nContext < 2)
		{
#ifdef DEBUG_MSG_ALLOCATION
			if (Config->Debug) fprintf(STD_OUT, "Allocating temporary device buffer for device %d context %d buffer %d\n", num_device, nContext, i);
#endif
			if (calResAllocLocal2D(&_Res[i], *device, tWidth / 2, tHeight, CAL_FORMAT_UNSIGNED_INT32_4, flag) != CAL_RESULT_OK)
			{
				fprintf(STD_OUT, "Error allocating GPU memory\n");
				return(1);
			}
			CHKERR(calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]), "binding temporary memory to context");
			resuse = &data[i].tmpres;
			memuse = &data[i].tmpmem;
		}
		else
#endif
		{
			resuse = &_Res[i];
			memuse = &data[i].dstMem;
		}

		switch(mem)
		{
		case 'g':
			r = calResAllocLocal2D(resuse, *device, tWidth, tHeight, bufferformat, flag);

			break;
		case 'c':
			r = calResAllocRemote2D(resuse, device, 1, tWidth, tHeight, bufferformat, flag);
			break;
		}
		if (r != CAL_RESULT_OK)
		{
			for (unsigned int j = dwBuffersA;j < i;j++)
			{
				calCtxReleaseMem(*ctx, data[j].dstMem);
				calResFree(_Res[j]);
			}

			if (nContext < obuffercount)
			{
				fprintf(STD_OUT, "There was an error in allocating resources and binding them to memory (Error code %d)\n", r);
			}
			else if (Config->Debug)
			{
				fprintf(STD_OUT, "No more memory available for bbuffers\n");
			}
			return(1);
		}
		CHKERR(calCtxGetMem(memuse, *ctx, *resuse), "binding memory to context");
		if ((Config->DstMemory == 'c' && i >= fStop) || (Config->DivideToGPU && i < bStop))
		{
			data[i].mem = data[i].dstMem;
			data[i].res = _Res[i];
		}
		if (Config->DstMemory == 'c' && i >= fStop && Config->KeepBuffersMapped)
		{
			CHKERR(calResMap(&data[i].ptr_void, &data[i].pitch, data[i].res, 0), "mapping of remote output memory");
		}
	}

	cpu_set_t tmpmask;
	CPU_ZERO(&tmpmask);
	CPU_SET(Config->GPUMapping[num_device], &tmpmask);
	sched_setaffinity(0, sizeof(tmpmask), &tmpmask);

	if (nContext >= 1) return(0);

	for (unsigned int i = bStop; i < fStop; ++i)
	{
		int cWidth = data[i].Width * data[i].Height;
#ifdef DEBUG_MSG_ALLOCATION
		if (Config->Debug) fprintf(STD_OUT, "Allocating Host Constant buffer device %d context %d buffer %d\n", num_device, nContext, i);
#endif
		CHKERR(calResAllocRemote1D(&_Res[i], device, 1, cWidth, CAL_FORMAT_FLOAT_4, 0), "allocating constant memory");
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
#ifdef DEBUG_MSG_ALLOCATION
			if (Config->Debug) fprintf(STD_OUT, "Getting module buffer name for device %d context %d kernel %d buffer %d name %s\n", num_device, nContext, i, j, buffer);
#endif
			CHKERR(calModuleGetName(&ctxProgNames[i][j], *ctx, module[i], buffer), "getting buffer name");
			if (j >= bStop && j < fStop)
			{
				CHKERR(calCtxSetMem(*ctx, ctxProgNames[i][j], data[j].dstMem), "setting memory buffer to context");
			}
		}
	}
	
	for (unsigned int j = 0;j < dwBuffersA;j++)
	{
		char buffer[10];
		sprintf(buffer, "i%d", j);
		CHKERR(calModuleGetName(&progNamesConvert[num_device][j], *ctx, modulesConvert[num_device], buffer), "getting buffer name");
		sprintf(buffer, "o%d", j);
		CHKERR(calModuleGetName(&progNamesConvert[num_device][j + dwBuffersA], *ctx, modulesConvert[num_device], buffer), "getting buffer name");
	}

	return(0);
}

int caldgemm_cal::ExitRuntime()
{
	for (int i = 0;i < nDevices;i++)
	{
#ifdef DEBUG_MSG_ALLOCATION
		if (Config->Debug) fprintf(STD_OUT, "Uninitializing context for device %d\n", i);
#endif
		if (ctxs[i]) calCtxDestroy(ctxs[i]);
		if (devices[i])
		{
			if (calDeviceClose(devices[i]) != CAL_RESULT_OK)
			{
				fprintf(STD_OUT, "There was an error closing the device.\n");
				fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
			}
		}
	}

#ifdef DEBUG_MSG_ALLOCATION
	if (Config->Debug) fprintf(STD_OUT, "Uninitializing CAL runtime\n");
#endif

	if (gpu_available && calShutdown() != CAL_RESULT_OK)
	{
		fprintf(STD_OUT, "There was an error during cal shutdown.\n");
		fprintf(STD_OUT, "Error string is %s\n", calGetErrorString());
	}

	for (int i = 0;i < nDevices;i++) for (int j = 0;j < kernel_count;j++) delete[] progNames[i][j];

	return(0);
}

int caldgemm_cal::FetchResult(int device, int j, int m, int n)
{
	return(CopyDataFromGPU(device, resourceHandlers[device][j] + numInputs + numConstantBuffers, datas[device][j] + numInputs + numConstantBuffers, numOutputs, j, m, n));
}

int caldgemm_cal::RunMergeBuffers(double* dst, int device, int j, int width, int height, int gpu_width, int gpu_height, int pitch)
{
	return(mergeBuffers(dst, datas[device][j] + numInputs + numConstantBuffers, width, height, gpu_width, gpu_height, pitch, dwBuffersC));
}

int caldgemm_cal::DGEMM_prepare_backend(size_t k, int j, unsigned int num_device, bool prepareM, bool prepareN, bool buffersSufficiant, bool buffersSufficiant0)
{
	size_t blockm, blockn;
	DGEMM_getblocks(k, blockm, blockn);

	if (prepareM)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer A (device = %d, k = %lld, buffer = %d)\n", num_device, (long long int) k, next_buffer_A[num_device] % 2);
		if (Config->VerboseTiming) Timers.CounterDivide.Start();
		Timers.divideA++;
#ifdef CALDGEMM_TRANSPOSED_A
		if (divideBuffer(Config->DivideToGPU && !DGEMM_favor_m && buffersSufficiant ? (datas[num_device][blockm] + dwBuffersA) : datas[num_device][next_buffer_A[num_device] % 2], A + blockm * Config->Height * (TransposeA ? 1 : A_pitch), (blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height, Config->Width, BufferHeight, BufferWidth, A_pitch, dwBuffersA, TransposeA == false)) return(1);
#else
		if (divideBuffer(Config->DivideToGPU && !DGEMM_favor_m && buffersSufficiant ? (datas[num_device][blockm] + dwBuffersA) : datas[num_device][next_buffer_A[num_device] % 2], A + blockm * Config->Height * (TransposeA ? 1 : A_pitch), Config->Width, (blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height, BufferWidth, BufferHeight, A_pitch, dwBuffersA, TransposeA)) return(1);
#endif
		if (Config->VerboseTiming) Timers.CounterDivide.Stop();
		if (Config->DivideToGPU == false)
		{
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of A to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
			if (Config->VerboseTiming) Timers.CounterCopyTo.Start();
			if (!DGEMM_favor_m && buffersSufficiant0)
			{
				if (CopyDataToGPU(&ctxs[num_device], resourceHandlers[num_device][j], datas[num_device][next_buffer_A[num_device] % 2], dwBuffersA, false, &events[num_device][j], num_device, datas[num_device][buffer_pointers_A[num_device][blockm] % (buffersSufficiant ? bbuffers[num_device] : 2)] + dwBuffersA)) {fprintf(STD_OUT, "Error copying A to GPU (minor)\n"); return(1);}
			}
			else
			{
				if (CopyDataToGPU(&ctxs[num_device], resourceHandlers[num_device][j], datas[num_device][next_buffer_A[num_device] % 2], dwBuffersA, false, &events[num_device][j], num_device)) {fprintf(STD_OUT, "Error copying A to GPU (major)\n"); return(1);}
			}
			if (Config->VerboseTiming) Timers.CounterCopyTo.Stop();
		}
	}

	if (prepareN)
	{
		if (Config->Debug) fprintf(STD_OUT, "\tDividing Buffer B (device = %d, k = %lld, buffer = %d)\n", num_device, (long long int) k, next_buffer_B[num_device] % 2);
		if (Config->VerboseTiming) Timers.CounterDivide.Start();
		Timers.divideB++;
#ifdef CALDGEMM_TRANSPOSED_B
		divideBuffer(Config->DivideToGPU && buffersSufficiant ? (datas[num_device][blockn] + (DGEMM_favor_m ? dwBuffersA : 0)) : (datas[num_device][next_buffer_B[num_device] % 2] + dwBuffersA), B + blockn * Config->Height * (TransposeB ? B_pitch : 1), Config->Width, (blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height, BufferWidth, BufferHeight, B_pitch, dwBuffersB, TransposeB == false);
#else
		divideBuffer(Config->DivideToGPU && buffersSufficiant ? (datas[num_device][blockn] + (DGEMM_favor_m ? dwBuffersA : 0)) : (datas[num_device][next_buffer_B[num_device] % 2] + dwBuffersA), B + blockn * Config->Height * (TransposeB ? B_pitch : 1), Config->Height, Config->Width, (blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height, BufferWidth, B_pitch, dwBuffersB, TransposeB);
#endif
		if (Config->VerboseTiming) Timers.CounterDivide.Stop();
		if (Config->DivideToGPU == false)
		{
			if (Config->Debug) fprintf(STD_OUT, "\tCopying part of B to GPU (k = %lld, m = %lld, n = %lld)\n", (long long int) k, (long long int) blockm, (long long int) blockn);
			if (Config->VerboseTiming) Timers.CounterCopyTo.Start();
			if (!DGEMM_favor_m && buffersSufficiant0)
			{
				if (CopyDataToGPU(&ctxs[num_device], resourceHandlers[num_device][j] + dwBuffersA, datas[num_device][next_buffer_B[num_device] % 2] + dwBuffersA, dwBuffersB, false, &events[num_device][j], num_device, datas[num_device][next_buffer_B[num_device] % 2])) {fprintf(STD_OUT, "Error copying B to GPU (major)\n"); return(1);}
			}
			else
			{
				if (CopyDataToGPU(&ctxs[num_device], resourceHandlers[num_device][j] + dwBuffersA, datas[num_device][next_buffer_B[num_device] % 2] + dwBuffersA, dwBuffersB, false, &events[num_device][j], num_device, datas[num_device][buffersSufficiant ? (buffer_pointers_B[num_device][blockn] % bbuffers[num_device]) : (next_buffer_B[num_device] % 2)] + dwBuffersA)) {fprintf(STD_OUT, "Error copying B to GPU (minor)\n"); return(1);}
			}
			if (Config->VerboseTiming) Timers.CounterCopyTo.Stop();
		}
	}
	calCtxFlush(ctxs[num_device]);
	
	return(0);
}

int caldgemm_cal::ExitDevices()
{
	for (int num_device = 0;num_device < nDevices;num_device++)
	{
		for (int i = 0;i < bbuffers[num_device];i++)
		{
#ifdef DEBUG_MSG_ALLOCATION
			if (Config->Debug) fprintf(STD_OUT, "Uninitializing buffers for device %d context %d\n", num_device, i);
#endif
			if (Cleanup(&devices[num_device], &ctxs[num_device], modules[num_device], resourceHandlers[num_device][i], datas[num_device][i], numInputs + numOutputs + numConstantBuffers, i, num_device))
			{
				return 1;
			}
		}
	}
	return(0);
}

int caldgemm_cal::UseOutputPthreads() {return(1);}
int caldgemm_cal::UseInputPthreads() {return(1);}
int caldgemm_cal::UseMutexPerDevice() {return(1);}

int caldgemm_cal::reserve_cpu_cores()
{
	int nthreads = 0;
	int mainfound = 0;
	for (int i = 0;i < nDevices;i++)
	{
		int offset = 0;
		for (int j = 0;j < i;j++)
		{
			if (Config->GPUMapping[i] == Config->GPUMapping[j] && Config->PostprocessMapping[j] != -1) offset++;
		}
		if (offset == 0)
		{
			if (Config->MultiThreadDivide || i == 0)
			{
				caldgemm_goto_reserve_cpu(Config->GPUMapping[i], 1);
				if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for DivideBuffer\n", Config->GPUMapping[i]);
				nthreads++;
			}
		}
		for (int j = 0;j < outputthreads;j++)
		{
			const int merge_core = Config->PostprocessMapping[i] == -1 ? (Config->GPUMapping[i] + 1 + offset * outputthreads + j) : (Config->PostprocessMapping[i] + j);
			caldgemm_goto_reserve_cpu(merge_core, 1);
			if (Config->Debug) fprintf(STD_OUT, "Reserving Core %d for MergeBuffer\n", merge_core);
		}
		nthreads += outputthreads;
		if (Config->GPUMapping[i] == Config->PinMainThread) mainfound = 1;
	}
	if (mainfound == 0)
	{
		caldgemm_goto_reserve_cpu(Config->PinMainThread, 1);
		nthreads++;
	}
	if (Config->Debug) fprintf(STD_OUT, "Reserved %d cores\n", nthreads);
	return(nthreads);
}

int caldgemm_cal::RunCALDGEMM_Init()
{
	return(0);
}

int caldgemm_cal::RunCALDGEMM_Exit()
{
	return(0);
}

bool caldgemm_cal::cpuUsed(int cpu)
{
	if (cpu == Config->PinMainThread) return(true);
	for (int i = 0;i < nDevices;i++)
	{
		int procsreq = 1;
		for (int j = i;j < nDevices;j++)
		{
			if (Config->GPUMapping[i] == Config->GPUMapping[j] && Config->PostprocessMapping[j] == -1) procsreq += outputthreads;
		}
		if (cpu >= Config->GPUMapping[i] && cpu < Config->GPUMapping[i] + procsreq) return(true);
		if (Config->PostprocessMapping[i] != -1 && cpu >= Config->PostprocessMapping[i] && cpu < Config->PostprocessMapping[i] + outputthreads) return(true);
	}

	return(false);
}
