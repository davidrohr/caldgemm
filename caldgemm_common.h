/**
 * This file is part of the CALDGEMM library.
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

#ifndef CALDGEMM_COMMON_H
#define CALDGEMM_COMMON_H

#include <signal.h>

#ifdef _WIN32
#define __INTRIN_H_
#define _Complex
#ifndef __CUDA_ARCH__
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#endif
#endif

#if !defined(_WIN32) & defined(USE_GOTO_BLAS)
extern "C" {
#define CBLAS
#define ASSEMBLER
#include <common_linux.h>
#undef ASSEMBLER
#include <common.h>
}
#else

#ifndef USE_GOTO_BLAS
#include <omp.h>
#endif

extern "C" int get_num_procs();
static inline void caldgemm_goto_reserve_cpu(int, int) {}
static inline void caldgemm_goto_reserve_cpus(int) {}

typedef int blasint;
extern "C" {
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
}

#ifndef _WIN32
void goto_set_num_threads(int num);
void caldgemm_goto_restrict_cpus(int);

#ifdef USE_MKL
#define CBLAS_ENUM
#else
#define CBLAS_ENUM enum
#endif

extern "C" {
void cblas_dtrsma(CBLAS_ENUM CBLAS_ORDER Order, CBLAS_ENUM CBLAS_SIDE Side, CBLAS_ENUM CBLAS_UPLO Uplo, CBLAS_ENUM CBLAS_TRANSPOSE TransA, CBLAS_ENUM CBLAS_DIAG Diag, blasint M, blasint N, double alpha, double *A, blasint lda, double *B, blasint ldb);
void cblas_dgemva(CBLAS_ENUM CBLAS_ORDER order,  CBLAS_ENUM CBLAS_TRANSPOSE trans,  blasint m, blasint n, double alpha, double  *a, blasint lda,  double  *x, blasint incx,  double beta,  double  *y, blasint incy);
void cblas_dgemma(CBLAS_ENUM CBLAS_ORDER Order, CBLAS_ENUM CBLAS_TRANSPOSE TransA, CBLAS_ENUM CBLAS_TRANSPOSE TransB, blasint M, blasint N, blasint K, double alpha, double *A, blasint lda, double *B, blasint ldb, double beta, double *C, blasint ldc);
void cblas_daxpya(blasint n, double, double *x, blasint incx, double *y, blasint incy);
void cblas_dscala(blasint N, double alpha, double *X, blasint incX);
}
#else
static inline void goto_set_num_threads(int) {}
static inline void caldgemm_goto_restrict_cpus(int) {}
#endif

#endif

#ifndef _WIN32
#define CAST_FOR_MMPREFETCH
#else
#define CAST_FOR_MMPREFETCH (char*)
#endif

#ifdef VTRACE
#include <vt_user.h>
#include <pthread.h>
extern pthread_mutex_t global_vt_mutex;
#define VT_USER_START_A(a) {pthread_mutex_lock(&global_vt_mutex);VT_USER_START(a);pthread_mutex_unlock(&global_vt_mutex);}
#define VT_USER_END_A(a) {pthread_mutex_lock(&global_vt_mutex);VT_USER_END(a);pthread_mutex_unlock(&global_vt_mutex);}
#else
#define VT_USER_START_A(a)
#define VT_USER_END_A(a)
#endif

#define mcat(a, b) a ## b
#define mxcat(a, b) mcat(a, b)

#define str(s) xstr(s)
#define xstr(s) #s

#define PASS_ARG(arg) arg
#define COMMA ,
#define EMPTY

#define RED "\033[22;31m"
#define BOLDRED "\033[1m\033[31m"
#define BOLDBLACK "\033[1m\033[30m"
#define RESET "\033[0m"

#define COMPARE_GENERAL(a, b) ((a) != (b))
#define COMPARE_STRING(a, b) (strcmp(a, b))

#define PRINT_CONFIG_BASE(name1, type, type2, name2_old, name2_new, name2_conf, compare) \
	{ \
		if (oldConfig) \
		{ \
			if (compare((name2_old), (name2_new))) \
				fprintf(STD_OUT, "%35s: " type " changed to " BOLDRED type RESET "\n", name1, (type2) (name2_old), (type2) (name2_new)); \
		} \
		else \
		{ \
		     fprintf(STD_OUT, "%35s: " type "\n", name1, (type2) name2_conf); \
		} \
	}
	
#define PRINT_CONFIG_BASE_WRAP(name1, name2, name1param, type, type2, conf, hide1, hide2, hide1val, hide2val) \
	{ \
		char tmpBuffer[256]; \
		sprintf(tmpBuffer, str(name1) name1param); \
		PRINT_CONFIG_BASE(tmpBuffer, type, type2, (hide1) ? (hide1val) : oldConfig->name2, (hide2) ? (hide2val) : newConfig->name2, conf->name2, COMPARE_GENERAL) \
	}


#define PRINT_CONFIG_BASE_THIS(name1, name2, name1param, type, type2, conf) \
	{ \
		char tmpBuffer[256]; \
		sprintf(tmpBuffer, str(name1) name1param); \
		if (oldConfig == NULL) fprintf(STD_OUT, "%35s: " type "\n", tmpBuffer, (type2) conf->name2); \
	}

#define PRINT_CONFIG_INT(name) PRINT_CONFIG_BASE_WRAP(name, name, EMPTY, "%5d", int, myConfig, 0, 0, 0, 0)
#define PRINT_CONFIG_CHAR(name) PRINT_CONFIG_BASE_WRAP(name, name, EMPTY, "%5c", char, myConfig, 0, 0, 0, 0)
#define PRINT_CONFIG_DOUBLE(name) PRINT_CONFIG_BASE_WRAP(name, name, EMPTY, "%2.3f", double, myConfig, 0, 0, 0, 0)
#define PRINT_CONFIG_STRING(name) \
	{ \
		const char* strEmpty = ""; \
		const char* str1 = (myConfig->name ? myConfig->name : strEmpty); \
		const char* str2 = (oldConfig && oldConfig->name ? oldConfig->name : strEmpty); \
		const char* str3 = (newConfig && newConfig->name ? newConfig->name : strEmpty); \
		PRINT_CONFIG_BASE(str(name), "%5s", char*, str2, str3, str1, COMPARE_STRING) \
	}

#define PRINT_CONFIG_INT_THIS(name) PRINT_CONFIG_BASE_THIS(name, name, EMPTY, "%5d", int, this)

#define PRINT_CONFIG_LOOP_INT(name, loopvar) \
	{ \
		for (int i = 0;i < mymax(oldConfig ? oldConfig->loopvar : 0, newConfig->loopvar);i++) \
		{ \
			PRINT_CONFIG_BASE_WRAP(name[%d], name[i], PASS_ARG(COMMA) i, "%5d", int, myConfig, oldConfig && oldConfig->loopvar <= i, newConfig->loopvar <= i, -1, -1) \
		} \
	}
	
#define CALDGEMM_PREPARE_BACKEND_VARS1 \
	size_t blockm, blockn; \
	DGEMM_getblocks(k, blockm, blockn); \
	const size_t HeightM = ((blockm == gpu_m / Config->Height) ? (gpu_m % Config->Height) : Config->Height); \
	const size_t HeightN = ((blockn == gpu_n / Config->Height) ? (gpu_n % Config->Height) : Config->Height);

#define CALDGEMM_PREPARE_BACKEND_VARS2 \
	char myMat                = iMat ? 'B'                           : 'A'; \
	int& my_next_buffer       = iMat ? next_buffer_B[num_device]     : next_buffer_A[num_device]; \
	int*& my_buffer_pointers  = iMat ? buffer_pointers_B[num_device] : buffer_pointers_A[num_device]; \
	size_t& myblock           = iMat ? blockn : blockm; \
	bool& myTranspose         = iMat ? TransposeB : TransposeA; \
	const bool myKernelTranspose = iMat ? KernelSettings.transposeB : KernelSettings.transposeA; \
	const size_t& myHeight    = iMat ? HeightN : HeightM; \
	const size_t pitch        = iMat ? B_pitch : A_pitch; \
	double* src_ptr           = iMat ? \
		(B + blockn * Config->Height * (myTranspose ? B_pitch : 1)) : \
		(A + blockm * Config->Height * (myTranspose ? 1 : A_pitch)); \
	const bool access_bbuffers = (bool) (!DGEMM_favor_m && buffersSufficiant0) ^ (bool) iMat; \
	const int destbuffer       = access_bbuffers ? \
		((!iMat || buffersSufficiant) ? (my_buffer_pointers[myblock] % ((iMat || buffersSufficiant) ? bbuffers[num_device] : ibuffercount)) : (my_next_buffer % ibuffercount)) : \
		my_next_buffer % ibuffercount; \
	if (iMat) Timers.divideB++; else Timers.divideA++;

#define PREALLOC_ALTERNATE_LOOKAHEAD 4

#endif
