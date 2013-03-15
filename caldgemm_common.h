#include <signal.h>

#ifdef _WIN32
#define __INTRIN_H_
#define _Complex
#ifndef __restrict__
#define __restrict__
#endif
#endif

#if !defined(_WIN32) & defined(USE_GOTO_BLAS)
extern "C" {
#define CBLAS
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
