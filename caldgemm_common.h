#include <signal.h>
#if !defined(_WIN32) & defined(USE_GOTO_BLAS)
extern "C" {
#include <common.h>
}
#else

#ifndef USE_GOTO_BLAS
#include <omp.h>
#endif

extern "C" int get_num_procs();
static inline void caldgemm_goto_reserve_cpu(int, int) {}
static inline void caldgemm_goto_reserve_cpus(int) {}

#ifndef _WIN32
static inline void goto_set_num_threads(int num) {omp_set_num_threads(num);}
void caldgemm_goto_restrict_cpus(int);

enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114};
enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142};

void cblas_dtrsma(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side, enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag, blasint M, blasint N, double alpha, double *A, blasint lda, double *B, blasint ldb);
void cblas_dgemva(enum CBLAS_ORDER order,  enum CBLAS_TRANSPOSE trans,  blasint m, blasint n, double alpha, double  *a, blasint lda,  double  *x, blasint incx,  double beta,  double  *y, blasint incy);
void cblas_dgemma(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, blasint M, blasint N, blasint K, double alpha, double *A, blasint lda, double *B, blasint ldb, double beta, double *C, blasint ldc);
void cblas_daxpya(blasint n, double, double *x, blasint incx, double *y, blasint incy);
void cblas_dscala(blasint N, double alpha, double *X, blasint incX);
#else
static inline void goto_set_num_threads(int) {}
static inline void caldgemm_goto_restrict_cpus(int) {}
#endif

#endif

#ifdef _WIN32
#define __INTRIN_H_
#define _Complex
#ifndef __restrict__
#define __restrict__
#endif
#endif

typedef int blasint;
extern "C" {
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
}

#ifndef _WIN32
#define CAST_FOR_MMPREFETCH
#else
#define CAST_FOR_MMPREFETCH (char*)
#endif
