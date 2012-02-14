#include <signal.h>
#if !defined(_WIN32) & defined(USE_GOTO_BLAS)
extern "C" {
#include <common.h>
}
#else

#ifndef USE_GOTO_BLAS
#ifdef USE_MKL_NOT_ACML

#else
#include <omp.h>
#endif
#endif

#ifndef _WIN32
extern "C" int get_num_procs();
static inline void caldgemm_goto_restrict_cpus(int) {}
static inline void caldgemm_goto_reserve_cpu(int, int) {}
static inline void caldgemm_goto_reserve_cpus(int) {}
static inline void goto_set_num_threads(int num) {omp_set_num_threads(num);}
#else
extern "C" int get_num_procs();
static inline void caldgemm_goto_reserve_cpu(int, int) {}
static inline void caldgemm_goto_reserve_cpus(int) {}
static inline void caldgemm_goto_restrict_cpus(int) {}
static inline void goto_set_num_threads(int) {}
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
