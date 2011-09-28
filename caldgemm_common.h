#include <signal.h>
typedef int blasint;
extern "C" {
#include <cblas.h>
}
#ifndef _WIN32
extern "C" {
#include <common.h>
}
#else
static inline int get_num_procs() {return(1);}
static inline void caldgemm_goto_reserve_cpu(int, int) {}
static inline void caldgemm_goto_reserve_cpus(int) {}
static inline void caldgemm_goto_restrict_cpus(int) {}
static inline void goto_set_num_threads(int) {}
#endif
