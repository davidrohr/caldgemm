#ifndef AFFINITY_H
#define AFFINITY_H

#ifdef _WIN32
typedef HANDLE pid_t;
#include "sched_affinity_win32_wrapper.h"
#else
#include <sched.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

void setThreadName(const char* name);
#ifdef __cplusplus
const char* getThreadName(int tid = -1, const char* defaultval = "Unknown Thread");
#else
const char* getThreadName(int tid, const char* defaultval);
#endif
void printThreadPinning();
void setUnknownNames(char* name);
void setUnknownAffinity(int count, int* cores);

inline int sched_setaffinity_set_core(int core)
{
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(core, &set);
	return sched_setaffinity(0, sizeof(set), &set);
}

pid_t gettid();
#ifdef _WIN32
pid_t getpid();
#endif

#ifdef __cplusplus
}
#endif

#endif
