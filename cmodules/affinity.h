#ifndef AFFINITY_H
#define AFFINITY_H

#ifdef _WIN32
typedef HANDLE pid_t;
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

pid_t gettid();
#ifdef _WIN32
pid_t getpid();
#endif

#ifdef __cplusplus
}
#endif

#endif
