#ifndef AFFINITY_H
#define AFFINITY_H

#ifdef _WIN32
typedef HANDLE pid_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

void setThreadName(char* name);
void printThreadPinning();
pid_t gettid();

#ifdef __cplusplus
}
#endif

#endif
