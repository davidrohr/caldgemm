#ifndef OS_LOW_LEVEL_HELPER_H
#define OS_LOW_LEVEL_HELPER_H

#ifndef _WIN32
#include <syscall.h>
#endif

inline int get_number_of_cpu_cores()
{
#ifdef _WIN32
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return(info.dwNumberOfProcessors);
#else
	return(sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

#endif
