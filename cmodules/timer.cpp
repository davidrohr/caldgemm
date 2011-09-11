#include "timer.h"
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <time.h>
#endif

HighResTimer::HighResTimer()
{
	ElapsedTime = 0;
}

HighResTimer::~HighResTimer() {}

void HighResTimer::Start()
{
#ifdef _WIN32
	__int64 istart;
	QueryPerformanceCounter((LARGE_INTEGER*)&istart);
	StartTime = (double) istart;
#else
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	StartTime = (double) tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
}

void HighResTimer::Stop()
{
	double EndTime = 0;
#ifdef _WIN32
	__int64 iend;
	QueryPerformanceCounter((LARGE_INTEGER*) &iend);
	EndTime = (double) iend;
#else
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	EndTime = (double) tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
	ElapsedTime += EndTime - StartTime;
}

void HighResTimer::Reset()
{
	ElapsedTime = 0;
	StartTime = 0;
}

double HighResTimer::GetElapsedTime()
{
	return ElapsedTime / Frequency;
}

double HighResTimer::GetFrequency()
{
#ifdef _WIN32
	__int64 ifreq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&ifreq);
	return((double) ifreq);
#else
	return(1.0E9);
#endif
}

double HighResTimer::Frequency = HighResTimer::GetFrequency();