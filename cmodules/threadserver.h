#ifndef THREADSERVER_H
#define THREADSERVER_H

#ifdef _WIN32
#include "pthread_mutex_win32_wrapper.h"
#include "sched_affinity_win32_wrapper.h"
#else
#include <pthread.h>
#include <sched.h>
#endif

class qThreadServerException
{

};

template <class S, class T> class qThreadCls;

class qThreadParam
{
	template <class S, class T> friend class qThreadCls;

public:
	qThreadParam()
	{
		for (int i = 0;i < 2;i++) if (pthread_mutex_init(&threadMutex[i], NULL)) {fprintf(STD_OUT, "Error creating mutex");throw(qThreadServerException());}
		for (int i = 0;i < 2;i++) if (pthread_mutex_lock(&threadMutex[i])) {fprintf(STD_OUT, "Error locking mutex");throw(qThreadServerException());}
		terminate = false;
		pinCPU = -1;
	}

	~qThreadParam()
	{
		for (int i = 0;i < 2;i++) if (pthread_mutex_destroy(&threadMutex[i])) {fprintf(STD_OUT, "Error destroying mutex");throw(qThreadServerException());}
	}

	bool WaitForTask()
	{
		if (pthread_mutex_unlock(&threadMutex[1])) {fprintf(STD_OUT, "Error unlocking mutex");throw(qThreadServerException());}
		if (pthread_mutex_lock(&threadMutex[0])) {fprintf(STD_OUT, "Error locking mutex");throw(qThreadServerException());}
		return(!terminate);
	}

	int threadNum;

protected:
	int pinCPU;
	pthread_mutex_t threadMutex[2];
	bool terminate;
};

template <class S> class qThreadParamCls : qThreadParam
{
	S* pCls;
	void (S::*pFunc)(void*);
};

template <class S, class T> static void* qThreadWrapperCls(void* arg);

template <class S, class T> class qThreadCls
{
public:
	qThreadCls() {started = false;};
	qThreadCls(S* pCls, void (S::*pFunc)(T*), int threadNum = 0, int pinCPU = -1) : threadParam() {started = false;Start(pCls, pFunc, threadNum, pinCPU);}

	void Start(S* pCls, void (S::*pFunc)(T*), int threadNum = 0, int pinCPU = -1)
	{
		threadParam.pCls = pCls;
		threadParam.pFunc = pFunc;
		threadParam.threadNum = threadNum;
		threadParam.pinCPU = pinCPU;
		pthread_t thr;
		pthread_create(&thr, NULL, (void(*)(void*)) &qThreadWrapperCls<S, T>, &threadParam);
		if (pthread_mutex_lock(&threadMutex[1])) {fprintf(STD_OUT, "Error locking mutex");throw(qThreadServerException());}
	}

	~qThreadCls()
	{
		if (started)
		{
			End();
		}
	}

	void End()
	{
		threadParam.terminate = true;
		if (pthread_mutex_unlock(&threadParam.threadMutex[0])) {fprintf(STD_OUT, "Error unlocking mutex");throw(qThreadServerException());}
		if (pthread_mutex_lock(&threadParam.threadMutex[1])) {fprintf(STD_OUT, "Error locking mutex");throw(qThreadServerException());}
	}

private:
	bool started;
	T threadParam;
};

template <class S, class T> static void* qThreadWrapperCls(T* arg)
{
	if (arg->pinCPU != -1)
	{
		cpu_set_t tmp_mask;
		CPU_ZERO(&tmp_mask);
		CPU_SET(0, &arg->pinCPU);
		sched_setaffinity(0, sizeof(tmp_mask), &tmp_mask);
	}

	arg->pCls->(*arg->pFunc)(arg);

	if (pthread_mutex_unlock(&threadMutex[1])) {fprintf(STD_OUT, "Error unlocking mutex");throw(qThreadServerException());}
	pthread_exit(NULL);
	return(NULL);
}

template <class S, class T> class qThreadClsArray
{
public:
	qThreadClsArray() {pArray = NULL;nThreadsRunning = 0;}
	qThreadClsArray(int n, S* pCls, void (S::*pFunc)(T*), int threadNumOffset = 0, int* pinCPU = NULL) {pArray = NULL;nThreadsRunning = 0;SetNumberOfThreads(n);}

	void SetNumberOfThreads(int n, S* pCls, void (S::*pFunc)(T*), int threadNumOffset = 0, int* pinCPU = NULL)
	{
		if (nThreadsRunning)
		{
			fprintf(STD_OUT, "Threads already started\n");throw(qThreadServerException());
		}
		pArray = new qThreadCls<S, T>[n];
		nThreadsRunning = n;
		for (int i = 0;i < n;i++)
		{
			pArray[i].Start(pCls, pFunc, threadNumOffset + i, pinCPU == NULL ? -1 : pinCPU[i]);
		}
	}

	~qThreadClsArray()
	{
		if (nThreadsRunning)
		{
			StopThreads();
		}
	}

	void StopThreads()
	{
		delete[] pArray;
		nThreadsRunning = NULL;
	}

private:
	qThreadCls<S, T>* pArray;
	int nThreadsRunning;
};

#endif
