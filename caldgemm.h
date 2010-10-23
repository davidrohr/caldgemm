/* ============================================================

The source code is property of the Frankfurt Institute for Advanced Studies (FIAS).
None of the material may be copied, reproduced, distributed, republished, downloaded,
displayed, posted or transmitted in any form or by any means, including, but not
limited to, electronic, mechanical, photocopying, recording, or otherwise,
without the prior written permission of FIAS.

Authors:
David Rohr (drohr@jwdt.org)
Matthias Bach (bach@compeng.uni-frankfurt.de)
Matthias Kretz (kretz@compeng.uni-frankfurt.de)

============================================================ */

#include "caldgemm_config_load.h"

#include <cal.h>
#include <cal_ext.h>
#include <calcl.h>
#include "calutil.h"

#include <emmintrin.h>
#include <mm3dnow.h>
#include <pthread.h>
#include <signal.h>
typedef int blasint;
extern "C" {
#include <cblas.h>
}

void* merge_wrapper(void* arg);
void* cblas_wrapper(void* arg);
void* linpack_wrapper(void* arg);

class caldgemm : public calutil
{
	friend void* merge_wrapper(void* arg);
	friend void* cblas_wrapper(void* arg);
	friend void* linpack_wrapper(void* arg);

public:

	caldgemm();
	~caldgemm();

	//CALDGEMM interface functions
	//Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
	//Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = CAL_TRUE, UseGPU = UseCPU = CAL_TRUE, GPURatio = 0.66
	//m and n can be defined in the RunCALDGEMM call
	//The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
	int InitCALDGEMM(SampleInfo* pInfo);
	int ExitCALDGEMM();
	int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = -1, size_t k = -1, size_t n = -1, size_t Apitch = -1, size_t Bpitch = -1, size_t Cpitch = -1, CBLAS_ORDER order = CblasRowMajor, CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans, int ExecuteLinpackCallbacks = 0);
	double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages);
	void FreeMemory(double* ptr);
	void ResetTimers();

private:

	int divideBuffer(Data* dst, CALdouble* src, CALint width, CALint height, CALint gpu_width, CALint gpu_height, CALint pitch, CALint numBuffers, bool transpose);
	int mergeBuffers(CALdouble* dst, Data* src, CALint width, CALint height, CALint gpu_width, CALint gpu_height, CALint pitch, CALint numBuffers);

	int DGEMM_prepare(size_t k, int j);
	inline void DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn);
	void checkCalPatch();
	void cal_init_constant_data(Data* &data, double alpha);
	virtual void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2 = NULL);
	int cpuScheduler();
	int getcpumask(cpu_set_t* set);

	struct mergeParameters
	{
		caldgemm* cls;
		CALdouble* dst;
		Data* src;
		int nMergeThread;
		int nContext;
		CALboolean terminate;
		pthread_mutex_t mergeThreadMutex[2];
	};

	pthread_mutex_t obufferMutex[ctxcount];

	struct structLinpackParameters
	{
		pthread_mutex_t linpackMutex[2];
		CALboolean terminate;
	} linpackParameters;

	mergeParameters mParam[max_outputthreads];

	cpu_set_t oldcpumask;
	cpu_set_t gpumask;

	size_t gpu_m, gpu_n;

	bool caldgemm_initialized;

	pthread_mutex_t scheduleMutex;
	volatile long long int gpu_k_barrier, cpu_k_barrier;

	static const int max_linpack_callback_types = 3;

	double linpack_last_mn[max_linpack_callback_types];
	double linpackGPURatios[max_linpack_callback_types];

#if (defined(CALDGEMM_TRANSPOSED_A) | defined(CALDGEMM_TRANSPOSED_B)) & !(defined(CALDGEMM_TRANSPOSED_A) & defined(CALDGEMM_TRANSPOSED_B))
	static const bool buffersSwitchable = true;
#else
	static const bool buffersSwitchable = false;
#endif
};
