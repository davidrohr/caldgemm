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

class caldgemm : public calutil
{
    friend void* merge_wrapper(void* arg);
    friend void* cblas_wrapper(void* arg);

    public:
    
    //CALDGEMM interface functions
    //Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
    //Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = CAL_TRUE, UseGPU = UseCPU = CAL_TRUE, GPURatio = 0.66
    //m and n can be defined in the RunCALDGEMM call
    //The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
    int InitCALDGEMM(SampleInfo* pInfo);
    int ExitCALDGEMM();
    int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = -1, size_t k = -1, size_t n = -1, size_t Apitch = -1, size_t Bpitch = -1, size_t Cpitch = -1, CBLAS_ORDER order = CblasRowMajor, CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans);
    double* AllocMemory(size_t nDoubles, bool page_locked, bool huge_pages);
    void FreeMemory(double* ptr);
    void ResetTimers();

    private:
    
    int divideBuffer(Data* dst, CALdouble* src, CALint width, CALint height, CALint gpu_width, CALint gpu_height, CALint pitch, CALint numBuffers, bool transpose);
    int mergeBuffers(CALdouble* dst, Data* src, CALint width, CALint height, CALint gpu_width, CALint gpu_height, CALint pitch, CALint numBuffers);
    
    int DGEMM_prepare(size_t k, int j);
    inline void DGEMM_getblocks(size_t k, size_t &blockm, size_t &blockn);
    void checkCalPatch();

    struct mergeParameters
    {
	caldgemm* cls;
	CALdouble* dst;
	Data* src;
	int nContext;
	CALboolean terminate;
	pthread_mutex_t mergeMutex[2];
    };
    
    mergeParameters mParam[ctxcount];
        
    cpu_set_t oldcpumask;
    cpu_set_t gpumask;
    
    char hostname[256];	//Store hostname of node for host dependant debug code
    
    size_t gpu_m, gpu_n;
    
#if (defined(CALDGEMM_TRANSPOSED_A) | defined(CALDGEMM_TRANSPOSED_B)) & !(defined(CALDGEMM_TRANSPOSED_A) & defined(CALDGEMM_TRANSPOSED_B))
    static const bool buffersSwitchable = true;
#else
    static const bool buffersSwitchable = false;
#endif
};
