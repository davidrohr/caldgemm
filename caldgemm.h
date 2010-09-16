/* ============================================================

The source code is property of the Frankfurt Institute for Advanced Studies (FIAS).
None of the material may be copied, reproduced, distributed, republished, downloaded,
displayed, posted or transmitted in any form or by any means, including, but not
limited to, electronic, mechanical, photocopying, recording, or otherwise,
without the prior written permission of FIAS.

Authors:
David Rohr (drohr@jwdt.org)
Mathias Bach (bach@compeng.uni-frankfurt.de)
Mathias Kretz (kretz@compeng.uni-frankfurt.de)

============================================================ */

#include "caldgemm_config.h"

#ifdef CALDGEMM_88
#define CALDGEMM_44
#define CALDGEMM_USE_MEMEXPORT
#define CALDGEMM_TRANSPOSED_A
#ifdef CALDGEMM_TRANSPOSED_B
#undef CALDGEMM_TRANSPOSED_B
#endif
#endif

#ifdef CALDGEMM_44
#ifdef CALDGEMM_TRANSPOSED_B
#ifdef CALDGEMM_TRANSPOSED_A
#undef CALDGEMM_TRANSPOSED_A
#endif
#else
#define CALDGEMM_TRANSPOSED_A
#endif
#endif

#if defined(CALDGEMM_DIAGONAL_TEXTURE) & (!defined(CALDGEMM_44) | defined(CALDGEMM_88) | !defined(CALDGEMM_TRANSPOSED_A))
#undef CALDGEMM_DIAGONAL_TEXTURE
#endif

#if defined(CALDGEMM_88) | !defined(CALDGEMM_44)
#define TILING_Y 8
#else
#define TILING_Y 4
#endif

#if defined(CALDGEMM_88)
#define TILING_X 8
#elif defined(CALDGEMM_44)
#define TILING_X 4
#else
#define TILING_X 2
#endif

#include "cal.h"
#include "cal_ext.h"
#include "calcl.h"

#include <emmintrin.h>
#include <mm3dnow.h>
#include <pthread.h>
#include <signal.h>
typedef int blasint;
extern "C" {
#include <cblas.h>
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

#ifdef ATI_OS_VISTA //define ATI_OS_WIN if ATI_OS_VISTA is defined.
	#ifndef ATI_OS_WIN
		#define ATI_OS_WIN
	#endif //ATI_OS_WIN
#endif //ATI_OS_VISTA

#ifdef ATI_OS_WIN
typedef __int64 i64 ;
#endif
#ifdef ATI_OS_LINUX
typedef long long i64;
#endif

class CPerfCounter {

public:
    CPerfCounter();
    ~CPerfCounter();
    void Start(void);
    void Stop(void);
    void Reset(void);
    double GetElapsedTime(void);

private:

    double _freq;
    double _clocks;
    double _start;
};

void* merge_wrapper(void* arg);
void* cblas_wrapper(void* arg);

class caldgemm
{
    friend void* merge_wrapper(void* arg);
    friend void* cblas_wrapper(void* arg);

    public:
    class SampleInfo		//Run Parameters
    {
	public:
	SampleInfo();
    
	CALint     Pin;
	CALboolean Verify;
	CALboolean Disassemble;
	CALboolean PrintILKernel;
	CALboolean Quiet;
	CALuint    DeviceNum;
	size_t     Width;		//k for matrix multiply
	size_t     Height;		//height of subblock od A, width of subblock of B
	CALboolean AutoHeight;		//Automatically adjust height
	CALuint    Iterations;
	CALchar	   DstMemory;
	CALboolean VerboseTiming;
	CALboolean TabularTiming;
	CALboolean Debug;
	CALboolean MultiThread;
	CALboolean UseGPU;
	CALboolean UseCPU;
	CALdouble  GPURatio;
	CALboolean DynamicSched;
	CALboolean MemPolicy;
	CALboolean DumpMatrix;
	size_t     m, n;		//height of A, width of B, must be multiple of height
    };
    
    //CALDGEMM interface functions
    //Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
    //Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = CAL_TRUE, UseGPU = UseCPU = CAL_TRUE, GPURatio = 0.66
    //m and n can be defined in the RunCALDGEMM call
    //The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
    int InitCALDGEMM(SampleInfo* pInfo);
    int ExitCALDGEMM();
    int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m = -1, size_t = -1, size_t n = -1, size_t Apitch = -1, size_t Bpitch = -1, size_t Cpitch = -1, CBLAS_ORDER order = CblasRowMajor, CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans);
    void ResetTimers();

    private:
    
    struct TimerInfo
    {
	CPerfCounter System, Kernel, CounterDivide, CounterMerge, CounterCopyTo, CounterCopyFrom, CPUTimer, GPUTimer;
    } Timers;

    typedef struct DataRec
    {
	union
	{
    	    CALfloat*  f_data;
	    CALuint*   u_data;
    	    CALint*    i_data;
	    CALdouble* d_data;
    	    CALchar*   c_data;
    	    CALvoid*   v_data;
	};
	CALuint Width;	//width of matrix
	CALuint Height;	//height of matrix
	CALuint ComponentSize;	//number of components in vector
	CALuint DataSize;	//size of data element (e.g. sizeof(CALfloat)
    
	CALboolean CALMemory;
	CALresource res;
	CALmem mem;
        CALmem dstMem;
        CALuint pitch;
    } Data;

    typedef struct SampleFeaturesRec
    {
	SampleFeaturesRec()
	{
		DoublePrecision = CAL_FALSE; 
		ComputeShaders = CAL_FALSE;
		LocalDataShares = CAL_FALSE;
		MemExport = CAL_FALSE;
		GlobalDataShares = CAL_FALSE;
	}

	CALboolean DoublePrecision;
	CALboolean ComputeShaders;
	CALboolean LocalDataShares;
	CALboolean MemExport;
	CALboolean GlobalDataShares;
    } SampleFeatures;

    typedef struct DeviceInfoRec DeviceInfo;

    typedef struct float4Rec { CALfloat x, y, z, w; } float4;
    typedef struct float2Rec { CALfloat x, y; } float2;
    typedef struct CALVersionRec {
	CALuint major;
	CALuint minor;
	CALuint imp;
    } CALVersion;

    CALint Initialize (CALdevice *device, CALcontext *ctx, CALuint deviceNum);
    CALint SetupKernel(const CALchar* ILKernel, CALmodule* module, CALcontext* ctx, CALboolean disassemble = CAL_FALSE);
    CALint RunProgram(CALcontext* ctx, CALmodule* module, CALuint Width, CALuint Height, CALevent* event);
    CALint CleanupData(CALcontext* ctx, CALresource* &resourceHandler, Data* &data, CALuint numHandles);
    CALint Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, Data* &data, CALuint numHandles);
    CALformat getFormat(CALuint formatSize, CALuint dataSize, CALboolean isInt = CAL_FALSE);
    CALuint AnalyzeResults(Data* data);
    CALint SetupData(CALmodule* module, CALresource* &_Res, Data* &data, CALdevice* device, CALcontext* ctx, CALuint numInputs, CALuint numOutputs, CALuint numConstantBuffers);
    CALint CopyDataFromGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALevent* event);
    CALint CopyDataToGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALboolean constants, CALevent* event);
    CALint BindIONames(CALcontext* ctx, CALmodule* module, CALuint iStop, CALuint cStop, CALuint oStop, Data* data);
    CALint AllocateResources(CALcontext* ctx, CALdevice* device, CALresource* &_Res, CALuint iStop, CALuint cStop, CALuint oStop, Data* data);
    int AllocateMemory(Data& data, CALdevice *device, CALcontext *ctx, CALuint tWidth, CALuint tHeight, CALuint CompSize, CALuint DataSize, CALresallocflags flags, CALuint i);
    CALint ParameterValidation(CALuint nInput, CALuint nOutput, CALdeviceattribs* attribs);
    CALvoid SupportedCALVersion(CALVersion *calVersion);
    CALint QueryDeviceCaps(CALuint DeviceNum, SampleFeatures *FeatureList);
    CALint QueryCALVersion(CALVersion required, const CALchar *comparison);
    CALint ValidateCALRuntime();
    CALvoid displayMatrixTiming(const CALchar* name);
    void copyFrom(CALchar* ptr, Data& data, CALuint pitch);
    void copyTo(CALchar* ptr, Data& data, CALuint pitch);
    void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey);
    int DumpMatrix(double* A, double* B, double* C, double alpha, double beta, int m, int k, int n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB);

    CALdouble* A;
    CALdouble* B;
    CALdouble* C;
    
    CALdouble Alpha, Beta;
    
    size_t A_pitch, B_pitch, C_pitch;
    CBLAS_TRANSPOSE TransposeA;
    CBLAS_TRANSPOSE TransposeB;

#ifdef CALDGEMM_44
#if defined(CALDGEMM_TRANSPOSED_A) & !defined(CALDGEMM_88)
    static const CALuint aPartsNum = 2;
    static const CALuint bPartsNum = 2;
#else
    static const CALuint aPartsNum = 4;
    static const CALuint bPartsNum = 4;
#endif
#else //CALDGEMM_44
#ifdef CALDGEMM_TRANSPOSED_A
    static const CALuint aPartsNum = 2;
#else
    static const CALuint aPartsNum = 8;
#endif
    static const CALuint bPartsNum = 2;
#endif //CALDGEMM_44

#ifdef CALDGEMM_USE_MEMEXPORT
    static const CALuint cPartsNum = 1;
#else
    static const CALuint cPartsNum = 8;
#endif
    static const int ctxcount = 3;
    static const int vcpysize = 16;
    static const int kernel_count = 2;
    
    SampleInfo* Info;
    SampleFeatures Features;

    CALvoid divideBuffer(Data* dst, CALdouble* src, CALint width, CALint height, CALint pitch, CALint numBuffers, bool transpose);
    bool isDoubleEqual(CALdouble a, CALdouble b);
    int mergeBuffers(CALdouble* dst, Data* src, CALint width, CALint height, CALint pitch, CALint numBuffers);
    
    struct mergeParameters
    {
	caldgemm* cls;
	CALdouble* dst;
	Data* src;
	int nContext;
	CALboolean terminate;
	pthread_mutex_t mergeMutex[2];
    };
    
    struct cblasParameters
    {
	caldgemm* cls;
	size_t cblas_size;
	size_t dynamic_run;	//Do an extra dynic cblas run?, works also as m for the dynamic run, set negative value to omit doing the border run
	size_t dynamic_size;	//n for dynamic run
	CALboolean borders_done;
	CALboolean terminate;
	pthread_mutex_t cblasMutex[2];
    };
    
    mergeParameters mParam[ctxcount];
    cblasParameters cParam;
        
    Data* datas[ctxcount];
    CALuint numInputs, numOutputs, numConstantBuffers;
    CALdevice device;
    CALcontext ctxs[ctxcount];
    CALresource* resourceHandlers[ctxcount];
    CALmodule modules[ctxcount][kernel_count];
    CALevent events[ctxcount];

    static const char *ILKernel, *ILKernelALPHA1;

    cpu_set_t oldcpumask;
    cpu_set_t gpumask;
    
    //For Verfify only
    CALdouble* D;

    //For Timing only
    bool CPUOnlyRun;
};
