/* ============================================================

Copyright (c) 2007 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERATION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
 commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be
modified or waived, and no breach of this license can be excused, unless done
so in a writing signed by all affected parties. Each term of this license is
separately enforceable. If any term of this license is determined to be or
becomes unenforceable or illegal, such term shall be reformed to the minimum
extent necessary in order for this license to remain in effect in accordance
with its terms as modified by such reformation. This license shall be governed
by and construed in accordance with the laws of the State of Texas without
regard to rules on conflicts of law of any state or jurisdiction or the United
Nations Convention on the International Sale of Goods. All disputes arising out
of this license shall be subject to the jurisdiction of the federal and state
courts in Austin, Texas, and all defenses are hereby waived concerning personal
jurisdiction and venue of these courts.

============================================================ */
/////////////////////////////////////////////////////////////////////////////
//
//  CAL include headers :
//      cal.h contains declarations for CAL runtime libarary functions
//      calcl.h contains declarations for CAL compiler libarary functions
//

//#define CALDGEMM_TRANSPOSED_A
//#define CALDGEMM_88
//#define CALDGEMM_44
//#define CALDGEMM_TRANSPOSED_B
//#define CALDGEMM_USE_MEMEXPORT
//#define TESTMODE

#ifdef CALDGEMM_88
#define CALDGEMM_44
#define CALDGEMM_USE_MEMEXPORT
#endif

#ifdef CALDGEMM_44
#ifdef CALDGEMM_TRANSPOSED_A
#ifdef CALDGEMM_TRANSPOSED_B
#undef CALDGEMM_TRANSPOSED_B
#endif
#else
#define CALDGEMM_TRANSPOSED_B
#endif
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
	CALuint    Width;		//k for matrix multiply
	CALuint    Height;		//height of subblock od A, width of subblock of B
	CALboolean AutoHeight;		//Automatically adjust height
	CALuint    Iterations;
	CALchar	   DstMemory;
	CALboolean VerboseTiming;
	CALboolean Debug;
	CALboolean MultiThread;
	CALboolean UseGPU;
	CALboolean UseCPU;
	CALdouble  GPURatio;
	CALboolean DynamicSched;
	CALboolean MemPolicy;
	CALboolean DumpMatrix;
	CALuint    m, n;		//height of A, width of B, must be multiple of height
    };
    
    //CALDGEMM interface functions
    //Initiate an appropriate sampleinfo struct and call InitCALDGEMM for initialization
    //Optimal parameters for big n,m are: DstMemory = 'c', Height = 2048, Width = 1024, MultiThread = CAL_TRUE, UseGPU = UseCPU = CAL_TRUE, GPURatio = 0.66
    //m and n can be defined in the RunCALDGEMM call
    //The Width (k in matrix multiply) is fixed and cannot be changed without reinitializing
    int InitCALDGEMM(SampleInfo* pInfo);
    int ExitCALDGEMM();
    int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, int m = -1, int k = -1, int n = -1, int Apitch = -1, int Bpitch = -1, int Cpitch = -1, CBLAS_ORDER order = CblasRowMajor, CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans);
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
    void print_submatrices(double* M, int width, int height, int pitch, int subx, int suby, int stridex, int stridey);
    int DumpMatrix(double* A, double* B, double* C, double alpha, double beta, int m, int k, int n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB);

    CALdouble* A;
    CALdouble* B;
    CALdouble* C;
    
    CALdouble Alpha, Beta;
    
    int A_pitch, B_pitch, C_pitch;
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
#else
#ifdef CALDGEMM_TRANSPOSED_A
    static const CALuint aPartsNum = 2;
#else
    static const CALuint aPartsNum = 8;
#endif
    static const CALuint bPartsNum = 2;
#endif

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
	CALint cblas_size;
	CALint dynamic_run;	//Do an extra dynic cblas run?, works also as m for the dynamic run, set negative value to omit doing the border run
	CALint dynamic_size;	//n for dynamic run
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
