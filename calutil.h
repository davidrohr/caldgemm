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
 
All modifications to the original source code are property of the Frankfurt Institute for Advanced Studies (FIAS).
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

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <cal.h>
#include <cal_ext.h>
#include <calcl.h>
typedef int blasint;
extern "C" {
#include <cblas.h>
}

#include <emmintrin.h>
#include <mm3dnow.h>
#include <pthread.h>

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

#ifdef ATI_OS_WIN
#include <windows.h>
#endif

#ifdef ATI_OS_LINUX
#include <sys/time.h>
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

class caldgemm;

class calutil
{
    public:
    class SampleInfo		//Run Parameters
    {
	public:
	SampleInfo();
    
	CALint     Pin;
	CALint     Pin_HackedLibUnavailable;
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
	CALboolean AsyncTiming;
	CALboolean TabularTiming;
	CALboolean Debug;
	CALboolean MultiThread;
	CALboolean UseGPU;
	CALboolean UseCPU;
	CALdouble  GPURatio;
	CALboolean DynamicSched;
	CALboolean MemPolicy;
	CALboolean DumpMatrix;
	CALboolean DivideToGPU;
	CALboolean AsyncDMA;
	CALboolean KeepBuffersMapped;
	CALboolean NoPerformanceWarnings;
	size_t     m, n;		//height of A, width of B, must be multiple of height
    };

protected:
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
    CALint CleanupData(CALcontext* ctx, CALresource* &resourceHandler, Data* &data, CALuint numHandles, int nContext);
    CALint Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, Data* &data, CALuint numHandles, int nContext);
    CALformat getFormat(CALuint formatSize, CALuint dataSize, CALboolean isInt = CAL_FALSE);
    CALuint AnalyzeResults(Data* data);
    CALint SetupData(CALmodule* module, CALresource* &_Res, Data* &data, CALdevice* device, CALcontext* ctx, CALuint numInputs, CALuint numOutputs, CALuint numConstantBuffers, CALname** ctxProgNames, int nContext);
    CALint CopyDataFromGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALevent* event);
    CALint CopyDataToGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALboolean constants, CALevent* event, Data* dest_data = NULL);
    CALint BindIONames(CALcontext* ctx, CALmodule* module, CALuint iStop, CALuint cStop, CALuint oStop, Data* data, CALname* ctxProgNames);
    CALint AllocateResources(CALcontext* ctx, CALdevice* device, CALresource* &_Res, CALuint iStop, CALuint cStop, CALuint oStop, Data* data, int nContext);
    int AllocateMemory(Data& data, CALdevice *device, CALcontext *ctx, CALuint tWidth, CALuint tHeight, CALuint CompSize, CALuint DataSize, CALresallocflags flags, CALuint i, int nContext);
    CALint QueryDeviceCaps(CALuint DeviceNum, SampleFeatures *FeatureList);
    CALvoid SupportedCALVersion(CALVersion *calVersion); 
    CALint QueryCALVersion(CALVersion required, const CALchar *comparison, bool silent = false);
    CALint ValidateCALRuntime();
    CALvoid displayMatrixTiming(const CALchar* name);
    void copyFrom(CALchar* ptr, Data& data, CALuint pitch);
    void copyTo(CALchar* ptr, Data& data, CALuint pitch);
    bool isDoubleEqual(CALdouble a, CALdouble b);
    
    struct TimerInfo
    {
	CPerfCounter System, Kernel, CounterDivide, CounterMerge, CounterCopyTo, CounterCopyFrom, CPUTimer, GPUTimer, ATime;
	int divideA, divideB;
    } Timers;

    void print_submatrices(double* M, size_t width, size_t height, size_t pitch, size_t subx, size_t suby, size_t stridex, size_t stridey, double* M2 = NULL);
    int DumpMatrix(double* A, double* B, double* C, double alpha, double beta, int m, int k, int n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB);

    CALdouble* A;
    CALdouble* B;
    CALdouble* C;
    
    CALdouble Alpha, Beta;
    
    size_t A_pitch, B_pitch, C_pitch;
    CBLAS_TRANSPOSE TransposeA;
    CBLAS_TRANSPOSE TransposeB;

#ifdef CALDGEMM_44
#if !defined(CALDGEMM_48)
    static const CALuint aPartsNum = 2;
#else
    static const CALuint aPartsNum = 4;
#endif
#if !defined(CALDGEMM_84)
    static const CALuint bPartsNum = 2;
#else
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
    static const int ctxcount = 3;		//Not cal context count but number of copies of data buffers etc.
    static const int max_outputthreads = 3;
    static const int vcpysize = 16;
    static const int kernel_count = 2;
    static const int max_bbuffers = 20;
    int bbuffers;
    int outputthreads;
    
    size_t BufferHeight;			//Height to which the buffers were originally initialized
    size_t BufferWidth;				//Same for width
    
    SampleInfo* Info;
    SampleFeatures Features;

    Data* datas[max_bbuffers];
    CALuint numInputs, numOutputs, numConstantBuffers;
    CALdevice device;
    CALcontext ctx_main;
    CALresource* resourceHandlers[max_bbuffers];
    CALmodule modules[1][kernel_count];
    CALmodule fakeModule;
    CALname *progNames[1][kernel_count];
    CALevent events[ctxcount];

    static const char *ILKernel, *ILKernelALPHA1, *ILFakeKernel;

    struct cblasParameters
    {
        caldgemm* cls;
        size_t cblas_size;
        size_t dynamic_run;     //Do an extra dynamic cblas run?, works also as m for the dynamic run
	size_t dynamic_size;    //n for dynamic run
	size_t cpu_k;		//k that cpu will take over from gpu in 3rd phase dynamic run
	size_t dynamic_run2;
        CALboolean borders_done;
        CALboolean terminate;
        pthread_mutex_t cblasMutex[2];
    };

    cblasParameters cParam;
    //For Verfify only
    CALdouble* D;

    //For Timing only
    bool CPUOnlyRun;
    
    char hostname[256]; //Store hostname of node for host dependant debug code
};
