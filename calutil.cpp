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

#include "calutil.h"

#define CHKERR(cmd, text) if (cmd != CAL_RESULT_OK) {printf("Error '%s' while " text "\n", calGetErrorString());return(1);}
#define WAITFOREVENT(ctx, event) { CALresult r; do { r = calCtxIsEventDone(ctx, event); if (r == CAL_RESULT_ERROR) { printf("Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}

CALvoid calutil::displayMatrixTiming(const CALchar* name)
{
    if (!Info->Quiet)
    {
        CALdouble gflops_CPU = (CALdouble)1e-09 * Info->m * Info->n * (2 * Info->Width + 2) * (CALdouble) Info->Iterations / Timers.System.GetElapsedTime();
        printf("Program: %s Sizes - A: %lldx%lld B: %lldx%lld C:%lldx%lld System Time %2.3lf System Gflops %2.3lf\n", name, 
                Info->m, Info->Width, Info->Width, Info->n, Info->m, Info->n, Timers.System.GetElapsedTime(), gflops_CPU);
        if (Info->UseCPU == CAL_TRUE && Info->UseGPU == CAL_TRUE)
        {
    	    double flopsc, flopsg;
    	    if (CPUOnlyRun)
    	    {
    		flopsc = (double) 1e-09 * Info->m * Info->n * (2 * Info->Width + 2) * Info->Iterations / Timers.CPUTimer.GetElapsedTime();
    		flopsg = 0.0;
    	    }
    	    else if (Info->m >= Info->n)
    	    {
    		flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Info->n + (Info->n % Info->Height) * (Info->m - cParam.cblas_size)) * (2 * Info->Width + 2) * Info->Iterations / Timers.CPUTimer.GetElapsedTime();
    		flopsg = (double) 1e-09 * ((Info->m - cParam.cblas_size) * (Info->n - Info->n % Info->Height) - cParam.dynamic_run * cParam.dynamic_size) * (2 * Info->Width + 2) * Info->Iterations / Timers.GPUTimer.GetElapsedTime();
    	    }
    	    else
    	    {
    		flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Info->m + (Info->m % Info->Height) * (Info->n - cParam.cblas_size)) * (2 * Info->Width + 2) * Info->Iterations / Timers.CPUTimer.GetElapsedTime();
    		flopsg = (double) 1e-09 * ((Info->n - cParam.cblas_size) * (Info->m - Info->m % Info->Height) - cParam.dynamic_run * cParam.dynamic_size) * (2 * Info->Width + 2) * Info->Iterations / Timers.GPUTimer.GetElapsedTime();
    	    }
    	    printf("GPU Time %2.4lf (%2.4lf Gflops)     CPU Time %2.4lf (%2.4lf Gflops)", Timers.GPUTimer.GetElapsedTime(), flopsg, Timers.CPUTimer.GetElapsedTime(), flopsc);
    	    if (Info->TabularTiming)
    	    {
    		printf("            GPU Ratio: %2.3lf%%, m*n: %E", (100.0 * flopsg / (flopsc + flopsg)), (double) (Info->m * Info->n));
    	    }
    	    printf("\n");
        }
	if (Info->VerboseTiming)
	{
	    CALdouble gflops = (CALdouble)1e-09 * Info->m * Info->n * (2 * Info->Width) * (CALdouble)Info->Iterations / Timers.Kernel.GetElapsedTime();
	    CALdouble copyto = Info->DivideToGPU ? 0 : ((CALdouble) 1e-09 * (Info->Height * Timers.divideA + Info->Height * Timers.divideB) * Info->Width * sizeof(CALdouble) * (CALdouble)Info->Iterations / Timers.CounterCopyTo.GetElapsedTime());
    	    CALdouble copyfrom = Info->DstMemory == 'g' ? ((CALdouble) 1e-09 * Info->m * Info->n * sizeof(CALdouble) * (CALdouble)Info->Iterations / Timers.CounterCopyFrom.GetElapsedTime()) : 0;
    	    CALdouble copyMerge = Info->MultiThread ? 0 :((CALdouble) 1e-09 * Info->m * Info->n * sizeof(CALdouble) * (CALdouble)Info->Iterations / Timers.CounterMerge.GetElapsedTime());
    	    CALdouble copyDivide = (CALdouble) 1e-09 * (Info->Height * Timers.divideA + Info->Height * Timers.divideB) * Info->Width * sizeof(CALdouble) * (CALdouble)Info->Iterations / Timers.CounterDivide.GetElapsedTime();
    	    printf("Times:  Kernel                    Divide (%d,%d)            Merge                   Copy To                 Copy From\n", Timers.divideA, Timers.divideB);
    	    printf("        %2.4lf (%2.4lf Gflops)  %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf Gb/s)\n", Timers.Kernel.GetElapsedTime(), gflops, Timers.CounterDivide.GetElapsedTime(), copyDivide, Timers.CounterMerge.GetElapsedTime(), copyMerge, Timers.CounterCopyTo.GetElapsedTime(), copyto, Timers.CounterCopyFrom.GetElapsedTime(), copyfrom);
    	    if (Info->TabularTiming)
    	    {
    		printf("TIMES:\tw\t%lld\th\t%lld\tkernel\t%2.4lf\tdivide\t%2.4lf\tmerge\t%2.4lf\tcopyto\t%2.4lf\tcopyfr\t%2.4lf\n", Info->Width, Info->Height, gflops, copyDivide, copyMerge, copyto, copyfrom);
    	    }
    	}
    }
}

CALuint calutil::AnalyzeResults(Data* data)
{
    size_t wrong = 0;
    size_t total = 0;

    displayMatrixTiming("caldgemm");
    if (Info->Verify) {
        printf("Verifying results can take a long time on large matrices.\n");
        CPerfCounter Timer;
        Timer.Reset();
        Timer.Start();
	cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m, Info->n, Info->Width, Alpha, A, A_pitch, B, B_pitch, Beta, D, C_pitch);
        Timer.Stop();
        printf("CPU Time: %lf Gflops: %lf\n", Timer.GetElapsedTime(), (CALdouble)1e-09 * 2 * Info->m * Info->n * Info->Width / Timer.GetElapsedTime());
        for (size_t i=0; i < Info->m; i++)
        {
            for (size_t j=0; j < Info->n; j++)
            {
                if (!isDoubleEqual(C[i * C_pitch + j],D[i * C_pitch + j]))
                {
            	    if (wrong < 1) printf("Error found at row %lld, col %lld: Expected: %le, Found: %le, Diff: %le\n", i, j, D[i * C_pitch + j], C[i * C_pitch + j], D[i * C_pitch + j] - C[i * C_pitch + j]);
                    ++wrong;
                }
                ++total;
            }
        }
        if (wrong)
        {
            printf("%lld out of %lld elements were incorrect\n", wrong, total);
        }
        else
        {
            printf("Passed!\n");
        }
        if (wrong || Info->Debug)
        {
    	    print_submatrices(C, Info->n, Info->m, Info->n, 2, 2, Info->Height, Info->Height);
    	    print_submatrices(D, Info->n, Info->m, Info->n, 2, 2, Info->Height, Info->Height, C);
        }
        
    }

    return !wrong;
}

CALint calutil::SetupData ( CALmodule *module, CALresource* &_Res, Data* &data, CALdevice *device, CALcontext *ctx, CALuint numInputs, CALuint numOutputs, CALuint numConstantBuffers, CALname** ctxProgNames, int nContext )
{
    BufferHeight = Info->Height;
    BufferWidth = Info->Width;
    // Fill in the dimensions
    const CALuint aStop = aPartsNum;
    const CALuint bStop = aStop + bPartsNum;
    const CALuint fStop = bStop + numConstantBuffers;
    const CALuint cStop = fStop + cPartsNum;
    CALresult r = CAL_RESULT_OK;
    
    for (CALuint i = 0; i < cStop; ++i)
    {
	if (nContext >= 1 && i == aPartsNum + bPartsNum) continue;
	if (nContext >= 2 && i < aStop) continue;
	if (nContext >= ctxcount && (i < aStop || i >= bStop)) continue;
        CALuint tWidth = 0;
        CALuint tHeight = 0;
        CALresallocflags flag = static_cast<CALresallocflags>(0);
        CALchar mem = 'g';
        CALuint mComponents = 2;
        if (i < aStop)
        {
#if defined(CALDGEMM_48) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 8;
	    tHeight = Info->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 4;
	    tHeight = Info->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
	    tHeight = Info->Height / 4;
	    tWidth = Info->Width;
#elif defined(CALDGEMM_TRANSPOSED_A)
            /* A matrix sizes are shrunk by 2 (double2) in the width and 8 (8 resources) in the height */
            tWidth = Info->Height / 2;
            tHeight = Info->Width / aPartsNum;
#else
            tWidth = Info->Width / 2;
            tHeight = Info->Height / aPartsNum;
#endif
            mem = 'g';
        }
        else if (i >= aStop && i < bStop)
        {
#if defined(CALDGEMM_84) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 8;
	    tHeight = Info->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 4;
	    tHeight = Info->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_B)
	    tHeight = Info->Height / 4;
	    tWidth = Info->Width;
#elif defined (CALDGEMM_TRANSPOSED_B)
            tWidth = Info->Width / 2;
            tHeight = Info->Height / bPartsNum;
#else
            /* B matrix sizes are shrunk by 2 (double2) in the width and 2 (2 resources) in the height */
            tWidth = Info->Height / 2;
            tHeight = Info->Width / bPartsNum;
#endif
            mem = 'g';
        }
        else if (i >= bStop && i < fStop)
        {
            tWidth = 8;
            tHeight = 1;
            flag = static_cast<CALresallocflags>(0);
        }
        else if (i >= fStop && i < cStop)
        {
#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)
	    tWidth = tHeight = Info->Height / 4;
#else
            tWidth = Info->Height / 2;
            tHeight = Info->Height / cPartsNum;
#endif
            mem = Info->DstMemory;
            flag = (CALresallocflags) (flag | CAL_RESALLOC_CACHEABLE);
        }
        else
        {
            fprintf(stderr, "Error: Path that should be unreachable is reached\n");
            return 0;
        }
        if (AllocateMemory(data[i], device, ctx, tWidth, tHeight, mComponents, sizeof(CALdouble), flag, i, nContext)) return(1);
    }

    if (nContext < 1) {
    // Setup the constants for the kernel
    data[bStop].f_data[0] = (float) TILING_Y / Info->Height;  //Scale factor for normalized y pos
    data[bStop].f_data[2] = (float) TILING_X / Info->Height;  //Scale factor for normalized x pos
#ifdef CALDGEMM_44
    data[bStop].f_data[1] = 1.f / Info->Width;  //Step in K direction
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width);				//Iterations of loop in IL Kernel
#else //CALDGEMM_44
    data[bStop].f_data[1] = 2.f / Info->Width;  //Step in K direction
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width / (bPartsNum << 2));	//Iterations of loop in IL Kernel
#endif //CALDGEMM_44
    data[bStop].f_data[3] = 0.f;
    data[bStop].f_data[5] = (float) aPartsNum / Info->Height;  //For transposed matrix finer y resolution is needed
    data[bStop].f_data[8] = 0.5f - 0.5f / (float) (TILING_Y / aPartsNum);
    
    //Constants for Memexport
    data[bStop].i_data[9] = TILING_Y * Info->Height / 2;		//2 for double2
    data[bStop].i_data[10] = TILING_X / 2;				//x tiling in double2
#if defined(CALDGEMM_84)
    data[bStop].i_data[12] = 0 + 0 * Info->Height / 2;			//8 consecutive entries in x
    data[bStop].i_data[13] = 1 + 0 * Info->Height / 2;
    data[bStop].i_data[14] = 2 + 0 * Info->Height / 2;
    data[bStop].i_data[15] = 3 + 0 * Info->Height / 2;

    data[bStop].i_data[16] = 0 + 1 * Info->Height / 2;			//Next row
    data[bStop].i_data[17] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[18] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[19] = 0 + 1 * Info->Height / 2;

    data[bStop].i_data[20] = 0 + 2 * Info->Height / 2;			//Proceed by two rows
    data[bStop].i_data[21] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[22] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[23] = 0 + 2 * Info->Height / 2;
#elif defined(CALDGEMM_44)
    data[bStop].i_data[12] = 0 + 0 * Info->Height / 2;
    data[bStop].i_data[13] = 1 + 0 * Info->Height / 2;
    data[bStop].i_data[14] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[15] = 1 + 1 * Info->Height / 2;
    data[bStop].i_data[16] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[17] = 1 + 2 * Info->Height / 2;
    data[bStop].i_data[18] = 0 + 3 * Info->Height / 2;
    data[bStop].i_data[19] = 1 + 3 * Info->Height / 2;
#ifdef CALDGEMM_48
    data[bStop].i_data[20] = 0 + 4 * Info->Height / 2;			//Proceed by 4 rows
    data[bStop].i_data[21] = 0 + 4 * Info->Height / 2;
    data[bStop].i_data[22] = 0 + 4 * Info->Height / 2;
    data[bStop].i_data[23] = 0 + 4 * Info->Height / 2;
#endif
#else
    data[bStop].i_data[12] = 0 + 0 * Info->Height / 2;
    data[bStop].i_data[13] = 0 + 4 * Info->Height / 2;
    data[bStop].i_data[14] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[15] = 0 + 5 * Info->Height / 2;
    data[bStop].i_data[16] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[17] = 0 + 6 * Info->Height / 2;
    data[bStop].i_data[18] = 0 + 3 * Info->Height / 2;
    data[bStop].i_data[19] = 0 + 7 * Info->Height / 2;
#endif
#ifdef CALDGEMM_DIAGONAL_TEXTURE
    data[bStop].f_data[11] = 8.f / Info->Height;  //Offset for diagonal texture read
#endif
    }
    
    //////////////////////////////////////////////////////////////////////////
    //
    //  setup the program's inputs and outputs
    //
    if (!AllocateResources(ctx, device, _Res, bStop, fStop, cStop, data, nContext)) {
        if (nContext < ctxcount) fprintf(stderr, "There was an error in allocating resources and binding them to memory\n");
        else if (Info->Debug) printf("No more memory available for bbuffers\n");
        return 0;
    }
    
    if (nContext >= 1) return 1;
    for (int i = 0;i < kernel_count;i++)
    {
	if (!BindIONames(ctx, &module[i], bStop, fStop, cStop, data, ctxProgNames[i]))
	{
    	    fprintf(stderr, "There was an error in binding the memory to I/O names (context %d, kernel %d).\n", nContext, i);
    	    return 0;
    	}
    }
    
    return 1;
}

bool calutil::isDoubleEqual(CALdouble a, CALdouble b)
{
    CALdouble epsilon = 1e-6;

    if(fabs(b) <1e-13)
       return (fabs(a-b) < epsilon);
    else
       return (fabs((a-b)/b) < epsilon);
}

CPerfCounter::CPerfCounter() : _clocks(0.0E0), _start(0.0E0)
{
#ifdef ATI_OS_WIN
    i64 ifreq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&ifreq);
    _freq = (double) ifreq;
#endif

#ifdef ATI_OS_LINUX
    _freq = 1.0E3;
#endif
}

CPerfCounter::~CPerfCounter()
{
    // EMPTY!
}

void CPerfCounter::Start(void)
{
#ifdef ATI_OS_WIN
    i64 istart;
    QueryPerformanceCounter((LARGE_INTEGER*)&istart);
    _start = (double) istart;
#endif

#ifdef ATI_OS_LINUX
    struct timeval s;
    gettimeofday(&s, 0);
    _start = (double)s.tv_sec * 1.0E3 + (double)s.tv_usec / 1.0E3;
#endif
}

void CPerfCounter::Stop(void)
{
    double n = 0;
#ifdef ATI_OS_WIN
    i64 in;
    QueryPerformanceCounter((LARGE_INTEGER*)&in);
    n = (double) in;
#endif

#ifdef ATI_OS_LINUX
    struct timeval s;
    gettimeofday(&s, 0);
    n = (double)s.tv_sec * 1.0E3+ (double)s.tv_usec / 1.0E3;
#endif
    n -= _start;
    _start = 0.0E0;
    _clocks += n;
}

void CPerfCounter::Reset(void)
{
    _clocks = 0.0E0;
}

double CPerfCounter::GetElapsedTime(void)
{
    return _clocks / _freq;
}

static void __logger(const CALchar *msg)
{
    fprintf(stderr, msg);
}

CALint calutil::Initialize(CALdevice *device, CALcontext *ctx, CALuint deviceNum )
{
    if (calInit() != CAL_RESULT_OK )
    {
        fprintf(stderr, "There was an error initializing CAL.\n");
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
        return 0;
    }

    // Open the first device
    if (calDeviceOpen(device, deviceNum) != CAL_RESULT_OK )
    {
        fprintf(stderr, "There was an error opening the device %d.\n", deviceNum);
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
        return 0;
    }

    // Create a CAL context
    if (calCtxCreate(&ctx_main, *device) != CAL_RESULT_OK )
    {
        fprintf(stderr, "There was an error creatint the context.\n");
	fprintf(stderr, "Error string is %s\n", calGetErrorString());
	return 0;
    }
    return 1;
}

CALint calutil::SetupKernel(const CALchar* ILKernel, CALmodule* module, CALcontext* ctx, CALboolean disassemble)
{
    CALimage image = NULL;
    CALboolean success = CAL_FALSE;

    // Get device specific information
    CALdeviceattribs attribs;

    attribs.struct_size = sizeof(CALdeviceattribs);
    if (calDeviceGetAttribs(&attribs, Info->DeviceNum) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error getting device attribs.\n");
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
        return 0;
    }
    
    // Compile IL kernel into object
    CALobject obj;
    if (Info->PrintILKernel && (module == modules[0] || module == &modules[0][1])) printf("Kernel:\n%s\n", ILKernel);
    if (calclCompile(&obj, CAL_LANGUAGE_IL, ILKernel, attribs.target) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error compiling the program.\n");
        fprintf(stderr, "Kernel: %s\n", ILKernel);
        fprintf(stderr, "Error string is %s\n", calclGetErrorString());
        return 0;
    }

    // Link object into an image
    if (calclLink(&image, &obj, 1) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error linking the program.\n");
        fprintf(stderr, "Error string is %s\n", calclGetErrorString());
        return 0;
    }

    if (calclFreeObject(obj) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error freeing the compiler object.\n");
        fprintf(stderr, "Error string: %s\n", calclGetErrorString());
        return 0;
    }
    if (disassemble == CAL_TRUE)
    {
        calclDisassembleImage(image, __logger);
    }

    // Load module into the context
    if (calModuleLoad(module, *ctx, image) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error loading the program module.\n");
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
        return 0;
    }

    if (calclFreeImage(image) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error freeing the program image.\n");
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
        return 0;
    }

    return 1;
}

CALint calutil::RunProgram(CALcontext *ctx, CALmodule *module, CALuint Width, CALuint Height, CALevent* event)
{
    CALfunc func;
    CALresult r = CAL_RESULT_ERROR;
    if (calModuleGetEntry(&func, *ctx, *module, "main") != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error finding the program entry point.\n");
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
        return 0;
    }

    // Setup a computation domain
    CALdomain rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = Width;
    rect.height = Height;

    // Execute the program iterations number of times
    if (Info->VerboseTiming) Timers.Kernel.Start();
    r = calCtxRunProgram(event, *ctx, func, &rect);
    if (r != CAL_RESULT_OK)
    {
	fprintf(stderr, "There was an error running the program, Error code: %d.\n", r);
	fprintf(stderr, "Error string is %s\n", calGetErrorString());
	return 0;
    }

    // Wait for the last run to complete.
    if (Info->VerboseTiming)
    {
	if (event) WAITFOREVENT(*ctx, *event);
	Timers.Kernel.Stop();
	if (Info->Debug) printf("\tTotal Kernel Time: %2.4lf\n", Timers.Kernel.GetElapsedTime());
    }

    return 1;
}

CALint calutil::CleanupData(CALcontext* ctx, CALresource* &resourceHandler, Data* &data, CALuint numHandles, int nContext)
{
    if (data)
    {
        for (CALuint i = 0; i < numHandles;++i)
        {
            if ((nContext == 0 || i != aPartsNum + bPartsNum) && (nContext < 2 || i >= aPartsNum) && (nContext < ctxcount || i < aPartsNum + bPartsNum) && data[i].c_data)
            {
        	if (data[i].CALMemory )
        	{
        	    if ((Info->DstMemory == 'g' || i <= aPartsNum + bPartsNum) && (Info->DivideToGPU == CAL_FALSE || i >= aPartsNum + bPartsNum + numConstantBuffers) && nContext < 2)
        	    {
        		calResUnmap(data[i].res);
        		calCtxReleaseMem(*ctx, data[i].mem);
        		calResFree(data[i].res);
        	    }
        	}
        	else
        	{
        	    if (nContext == 0) delete [] data[i].c_data;
        	}
        	data[i].c_data = NULL;
            }
        }
    }

    // Free up the CALresource
    if (resourceHandler)
    {
        for (CALuint i = 0; i < numHandles; i++ )
        {
            if ((nContext == 0 || i != aPartsNum + bPartsNum) && (nContext < 2 || i >= aPartsNum) && (nContext < ctxcount || i < aPartsNum + bPartsNum) && resourceHandler[i])
            {
    		if (Info->DstMemory == 'c' && i >= aPartsNum + bPartsNum + numConstantBuffers && Info->KeepBuffersMapped)
    		{
		    CHKERR(calResUnmap(data[i].res), "mapping of remote output memory");
		}
        	if (calCtxReleaseMem(*ctx, data[i].dstMem) != CAL_RESULT_OK )
                {
                    fprintf(stderr, "There was an error releasing memory handle %d.\n", i);
                    fprintf(stderr, "Error string is %s\n", calGetErrorString());
                }
                if (calResFree(resourceHandler[i]) != CAL_RESULT_OK )
                {
                    fprintf(stderr, "There was an error releasing resource handle %d.\n", i);
                    fprintf(stderr, "Error string is %s\n", calGetErrorString());
                }
            }
        }
    }
}

CALint calutil::Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, Data* &data, CALuint numHandles, int nContext)
{
    CleanupData(ctx, resourceHandler, data, numHandles, nContext);

    // Unload the module from the context
    
    if (nContext < 1)
    for (int i = 0;i < kernel_count;i++)
    {
	if (module[i])
	{
    	    if (calModuleUnload(*ctx, module[i]) != CAL_RESULT_OK )
    	    {
    		printf("Error unloading module\n");
        	fprintf(stderr, "Error string is %s\n", calGetErrorString());
    	    }
    	}
    }

    delete[] resourceHandler;
    delete[] data;

    return 1;
}

CALformat calutil::getFormat(CALuint formatSize, CALuint dataSize, CALboolean isInt)
{
    CALformat format; // = CAL_FORMAT_FLOAT_1;
    
    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_1 : CAL_FORMAT_FLOAT_1);

    switch(dataSize)
    {
        case 4:

            switch(formatSize)
            {
                case 1:
                    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_1 : CAL_FORMAT_FLOAT_1);
                    break;
                case 2:
                    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_2 : CAL_FORMAT_FLOAT_2);
                    //format = CAL_FORMAT_FLOAT_2;
                    break;
                case 4:
                    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_4 : CAL_FORMAT_FLOAT_4);
                    //format = CAL_FORMAT_FLOAT_4;
                    break;
                default:
                    assert(!"attempted to use invalid format!" );
                    break;
            };
            break;
        case 8:
            switch(formatSize)
            {
                case 1:
                    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_2 : CAL_FORMAT_FLOAT_2);
                    //format = CAL_FORMAT_FLOAT_2;
                    break;
                case 2:
                    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_4 : CAL_FORMAT_FLOAT_4);
                    //format = CAL_FORMAT_FLOAT_4;
                    break;
                default:
                    assert(!"attempted to use invalid format!" );
                    break;
            };

            break;
        case 16:
            switch(formatSize)
            {
                case 1:
                    format = (isInt? CAL_FORMAT_UNSIGNED_INT32_4 : CAL_FORMAT_FLOAT_4);
                    //format = CAL_FORMAT_FLOAT_4;
                    break;
                default:
                    assert(!"attempted to use invalid format!" );
                    break;
            };
            break;
    }

    return format;
}

void calutil::copyFrom(CALchar* ptr, Data& data, CALuint pitch)
{
    // Fast path that just does a memcpy of the whole data set
    if (pitch == data.Width)
    {
        memcpy(data.c_data, ptr, data.DataSize * data.ComponentSize * data.Width * data.Height);
    }
    else
    {
        CALuint d_idx = 0;
        CALuint s_idx = 0;
        CALuint elem_size = data.ComponentSize * data.DataSize;
        for (CALuint y = 0; y < data.Height; ++y)
        {
            for (CALuint x = 0; x <(data.Width * data.ComponentSize * data.DataSize); ++x)
            {
                data.c_data[d_idx + x] = ptr[s_idx + x];
                //memcpy(data.c_data + d_idx, ptr + s_idx, elem_size * data.Width);
            }
            d_idx += (data.Width * elem_size);
            s_idx += (pitch * elem_size);
        }
    }
}

CALint calutil::CopyDataFromGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALevent* event)
{
    if (Info->DstMemory == 'c') return 0;
    CALuint pitch;
    CALresult r;
    CALchar* ptr;
    WAITFOREVENT(*ctx, *event);
    for (CALuint i = 0; i < num; ++i)
    {
	if (data[i].CALMemory)
	{
	    //if (Info->Debug) printf("GPUHandle: %d, CPUHandle: %d\n", data[i].dstMem, data[i].mem);
	    CHKERR(calMemCopy(event, *ctx, data[i].dstMem, data[i].mem, NULL), "copying data from gpu");
	    continue;
	}
        r = calResMap((CALvoid**)&ptr, &pitch, _Res[i], 0);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 1;
        }
        copyFrom(ptr, data[i], pitch);
        r = calResUnmap(_Res[i]);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 1;
        }
    }
    if (Info->VerboseTiming) WAITFOREVENT(*ctx, *event);
    return 0;
}

void calutil::copyTo(CALchar* ptr, Data& data, CALuint pitch)
{
    // Fast path that just does a memcpy of the whole data set
    if (pitch == data.Width)
    {
        memcpy(ptr, data.c_data, data.DataSize * data.ComponentSize * data.Width * data.Height);
    }
    else
    {
        CALuint d_idx = 0;
        CALuint s_idx = 0;
        CALuint elem_size = data.ComponentSize * data.DataSize;
        for (CALuint y = 0; y < data.Height; ++y)
        {
            for (CALuint x = 0; x <(data.Width * data.ComponentSize * data.DataSize); ++x)
            {
                ptr[d_idx + x] = data.c_data[s_idx + x];
                //memcpy(ptr + d_idx, data.c_data + s_idx, elem_size * data.Width);
            }
            s_idx += (data.Width * elem_size);
            d_idx += (pitch * elem_size);
        }
    }
}


CALint calutil::CopyDataToGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALboolean constants, CALevent* event, Data* dest_data)
{
    if (dest_data == NULL) dest_data = data;
    if (Info->AsyncTiming)
    {
	Timers.ATime.Reset();
	Timers.ATime.Start();
    }
    CALuint pitch;
    CALresult r;
    CALchar* ptr;
    for (CALuint i = 0; i < num; ++i)
    {
	if (data[i].CALMemory == constants) continue;
	if (data[i].CALMemory)
	{
	    CHKERR(calMemCopy(event, *ctx, data[i].mem, dest_data[i].dstMem, NULL), "copying data to gpu");
	    continue;
	}
        r = calResMap((CALvoid**)&ptr, &pitch, _Res[i], 0);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 1;
        }
        copyTo(ptr, data[i], pitch);
        r = calResUnmap(_Res[i]);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 1;
        }
    }
    if (Info->AsyncTiming && constants == CAL_FALSE)
    {
	Timers.ATime.Stop();
	printf("\t\tCopyToGPU: Time until command issued: %2.4lf\n", Timers.ATime.GetElapsedTime());
	Timers.ATime.Start();
    }
    if (Info->VerboseTiming && constants == CAL_FALSE) WAITFOREVENT(*ctx, *event);

    if (Info->AsyncTiming && constants == CAL_FALSE)
    {
	WAITFOREVENT(*ctx, *event);
	Timers.ATime.Stop();
	printf("\t\tCopyToGPU: Time until event done: %2.4lf\n", Timers.ATime.GetElapsedTime());
    }
    return 0;
}

/*typedef enum calSamplerParameterEnum {
    CAL_SAMPLER_PARAM_FETCH4 = 0,
    CAL_SAMPLER_PARAM_DEFAULT = 0,
    CAL_SAMPLER_PARAM_MIN_FILTER,
    CAL_SAMPLER_PARAM_MAG_FILTER,
    CAL_SAMPLER_PARAM_WRAP_S,
    CAL_SAMPLER_PARAM_WRAP_T,
    CAL_SAMPLER_PARAM_WRAP_R,
    CAL_SAMPLER_PARAM_BORDER_COLOR,
    CAL_SAMPLER_PARAM_LAST
} CALsamplerParameter;

typedef enum calSamplerParamWrapMode {
    CAL_SAMPLER_WRAP_REPEAT,
    CAL_SAMPLER_WRAP_MIRRORED_REPEAT,
    CAL_SAMPLER_WRAP_CLAMP_TO_EDGE,
    CAL_SAMPLER_WRAP_MIRROR_CLAMP_TO_EDGE_EXT,
    CAL_SAMPLER_WRAP_CLAMP,
    CAL_SAMPLER_WRAP_MIRROR_CLAMP_EXT,
    CAL_SAMPLER_WRAP_CLAMP_TO_BORDER,
    CAL_SAMPLER_WRAP_MIRROR_CLAMP_TO_BORDER_EXT
} CALsamplerParamWrapMode;*/

CALint calutil::BindIONames(CALcontext* ctx, CALmodule* module, CALuint iStop, CALuint cStop, CALuint oStop, Data* data, CALname* ctxProgNames)
{
    CALresult r = CAL_RESULT_ERROR;
    for (CALuint i = 0; i < oStop; ++i)
    {
        CALchar buffer[10];
        if (i < iStop)
        {
            sprintf(buffer,"i%d", i);
        }
        else if (i >= cStop && i < oStop)
        {
#ifdef CALDGEMM_USE_MEMEXPORT
	    sprintf(buffer, "g[]", i - cStop);
#else
            sprintf(buffer,"o%d", i - cStop);
#endif
        }
        else if (i >= iStop && i < cStop)
        {
            sprintf(buffer,"cb%d", i - iStop);
        }
        else
        {
            fprintf(stderr, "Error: Path that should be unreachable is reached\n");
            return 0;
        }
        r = calModuleGetName(&ctxProgNames[i], *ctx, *module, buffer);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            fprintf(stderr, "Failing name binding was %s\n", buffer);
            return 0;
        }
        //if (Info->Debug) printf("Setting Kernel Memory Resource: Memory Handle: %d, CALname handle: %d\n", data[i].dstMem, ctxProgNames[i]);
        if (i >= iStop && i < cStop)
        {
    	    r = calCtxSetMem(*ctx, ctxProgNames[i], data[i].dstMem);
    	    if (r != CAL_RESULT_OK)
    	    {
    		fprintf(stderr, "Error setting memory buffer %d\n", i);
    		fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
        	fprintf(stderr, "Error string is %s\n",calGetErrorString());
        	fprintf(stderr, "Memory Handle: %d, CALname handle: %d\n", data[i].dstMem, ctxProgNames[i]);
        	return 0;
    	    }
        }

/*	CALresult CALAPIENTRY (*calCtxSetSamplerParams) (CALcontext ctx, CALname name, CALsamplerParameter param, CALvoid* vals);
	r = calExtGetProc((CALextproc*) &calCtxSetSamplerParams, (CALextid) CAL_EXT_SAMPLER, "calCtxSetSamplerParams");
	if (r != CAL_RESULT_OK) printf("Error getting sampler extension\n");
	else
	{
    	    CALsamplerParamWrapMode wrapMode = CAL_SAMPLER_WRAP_REPEAT;
    	    r = calCtxSetSamplerParams(*ctx, progName, CAL_SAMPLER_PARAM_WRAP_S, &wrapMode);
    	    if (r != CAL_RESULT_OK) printf("Error setting wrapping mode\n");
    	    r = calCtxSetSamplerParams(*ctx, progName, CAL_SAMPLER_PARAM_WRAP_T, &wrapMode);
    	    if (r != CAL_RESULT_OK) printf("Error setting wrapping mode\n");
    	    r = calCtxSetSamplerParams(*ctx, progName, CAL_SAMPLER_PARAM_WRAP_R, &wrapMode);
    	    if (r != CAL_RESULT_OK) printf("Error setting wrapping mode\n");
    	}*/
    }
    return 1;
}

CALint calutil::AllocateResources(CALcontext* ctx, CALdevice* device, CALresource* &_Res, CALuint iStop, CALuint cStop, CALuint oStop, Data* data, int nContext)
{
    CALresult r = CAL_RESULT_ERROR;
    //////////////////////////////////////////////////////////////////////////
    //
    //  allocate input and output resources and map them into the context
    //
    for (CALuint i = 0; i < oStop; ++i)
    {
	if (nContext >= 2 && i < aPartsNum) continue;
	if (nContext >= ctxcount && (i < aPartsNum || i >= aPartsNum + bPartsNum)) continue;
	if (nContext >= 1 && i == aPartsNum + bPartsNum) continue;
        CALint tWidth = data[i].Width;;
        CALint tHeight = data[i].Height;
        CALresallocflags flag = (CALresallocflags) NULL;
        CALint mComponents = data[i].ComponentSize;
        CALchar mem = 'g';
        if (i >= cStop && i < oStop)
        {
            mem = Info->DstMemory;
            if (mem == 'c') flag = static_cast<CALresallocflags>(flag | CAL_RESALLOC_CACHEABLE);
        }
        else if (i >= iStop && i < cStop)
        {
            continue;
        }
        
#ifdef CALDGEMM_USE_MEMEXPORT
	if (i >= cStop)
	{
	    flag = (CALresallocflags) (flag | CAL_RESALLOC_GLOBAL_BUFFER);
	}
#endif
        switch(mem)
        {
            case 'g':
                r = calResAllocLocal2D(&_Res[i], *device, tWidth, tHeight, getFormat(mComponents, data[i].DataSize, CAL_TRUE), flag);
                    
                break;
            case 'c':
                r = calResAllocRemote2D(&_Res[i], device, 1, tWidth, tHeight, getFormat(mComponents, data[i].DataSize, CAL_TRUE), flag);
                break;
        }
        if (r != CAL_RESULT_OK)
        {
    	    for (CALuint j = aPartsNum;j < i;j++)
    	    {
	        calCtxReleaseMem(*ctx, data[j].dstMem);
    	        calResFree(_Res[j]);
    	    }
    	    if (nContext < ctxcount || Info->Debug)
    	    {
        	fprintf(stderr, "%s:%d - An error occured while allocating memory (context %d, i %d): %d\n", __FILE__, __LINE__, nContext, i, r);
        	fprintf(stderr, "Error string is %s\n",calGetErrorString());
    	    }
            return 0;
        }
        r = calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured while binding the allocated memory to the context: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
        //if (Info->Debug) printf("Memory Handle Context %d Buffer %d Handle %d\n", nContext, i, data[i].dstMem);
        if ((Info->DstMemory == 'c' && i >= cStop) || (Info->DivideToGPU && i < iStop))
        {
    	    data[i].mem = data[i].dstMem;
    	    data[i].res = _Res[i];
        }
        if (Info->DstMemory == 'c' && i >= cStop && Info->KeepBuffersMapped)
        {
	    CHKERR(calResMap(&data[i].v_data, &data[i].pitch, data[i].res, NULL), "mapping of remote output memory");
	}
    }

    if (nContext >= 1) return 1;
    /* Setup constant resources/memory handles */
    for (CALuint i = iStop; i < cStop; ++i)
    {
        CALint cWidth = data[i].Width * data[i].Height;
        r = calResAllocRemote1D(&_Res[i], device, 1, cWidth, getFormat(data[i].ComponentSize,data[i].DataSize), 0);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured while allocating constant memory: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
        r = calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured while binding the allocated constant memory to the context: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
    }
    return 1;
}

int calutil::AllocateMemory(Data& data, CALdevice *device, CALcontext *ctx, CALuint tWidth, CALuint tHeight, CALuint CompSize, CALuint DataSize, CALresallocflags flags, CALuint i, int nContext)
{
    data.DataSize = DataSize;
    data.Width = tWidth;
    data.Height = tHeight;
    data.ComponentSize = CompSize;
    if (tHeight > 1)
    {
	data.CALMemory = CAL_TRUE;
	if ((Info->DstMemory == 'g' || i < aPartsNum + bPartsNum) && (Info->DivideToGPU == CAL_FALSE || i >= aPartsNum + bPartsNum) && (nContext < 2 || (Info->DstMemory == 'g' && i >= aPartsNum + bPartsNum + numConstantBuffers)))
	{
		CHKERR(calResAllocRemote2D(&data.res, device, 1, tWidth, tHeight, getFormat(CompSize, data.DataSize, CAL_TRUE), flags), "allocattion of remote memory");
		CHKERR(calCtxGetMem(&data.mem, *ctx, data.res), "getting remote memory for context");
		CHKERR(calResMap(&data.v_data, &data.pitch, data.res, NULL), "mapping of remote memory");
		if (((size_t) data.v_data) & (vcpysize - 1))
		{
			printf("Memory not aligned correctly\n");
			return(1);
		}
	}
    }
    else
    {
	if (nContext == 0) data.c_data = new CALchar[tWidth * DataSize * CompSize * tHeight];
	data.CALMemory = CAL_FALSE;
    }
    if (Info->Debug && nContext < 2 && (data.CALMemory == CAL_TRUE ? ((Info->DstMemory == 'g' || i <= aPartsNum + bPartsNum) && (Info->DivideToGPU == CAL_FALSE || i >= aPartsNum + bPartsNum)) : nContext == 0))
    {
	memset((void*)data.c_data, 0, tWidth * DataSize * CompSize * tHeight);
    }
    return(0);
}

CALint calutil::QueryDeviceCaps(CALuint DeviceNum, SampleFeatures *FeatureList)
{
	CALboolean capable = CAL_TRUE;

	// Get device attributes
	CALdeviceattribs attribs;
    attribs.struct_size = sizeof(CALdeviceattribs);
    if (calDeviceGetAttribs(&attribs, DeviceNum) != CAL_RESULT_OK)
    {
		fprintf(stderr, "Could not get device attributes.\n");
		capable = CAL_FALSE;
        return capable;
    }
	
	// Check for requested features
	if(FeatureList->DoublePrecision == CAL_TRUE)	{
		if(!attribs.doublePrecision)		{
			capable = CAL_FALSE;
		}
	}
	if(FeatureList->ComputeShaders == CAL_TRUE)	{
		if(!attribs.computeShader)		{
			capable = CAL_FALSE;
		}
	}
	if(FeatureList->LocalDataShares == CAL_TRUE)	{
		if(!attribs.localDataShare)		{
			capable = CAL_FALSE;
		}
	}
	if(FeatureList->GlobalDataShares == CAL_TRUE)	{
		if(!attribs.globalDataShare)	{
			capable = CAL_FALSE;
		}
	}
	if(FeatureList->MemExport == CAL_TRUE)	{
		if(!attribs.memExport)	{
			capable = CAL_FALSE;
		}
	}

	return capable;
}

CALint calutil::QueryCALVersion(CALVersion required, const CALchar* comparison)
{
	CALVersion available;
	calGetVersion(&available.major, &available.minor, &available.imp);
	if (Info->Debug) printf("Found CAL Runtime Version: %d.%d.%d\n", available.major, available.minor, available.imp);

	if( strcmp(comparison,">") == 0 )
	{
		if( (available.major > required.major) ||
			(available.major == required.major && available.minor > required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			 available.imp > required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,">=") == 0 )
	{
		if( (available.major > required.major) ||
			(available.major == required.major && available.minor > required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			 available.imp >= required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,"<") == 0 )
	{
		if( (available.major < required.major) ||
			(available.major == required.major && available.minor < required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			 available.imp < required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,"<=") == 0 )
	{
		if( (available.major < required.major) ||
			(available.major == required.major && available.minor < required.minor) ||
			(available.major == required.major && available.minor == required.minor && 
			 available.imp <= required.imp))
		{
			return 1;
		}
	}
	else if( strcmp(comparison,"==") == 0 )
	{
		if( available.major == required.major && available.minor == required.minor &&
			 available.imp == required.imp )
		{
			return 1;
		}
	}
	else 
	{
		fprintf(stderr, "Error. Invalid comparison operator: %s (QueryCALVersion)\n", comparison);
	}

	return 0;
}

CALvoid calutil::SupportedCALVersion(CALVersion *calVersion)
{
    calVersion->major = 1;
    calVersion->minor = 3;
    calVersion->imp = 185;
    if (Info->Debug) printf("Supported CAL Runtime Version: %d.%d.%d\n", calVersion->major, calVersion->minor, calVersion->imp);
}

CALint calutil::ValidateCALRuntime()
{
	CALVersion supportedCALRuntime;
	
	supportedCALRuntime.major = 1;
	supportedCALRuntime.minor = 4;
	supportedCALRuntime.imp = 815;
	if (QueryCALVersion(supportedCALRuntime, ">=") == 0)
	{
	    if (Info->AsyncDMA && !Info->NoPerformanceWarnings) printf("WARNING: Asynchronous DMA not supported by CAL Runtime Version\n");
	    Info->AsyncDMA = CAL_FALSE;
	}

	// Get the CAL runtime currently supported by the SDK 
	SupportedCALVersion( &supportedCALRuntime );

	// Check if this runtime is available 
	return QueryCALVersion( supportedCALRuntime, ">=" );
}
