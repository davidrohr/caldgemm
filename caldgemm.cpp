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

#include "caldgemm.h"

#define ILKernelName ILKernel
#include "caldgemm.il"
#undef ILKernelName
#define ILKernelName ILKernelALPHA1
#define CALDGEMM_ALPHA1
#include "caldgemm.il"
#undef CALDGEMM_ALPHA1
#undef ILKernelName

#ifdef ATI_OS_WIN
#include <windows.h>
#endif

#ifdef ATI_OS_LINUX
#include <sys/time.h>
#endif

#include <syscall.h>
#include <errno.h>
extern "C" {
#include <common.h>
}
#include <math.h>

#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3

template <class T> T mymin(const T a, const T b) {return(a < b ? a : b);}
template <class T> T mymax(const T a, const T b) {return(a > b ? a : b);}

#define CHKERR(cmd, text) if (cmd != CAL_RESULT_OK) {printf("Error '%s' while " text "\n", calGetErrorString());return(1);}
#define WAITFOREVENT(ctx, event) { CALresult r; do { r = calCtxIsEventDone(ctx, event); if (r == CAL_RESULT_ERROR) { printf("Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}

caldgemm::SampleInfo::SampleInfo()
{
    Pin = -3;
    Verify = CAL_FALSE;
    Disassemble = CAL_FALSE;
    PrintILKernel = CAL_FALSE;
    Quiet = CAL_TRUE;
    DeviceNum = 0;
    Width = 1024;
    Height = 2048;
    AutoHeight = CAL_TRUE;
    Iterations = 1;
    DstMemory = 'c';
    VerboseTiming = CAL_FALSE;
    TabularTiming = CAL_FALSE;
    Debug = CAL_FALSE;
    MultiThread = CAL_TRUE;
    UseGPU = CAL_TRUE;
    UseCPU = CAL_TRUE;
    GPURatio = -1.0;
    DynamicSched = CAL_TRUE;
    MemPolicy = CAL_TRUE;
    DumpMatrix = CAL_FALSE;
    m = 0;
    n = 0;
}

CALvoid caldgemm::displayMatrixTiming(const CALchar* name)
{
    if (!Info->Quiet)
    {
        CALdouble gflops_CPU = (CALdouble)1e-09 * Info->m * Info->n * (2 * Info->Width + 2) * (CALdouble) Info->Iterations / Timers.System.GetElapsedTime();
        printf("Program: %s Sizes - A: %dx%d B: %dx%d C:%dx%d System Time %2.3lf System Gflops %2.3lf\n", name, 
                Info->m, Info->Width, Info->Width, 
                Info->n, Info->m, Info->n, Timers.System.GetElapsedTime(), gflops_CPU);
        if (Info->UseCPU == CAL_TRUE && Info->UseGPU == CAL_TRUE)
        {
    	    double flopsc, flopsg;
    	    if (CPUOnlyRun)
    	    {
    		flopsc = (double) 1e-09 * Info->m * Info->n * (2 * Info->Width + 2) * Info->Iterations / Timers.CPUTimer.GetElapsedTime();
    		flopsg = 0.0;
    	    }
    	    else if (Info->m >= Info->n / 2)
    	    {
    		flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Info->n + (Info->n % Info->Height) * (Info->m - cParam.cblas_size)) * (2 * Info->Width + 2) * Info->Iterations / Timers.CPUTimer.GetElapsedTime();
    		flopsg = (double) 1e-09 * ((Info->m - cParam.cblas_size) * (Info->n - Info->n % Info->Height) - cParam.dynamic_run * cParam.dynamic_size) * (2 * Info->Width + 2) * Info->Iterations / Timers.GPUTimer.GetElapsedTime();
    	    }
    	    else
    	    {
    		flopsc = (double) 1e-09 * (cParam.dynamic_run * cParam.dynamic_size + cParam.cblas_size * Info->m + (Info->m % Info->Height) * (Info->n - cParam.cblas_size)) * (2 * Info->Width + 2) * Info->Iterations / Timers.CPUTimer.GetElapsedTime();
    		flopsg = (double) 1e-09 * ((Info->n - cParam.cblas_size) * (Info->m - Info->m % Info->Height) - cParam.dynamic_run * cParam.dynamic_size) * (2 * Info->Width + 2) * Info->Iterations / Timers.GPUTimer.GetElapsedTime();
    	    }
    	    printf("GPU Time %2.4lf (%2.4lf Gflops)     CPU Time %2.4lf (%2.4lf Gflops)\n", Timers.GPUTimer.GetElapsedTime(), flopsg, Timers.CPUTimer.GetElapsedTime(), flopsc);
        }
	if (Info->VerboseTiming)
	{
	    CALdouble gflops = (CALdouble)1e-09 * Info->m * Info->n * (2 * Info->Width + 2) * (CALdouble)Info->Iterations / Timers.Kernel.GetElapsedTime();
	    CALdouble copyto = (CALdouble) 1e-09 * (Info->m * (1 + (double) (Info->n > Info->Height)) + Info->n * (Info->m / Info->Height)) * Info->Width * sizeof(CALdouble) * (CALdouble)Info->Iterations/Timers.CounterCopyTo.GetElapsedTime();
    	    CALdouble copyfrom = Info->DstMemory == 'g' ? ((CALdouble) 1e-09 * Info->m * Info->n * sizeof(CALdouble) * (CALdouble)Info->Iterations/Timers.CounterCopyFrom.GetElapsedTime()) : 0;
    	    CALdouble copyMerge = Info->MultiThread ? 0 :((CALdouble) 1e-09 * Info->m * Info->n * sizeof(CALdouble) * (CALdouble)Info->Iterations/Timers.CounterMerge.GetElapsedTime());
    	    CALdouble copyDivide = (CALdouble) 1e-09 * (Info->m * (1 + (double) (Info->n > Info->Height)) + Info->n * (Info->m / Info->Height)) * Info->Width * sizeof(CALdouble) * (CALdouble)Info->Iterations/Timers.CounterDivide.GetElapsedTime();
    	    printf("Times:  Kernel                    Divide                  Merge                   Copy To                 Copy From\n");
    	    printf("        %2.4lf (%2.4lf Gflops)  %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf Gb/s)\n", Timers.Kernel.GetElapsedTime(), gflops, Timers.CounterDivide.GetElapsedTime(), copyDivide, Timers.CounterMerge.GetElapsedTime(), copyMerge, Timers.CounterCopyTo.GetElapsedTime(), copyto, Timers.CounterCopyFrom.GetElapsedTime(), copyfrom);
    	    if (Info->TabularTiming)
    	    {
    		printf("TIMES:\tw\t%d\th\t%d\tkernel\t%2.4lf\tdivide\t%2.4lf\tmerge\t%2.4lf\tcopyto\t%2.4lf\tcopyfr\t%2.4lf\n", Info->Width, Info->Height, gflops, copyDivide, copyMerge, copyto, copyfrom);
    	    }
    	}
    }
}

void caldgemm::print_submatrices(double* M, int width, int height, int pitch, int subx, int suby, int stridex, int stridey)
{
    printf("Matrix %d x %d, Subblocks %d x %d, Strides: %d / %d\n", width, height, subx, suby, stridex, stridey);
    for (int j = 0;j < height;j += stridey)
    {
	for (int jj = j;jj < j + suby && jj < height;jj++)
	{
	    for (int i = 0;i < width;i += stridex)
	    {
		for (int ii = i;ii < i + subx && ii < width;ii++)
		{
		    printf("%+ 10.3lf\t", M[jj * pitch + ii]);
		}
	    }
	    printf("\n");
	}
    }
    printf("Done\n");
}

CALuint caldgemm::AnalyzeResults(Data* data)
{
    CALuint wrong = 0;
    CALuint total = 0;

    displayMatrixTiming("caldgemm");
    if (Info->Verify) {
        printf("Verifying results can take a long time on large matrices.\n");
        CPerfCounter Timer;
        Timer.Reset();
        Timer.Start();
	cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m, Info->n, Info->Width, Alpha, A, A_pitch, B, B_pitch, Beta, D, C_pitch);
        Timer.Stop();
        printf("CPU Time: %lf Gflops: %lf\n", Timer.GetElapsedTime(), (CALdouble)1e-09 * 2 * Info->m * Info->n * Info->Width / Timer.GetElapsedTime());
        for (CALuint i=0; i < Info->m; i++)
        {
            for (CALuint j=0; j < Info->n; j++)
            {
                if (!isDoubleEqual(C[i * C_pitch + j],D[i * C_pitch + j]))
                {
            	    if (wrong < 1) printf("Error found at %d, %d: Expected: %le, Found: %le, Diff: %le\n", i, j, D[i * C_pitch + j], C[i * C_pitch + j], D[i * C_pitch + j] - C[i * C_pitch + j]);
                    ++wrong;
                }
                ++total;
            }
        }
        if (wrong)
        {
            printf("%d out of %d elements were incorrect\n", wrong, total);
        }
        else
        {
            printf("Passed!\n");
        }
        if (wrong || Info->Debug)
        {
    	    print_submatrices(C, Info->n, Info->m, Info->n, 2, 2, Info->Height, Info->Height);
    	    print_submatrices(D, Info->n, Info->m, Info->n, 2, 2, Info->Height, Info->Height);
        }
        
    }

    return !wrong;
}

CALint caldgemm::SetupData ( CALmodule *module, CALresource* &_Res, Data* &data, CALdevice *device, CALcontext *ctx, CALuint numInputs, CALuint numOutputs, CALuint numConstantBuffers )
{
    // Fill in the dimensions
    const CALuint aStop = aPartsNum;
    const CALuint bStop = aStop + bPartsNum;
    const CALuint fStop = bStop + numConstantBuffers;
    const CALuint cStop = fStop + cPartsNum;
    CALresult r = CAL_RESULT_OK;
    
    for (CALuint i = 0; i < cStop; ++i)
    {
        CALuint tWidth = 0;
        CALuint tHeight = 0;
        CALresallocflags flag = static_cast<CALresallocflags>(0);
        CALchar mem = 'g';
        CALuint mComponents = 2;
        if (i < aStop)
        {
#if defined(CALDGEMM_88) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 8;
	    tHeight = Info->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 4;
	    tHeight = Info->Width;
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
#if defined(CALDGEMM_88) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 8;
	    tHeight = Info->Width;
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
	    tWidth = Info->Height / 4;
	    tHeight = Info->Width;
#elif defined (CALDGEMM_TRANSPOSED_B)
            /* B matrix sizes are shrunk by 2 (double2) in the width and 2 (2 resources) in the height */
            tWidth = Info->Width / 2;
            tHeight = Info->Height / bPartsNum;
#else
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
        AllocateMemory(data[i], device, ctx, tWidth, tHeight, mComponents, sizeof(CALdouble), flag, i);
    }

    // Setup the constants for the kernel
#ifdef CALDGEMM_44
#ifdef CALDGEMM_88
    data[bStop].f_data[0] = 8.f / Info->Height;  //Scale factor for normalized y pos
    data[bStop].f_data[2] = 8.f / Info->Height;  //Scale factor for normalized x pos
#else
    data[bStop].f_data[0] = 4.f / Info->Height;  //Scale factor for normalized y pos
    data[bStop].f_data[2] = 4.f / Info->Height;  //Scale factor for normalized x pos
#endif
#ifdef CALDGEMM_TRANSPOSED_A
    data[bStop].f_data[1] = 1.f / Info->Width;  //Step in K direction
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width);				//Iterations of loop in IL Kernel
#else
    data[bStop].f_data[1] = 2.f / Info->Width;  //Step in K direction
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width / 2);			//Iterations of loop in IL Kernel, factor 2 for double2
#endif
#else //CALDGEMM_44
    data[bStop].f_data[0] = (float) TILING_Y / Info->Height;  //Scale factor for normalized y pos, factor cPartsNum for resources
    data[bStop].f_data[1] = 2.f / Info->Width;  //Step in K direction
    data[bStop].f_data[2] = 2.f / Info->Height;  //Scale factor for normalized x pos, factor 2 for double2
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width / (bPartsNum << 2));	//Iterations of loop in IL Kernel
#endif //CALDGEMM_44
    data[bStop].f_data[3] = 0.f;
    data[bStop].f_data[5] = (float) aPartsNum / Info->Height;  //For transposed matrix finer y resolution is needed
    data[bStop].f_data[8] = 0.5f - 0.5f / (float) (TILING_Y / aPartsNum);
    
    //Constants for Memexport
#ifdef CALDGEMM_88
    data[bStop].i_data[9] = TILING_Y * Info->Height / 2;		//2 for double2
    data[bStop].i_data[10] = 4;						//x tiling in double2

    data[bStop].i_data[12] = 0 + 0 * Info->Height / 2;			//8 consecutive entries in x
    data[bStop].i_data[13] = 1 + 0 * Info->Height / 2;
    data[bStop].i_data[14] = 2 + 0 * Info->Height / 2;
    data[bStop].i_data[15] = 3 + 0 * Info->Height / 2;

    data[bStop].i_data[16] = 0 + 1 * Info->Height / 2;			//Next row
    data[bStop].i_data[17] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[18] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[19] = 0 + 1 * Info->Height / 2;

    data[bStop].i_data[20] = 0 + 2 * Info->Height / 2;			//Skip one row
    data[bStop].i_data[21] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[22] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[23] = 0 + 2 * Info->Height / 2;
#elif defined(CALDGEMM_44)
    data[bStop].i_data[9] = TILING_Y * Info->Height / 2;		//2 for double2
    data[bStop].i_data[10] = 2;						//x tiling in double2
    data[bStop].i_data[12] = 0 + 0 * Info->Height / 2;
    data[bStop].i_data[13] = 1 + 0 * Info->Height / 2;
    data[bStop].i_data[14] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[15] = 1 + 1 * Info->Height / 2;
    data[bStop].i_data[16] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[17] = 1 + 2 * Info->Height / 2;
    data[bStop].i_data[18] = 0 + 3 * Info->Height / 2;
    data[bStop].i_data[19] = 1 + 3 * Info->Height / 2;
#else
    data[bStop].i_data[9] = TILING_Y * Info->Height / 2;		//2 for double2
    data[bStop].i_data[10] = 1;						//x tiling in double2

    data[bStop].i_data[12] = 0 + 0 * Info->Height / 2;
    data[bStop].i_data[13] = 0 + 4 * Info->Height / 2;
    data[bStop].i_data[14] = 0 + 1 * Info->Height / 2;
    data[bStop].i_data[15] = 0 + 5 * Info->Height / 2;
    data[bStop].i_data[16] = 0 + 2 * Info->Height / 2;
    data[bStop].i_data[17] = 0 + 6 * Info->Height / 2;
    data[bStop].i_data[18] = 0 + 3 * Info->Height / 2;
    data[bStop].i_data[19] = 0 + 7 * Info->Height / 2;
#endif
    //////////////////////////////////////////////////////////////////////////
    //
    //  setup the program's inputs and outputs
    //
    if (!AllocateResources(ctx, device, _Res, bStop, fStop, cStop, data)) {
        fprintf(stderr, "There was an error in allocating resources and binding them to memory\n");
        return 0;
    }
    
    for (int i = 0;i < kernel_count;i++)
    if (!BindIONames(ctx, &module[i], bStop, fStop, cStop, data))
    {
        fprintf(stderr, "There was an error in binding the memory to I/O names.\n");
        return 0;
    }
    
    
    return 1;
}

#define _mm_store_pd_use _mm_stream_pd
#define CALDGEMM_USE_VEC_MEMCPY_PREFETCH

CALvoid caldgemm::divideBuffer(Data* dst, CALdouble* src, CALint width, CALint height, CALint pitch, CALint numBuffers, bool transpose)
{
    if (transpose)
    {
	for (CALint y=0; y < width; y += 2)
	{
    	    double* saddr = src + (y * pitch);
    	    double* saddr2 = src + ((y + 1) * pitch);
        
    	    for (int i = 0;i < height;i += 2)
    	    {
#if defined(CALDGEMM_88) & defined(CALDGEMM_TRANSPOSED_A)
    		CALint bank = (y / 2) % 4;
    		double* daddr = dst[bank].d_data + (i * width / 4 + ((y / 4) & 0xFFFFFFFE));
    		double* daddr2 = dst[bank].d_data + ((i + 1) * width / 4 + ((y / 4) & 0xFFFFFFFE));
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
    		CALint bank = (y / 2) % 2;
    		double* daddr = dst[bank].d_data + (i * width / 2 + ((y / 2) & 0xFFFFFFFE));
    		double* daddr2 = dst[bank].d_data + ((i + 1) * width / 2 + ((y / 2) & 0xFFFFFFFE));
#else
    		CALint bank = (i) % numBuffers;
    		CALint bank2 = (i + 1) % numBuffers;
    		double* daddr = dst[bank].d_data + (i / numBuffers) * width + y;
    		double* daddr2 = dst[bank2].d_data + (i / numBuffers) * width + y;
#endif

#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 100, _MM_HINT_NTA);
    		_mm_prefetch(saddr2 + 100, _MM_HINT_NTA);
#endif
		__m128d x1, x2, x3, x4;
		x1 = _mm_load_pd(saddr);
		x2 = _mm_load_pd(saddr2);
		x3 = _mm_unpacklo_pd(x1, x2);
		x4 = _mm_unpackhi_pd(x1, x2);
    		_mm_store_pd_use(daddr, x3);
    		_mm_store_pd_use(daddr2, x4);
    		saddr += 2;
    		saddr2 += 2;
    	    }
    	}
    }
    else
    {
#if defined(CALDGEMM_88) & defined(CALDGEMM_TRANSPOSED_A)
	double* daddr = dst[0].d_data;
	double* daddr2 = dst[1].d_data;
	double* daddr3 = dst[2].d_data;
	double* daddr4 = dst[3].d_data;
	for (CALint y=0; y < height; y++)
	{
	    int count = dst[0].DataSize * width;
	    double* saddr = src + (y * pitch);
        
	    for (int i = 0;i < count;i += 256)
	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 25, _MM_HINT_NTA);
#endif
    		_mm_store_pd_use(daddr, _mm_load_pd(saddr));
    		_mm_store_pd_use(daddr2, _mm_load_pd(saddr + 2));
    		_mm_store_pd_use(daddr3, _mm_load_pd(saddr + 4));
    		_mm_store_pd_use(daddr4, _mm_load_pd(saddr + 6));
    		_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr + 8));
    		_mm_store_pd_use(daddr2 + 2, _mm_load_pd(saddr + 10));
    		_mm_store_pd_use(daddr3 + 2, _mm_load_pd(saddr + 12));
    		_mm_store_pd_use(daddr4 + 2, _mm_load_pd(saddr + 14));
    		_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 16));
    		_mm_store_pd_use(daddr2 + 4, _mm_load_pd(saddr + 18));
    		_mm_store_pd_use(daddr3 + 4, _mm_load_pd(saddr + 20));
    		_mm_store_pd_use(daddr4 + 4, _mm_load_pd(saddr + 22));
    		_mm_store_pd_use(daddr + 6, _mm_load_pd(saddr + 24));
    		_mm_store_pd_use(daddr2 + 6, _mm_load_pd(saddr + 26));
    		_mm_store_pd_use(daddr3 + 6, _mm_load_pd(saddr + 28));
    		_mm_store_pd_use(daddr4 + 6, _mm_load_pd(saddr + 30));
    		saddr += 32;
    		daddr += 8;
    		daddr2+= 8;
    		daddr3 += 8;
    		daddr4 += 8;
    	    }
        }
#elif defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A)
	double* daddr = dst[0].d_data;
	double* daddr2 = dst[1].d_data;
	for (CALint y=0; y < height; y++)
	{
	    int count = dst[0].DataSize * width;
    	    double* saddr = src + (y * pitch);
        
    	    for (int i = 0;i < count;i += 128)
    	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 50, _MM_HINT_NTA);
#endif
    		_mm_store_pd_use(daddr, _mm_load_pd(saddr));
    		_mm_store_pd_use(daddr2, _mm_load_pd(saddr + 2));
    		_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr + 4));
    		_mm_store_pd_use(daddr2 + 2, _mm_load_pd(saddr + 6));
    		_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 8));
    		_mm_store_pd_use(daddr2 + 4, _mm_load_pd(saddr + 10));
    		_mm_store_pd_use(daddr + 6, _mm_load_pd(saddr + 12));
    		_mm_store_pd_use(daddr2 + 6, _mm_load_pd(saddr + 14));
    		saddr += 16;
    		daddr += 8;
    		daddr2+= 8;
    	    }
        }
#else
	// Array to store the position from which data will be filled in the various output buffers.
	CALint* position = new CALint[numBuffers];
        memset((CALvoid*) position, 0, numBuffers * sizeof(CALint));
	for (CALint y=0; y < height; y++)
	{
    	    CALint bank = y % numBuffers;
    	    double* daddr = dst[bank].d_data + position[bank];
    	    double* saddr = src + (y * pitch);
    	    int count = dst[bank].DataSize * width;
        
    	    for (int i = 0;i < count;i += 64)
    	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 100, _MM_HINT_NTA);
#endif
    		_mm_store_pd_use(daddr, _mm_load_pd(saddr));
    		_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr + 2));
    		_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 4));
    		_mm_store_pd_use(daddr + 6, _mm_load_pd(saddr + 6));
    		saddr += 8;
    		daddr += 8;
    	}
        
    	    position[bank] += width;
	}
	delete[] position;
#endif
    }
}

int caldgemm::mergeBuffers(CALdouble* dst, Data* src, CALint width, CALint height, CALint pitch, CALint numBuffers)
{
    // Array to store the position from which data will be pulled in from the input buffers
    CALint* position = new CALint[numBuffers];
    memset((CALvoid*) position, 0, numBuffers * sizeof(CALint));
    
    if (Info->DstMemory == 'c')
    for (CALuint i = 0;i < cPartsNum;i++)
    {
        CHKERR(calResMap(&src[i].v_data, &src[i].pitch, src[i].res, 0), "mapping output buffer for merging");
	if (((size_t) src[i].v_data) & (vcpysize - 1))
	{
	    printf("Invalid alignment\n");
	    return(1);
	}
    }

    for (CALint y=0; y < height; y++)
    {
#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)
	CALint bank = y % 4;
	double* saddr2 = src[bank + 4].d_data + position[bank];
#else
        CALint bank = y % numBuffers;
#endif

        double* daddr = dst + (y * pitch);
        double* saddr = src[bank].d_data + position[bank];
        int count = src[bank].DataSize * width;
        
#if defined(CALDGEMM_44) & !defined(CALDGEMM_USE_MEMEXPORT)

        if (__fpclassify(Beta) == FP_ZERO)
        {
    	    for (int i = 0;i < count;i += 128)
    	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 50, _MM_HINT_NTA);
    		_mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
#endif
    		_mm_store_pd_use(daddr, _mm_load_pd(saddr));
    		_mm_store_pd_use(daddr + 2, _mm_load_pd(saddr2));
    	        _mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 2));
    	        _mm_store_pd_use(daddr + 6, _mm_load_pd(saddr2 + 2));
    		_mm_store_pd_use(daddr + 8, _mm_load_pd(saddr + 4));
    		_mm_store_pd_use(daddr + 10, _mm_load_pd(saddr2 + 4));
    	        _mm_store_pd_use(daddr + 12, _mm_load_pd(saddr + 6));
    	        _mm_store_pd_use(daddr + 14, _mm_load_pd(saddr2 + 6));
    		saddr += 8;
    		saddr2 += 8;
    	        daddr += 16;
    	    }
        }
        else
        {
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
    	    __m128d beta = _mm_set1_pd(Beta);
    	    for (int i = 0;i < count;i += 128)
    	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 50, _MM_HINT_NTA);
    		_mm_prefetch(saddr2 + 50, _MM_HINT_NTA);
    	        _m_prefetchw(daddr + 100);
#endif
    		_mm_store_pd_use(daddr, _mm_add_pd(_mm_load_pd(saddr), _mm_mul_pd(beta, _mm_load_pd(daddr))));
    		_mm_store_pd_use(daddr + 2, _mm_add_pd(_mm_load_pd(saddr2), _mm_mul_pd(beta, _mm_load_pd(daddr + 2))));
    	        _mm_store_pd_use(daddr + 4, _mm_add_pd(_mm_load_pd(saddr + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 4))));
    	        _mm_store_pd_use(daddr + 6, _mm_add_pd(_mm_load_pd(saddr2 + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 6))));
    		_mm_store_pd_use(daddr + 8, _mm_add_pd(_mm_load_pd(saddr + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 8))));
    		_mm_store_pd_use(daddr + 10, _mm_add_pd(_mm_load_pd(saddr2 + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 10))));
    	        _mm_store_pd_use(daddr + 12, _mm_add_pd(_mm_load_pd(saddr + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 12))));
    	        _mm_store_pd_use(daddr + 14, _mm_add_pd(_mm_load_pd(saddr2 + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 14))));
    		saddr += 8;
    		saddr2 += 8;
    	        daddr += 16;
    	    }
    	}
    	    
	position[bank] += width / 2;
#else        
        if (__fpclassify(Beta) == FP_ZERO)
        {
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_stream_pd
    	    for (int i = 0;i < count;i += 64)
    	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 100, _MM_HINT_NTA);
#endif
    		_mm_store_pd_use(daddr, _mm_load_pd(saddr));
    	        _mm_store_pd_use(daddr + 2, _mm_load_pd(saddr + 2));
    		_mm_store_pd_use(daddr + 4, _mm_load_pd(saddr + 4));
    	        _mm_store_pd_use(daddr + 6, _mm_load_pd(saddr + 6));
    		saddr += 8;
    	        daddr += 8;
    	    }
        }
        else
        {
#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd
    	    __m128d beta = _mm_set1_pd(Beta);
    	    for (int i = 0;i < count;i += 64)
    	    {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    		_mm_prefetch(saddr + 100, _MM_HINT_NTA);
    	        _m_prefetchw(daddr + 100);
#endif
    		_mm_store_pd_use(daddr, _mm_add_pd(_mm_load_pd(saddr), _mm_mul_pd(beta, _mm_load_pd(daddr))));
    	        _mm_store_pd_use(daddr + 2, _mm_add_pd(_mm_load_pd(saddr + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 2))));
    		_mm_store_pd_use(daddr + 4, _mm_add_pd(_mm_load_pd(saddr + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 4))));
    	        _mm_store_pd_use(daddr + 6, _mm_add_pd(_mm_load_pd(saddr + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 6))));
    		saddr += 8;
    	        daddr += 8;
    	    }
    	}
    	
        position[bank] += width;
#endif //CALDGEMM_44
    }

    if (Info->DstMemory == 'c')
    for (CALuint i = 0;i < cPartsNum;i++)
    {
        CHKERR(calResUnmap(src[i].res), "unmapping output buffer for merging");
    }
    delete[] position;
    return(0);
}

bool caldgemm::isDoubleEqual(CALdouble a, CALdouble b)
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

CALint caldgemm::Initialize(CALdevice *device, CALcontext *ctxs, CALuint deviceNum )
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
    for (int i = 0;i < ctxcount;i++)
    {
	if (calCtxCreate(&ctxs[i], *device) != CAL_RESULT_OK )
	{
	    fprintf(stderr, "There was an error creatint the context.\n");
	    fprintf(stderr, "Error string is %s\n", calGetErrorString());
	    return 0;
	}
    }

    return 1;
}

CALint caldgemm::SetupKernel(const CALchar* ILKernel, CALmodule* module, CALcontext* ctx, CALboolean disassemble)
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
    if (Info->PrintILKernel && ctx == ctxs) printf("Kernel:\n%s\n", ILKernel);
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

CALint caldgemm::RunProgram(CALcontext *ctx, CALmodule *module, CALuint Width, CALuint Height, CALevent* event)
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
	WAITFOREVENT(*ctx, *event);
	Timers.Kernel.Stop();
	if (Info->Debug) printf("\tTotal Kernel Time: %2.4lf\n", Timers.Kernel.GetElapsedTime());
    }

    return 1;
}

CALint caldgemm::CleanupData(CALcontext* ctx, CALresource* &resourceHandler, Data* &data, CALuint numHandles)
{
    if (data)
    {
        for (CALuint i = 0; i < numHandles; ++i)
        {
            if (data[i].c_data)
            {
        	if (data[i].CALMemory )
        	{
        	    if (Info->DstMemory == 'g' || i <= aPartsNum)
        	    {
        		calCtxReleaseMem(*ctx, data[i].mem);
        		calResUnmap(data[i].res);
        		calResFree(data[i].res);
        	    }
        	}
        	else
        	{
        	    delete [] data[i].c_data;
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
            if (resourceHandler[i] )
            {
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

CALint caldgemm::Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, Data* &data, CALuint numHandles)
{
    CleanupData(ctx, resourceHandler, data, numHandles);

    // Unload the module from the context
    
    for (int i = 0;i < kernel_count;i++)
    {
	if (module[i])
	{
    	    if (calModuleUnload(*ctx, module[i]) != CAL_RESULT_OK )
    	    {
        	fprintf(stderr, "Error string is %s\n", calGetErrorString());
    	    }
    	}
    }

    // Destroy the context
    if (ctx )
    {
        if (calCtxDestroy(*ctx) != CAL_RESULT_OK )
        {
            fprintf(stderr, "Error string is %s\n", calGetErrorString());
        }
    }

    delete[] resourceHandler;
    delete[] data;

    return 1;
}

CALformat caldgemm::getFormat(CALuint formatSize, CALuint dataSize, CALboolean isInt)
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

void caldgemm::copyFrom(CALchar* ptr, Data& data, CALuint pitch)
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

CALint caldgemm::CopyDataFromGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALevent* event)
{
    if (Info->DstMemory == 'c') return 0;
    CALuint pitch;
    CALresult r;
    CALchar* ptr;
    for (CALuint i = 0; i < num; ++i)
    {
	if (data[i].CALMemory)
	{
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

void caldgemm::copyTo(CALchar* ptr, Data& data, CALuint pitch)
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


CALint caldgemm::CopyDataToGPU(CALcontext* ctx, CALresource* _Res, Data* data, CALuint num, CALboolean constants, CALevent* event)
{
    CALuint pitch;
    CALresult r;
    CALchar* ptr;
    for (CALuint i = 0; i < num; ++i)
    {
	if (data[i].CALMemory == constants) continue;
	if (data[i].CALMemory)
	{
	    CHKERR(calMemCopy(event, *ctx, data[i].mem, data[i].dstMem, NULL), "copying data to gpu");
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
    if (Info->VerboseTiming && constants == CAL_FALSE) WAITFOREVENT(*ctx, *event);
    return 0;
}

CALint caldgemm::BindIONames(CALcontext* ctx, CALmodule* module, CALuint iStop, CALuint cStop, CALuint oStop, Data* data)
{
    CALname progName = 0;
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
        r = calModuleGetName(&progName, *ctx, *module, buffer);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            fprintf(stderr, "Failing name binding was %s\n", buffer);
            return 0;
        }
        r = calCtxSetMem(*ctx, progName, data[i].dstMem);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
    }
    return 1;
}

CALint caldgemm::AllocateResources(CALcontext* ctx, CALdevice* device, CALresource* &_Res, CALuint iStop, CALuint cStop, CALuint oStop, Data* data)
{
    CALresult r = CAL_RESULT_ERROR;
    //////////////////////////////////////////////////////////////////////////
    //
    //  allocate input and output resources and map them into the context
    //
    for (CALuint i = 0; i < oStop; ++i)
    {
        CALint tWidth = data[i].Width;;
        CALint tHeight = data[i].Height;
        CALresallocflags flag = (CALresallocflags) NULL;
        CALint mComponents = data[i].ComponentSize;
        CALchar mem = 'g';
        if (i < iStop)
        {
            mem = 'g';
        }
        else if (i >= cStop && i < oStop)
        {
            mem = Info->DstMemory;
            flag = static_cast<CALresallocflags>(flag | CAL_RESALLOC_CACHEABLE);
        }
        else if (i >= iStop && i < cStop)
        {
            continue;
        }
        else
        {
            fprintf(stderr, "Error: Path that should be unreachable is reached\n");
            return 0;
        }
#ifdef CALDGEMM_USE_MEMEXPORT
	flag = (CALresallocflags) (flag | CAL_RESALLOC_GLOBAL_BUFFER);
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
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
        r = calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
        if (Info->DstMemory == 'c' && i >= cStop)
        {
    	    data[i].mem = data[i].dstMem;
    	    data[i].res = _Res[i];
        }
    }

    /* Setup constant resources/memory handles */
    for (CALuint i = iStop; i < cStop; ++i)
    {
        CALint cWidth = data[i].Width * data[i].Height;
        r = calResAllocRemote1D(&_Res[i], device, 1, cWidth, getFormat(data[i].ComponentSize,data[i].DataSize), 0);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
        r = calCtxGetMem(&data[i].dstMem, *ctx, _Res[i]);
        if (r != CAL_RESULT_OK)
        {
            fprintf(stderr, "%s:%d - An error occured: %d\n",__FILE__, __LINE__, r);
            fprintf(stderr, "Error string is %s\n",calGetErrorString());
            return 0;
        }
    }
    return 1;
}

int caldgemm::AllocateMemory(Data& data, CALdevice *device, CALcontext *ctx, CALuint tWidth, CALuint tHeight, CALuint CompSize, CALuint DataSize, CALresallocflags flags, CALuint i)
{
    data.DataSize = DataSize;
    data.Width = tWidth;
    data.Height = tHeight;
    data.ComponentSize = CompSize;
    if (tHeight > 1)
    {
		data.CALMemory = CAL_TRUE;
		if (Info->DstMemory == 'g' || i < aPartsNum + bPartsNum)
		{
			CHKERR(calResAllocRemote2D(&data.res, device, 1, tWidth, tHeight, getFormat(CompSize, data.DataSize, CAL_TRUE), flags), "allocating of remote memory");
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
		data.c_data = new CALchar[tWidth * DataSize * CompSize * tHeight];
		data.CALMemory = CAL_FALSE;
    }
    if (data.CALMemory != CAL_TRUE || Info->DstMemory == 'g' || i <= aPartsNum)
    {
		memset((void*)data.c_data, 0, tWidth * DataSize * CompSize * tHeight);
    }
    return(0);
}

CALint caldgemm::ParameterValidation(CALuint nInput, CALuint nOutput, CALdeviceattribs* attribs)
{
    CALint retval = 1;
    CALuint mult = 0;
    CALuint mega = 1024 * 1024;
    CALuint pitch = (Info->Width + 63) &(~63);
    CALuint single = (pitch * Info->Height * sizeof(CALfloat));
    CALuint srcbytes = 2 *(CALuint)single * nInput / mega;
    CALuint dstbytes = 2 *(CALuint)single * nOutput / mega;
    mult += 1;
    if (srcbytes >= attribs->uncachedRemoteRAM)
    {
        retval = 0;
    }
    else if (srcbytes >= attribs->localRAM)
    {
        retval = 0;
    }

    if (Info->DstMemory == 'c')
    {
        if (mult * dstbytes >= attribs->cachedRemoteRAM)
        {
            retval = 0;
        }
        else if (dstbytes >= attribs->uncachedRemoteRAM)
        {
            retval = 0;
        }
    }
    else
    {
        if (mult * dstbytes >= attribs->cachedRemoteRAM)
        {
            retval = 0;
        }
        else if (dstbytes >= attribs->uncachedRemoteRAM)
        {
            retval = 0;
        }
        else if (dstbytes >= attribs->localRAM)
        {
            retval = 0;
        }
    }
    return retval;
}


CALvoid caldgemm::SupportedCALVersion(CALVersion *calVersion)
{
	calVersion->major = 1;
	calVersion->minor = 3;
	calVersion->imp = 185;
	if (Info->Debug) printf("Supported CAL Runtime Version: %d.%d.%d\n", calVersion->major, calVersion->minor, calVersion->imp);
}

CALint caldgemm::QueryDeviceCaps(CALuint DeviceNum, SampleFeatures *FeatureList)
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

CALint caldgemm::QueryCALVersion(CALVersion required, const CALchar* comparison)
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

CALint caldgemm::ValidateCALRuntime()
{
	CALVersion supportedCALRuntime;

	// Get the CAL runtime currently supported by the SDK 
	SupportedCALVersion( &supportedCALRuntime );

	// Check if this runtime is available 
	return QueryCALVersion( supportedCALRuntime, ">=" );
}

int caldgemm::InitCALDGEMM(SampleInfo* pInfo)
{
    Info = pInfo;

    sched_getaffinity(0, sizeof(oldcpumask), &oldcpumask);
    
    if (Info->Pin != -100)
    {
        CPU_ZERO(&gpumask);
        if (Info->Pin < 0)
        {
            for (int i = 0;i < -Info->Pin;i++) CPU_SET(i, &gpumask);
        }
        else
        {
            CPU_SET(Info->Pin, &gpumask);
        }
	if (Info->Debug) printf("Setting affinitiy to restrict on CPU %d\n", Info->Pin);
	if (0 != sched_setaffinity(0, sizeof(gpumask), &gpumask))
	{
    	    printf("Error setting CPU affinity\n");
    	    return(1);
    	}
    }

    if(!ValidateCALRuntime())
    {
        fprintf(stdout, "Error. Could not find a compatible CAL runtime.\n");
	return 0;
    }

    if (Info->Width & 0xf)
    {
        fprintf(stderr, "Only width of size 16 are computable.\n");
        Info->Width = (Info->Width + 0xf) & (~0xf);
    }
    if (Info->Height & 0x7)
    {
        fprintf(stderr, "Only heights with multiple of 8 are computable.\n" );
        Info->Height = (Info->Height + 0x7) & (~0x7);
    }
    
    numInputs = aPartsNum + bPartsNum;
    numOutputs = cPartsNum;
    numConstantBuffers = 1;
    device = 0;

    if (Info->Debug) printf("Initializing CAL\n");
    if (!Initialize(&device, ctxs, Info->DeviceNum))
    {
        return 1;
    }

    CALdeviceattribs attribs;
    attribs.struct_size = sizeof(CALdeviceattribs);
    if (calDeviceGetAttribs(&attribs, Info->DeviceNum) != CAL_RESULT_OK)
    {
	printf("Error getting device attributes\n");
        return 1;
    }

    // Verify that there is space on the graphics card to run the kernel
    // with the given parameters
    if (!ParameterValidation(numInputs, numOutputs, &attribs))
    {
        printf("There is not enough memory on the card to run the sample\n"
               "with the given command line options. Please try again.\n");
        return 1;
    }
    Features.DoublePrecision = CAL_TRUE;
    if(QueryDeviceCaps(Info->DeviceNum, &Features) != CAL_TRUE)
    {
	printf("The Device Number is invalid or this device is not capable of running this sample.");
	return 1;
    }

    for (int i = 0;i < ctxcount;i++)
    {
	if (!SetupKernel(ILKernel, &modules[i][0], &ctxs[i], (CALboolean) (Info->Disassemble && i == 0)) ||
	    !SetupKernel(ILKernelALPHA1, &modules[i][1], &ctxs[i], (CALboolean) (Info->Disassemble && i == 0)))
	{
	    return 1;
	}
	

	datas[i] = new Data[numInputs + numOutputs + numConstantBuffers];
	resourceHandlers[i] = new CALresource[numInputs + numOutputs + numConstantBuffers];

	if (!SetupData(modules[i], resourceHandlers[i], datas[i], &device, &ctxs[i], numInputs, numOutputs, numConstantBuffers))
	{
	    return 1;
	}
    
	if (Info->MultiThread)
	{
	    for (int j = 0;j < 2;j++) pthread_mutex_init(&mParam[i].mergeMutex[j], NULL);
	    mParam[i].cls = this;
	    mParam[i].nContext = i;
	    mParam[i].terminate = CAL_FALSE;
	    pthread_t thr;
	    pthread_create(&thr, NULL, merge_wrapper, &mParam[i]);
	}
    }
    cParam.cls = this;
    cParam.terminate = CAL_FALSE;
    for (int j = 0;j < 2;j++) pthread_mutex_init(&cParam.cblasMutex[j], NULL);
    pthread_mutex_lock(&cParam.cblasMutex[0]);
    pthread_t thr;
    pthread_create(&thr, NULL, cblas_wrapper, &cParam);
    
    if (Info->MemPolicy)
    {
	unsigned long nodemask = 0xffffff;
	syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
    }

    if (Info->Pin != -100) sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

    return(0);
}

void* cblas_wrapper(void* arg)
{
    volatile caldgemm::cblasParameters* par = (caldgemm::cblasParameters*) arg;
    volatile caldgemm::SampleInfo* Info = par->cls->Info;
    
    sched_setaffinity(0, sizeof(par->cls->oldcpumask), &par->cls->oldcpumask);
    
    pthread_mutex_lock(&par->cls->cParam.cblasMutex[1]);
    while (pthread_mutex_lock(&par->cls->cParam.cblasMutex[1]) == 0 && par->terminate == CAL_FALSE)
    {
	const CALdouble Alpha = par->cls->Alpha;
	const CALdouble Beta = par->cls->Beta;
	double* const A = par->cls->A;
	double* const B = par->cls->B;
	double* const C = par->cls->C;
	const int A_pitch = par->cls->A_pitch;
	const int B_pitch = par->cls->B_pitch;
	const int C_pitch = par->cls->C_pitch;
	const int A_pitch_use = (par->cls->TransposeA == CblasTrans ? 1 : A_pitch);
	const int B_pitch_use = (par->cls->TransposeB == CblasTrans ? B_pitch : 1);
	const CBLAS_TRANSPOSE TransposeA = par->cls->TransposeA;
	const CBLAS_TRANSPOSE TransposeB = par->cls->TransposeB;
	if (!Info->Quiet) printf("\t\tSlave thread starting cblas (m: %d, n: %d, cblas_size: %d, dynamic: %d/%d)\n", Info->m, Info->n, par->cblas_size, par->dynamic_run, par->dynamic_size);

	par->cls->Timers.CPUTimer.Start();
	int old_goto_threads = get_num_procs();
	if (Info->Pin != -100)
	{
	    goto_set_num_threads(old_goto_threads - (Info->Pin < 0 ? -Info->Pin : 1));
	    caldgemm_goto_reserve_cpus(Info->Pin < 0 ? -Info->Pin : 1);
	}
	if (Info->m >= Info->n / 2)	//favor splitting m because of consecutive memory
	{
	    if (par->dynamic_run == 0)
	    {
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->cblas_size, Info->n, Info->Width, Alpha, A + (Info->m - par->cblas_size) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Info->m - par->cblas_size) * C_pitch, C_pitch);
	    }
	    else
	    {
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->dynamic_run, par->dynamic_size, Info->Width, Alpha, A + (Info->m - par->cblas_size - par->dynamic_run) * A_pitch_use, A_pitch, B + (Info->n - Info->n % Info->Height - par->dynamic_size) * B_pitch_use, B_pitch, Beta, C + (Info->m - par->cblas_size - par->dynamic_run) * C_pitch + Info->n - Info->n % Info->Height - par->dynamic_size, C_pitch);
	    }
	    
	    if (Info->n % Info->Height && par->borders_done == CAL_FALSE)
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m - par->cblas_size, Info->n % Info->Height, Info->Width, Alpha, A, A_pitch, B + (Info->n - Info->n % Info->Height) * B_pitch_use, B_pitch, Beta, C + Info->n - Info->n % Info->Height, C_pitch);
	}
	else
	{
	    if (par->dynamic_run == 0)
	    {
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m, par->cblas_size, Info->Width, Alpha, A, A_pitch, B + (Info->n - par->cblas_size) * B_pitch_use, B_pitch, Beta, C + Info->n - par->cblas_size, C_pitch);
	    }
	    else
	    {
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, par->dynamic_size, par->dynamic_run, Info->Width, Alpha, A + (Info->m - Info->m % Info->Height - par->dynamic_size) * A_pitch_use, A_pitch, B + (Info->n - par->cblas_size - par->dynamic_run) * B_pitch_use, B_pitch, Beta, C + (Info->m - Info->m % Info->Height - par->dynamic_size) * C_pitch + Info->n - par->cblas_size - par->dynamic_run, C_pitch);
	    }
	    
	    if (Info->m % Info->Height && par->borders_done == CAL_FALSE)
		cblas_dgemm(CblasRowMajor, TransposeA, TransposeB, Info->m % Info->Height, Info->n - par->cblas_size, Info->Width, Alpha, A + (Info->m - Info->m % Info->Height) * A_pitch_use, A_pitch, B, B_pitch, Beta, C + (Info->m - Info->m % Info->Height) * B_pitch, C_pitch);
	}
	goto_set_num_threads(old_goto_threads);
	caldgemm_goto_reserve_cpus(0);
	par->cls->Timers.CPUTimer.Stop();

        if (Info->Debug) printf("\t\tUnlocking cblasmutex\n");
        pthread_mutex_unlock(&par->cls->cParam.cblasMutex[0]);
    }
    if (Info->Debug) printf("blas slave terminating\n");
    pthread_exit(NULL);
    return(NULL);
}

void* merge_wrapper(void* arg)
{
    caldgemm::mergeParameters* par = (caldgemm::mergeParameters*) arg;
    
    if (par->cls->Info->Pin != -100) sched_setaffinity(0, sizeof(par->cls->gpumask), &par->cls->gpumask);
    
    pthread_mutex_lock(&par->mergeMutex[1]);
    while (pthread_mutex_lock(&par->mergeMutex[1]) == 0 && par->terminate == CAL_FALSE)
    {
	if (par->cls->Info->Debug) printf("\t\tSlave thread starting merge process\n");
        par->cls->mergeBuffers(par->dst, par->src, par->cls->Info->Height, par->cls->Info->Height, par->cls->C_pitch, par->cls->cPartsNum);
        if (par->cls->Info->Debug) printf("\t\tUnlocking mutex %d\n", par->nContext);
        pthread_mutex_unlock(&par->mergeMutex[0]);
    }
    if (par->cls->Info->Debug) printf("merge slave %d terminating\n", par->nContext);
    pthread_exit(NULL);
    return(NULL);
}

int caldgemm::DumpMatrix(double* a, double* b, double* c, double alpha, double beta, int tmp_m, int tmp_k, int tmp_n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB)
{
    int i = 0;
    char filename[256];
    FILE* fp = NULL;
    do
    {
	if (fp) fclose(fp);
	sprintf(filename, "dump%d.out", i++);
    } while ((fp = fopen(filename, "r")) != NULL && i < 100);
    if (i == 100)
    {
	if (fp) fclose(fp);
	return(1);
    }
    fp = fopen(filename, "w+b");
    fwrite(&a, sizeof(a), 1, fp);
    fwrite(&b, sizeof(b), 1, fp);
    fwrite(&c, sizeof(c), 1, fp);
    fwrite(&alpha, sizeof(alpha), 1, fp);
    fwrite(&beta, sizeof(beta), 1, fp);
    fwrite(&tmp_m, sizeof(tmp_m), 1, fp);
    fwrite(&tmp_k, sizeof(tmp_k), 1, fp);
    fwrite(&tmp_n, sizeof(tmp_n), 1, fp);
    fwrite(&Apitch, sizeof(Apitch), 1, fp);
    fwrite(&Bpitch, sizeof(Bpitch), 1, fp);
    fwrite(&Cpitch, sizeof(Cpitch), 1, fp);
    for (i = 0;i < tmp_m;i++)
    {
	fwrite(a + i * Apitch, sizeof(double), tmp_k, fp);
    }
    for (i = 0;i < tmp_k;i++)
    {
	fwrite(b + i * Bpitch, sizeof(double), tmp_n, fp);
    }
    fclose(fp);
    return(0);
}

int caldgemm::RunCALDGEMM(double* a, double* b, double* c, double alpha, double beta, int tmp_m, int tmp_k, int tmp_n, int Apitch, int Bpitch, int Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB)
{
    if (tmp_m == 0 || tmp_k == 0 || tmp_n == 0) return(0);		//Do Nothing
    
    bool forceCPU = false;
    bool forceReinit = false;
    int old_k = Info->Width;
    int old_height = Info->Height;
    double GPURatio;
    
    A = a;
    B = b;
    C = c;
    Alpha = alpha;
    Beta = beta;
    if (tmp_m != -1) Info->m = tmp_m;
    if (tmp_n != -1) Info->n = tmp_n;
    if (tmp_k != -1 && tmp_k != Info->Width)
    {
	Info->Width = tmp_k;
	forceReinit = true;
    }
    A_pitch = Apitch != -1 ? Apitch : Info->Width;
    B_pitch = Bpitch != -1 ? Bpitch : Info->n;
    C_pitch = Cpitch != -1 ? Cpitch : Info->n;
    TransposeA = TransA;
    TransposeB = TransB;    
    ResetTimers();
    
    //Check for double == 1.0 is unsafe and causes compiler warning
    const unsigned long long int double_one = 0x3FF0000000000000;	//1.0 in double
    const int kernel_num = (reinterpret_cast<long long int &>(Alpha) == double_one);
    if (kernel_num && Info->Debug) printf("Using Kernel for ALPHA = 1\n");
    
    if (Info->Debug) printf("Starting DGEMM Run m=%d k=%d n=%d Alpha=%lf Beta=%lf\n", Info->m, Info->Width, Info->n, Alpha, Beta);
    if (Info->Verify)
    {
	D = new CALdouble[(size_t) Info->m * (size_t) C_pitch];
	if (D == NULL)
	{
	    printf("Memory allocation error\n");
	    return(1);
	}
	memcpy(D, C, Info->m * C_pitch * sizeof(CALdouble));
    }

    Timers.System.Start();
    
    if (order == CblasColMajor)
    {
	double* tmpd;
	int tmpi;
	CBLAS_TRANSPOSE tmpt;
	tmpd = A; A = B; B = tmpd;
	tmpi = Info->m; Info->m = Info->n; Info->n = tmpi;
	tmpi = A_pitch; A_pitch = B_pitch; B_pitch = tmpi;
	tmpt = TransA;TransA = TransB;TransB = tmpt;
    }
    
    if (Info->DumpMatrix) DumpMatrix(A, B, C, Alpha, Beta, Info->m, Info->Width, Info->n, A_pitch, B_pitch, C_pitch, CblasRowMajor, TransA, TransB);

#ifndef TESTMODE    
    //Check if the GPU can/shall process the required dgemm task
    if (Info->Iterations > 1);
    else if (Info->Width % 256 || Info->Width < 256) forceCPU = true;
    else if (Info->m < 1024 || Info->n < 1024) forceCPU = true;
    else if (__fpclassify(Alpha) == FP_ZERO) forceCPU = true;
    else if (((size_t) A) & (vcpysize - 1) || ((size_t) B) & (vcpysize - 1) || ((size_t) C) & (vcpysize - 1) ||
	A_pitch & (vcpysize / sizeof(CALdouble) - 1) || B_pitch & (vcpysize / sizeof(CALdouble) - 1)|| C_pitch & (vcpysize / sizeof(CALdouble) - 1))
    {
	printf("Input addresses not aligned correctly: A 0x%llX B 0x%llX C 0x%llX Pitch 0x%X 0x%X 0x%X\n", A, B, C, A_pitch, B_pitch, C_pitch);
	forceCPU = true;
    }
#endif
    
    if (Info->AutoHeight)
    {
	if (Info->m < 2048 || Info->n < 2048 || Info->m * Info->n < 9 * 1024 * 1024)
        {
    	    Info->Height = 1024;
	}
        else
	{
	    Info->Height = 2048;
	}
	if (Info->Height != old_height)
	{
	    if (Info->Debug) printf("Height changed from %d to %d\n", old_height, Info->Height);
	    forceReinit = true;
	}
    }

    if (forceCPU || Info->UseGPU == CAL_FALSE || Info->m < Info->Height || Info->n < Info->Height || (forceReinit && (long long int) Info->m * (long long int) Info->n * (long long int) Info->Width < (long long int) 24 * 1024 * 1024 * 1024))
    {
	if (Info->Debug) printf("Running CPU only DGEMM\n");
	Timers.CPUTimer.Start();
	cblas_dgemm(CblasRowMajor, TransA, TransB, Info->m, Info->n, Info->Width, Alpha, A, A_pitch, B, B_pitch, Beta, C, C_pitch);
	Timers.CPUTimer.Stop();
	CPUOnlyRun = true;
	Info->Width = old_k;
	Info->Height = old_height;
	goto RunCALDGEMM_end;
    }
    CPUOnlyRun = false;
    
    if (Info->Pin != -100) sched_setaffinity(0, sizeof(gpumask), &gpumask);
    
    if (forceReinit)
    {
	if (Info->Debug) printf("Reinit for changed width / height\n");
	for (int i = 0;i < ctxcount;i++)
	{
    	    CleanupData(&ctxs[i], resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers);
	    SetupData(modules[i], resourceHandlers[i], datas[i], &device, &ctxs[i], numInputs, numOutputs, numConstantBuffers);
	}
    }

    if (Info->Debug) printf("Initiliazing GPU Constant Buffers...");
    for (int i = 0;i < ctxcount;i++)
    {
	if (Info->Debug) printf("%d", i);
	datas[i][aPartsNum + bPartsNum].d_data[3] = alpha;
	if (CopyDataToGPU(&ctxs[i], resourceHandlers[i] + numInputs, datas[i] + numInputs, numConstantBuffers, CAL_TRUE, &events[i])) return(1);
    }
    if (Info->Debug) printf("   Done\n");
    
    if (Info->GPURatio < 0)
    {
	/* //Optimal ratio found using seperated runs
	if ((long long int) Info->m * (long long int) Info->n > (long long int) 600000000) GPURatio = 0.66;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 150000000) GPURatio = 0.63;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 50000000) GPURatio = 0.60;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 20000000) GPURatio = 0.55;
	else GPURatio = 0.5;*/
	//Optimal ratio found using combined runs
	if ((long long int) Info->m * (long long int) Info->n > (long long int) 2000000000) GPURatio = 0.66;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 1000000000) GPURatio = 0.65;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 350000000) GPURatio = 0.64;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 200000000) GPURatio = 0.63;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 50000000) GPURatio = 0.60;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 25000000) GPURatio = 0.55;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 15000000) GPURatio = 0.50;
	else if ((long long int) Info->m * (long long int) Info->n > (long long int) 10000000) GPURatio = 0.40;
	else GPURatio = 0.30;
	if (Info->Debug) printf("GPURatio automatically set to %1.2lf\n", GPURatio);
    }
    else
    {
	GPURatio = Info->GPURatio;
    }
    
    int usem, usen; //m and n for gpu, rest done by cblas
    cParam.dynamic_run = 0;
    cParam.borders_done = CAL_FALSE;
    if (Info->UseCPU == CAL_TRUE && Info->UseGPU == CAL_TRUE)
    {
	if (Info->m >= Info->n / 2)
	{
	    usem = (int) (GPURatio * (float) Info->m + (Info->Height - 1));
	    usem -= usem % Info->Height;
	    cParam.cblas_size = Info->m - usem;
	    usen = Info->n;
	    usen -= usen % Info->Height;
	    if (Info->Debug) printf("Splitting: GPU: %d x %d, CPU: %d x %d\n", usem, usen, Info->m - usem, usen);
	}
        else
        {
	    usen = (int) (GPURatio * (float) Info->n + (Info->Height - 1));
	    usen -= usen % Info->Height;
	    cParam.cblas_size = Info->n - usen;
	    usem = Info->m;
	    usem -= usem % Info->Height;
	    if (Info->Debug) printf("Splitting: GPU: %d x %d, CPU: %d x %d\n", usem, usen, Info->m, Info->n - usen);
	}
	if (cParam.cblas_size == 0 && Info->DynamicSched == CAL_TRUE)
	{
	    cParam.dynamic_run = Info->Height;
	    cParam.dynamic_size = mymin((int) (Info->m >= Info->n / 2 ? Info->m : Info->n), (int) ((1.0f - GPURatio) * (float) Info->m * Info->n / Info->Height));
	    cParam.dynamic_size -= cParam.dynamic_size % Info->Height;
	    if (!Info->Quiet) printf("Scheduling initial dynamic run over %dx%d blocks\n", cParam.dynamic_run, cParam.dynamic_size);
	}
    }
    else
    {
	if (Info->n % Info->Height || Info->m % Info->Height)
	{
	    printf("Invalid matrix size for GPU only\n");
	    return(1);
	}
	usen = Info->n;
	usem = Info->m;
    }
    
    if (Info->UseCPU) pthread_mutex_unlock(&cParam.cblasMutex[1]);

    Timers.GPUTimer.Start();

    for (CALuint i = 0; i < Info->Iterations; ++i)
    {
	int oldj;
	int j = 0;
	
	int mb = usem / Info->Height;
	int nb = usen / Info->Height;
	int blockm, blockn, lastm, lastn;
	
	if (usen && usem)
	for (int k = 0;k <= mb * nb;k ++)
	{
	    if (cParam.dynamic_run && k < nb * mb)
	    {
		if (Info->m >= Info->n / 2)
		{
		    if (k / nb * Info->Height >= usem - cParam.dynamic_run && (k % nb) * Info->Height >= usen - cParam.dynamic_size)
		    {
			if (Info->Debug) printf("GPU skipping k = %d\n", k);
			continue;
		    }
		}
		else
		{
		    if ((k % nb) * Info->Height >= usen - cParam.dynamic_run && k / nb * Info->Height >= usem - cParam.dynamic_size)
		    {
			if (Info->Debug) printf("GPU skipping k = %d\n", k);
			continue;
		    }
		}
	    }
	
	    lastm = blockm;
    	    lastn = blockn;
	    if (k < nb * mb)
	    {
		blockm = k % nb;
		blockn = k / nb;
		if (Info->Debug) printf("Iteration k = %d, m = %d, n = %d (Context %d)\n", k, blockm, blockn, j);
		
		if (Info->VerboseTiming) Timers.CounterDivide.Start();
		if (Info->Debug) printf("\tDividing Buffer A\n");
#ifdef CALDGEMM_TRANSPOSED_A
		if (blockm < ctxcount) divideBuffer(datas[j], A + blockn * Info->Height * (TransposeA == CblasTrans ? 1 : A_pitch), Info->Height, Info->Width, A_pitch, aPartsNum, TransposeA == CblasNoTrans);
#else
		if (blockm < ctxcount) divideBuffer(datas[j], A + blockn * Info->Height * (TransposeA == CblasTrans ? 1 : A_pitch), Info->Width, Info->Height, A_pitch, aPartsNum, TransposeA == CblasTrans);
#endif
		if (Info->Debug) printf("\tDividing Buffer B\n");
#ifdef CALDGEMM_TRANSPOSED_B
		divideBuffer(datas[j] + aPartsNum, B + blockm * Info->Height * (TransposeB == CblasTrans ? B_pitch : 1), Info->Width, Info->Height, B_pitch, bPartsNum, TransposeB == CblasNoTrans);
#else
		divideBuffer(datas[j] + aPartsNum, B + blockm * Info->Height * (TransposeB == CblasTrans ? B_pitch : 1), Info->Height, Info->Width, B_pitch, bPartsNum, TransposeB == CblasTrans);
#endif
	        if (Info->VerboseTiming) Timers.CounterDivide.Stop();

		if (Info->VerboseTiming) Timers.CounterCopyTo.Start();
    	        if (blockm < ctxcount)
	        {
	    	    if (Info->Debug) printf("\tCopying part of A to GPU\n");
    	    	    if (CopyDataToGPU(&ctxs[j], resourceHandlers[j], datas[j], aPartsNum, CAL_FALSE, &events[j])) {printf("Error copying to GPU\n"); return(1);}
    	    	}
    	    	if (Info->Debug) printf("\tCopying part of B to GPU\n");
    	        if (CopyDataToGPU(&ctxs[j], resourceHandlers[j] + aPartsNum, datas[j] + aPartsNum, bPartsNum, CAL_FALSE, &events[j])) {printf("Error copying to GPU\n"); return(1);}
    		if (Info->VerboseTiming) Timers.CounterCopyTo.Stop();
    	        calCtxFlush(ctxs[j]);
    	    	if (Info->Debug) printf("\tLocking mutex %d\n", j);
    	        if (Info->MultiThread) pthread_mutex_lock(&mParam[j].mergeMutex[0]);
    	        if (Info->Debug) printf("\tExecuting MM kernel\n");
    		if (!RunProgram(&ctxs[j], &modules[j][kernel_num], Info->Height / TILING_X, Info->Height / TILING_Y, &events[j])) {printf("Error running program\n"); return 1;}
    	        if (Info->VerboseTiming) Timers.CounterCopyFrom.Start();
    	        if (Info->DstMemory == 'g' && Info->Debug == CAL_TRUE) printf("Fething part of C from GPU\n");
    		if (CopyDataFromGPU(&ctxs[j], resourceHandlers[j] + numInputs + numConstantBuffers, datas[j] + numInputs + numConstantBuffers, numOutputs, &events[j])) {printf("Error copying from GPU\n"); return(1);}
    	        if (Info->VerboseTiming) Timers.CounterCopyFrom.Stop();
    		calCtxFlush(ctxs[j]);
    		
    		if (Info->UseCPU && Info->DynamicSched && cParam.dynamic_run == 0 && k >= 0.75f * GPURatio * nb * mb)
    		{
    		    if (pthread_mutex_trylock(&cParam.cblasMutex[0]) == 0)
    		    {
    			cParam.dynamic_size = (((int) ((1.0f - GPURatio) * (float) (nb * mb - k - 1)))) * Info->Height;
    			if (cParam.dynamic_size > Info->Height)
    			{
    			    cParam.dynamic_run = 1 + cParam.dynamic_size / (Info->m >= Info->n / 2 ? Info->n : Info->m);
    			    cParam.dynamic_size /= cParam.dynamic_run;
    			    cParam.dynamic_size -= cParam.dynamic_size % Info->Height;
    			    cParam.dynamic_run *= Info->Height;
    			    cParam.borders_done = CAL_TRUE;
    			    if (!Info->Quiet) printf("Scheduling Additional CPU DGEMM Run over %dx%d blocks\n", cParam.dynamic_run, cParam.dynamic_size);
    			    pthread_mutex_unlock(&cParam.cblasMutex[1]);
    			}
    			else
    			{
    			    pthread_mutex_unlock(&cParam.cblasMutex[0]);
    			}
    		    }
    		}
    	    }
    	    if (ctxcount == 1)
    	    {
    		oldj = j;
    		lastm = blockm;
    		lastn = blockn;
    	    }
    	    if ((ctxcount > 1) ? (k > 0) : (k < nb * mb))
    	    {
    		WAITFOREVENT(ctxs[oldj], events[oldj]);
    		if (Info->VerboseTiming) Timers.CounterMerge.Start();
    		if (Info->Debug) printf("\tMerging buffer\n");

		if (k == nb * mb || Info->MultiThread == CAL_FALSE)
		{
	    	    if (mergeBuffers(C + lastm * Info->Height + lastn * C_pitch * Info->Height, datas[oldj] + numInputs + numConstantBuffers, Info->Height, Info->Height, C_pitch, cPartsNum)) {printf("Error merging\n"); return(1);}
	    	    if (Info->MultiThread)
	    	    {
	    		pthread_mutex_unlock(&mParam[oldj].mergeMutex[0]);
	    		for (int l = 1;l < ctxcount;l++)
	    		{
	    		    pthread_mutex_lock(&mParam[(oldj + l) % ctxcount].mergeMutex[0]);
	    		    pthread_mutex_unlock(&mParam[(oldj + l) % ctxcount].mergeMutex[0]);
	    		}
	    	    }
	    	}
	    	else
	    	{
		    mParam[oldj].dst = C + ((size_t) lastm * (size_t) Info->Height + (size_t) lastn * (size_t) C_pitch * (size_t) Info->Height);
		    mParam[oldj].src = datas[oldj] + numInputs + numConstantBuffers;
		    pthread_mutex_unlock(&mParam[oldj].mergeMutex[1]);
		}

	        if (Info->VerboseTiming) Timers.CounterMerge.Stop();
	    }
	    oldj = j;
    	    j = (j + 1) % ctxcount;
	}
    }
    Timers.GPUTimer.Stop();
    
    if (Info->Pin != -100) sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

    if (Info->UseCPU) pthread_mutex_lock(&cParam.cblasMutex[0]);

RunCALDGEMM_end:
    Timers.System.Stop();

    if (Info->Debug) printf("DGEMM Run Complete\n");
    
#ifdef TESTMODE
    print_submatrices(C, 12, 24, Info->n, 1, 1, 1, 1);
#endif
    
    if( Info->Quiet == CAL_FALSE && !AnalyzeResults(datas[0]) )
    {
        return 1;
    }
    if (Info->Verify) delete[] D;
    return(0);
}

int caldgemm::ExitCALDGEMM()
{
    for (int i = 0;i < ctxcount;i++)
    {
	if (!Cleanup(&device, &ctxs[i], modules[i], resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers))
	{
	    return 1;
	}
	if (Info->MultiThread)
	{
	    if (Info->Debug) printf("Trying to terminate merge slave %d\n", i);
	    mParam[i].terminate = CAL_TRUE;
	    if (pthread_mutex_unlock(&mParam[i].mergeMutex[1])) printf("Error unlocking mergemutex %d/1 to terminate slave\n", i);
	}
    }
    if (Info->Debug) printf("Trying to terminate blas slave\n");
    cParam.terminate = CAL_TRUE;
    if (pthread_mutex_unlock(&cParam.cblasMutex[1])) printf("Error unlocking blas mutex 1 to terminate thread\n");
    if (pthread_mutex_unlock(&cParam.cblasMutex[0])) printf("Error unlocking blas mutex 0 to terminate thread\n");
    
    if (Info->MultiThread)
    for (int i = 0;i < ctxcount;i++)
    {
	while (pthread_mutex_trylock(&mParam[i].mergeMutex[1]) != EBUSY) pthread_mutex_unlock(&mParam[i].mergeMutex[1]);
	if (pthread_mutex_unlock(&mParam[i].mergeMutex[1])) printf("Error unlocking mergeMutex %d/1\n", i);
    }
    while (pthread_mutex_trylock(&cParam.cblasMutex[1]) != EBUSY) pthread_mutex_unlock(&cParam.cblasMutex[1]);
    if (pthread_mutex_unlock(&cParam.cblasMutex[1])) printf("Error unlocking blasMutex 1\n");
    
    for (int j = 0;j < 2;j++)
    {
	if (Info->MultiThread) for (int i = 0;i < ctxcount;i++) if (pthread_mutex_destroy(&mParam[i].mergeMutex[j])) printf("Error destroying mergemutex %d/%d\n", i, j);
	if (pthread_mutex_destroy(&cParam.cblasMutex[j])) printf("Error destroying blas mutex %d\n", j);
    }

    // Close the device
    if (device)
    {
        if (calDeviceClose(device) != CAL_RESULT_OK )
        {
            fprintf(stderr, "There was an error closing the device.\n");
            fprintf(stderr, "Error string is %s\n", calGetErrorString());
        }
    }

    // Shutdown cal device
    if (calShutdown() != CAL_RESULT_OK )
    {
        fprintf(stderr, "There was an error during cal shutdown.\n");
        fprintf(stderr, "Error string is %s\n", calGetErrorString());
    }

    return(0);
}

void caldgemm::ResetTimers()
{
    //Reset Timers
    Timers.System.Reset();
    Timers.Kernel.Reset();
    Timers.CounterDivide.Reset();
    Timers.CounterMerge.Reset();
    Timers.CounterCopyTo.Reset();
    Timers.CounterCopyFrom.Reset();
    Timers.CPUTimer.Reset();
    Timers.GPUTimer.Reset();
}
