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
#include "caldgemm.il"
#ifdef ATI_OS_WIN
#include <windows.h>
#endif

#ifdef ATI_OS_LINUX
#include <sys/time.h>
#endif

#include <syscall.h>

#define CHKERR(cmd, text) if (cmd != CAL_RESULT_OK) {printf("Error '%s' while " text "\n", calGetErrorString());return(1);}
#define WAITFOREVENT(ctx, event) { CALresult r; do { r = calCtxIsEventDone(ctx, event); if (r == CAL_RESULT_ERROR) { printf("Error while waiting for event\nError String: %s\n", calGetErrorString()); return(1);} } while (r == CAL_RESULT_PENDING);}

CALvoid caldgemm::displayMatrixTiming(SampleInfo* info, const CALchar* name)
{
    if (!info->Quiet)
    {
	CALdouble gflops_CPU = (CALdouble)1e-09 * info->m * info->n * (2 * info->Width + 2) * (CALdouble)info->Iterations/info->System.GetElapsedTime();
        printf("Program: %s Sizes - A: %dx%d B: %dx%d C:%dx%d System Time %2.3lf System Gflops %2.3lf\n", name, 
                info->m, info->Width, info->Width, 
                info->n, info->m, info->n, info->System.GetElapsedTime(), gflops_CPU);
        if (info->UseCPU == CAL_TRUE && info->UseGPU == CAL_TRUE)
        {
    	    double flopsc, flopsg;
    	    if (info->m >= info->n / 2)
    	    {
    		flopsc = (double) 1e-09 * (cParam.cblas_size * info->n + (info->n % info->Height) * (info->m - cParam.cblas_size)) * (2 * info->Width + 2) * info->Iterations / info->CPUTimer.GetElapsedTime();
    		flopsg = (double) 1e-09 * (Info->m - cParam.cblas_size) * (info->n - info->n % info->Height) * (2 * info->Width + 2) * info->Iterations / info->GPUTimer.GetElapsedTime();
    	    }
    	    else
    	    {
    		flopsc = (double) 1e-09 * (cParam.cblas_size * info->m + (info->m % info->Height) * (info->n - cParam.cblas_size)) * (2 * info->Width + 2) * info->Iterations / info->CPUTimer.GetElapsedTime();
    		flopsg = (double) 1e-09 * (Info->n - cParam.cblas_size) * (info->m - info->m % info->Height) * (2 * info->Width + 2) * info->Iterations / info->GPUTimer.GetElapsedTime();
    	    }
    	    printf("GPU Time %2.4lf (%2.4lf Gflops)     CPU Time %2.4lf (%2.4lf Gflops)\n", info->GPUTimer.GetElapsedTime(), flopsg, info->CPUTimer.GetElapsedTime(), flopsc);
        }
	if (Info->VerboseTiming)
	{
	    CALdouble gflops = (CALdouble)1e-09 * info->m * info->n * (2 * info->Width + 2) * (CALdouble)info->Iterations / info->Kernel.GetElapsedTime();
	    CALdouble copyto = (CALdouble) 1e-09 * (info->m * (1 + (double) (info->n > info->Height)) + info->n * (info->m / info->Height)) * info->Width * sizeof(CALdouble) * (CALdouble)info->Iterations/info->CounterCopyTo.GetElapsedTime();
    	    CALdouble copyfrom = Info->DstMemory == 'g' ? ((CALdouble) 1e-09 * info->m * info->n * sizeof(CALdouble) * (CALdouble)info->Iterations/info->CounterCopyFrom.GetElapsedTime()) : 0;
    	    CALdouble copyMerge = Info->MultiThread ? 0 :((CALdouble) 1e-09 * info->m * info->n * sizeof(CALdouble) * (CALdouble)info->Iterations/info->CounterMerge.GetElapsedTime());
    	    CALdouble copyDivide = (CALdouble) 1e-09 * (info->m * (1 + (double) (info->n > info->Height)) + info->n * (info->m / info->Height)) * info->Width * sizeof(CALdouble) * (CALdouble)info->Iterations/info->CounterDivide.GetElapsedTime();
    	    printf("Times:  Kernel                    Divide                  Merge                   Copy To                 Copy From\n");
    	    printf("        %2.4lf (%2.4lf Gflops)  %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf GB/s)    %2.4lf (%2.4lf Gb/s)\n", info->Kernel.GetElapsedTime(), gflops, info->CounterDivide.GetElapsedTime(), copyDivide, info->CounterMerge.GetElapsedTime(), copyMerge, info->CounterCopyTo.GetElapsedTime(), copyto, info->CounterCopyFrom.GetElapsedTime(), copyfrom);
    	}
    }
}

void caldgemm::print_submatrices(double* M, int width, int height, int pitch, int subx, int suby, int stridex, int stridey)
{
    printf("Matrix %d x %d, Subblocks %d x %d, Strides: %d / %d\n", width, height, subx, suby, stridex, stridey);
    for (int j = 0;j < height;j += stridey)
    {
	for (int jj = j;jj < j + suby;jj++)
	{
	    for (int i = 0;i < width;i += stridex)
	    {
		for (int ii = i;ii < i + subx;ii++)
		{
		    printf("%+07.3lf\t", M[jj * pitch + ii]);
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

    displayMatrixTiming(Info, "caldgemm");
    if (Info->Verify) {
        printf("Verifying results can take a long time on large matrices.\n");
        CPerfCounter Timer;
        Timer.Reset();
        Timer.Start();
        matmultCPU(A, B, D, Alpha, Beta, Info->m, Info->Width, Info->n);
        Timer.Stop();
        printf("CPU Time: %lf Gflops: %lf\n", Timer.GetElapsedTime(), (CALdouble)1e-09 * 2 * Info->m * Info->n * Info->Width / Timer.GetElapsedTime());
        for (CALuint i=0; i < Info->m; i++)
        {
            for (CALuint j=0; j < Info->n; j++)
            {
                if (!isDoubleEqual(C[i * C_pitch + j],D[i * C_pitch + j]))
                {
                    ++wrong;
                }
                ++total;
            }
        }
        if (wrong || Info->Debug)
        {
    	    print_submatrices(C, Info->n, Info->m, Info->n, 2, 2, Info->Height, Info->Height);
    	    print_submatrices(D, Info->n, Info->m, Info->n, 2, 2, Info->Height, Info->Height);
        }
        
        if (wrong)
        {
            printf("%d out of %d elements were incorrect\n", wrong, total);
        }
        else
        {
            printf("Passed!\n");
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
            /* A matrix sizes are shrunk by 2 (double2) in the width and 8 (8 resources) in the height */
            tWidth = Info->Width / 2;
            tHeight = Info->Height / aPartsNum;
            mem = 'g';
        }
        else if (i >= aStop && i < bStop)
        {
            /* B matrix sizes are shrunk by 2 (double2) in the width and 2 (2 resources) in the height */
            tWidth = Info->Height / 2;
            tHeight = Info->Width / bPartsNum;
            mem = 'g';
        }
        else if (i >= bStop && i < fStop)
        {
            tWidth = 2;
            tHeight = 1;
            flag = static_cast<CALresallocflags>(0);
        }
        else if (i >= fStop && i < cStop)
        {
            tWidth = Info->Height / 2;
            tHeight = Info->Height / aPartsNum;
            mem = Info->DstMemory;
            flag = static_cast<CALresallocflags>(flag | CAL_RESALLOC_CACHEABLE);
        }
        else
        {
            fprintf(stderr, "Error: Path that should be unreachable is reached\n");
            return 0;
        }
        AllocateMemory(data[i], device, ctx, tWidth, tHeight, mComponents, sizeof(CALdouble), flag, i);
    }

    // Setup the constants for the kernel
    data[bStop].f_data[0] = (float) aPartsNum / Info->Height;  //Scale factor for normalized y pos, factor 8 for 8 resources
    data[bStop].f_data[1] = 2.f / Info->Width;
    data[bStop].f_data[2] = 2.f / Info->Height;  //Scale factor for normalized x pos, factor 2 for double2
    data[bStop].f_data[3] = 0.f;
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width / (bPartsNum << 2));
    //////////////////////////////////////////////////////////////////////////
    //
    //  setup the program's inputs and outputs
    //
    if (!AllocateResources(ctx, device, _Res, bStop, fStop, cStop, data, *Info)) {
        fprintf(stderr, "There was an error in allocating resources and binding them to memory\n");
        return 0;
    }
    if (!BindIONames(ctx, module, bStop, fStop, cStop, data))
    {
        fprintf(stderr, "There was an error in binding the memory to I/O names.\n");
        return 0;
    }
    
    
    return 1;
}

#define _mm_store_pd_use _mm_stream_pd
#define CALDGEMM_USE_VEC_MEMCPY
#define CALDGEMM_USE_VEC_MEMCPY_PREFETCH

CALvoid caldgemm::divideBuffer(Data* dst, CALdouble* src, CALint width, CALint height, CALint pitch, CALint numBuffers)
{
    // Array to store the position from which data will be filled in the various output buffers.
    CALint* position = new CALint[numBuffers];
    memset((CALvoid*) position, 0, numBuffers * sizeof(CALint));

    for (CALint y=0; y < height; y++)
    {
        CALint bank = y % numBuffers;
        CALint nextbank = (y + 1) % numBuffers;
#ifndef CALDGEMM_USE_VEC_MEMCPY
        memcpy(dst[bank].d_data + position[bank], src + (y * pitch), dst[bank].DataSize * width);
#else
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
#endif
        
        position[bank] += width;
    }

    delete[] position;
}

#undef _mm_store_pd_use
#define _mm_store_pd_use _mm_store_pd

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
        CALint bank = y % numBuffers;
#ifndef CALDGEMM_USE_VEC_MEMCPY
        memcpy(dst + (y * pitch), src[bank].d_data + position[bank], src[bank].DataSize * width);
#else
        double* daddr = dst + (y * pitch);
        double* saddr = src[bank].d_data + position[bank];
        CALint nextbank = (y + 1) % numBuffers;
        int count = src[bank].DataSize * width;
        
        __m128d beta = _mm_set1_pd(Beta);
        
        for (int i = 0;i < count;i += 64)
        {
#ifdef CALDGEMM_USE_VEC_MEMCPY_PREFETCH
    	    _mm_prefetch(saddr + 100, _MM_HINT_NTA);
    	    _mm_prefetch(daddr + 100, _MM_HINT_T0);
#endif
    	    _mm_store_pd_use(daddr, _mm_add_pd(_mm_load_pd(saddr), _mm_mul_pd(beta, _mm_load_pd(daddr))));
    	    _mm_store_pd_use(daddr + 2, _mm_add_pd(_mm_load_pd(saddr + 2), _mm_mul_pd(beta, _mm_load_pd(daddr + 2))));
    	    _mm_store_pd_use(daddr + 4, _mm_add_pd(_mm_load_pd(saddr + 4), _mm_mul_pd(beta, _mm_load_pd(daddr + 4))));
    	    _mm_store_pd_use(daddr + 6, _mm_add_pd(_mm_load_pd(saddr + 6), _mm_mul_pd(beta, _mm_load_pd(daddr + 6))));
    	    saddr += 8;
    	    daddr += 8;
        }
#endif
        position[bank] += width;
    }

    if (Info->DstMemory == 'c')
    for (CALuint i = 0;i < cPartsNum;i++)
    {
        CHKERR(calResUnmap(src[i].res), "unmapping output buffer for merging");
    }
    delete[] position;
    return(0);
}


// On windows, make sure that we are in precise CALfloating point
// mode so that the calculations agree with the gpu
#ifdef WIN32
#pragma optimize("p", on)
#endif
CALvoid caldgemm::matmultCPU(CALdouble* a, CALdouble* b, CALdouble* c, CALdouble alpha, CALdouble beta, CALint m, CALint k, CALint n)
{
    for (CALint y=0; y < m; y++)
    {
        for (CALint x=0; x < n; x++)
        {
            CALdouble temp = 0;
            for (CALint z=0; z < k; z++)
            {
                temp += a[y * A_pitch + z] * b[z * B_pitch + x];
            }
            c[y * C_pitch + x] = c[y * C_pitch + x] * beta + temp * alpha;
        }
    }
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
    if (calclCompile(&obj, CAL_LANGUAGE_IL, ILKernel, attribs.target) != CAL_RESULT_OK)
    {
        fprintf(stderr, "There was an error compiling the program.\n");
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

CALint caldgemm::RunProgram(CALcontext *ctx, CALmodule *module, CALuint Width, CALuint Height, SampleInfo* Info, CALevent* event)
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
    if (Info->VerboseTiming) Info->Kernel.Start();
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
	Info->Kernel.Stop();
    }

    return 1;
}

CALint caldgemm::Cleanup(CALdevice* device, CALcontext* ctx, CALmodule* module, CALresource* &resourceHandler, Data* &data, CALuint numHandles, CALboolean waitForUser )
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

    // Unload the module from the context
    if (module)
    {
        if (calModuleUnload(*ctx, *module) != CAL_RESULT_OK )
        {
            fprintf(stderr, "Error string is %s\n", calGetErrorString());
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


void caldgemm::PrintKernel(const CALchar* string, const CALchar* name)
{
    fprintf(stderr, "Kernel: %s\n", name);
    fprintf(stderr, "%s\n", string);
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
    if (Info->VerboseTiming) WAITFOREVENT(*ctx, *event);
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
            sprintf(buffer,"o%d", i - cStop);
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

CALint caldgemm::AllocateResources(CALcontext* ctx, CALdevice* device, CALresource* &_Res, CALuint iStop, CALuint cStop, CALuint oStop, Data* data, const SampleInfo& Info)
{
    CALresult r = CAL_RESULT_ERROR;
    _Res = new CALresource[oStop];
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
            mem = Info.DstMemory;
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
        if (Info.DstMemory == 'c' && i >= cStop)
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
    if (tWidth > 16 || tHeight > 16)
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

CALint caldgemm::ParameterValidation(CALuint nInput, CALuint nOutput, SampleInfo* Info, CALdeviceattribs* attribs)
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
	printf("Supported CAL Runtime Version: %d.%d.%d\n", 
			calVersion->major, calVersion->minor, calVersion->imp);
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
	printf("Found CAL Runtime Version: %d.%d.%d\n", available.major, available.minor, available.imp);

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
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (Info->Pin < 0)
        {
            for (int i = 0;i < -Info->Pin;i++) CPU_SET(i, &mask);
        }
        else
        {
            CPU_SET(Info->Pin, &mask);
        }
        printf("Setting affinitiy to restrict on CPU %d\n", Info->Pin);
        if (0 != sched_setaffinity(0, sizeof(mask), &mask))
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

    for (int i = 0;i < ctxcount;i++)
    {
	datas[i] = new Data[numInputs + numOutputs + numConstantBuffers];
    }
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
    if (!ParameterValidation(numInputs, numOutputs, Info, &attribs))
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

    if (Info->PrintIL == CAL_TRUE)
    {
        PrintKernel(ILKernel, "Simple Matrix Multiplication");
    }
    
    for (int i = 0;i < ctxcount;i++)
    {
	if (!SetupKernel(ILKernel, &modules[i], &ctxs[i], (CALboolean) (Info->Disassemble && i == 0)))
	{
	    return 1;
	}
	resourceHandlers[i] = NULL;
	if (!SetupData(&modules[i], resourceHandlers[i], datas[i], &device, &ctxs[i], numInputs, numOutputs, numConstantBuffers))
	{
	    return 1;
	}
    
	if (Info->MultiThread)
	{
	    for (int i = 0;i < ctxcount;i++)
	    {
		for (int j = 0;j < 2;j++) pthread_mutex_init(&mergeMutex[j][i], NULL);
		mParam[i].cls = this;
		mParam[i].width = Info->Height;
		mParam[i].height = Info->Height;
		mParam[i].numBuffers = cPartsNum;
		mParam[i].nContext = i;
		mParam[i].terminate = CAL_FALSE;
		pthread_create(&mParam[i].thr, NULL, merge_wrapper, &mParam[i]);
	    }
	}
    }
    cParam.cls = this;
    cParam.terminate = CAL_FALSE;
    for (int j = 0;j < 2;j++) pthread_mutex_init(&cParam.cblasMutex[j], NULL);
    pthread_mutex_lock(&cParam.cblasMutex[0]);
    pthread_t thr;
    pthread_create(&thr, NULL, cblas_wrapper, &cParam);
    
    //unsigned long nodemask = 0xffffff;
    //syscall(SYS_set_mempolicy, 3, &nodemask, sizeof(nodemask) * 8);

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
	if (Info->Debug) printf("\t\tSlave thread starting cblas\n");

	par->cls->Info->CPUTimer.Start();
	if (Info->m >= Info->n / 2)	//favor splitting m because of consecutive memory
	{
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, par->cblas_size, Info->n, Info->Width, Alpha, A + (Info->m - par->cblas_size) * A_pitch, A_pitch, B, B_pitch, Beta, C + (Info->m - par->cblas_size) * B_pitch, C_pitch);
	    if (Info->n % Info->Height)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Info->m - par->cblas_size, Info->n % Info->Height, Info->Width, Alpha, A, A_pitch, B + Info->n - Info->n % Info->Height, B_pitch, Beta, C + Info->n - Info->n % Info->Height, C_pitch);
	}
	else
	{
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Info->m, par->cblas_size, Info->Width, Alpha, A, A_pitch, B + Info->n - par->cblas_size, B_pitch, Beta, C + Info->n - par->cblas_size, C_pitch);
	    if (Info->m % Info->Height)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Info->m % Info->Height, Info->n - par->cblas_size, Info->Width, Alpha, A + (Info->m - Info->m % Info->Height) * A_pitch, A_pitch, B, B_pitch, Beta, C + (Info->m - Info->m % Info->Height) * B_pitch, C_pitch);
	}
	par->cls->Info->CPUTimer.Stop();

        if (Info->Debug) printf("\t\tUnlocking cblasmutex\n");
        pthread_mutex_unlock(&par->cls->cParam.cblasMutex[0]);
    }
    pthread_exit(NULL);
    return(NULL);
}

void* merge_wrapper(void* arg)
{
    caldgemm::mergeParameters* par = (caldgemm::mergeParameters*) arg;
    
    pthread_mutex_lock(&par->cls->mergeMutex[1][par->nContext]);
    while (pthread_mutex_lock(&par->cls->mergeMutex[1][par->nContext]) == 0 && par->terminate == CAL_FALSE)
    {
	if (par->cls->Info->Debug) printf("\t\tSlave thread starting merge process\n");
        par->cls->mergeBuffers(par->dst, par->src, par->width, par->height, par->cls->C_pitch, par->numBuffers);
        if (par->cls->Info->Debug) printf("\t\tUnlocking mutex %d\n", par->nContext);
        pthread_mutex_unlock(&par->cls->mergeMutex[0][par->nContext]);
    }
    pthread_exit(NULL);
    return(NULL);
}

int caldgemm::RunCALDGEMM(double* a, double* b, double* c, double alpha, double beta, int tmp_m, int tmp_n, int Apitch, int Bpitch, int Cpitch)
{
    A = a;
    B = b;
    C = c;
    Alpha = alpha;
    Beta = beta;
    if (tmp_m != -1) Info->m = tmp_m;
    if (tmp_n != -1) Info->n = tmp_n;
    A_pitch = Apitch != -1 ? Apitch : Info->Width;
    B_pitch = Bpitch != -1 ? Bpitch : Info->n;
    C_pitch = Cpitch != -1 ? Cpitch : Info->n;
    
    if (((size_t) A) & (vcpysize - 1) || ((size_t) B) & (vcpysize - 1) || ((size_t) C) & (vcpysize - 1) ||
	A_pitch & (vcpysize / sizeof(CALdouble) - 1) || B_pitch & (vcpysize / sizeof(CALdouble) - 1)|| C_pitch & (vcpysize / sizeof(CALdouble) - 1))
    {
	printf("Input addresses not aligned correctly: A 0x%llX B 0x%llX C 0x%llX Pitch 0x%X 0x%X 0x%X\n", A, B, C, A_pitch, B_pitch, C_pitch);
	return(1);
    }
    
    if (Info->Debug) printf("Starting DGEMM Run m=%d k=%d n=%d Alpha=%lf Beta=%lf\n", Info->m, Info->Width, Info->n, Alpha, Beta);
    
    if (Info->Verify)
    {
	D = new CALdouble[Info->m * C_pitch];
	memcpy(D, C, Info->m * C_pitch * sizeof(CALdouble));
    }
        
    Info->System.Start();
    
    for (int i = 0;i < ctxcount;i++)
    {
	datas[i][aPartsNum + bPartsNum].d_data[3] = alpha;
	if (CopyDataToGPU(&ctxs[i], resourceHandlers[i] + numInputs, datas[i] + numInputs, numConstantBuffers, CAL_TRUE, &events[i])) return(1);
    }
    
    int usem, usen; //m and n for gpu, rest done by cblas
    if (Info->UseCPU == CAL_TRUE && Info->UseGPU == CAL_TRUE)
    {
	if (Info->m >= Info->n / 2)
	{
	    usem = (int) (Info->GPURatio * (float) Info->m);
	    usem -= usem % Info->Height;
	    cParam.cblas_size = Info->m - usem;
	    usen = Info->n;
	    usen -= usen % Info->Height;
	}
        else
        {
	    usen = (int) (0.66f * (float) Info->n);
	    usen -= usen % Info->Height;
	    cParam.cblas_size = Info->n - usen;
	    usem = Info->m;
	    usem -= usem % Info->Height;
	}
    }
    else if (Info->UseCPU)
    {
	usen = usem = 0;
	cParam.cblas_size = (Info->m >= Info->n / 2) ? Info->m : Info->n;
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

    Info->GPUTimer.Start();

    int oldj;
    int j = 0;
    for (CALuint i = 0; i < Info->Iterations; ++i)
    {
	int mb = usem / Info->Height;
	int nb = usen / Info->Height;
	int blockm, blockn, lastm, lastn;
	for (int k = 0;k <= mb * nb;k ++)
	{
	    lastm = blockm;
    	    lastn = blockn;
	    if (k < nb * mb)
	    {
		blockm = k % nb;
		blockn = k / nb;
		if (Info->Debug) printf("Iteration k = %d, m = %d, n = %d (Context %d)\n", k, blockm, blockn, j);
		
		if (Info->VerboseTiming) Info->CounterDivide.Start();
		if (blockm < ctxcount) divideBuffer(datas[j], A + blockn * A_pitch * Info->Height, Info->Width, Info->Height, A_pitch, aPartsNum);
		if (Info->Debug) printf("\tDividing Buffer\n");
		divideBuffer(datas[j] + aPartsNum, B + blockm * Info->Height, Info->Height, Info->Width, B_pitch, bPartsNum);
	        if (Info->VerboseTiming) Info->CounterDivide.Stop();
    
		if (Info->VerboseTiming) Info->CounterCopyTo.Start();
    	        if (blockm < ctxcount)
    	        {
    	    	    if (Info->Debug) printf("\tCopying part of A to GPU\n");
    	    	    if (CopyDataToGPU(&ctxs[j], resourceHandlers[j], datas[j], aPartsNum, CAL_FALSE, &events[j])) {printf("Error copying to GPU\n"); return(1);}
    	    	}
    	    	if (Info->Debug) printf("\tCopying part of B to GPU\n");
    	        if (CopyDataToGPU(&ctxs[j], resourceHandlers[j] + aPartsNum, datas[j] + aPartsNum, bPartsNum, CAL_FALSE, &events[j])) {printf("Error copying to GPU\n"); return(1);}
    		if (Info->VerboseTiming) Info->CounterCopyTo.Stop();
    	        calCtxFlush(ctxs[j]);
    	    	if (Info->Debug) printf("\tLocking mutex %d\n", j);
    	        if (Info->MultiThread) pthread_mutex_lock(&mergeMutex[0][j]);
    	        if (Info->Debug) printf("\tExecuting MM kernel\n");
    		if (!RunProgram(&ctxs[j], &modules[j], Info->Height / bPartsNum , Info->Height / aPartsNum, Info, &events[j])) {printf("Error running program\n"); return 1;}
    	        if (Info->VerboseTiming) Info->CounterCopyFrom.Start();
    	        if (Info->DstMemory == 'g' && Info->Debug == CAL_TRUE) printf("Fething part of C from GPU\n");
    		if (CopyDataFromGPU(&ctxs[j], resourceHandlers[j] + numInputs + numConstantBuffers, datas[j] + numInputs + numConstantBuffers, numOutputs, &events[j])) {printf("Error copying from GPU\n"); return(1);}
    	        if (Info->VerboseTiming) Info->CounterCopyFrom.Stop();
    		calCtxFlush(ctxs[j]);
    	    }
    	    if (k > 0)
    	    {
    		WAITFOREVENT(ctxs[oldj], events[oldj]);
    		if (Info->VerboseTiming) Info->CounterMerge.Start();
    		if (Info->Debug) printf("\tMerging buffer\n");

		if (k == nb * mb || Info->MultiThread == CAL_FALSE)
		{
	    	    if (mergeBuffers(C + lastm * Info->Height + lastn * C_pitch * Info->Height, datas[oldj] + numInputs + numConstantBuffers, Info->Height, Info->Height, C_pitch, cPartsNum)) {printf("Error merging\n"); return(1);}
	    	    if (Info->MultiThread)
	    	    {
	    		pthread_mutex_unlock(&mergeMutex[0][oldj]);
	    		for (int l = 1;l < ctxcount;l++)
	    		{
	    		    pthread_mutex_lock(&mergeMutex[0][(oldj + l) % ctxcount]);
	    		    pthread_mutex_unlock(&mergeMutex[0][(oldj + l) % ctxcount]);
	    		}
	    	    }
	    	}
	    	else
	    	{
		    mParam[oldj].dst = C + ((size_t) lastm * (size_t) Info->Height + (size_t) lastn * (size_t) C_pitch * (size_t) Info->Height);
		    mParam[oldj].src = datas[oldj] + numInputs + numConstantBuffers;
		    pthread_mutex_unlock(&mergeMutex[1][oldj]);
		}

	        if (Info->VerboseTiming) Info->CounterMerge.Stop();
	    }
	    oldj = j;
    	    j = (j + 1) % ctxcount;
	}
    }
    Info->GPUTimer.Stop();

    if (Info->UseCPU) pthread_mutex_lock(&cParam.cblasMutex[0]);

    Info->System.Stop();

    if (Info->Debug) printf("DGEMM Run Complete\n");
    
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
	if (!Cleanup(&device, &ctxs[i], &modules[i], resourceHandlers[i], datas[i], numInputs + numOutputs + numConstantBuffers, CAL_TRUE))
	{
	    return 1;
	}
	if (Info->MultiThread)
	{
	    mParam[i].terminate = CAL_TRUE;
	    pthread_mutex_unlock(&mergeMutex[1][i]);
	}
    }
    cParam.terminate = CAL_TRUE;
    pthread_mutex_unlock(&cParam.cblasMutex[1]);
    
    //Fixme: must wait for threads to terminate
    /*for (int j = 0;j < 2;j++)
    {
	pthread_mutex_destroy(&mergeMutex[j][i]);
	pthread_mutex_destroy(&cParam.cblasMutex[j]);
    }*/

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

    sched_setaffinity(0, sizeof(oldcpumask), &oldcpumask);

    return(0);
}

void caldgemm::ResetTimers()
{
    //Reset Timers
    Info->System.Reset();
    Info->Kernel.Reset();
    Info->CounterDivide.Reset();
    Info->CounterMerge.Reset();
    Info->CounterCopyTo.Reset();
    Info->CounterCopyFrom.Reset();
    Info->CPUTimer.Reset();
    Info->GPUTimer.Reset();
}

