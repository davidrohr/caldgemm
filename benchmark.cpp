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
#include <sys/mman.h>
#include <common.h>

double *AA = NULL, *BB = NULL, *CC = NULL;
bool benchmark = false;
bool fastinit = false;
bool loadmatrix = false;
bool transa = false;
bool transb = false;
bool initialrun = true;
bool verifylarge = false;
char* matrixfile;

long seedused;

CALvoid Usage(const CALchar* name)
{
    fprintf(stderr,"Usage: %s [-h|-e|-v|-p|-t|-a]"
            " [-oc <1|2|4>] [-ic <1|2|4>] [-ol <c|g>] [-il <c|g>]"
            " [-w <integer>] [-h <integer>] [-d <deviceNum>]"
            " [-r <Iteration Count>]\n\n", name);
    fprintf(stderr, "\t-?        Display this help information\n" );
    fprintf(stderr, "\t-e        Verify Computational Correctness\n" );
    fprintf(stderr, "\t-q        Supress Display Output\n" );
    fprintf(stderr, "\t-a        Print the disassembled kernel image\n" );
    fprintf(stderr, "\t-i        Print IL Kernel used\n" );
    fprintf(stderr, "\t-o  <c|g> Specify the output location, c = CPU, g = GPU, default GPU\n" );
    fprintf(stderr, "\t-w  <int> k for matrix multiply, default 1024\n" );
    fprintf(stderr, "\t-h  <int> block size for matrix multiply, default 1024\n" );
    fprintf(stderr, "\t-l        Automatically select height for good performance\n" );
    fprintf(stderr, "\t-m  <int> m for matrix multiply, must be multiple of h, default 1024\n" );
    fprintf(stderr, "\t-n  <int> n for matrix multiply, must be multiple of h, default 1024\n" );
    fprintf(stderr, "\t-v        Verbose Symchronous Timing for Single Kernels / Transfers\n" );
    fprintf(stderr, "\t-r  <int> Number of iterations to run the program\n" );
    fprintf(stderr, "\t-y  <int> Force Device ID\n" );
    fprintf(stderr, "\t-d        Enable Debug Mode\n" );
    fprintf(stderr, "\t-z        Enable Multithreading\n" );
    fprintf(stderr, "\t-b        Enable Benchmarking\n" );
    fprintf(stderr, "\t-c        Use CPU\n" );
    fprintf(stderr, "\t-g        Use GPU\n" );
    fprintf(stderr, "\t-f        Fast Init (Empty Matrices)\n" );
    fprintf(stderr, "\t-t  <int> Pin to a CPU core (-100 for no pinning, -x to use cpu 0 to x - 1)\n" );
    fprintf(stderr, "\t-j  <dbl> GPU to CPU ratio\n" );
    fprintf(stderr, "\t-s        Dynamic CPU GPU scheduling\n" );
    fprintf(stderr, "\t-p        Interleaving Memory Policy\n" );
    fprintf(stderr, "\t-u        Dump Test Matrix\n" );
    fprintf(stderr, "\t-1        Transpose A Matrix\n" );
    fprintf(stderr, "\t-2        Transpose B Matrix\n" );
    fprintf(stderr, "\t-7        Verify Large Matrices\n" );
    fprintf(stderr, "\t-8        No initial run to negate cache effects\n" );
    fprintf(stderr, "\t-9        Output a table with timing information\n" );
    fprintf(stderr, "\t-x <file> Load Matrix\n" );
    
    fprintf(stderr, "*The cacheable memory flags may cause failures if the amount\n"
            " of cacheable memory is smaller than the requested memory\n"
            " size. Cacheable memory is machine dependent, so use with\n"
            " caution.\n");
}

CALboolean ParseCommandLine(CALuint argc, CALchar* argv[], caldgemm::SampleInfo* Info)
{
    Info->Verify = CAL_FALSE;
    Info->MemPolicy = CAL_FALSE;
    Info->Disassemble = CAL_FALSE;
    Info->PrintILKernel = CAL_FALSE;
    Info->Quiet = CAL_FALSE;
    Info->Pin = -3;
    Info->MultiThread = CAL_FALSE;
    Info->DeviceNum = 0;
    Info->Width = 1024;
    Info->Height = 4096;
    Info->AutoHeight = CAL_FALSE;
    Info->DynamicSched = CAL_FALSE;
    Info->VerboseTiming = CAL_FALSE;
    Info->TabularTiming = CAL_FALSE;
    Info->Debug = CAL_FALSE;
    Info->m = Info->n = 4096;
    Info->Iterations = 1;
    Info->DstMemory = 'g';
    Info->UseCPU = Info->UseGPU = CAL_FALSE;
    Info->GPURatio = -1;
    Info->DumpMatrix = CAL_FALSE;

    printf("Use -? for help\n");

    for (CALuint x = 1; x < argc; ++x)
    {
        switch(argv[x][1])
        {
            default:
				Usage(argv[0]);
                return CAL_FALSE;
            case 'q':
                Info->Quiet = CAL_TRUE;
                break;
            case '?':
                Usage(argv[0]);
                return CAL_FALSE;
            case 'e':
                Info->Verify = CAL_TRUE;
                Info->Iterations = 1;
                break;
            case 'p':
                Info->MemPolicy = CAL_TRUE;
                break;
            case 'b':
		benchmark = true;
                break;
            case 'u':
		Info->DumpMatrix = CAL_TRUE;
                break;
            case 'a':
                Info->Disassemble = CAL_TRUE;
                break;
            case '1':
		transa = true;
                break;
            case '2':
		transb = true;
                break;
            case '9':
		Info->TabularTiming = CAL_TRUE;
                break;
            case '8':
		initialrun = false;
                break;
            case '7':
		verifylarge = true;
                break;
            case 'i':
                Info->PrintILKernel = CAL_TRUE;
                break;
            case 'c':
		Info->UseCPU = CAL_TRUE;
                break;
            case 'l':
		Info->AutoHeight = CAL_TRUE;
                break;
            case 's':
		Info->DynamicSched = CAL_TRUE;
                break;
            case 'g':
                Info->UseGPU = CAL_TRUE;
                break;
            case 'f':
                fastinit = true;
                break;
            case 'o':
                if (++x < argc)
                {
                    Info->DstMemory = argv[x][0];
                    if (Info->DstMemory != 'c' && Info->DstMemory != 'g')
                    {
                        fprintf(stderr, "Invalid destination memory type\n" );
                        return CAL_FALSE;
                    }
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'w':
                if (++x < argc)
                {
                    sscanf(argv[x], "%u", &Info->Width);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'h':
                if (++x < argc)
                {
                    sscanf(argv[x], "%u", &Info->Height);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'm':
                if (++x < argc)
                {
                    sscanf(argv[x], "%u", &Info->m);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'n':
                if (++x < argc)
                {
                    sscanf(argv[x], "%u", &Info->n);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'x':
                if (++x < argc)
                {
        	    loadmatrix = true;
        	    matrixfile = argv[x];
        	}
        	else
        	{
        	    return(CAL_FALSE);
        	}
        	break;
            case 'v':
        	Info->VerboseTiming = CAL_TRUE;
        	break;
            case 'd':
        	Info->Debug = CAL_TRUE;
        	break;
            case 'z':
        	Info->MultiThread = CAL_TRUE;
        	break;
            case 'r':
                if (++x < argc)
                {
                    sscanf(argv[x], "%u", &Info->Iterations);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'y':
                if (++x < argc)
                {
                    sscanf(argv[x], "%u", &Info->DeviceNum);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
            case 'j':
                if (++x < argc)
                {
                    sscanf(argv[x], "%lf", &Info->GPURatio);
                    printf("Using GPU Ratio %lf\n", Info->GPURatio);
                }
                else
                {
                    return CAL_FALSE;
                }
                break;
                
    
	    case 't':
		Info->Pin = argc > x + 1 ? atoi(argv[++x]) : 0;
		break;
        };
    }
    
    if (Info->UseCPU == CAL_FALSE && Info->UseGPU == CAL_FALSE) Info->UseGPU = CAL_TRUE;
    
    return CAL_TRUE;
}

int SetupUserData(caldgemm::SampleInfo &Info)
{
    timespec randtime;
    clock_gettime(CLOCK_REALTIME, &randtime);
    srand((int) (seedused = randtime.tv_nsec));
    
    if (AA) delete[] AA;
    if (BB) delete[] BB;
    if (CC) delete[] CC;
    AA = new CALdouble[(size_t) Info.m * (size_t) Info.Width];
    BB = new CALdouble[(size_t) Info.Width * (size_t) Info.n];
    CC = new CALdouble[(size_t) Info.m * (size_t) Info.n];
    
    if (mlock(AA, (size_t) Info.m * (size_t) Info.Width * sizeof(double)) ||
    mlock(BB, (size_t) Info.Width * (size_t) Info.n * sizeof(double)) ||
    mlock(CC, (size_t) Info.m * (size_t) Info.n * sizeof(double))) printf("Error locking memory\n");
    
    if (AA == NULL || BB == NULL || CC == NULL)
    {
	printf("Memory allocation error allocating matrices\n");
	return(1);
    }
    
    if (fastinit)
    {
	memset(AA, 0, (size_t) Info.m * (size_t) Info.Width * sizeof(double));
	memset(BB, 0, (size_t) Info.Width * (size_t) Info.n * sizeof(double));
	memset(CC, 0, (size_t) Info.m * (size_t) Info.n * sizeof(double));
    }
    else
    {
	for (long long int i = 0;i < (long long int) Info.m * (long long int) Info.n;i++)
        {
#ifdef TESTMODE
	    CC[i] = 0;
#else
	    CC[i] = (CALdouble) (i % 16);
#endif
	}
    
	for (CALuint y = 0; y < Info.Width; y++)
        {
    	    for (CALuint x = 0; x < Info.m; x++)
    	    {
#ifdef TESTMODE
        	AA[x * Info.Width + y] = 1;
#else
        	AA[x * Info.Width + y] = (x&1? -1.0 : 0) + (rand() / static_cast<CALdouble>(RAND_MAX + 1.0));
#endif
    	    }
    	    for (CALuint x = 0; x < Info.n; x++)
    	    {
#ifdef TESTMODE
        	BB[y * Info.n + x] = 1;
#else
        	BB[y * Info.n + x] = (x&1? -1.0 : 0) + (rand() / static_cast<CALdouble>(RAND_MAX + 1.0));
#endif
    	    }
	}
    }
    if (Info.Debug) printf("User Data Initialized\n");
    return(0);
}

bool isDoubleEqual(CALdouble a, CALdouble b)
{
    CALdouble epsilon = 1e-6;
    
    if(fabs(b) <1e-13)
	return (fabs(a-b) < epsilon);
    else
	return (fabs((a-b)/b) < epsilon);
}

int main(CALint argc, CALchar** argv)
{
    caldgemm::SampleInfo Info;
    caldgemm dgemm;

    if (!ParseCommandLine(argc, argv, &Info))
    {
        return 1;
    }
    
    if (dgemm.InitCALDGEMM(&Info))
    {
	printf("Error initializing CALDGEMM\n");
	return(1);
    }

    if (loadmatrix)
    {
	FILE* fp;
	double* a, b, c;
	double alpha, beta;
	int tmp_m, tmp_k, tmp_n;
	int Apitch, Bpitch, Cpitch;
	
	if ((fp = fopen(matrixfile, "rb")) == NULL)
	{
	    printf("Error opening matrix dump\n");
	    return(1);
	}
	fread(&a, sizeof(a), 1, fp);
	fread(&b, sizeof(b), 1, fp);
	fread(&c, sizeof(c), 1, fp);
	fread(&alpha, sizeof(alpha), 1, fp);
	fread(&beta, sizeof(beta), 1, fp);
	fread(&tmp_m, sizeof(tmp_m), 1, fp);
	fread(&tmp_k, sizeof(tmp_k), 1, fp);
	fread(&tmp_n, sizeof(tmp_n), 1, fp);
	fread(&Apitch, sizeof(Apitch), 1, fp);
	fread(&Bpitch, sizeof(Bpitch), 1, fp);
	fread(&Cpitch, sizeof(Cpitch), 1, fp);
	
	Apitch = 1536;
	
	AA = new CALdouble[(size_t) tmp_m * (size_t) Apitch];
	BB = new CALdouble[(size_t) tmp_k * (size_t) Bpitch];
	CC = new CALdouble[(size_t) tmp_m * (size_t) Cpitch];
	
	for (int i = 0;i < tmp_m;i++)
	{
	    fread(AA + i * Apitch, tmp_k, sizeof(double), fp);
	}
	for (int i = 0;i < tmp_k;i++)
	{
	    fread(BB + i * Bpitch, tmp_n, sizeof(double), fp);
	}
	fclose(fp);
	memset(CC, 0, (size_t) tmp_m * (size_t) Cpitch * sizeof(double));
	
	printf("matrix loaded: m=%d k=%d n=%d lda=%d ldb=%d ldc=%d alpha=%2.4lf beta=%2.4lf\n", tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch, alpha, beta);
	
	dgemm.RunCALDGEMM(AA, BB, CC, alpha, beta, tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch);
    }
    else
    {
	printf("Initializing Data... ");
	if (SetupUserData(Info))
	{
	    return(1);
	}
	printf("Done\n");
	
	//Initial run to negate cache effects
#ifndef TESTMODE
        if (Info.Debug == CAL_FALSE && Info.DumpMatrix == CAL_FALSE && initialrun)
        {
    	    printf("Doing initial run... ");
	    CALboolean tmpquiet = Info.Quiet;
    	    CALuint tmpiter = Info.Iterations;
    	    CALuint tmpm = Info.m, tmpn = Info.n;
    	    Info.Quiet = CAL_TRUE;
    	    Info.Iterations = 2;
    	    if (Info.m > 2 * Info.Height) Info.m = 2 * Info.Height;
    	    if (Info.n > 2 * Info.Height) Info.n = 2 * Info.Height;
    	    if (dgemm.RunCALDGEMM(AA, BB, CC, 0.0, 1.0, Info.m, Info.Width, Info.n, transa ? Info.m : Info.Width, transb ? Info.Width : Info.n, Info.n, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans))
    	    {
	        printf("Error running CALDGEMM\n");
		return(1);
	    }
	    Info.m = tmpm;
	    Info.n = tmpn;
	    Info.Quiet = tmpquiet;
	    Info.Iterations = tmpiter;
	    printf("Done\n");
	}
#endif
	dgemm.ResetTimers();
    
	printf("Running Benchmark\n");
	do
        {
#ifdef TESTMODE
	    if (dgemm.RunCALDGEMM(AA, BB, CC, 1.0, 0.0, Info.m, Info.Width, Info.n, transa ? Info.m : Info.Width, transb ? Info.Width : Info.n, Info.n, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans))
#else
	    if (dgemm.RunCALDGEMM(AA, BB, CC, 0.5, 1.0, Info.m, Info.Width, Info.n, transa ? Info.m : Info.Width, transb ? Info.Width : Info.n, Info.n, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans))
#endif
	    {
		printf("Error running CALDGEMM\n");
		return(1);
	    }
	    dgemm.ResetTimers();
	} while (benchmark && (Info.n += Info.Height) < 70000 && (Info.m += Info.Height) < 70000 && SetupUserData(Info) == 0);
    }
    
    if (verifylarge)
    {
	printf("Running verification for large matrices\n");
	srand((int) seedused);
	Info.UseGPU = CAL_FALSE;
	Info.UseCPU = CAL_TRUE;
	Info.Verify = CAL_FALSE;
	Info.Quiet = CAL_TRUE;
	dgemm.RunCALDGEMM(AA, BB, CC, -0.5, 1.0, Info.m, Info.Width, Info.n, transa ? Info.m : Info.Width, transb ? Info.Width : Info.n, Info.n, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans);
	int verifyok = 1;
	for (long long int i = 0;i < (long long int) Info.m * (long long int) Info.n;i++)
        {
	    if (!isDoubleEqual(CC[i] * 1.0, (CALdouble) (i % 16)))
	    {
		printf("Verification failed at i = %lld, m = %lld, n = %lld\n", i, i % Info.n, i / Info.n);
		verifyok = 0;
		break;
	    }
	}
	if (verifyok) printf("Verification succeeded\n");
    }
    
    dgemm.ExitCALDGEMM();

    delete[] AA;
    delete[] BB;
    delete[] CC;
    return 0;
}
