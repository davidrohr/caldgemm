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
bool quietbench = false;
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
    fprintf(stderr, "\t-5        Quiet Benchmark mode (different from quiet caldgemm mode)\n" );
    fprintf(stderr, "\t-6  <int> Set m/n to value * height\n" );
    fprintf(stderr, "\t-4  <int> Set m/n to the closest multiple of height to value\n" );
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
            case '6':
		printf("Set m and n to %lld\n", Info->m = Info->n = Info->Height * atoi(argv[++x]));
                break;
            case '4':
        	Info->m = atoi(argv[++x]);
        	Info->m -= Info->m % Info->Height;
		printf("Set m and n to %lld\n", Info->n = Info->m);
                break;
            case '5':
		quietbench = true;
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
    
    if (!quietbench) printf("Use -? for help\n");
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
	if (!quietbench) printf("Initializing Data... ");
	if (SetupUserData(Info))
	{
	    return(1);
	}
	if (!quietbench) printf("Done\n");
	
	//Initial run to negate cache effects
#ifndef TESTMODE
        if (Info.Debug == CAL_FALSE && Info.DumpMatrix == CAL_FALSE && initialrun)
        {
    	    if (!quietbench) printf("Doing initial run... ");
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
	    if (!quietbench) printf("Done\n");
	}
#endif
	dgemm.ResetTimers();
    
	if (!quietbench) printf("Running Benchmark\n");
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
