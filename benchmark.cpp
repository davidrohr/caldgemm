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
bool alphaone = false;
bool betazero = false;
bool linpackmemory = false;
double* linpackmem = NULL;
int reduced_height = -1;
int reduced_width = -1;
int iterations = 1;
size_t pitch_a, pitch_b, pitch_c;
bool linpackpitch = false;
size_t height_a, height_b;

bool mem_page_lock = true;
bool mem_huge_table = false;
bool linpack_callbacks = false;

int torture = 0;

char* matrixfile;

long seedused;

caldgemm dgemm;

CALvoid Usage(const CALchar* name)
{
	fprintf(STD_OUT,"Usage: %s", name);
	fprintf(STD_OUT, "\t-?        Display this help information\n" );
	fprintf(STD_OUT, "\t-e        Verify Computational Correctness\n" );
	fprintf(STD_OUT, "\t-q        Supress Display Output\n" );
	fprintf(STD_OUT, "\t-a        Print the disassembled kernel image\n" );
	fprintf(STD_OUT, "\t-i        Print IL Kernel used\n" );
	fprintf(STD_OUT, "\t-o  <c|g> Specify the output location, c = CPU, g = GPU, default GPU\n" );
	fprintf(STD_OUT, "\t-h  <int> block size for matrix multiply, default 4096\n" );
	fprintf(STD_OUT, "\t-H  <int> Reduced block size for actual matrix multiply (buffer size given by -h)\n" );
	fprintf(STD_OUT, "\t-w  <int> k for matrix multiply, default 1024\n" );
	fprintf(STD_OUT, "\t-W  <int> reduced width, see H\n" );
	fprintf(STD_OUT, "\t-l        Automatically select height for good performance\n" );
	fprintf(STD_OUT, "\t-m  <int> m for matrix multiply, must be multiple of h, default 1024\n" );
	fprintf(STD_OUT, "\t-n  <int> n for matrix multiply, must be multiple of h, default 1024\n" );
	fprintf(STD_OUT, "\t-v        Verbose Synchronous Timing for Single Kernels / Transfers\n" );
	fprintf(STD_OUT, "\t-k        Print Timing of Asynchronous DGEMM Operation\n" );
	fprintf(STD_OUT, "\t-r  <int> Number of iterations to run the program (inside caldgemm)\n" );
	fprintf(STD_OUT, "\t-R  <int> Number of iterations to run the program (seperate caldgemm calls)\n" );
	fprintf(STD_OUT, "\t-y  <int> Force Device ID\n" );
	fprintf(STD_OUT, "\t-d        Enable Debug Mode\n" );
	fprintf(STD_OUT, "\t-z        Enable Multithreading\n" );
	fprintf(STD_OUT, "\t-b        Enable Benchmarking\n" );
	fprintf(STD_OUT, "\t-c        Use CPU\n" );
	fprintf(STD_OUT, "\t-g        Use GPU\n" );
	fprintf(STD_OUT, "\t-f        Fast Init (Empty Matrices)\n" );
	fprintf(STD_OUT, "\t-j  <dbl> GPU to CPU ratio\n" );
	fprintf(STD_OUT, "\t-s        Dynamic CPU GPU scheduling\n" );
	fprintf(STD_OUT, "\t-p        Interleaving Memory Policy\n" );
	fprintf(STD_OUT, "\t-u        Dump Test Matrix\n" );
	fprintf(STD_OUT, "\t-1        Transpose A Matrix\n" );
	fprintf(STD_OUT, "\t-2        Transpose B Matrix\n" );
	fprintf(STD_OUT, "\t-3        Set alpha parameter to 1.0 to test optimized kernel\n" );
	fprintf(STD_OUT, "\t-#        Set beta parameter to 0.0 to test optimized memcpy\n" );
	fprintf(STD_OUT, "\t-5        Quiet Benchmark mode (different from quiet caldgemm mode)\n" );
	fprintf(STD_OUT, "\t-6  <int> Set m/n to value * height\n" );
	fprintf(STD_OUT, "\t-4  <int> Set m/n to the closest multiple of height to value\n" );
	fprintf(STD_OUT, "\t-7        Verify Large Matrices\n" );
	fprintf(STD_OUT, "\t-8        No initial run to negate cache effects\n" );
	fprintf(STD_OUT, "\t-9        Output a table with timing information\n" );
	fprintf(STD_OUT, "\t-0        Write the output of divideBuffers directly to GPU instead of a seperate DMA transfer\n" );
	fprintf(STD_OUT, "\t-A        Do the DMA transfer to GPU asynchronously\n" );
	fprintf(STD_OUT, "\t-L        Memory Organisation like in HPL (LINPACK)\n" );
	fprintf(STD_OUT, "\t-C        Call fake LINPACK callback functions\n" );
	fprintf(STD_OUT, "\t-P  <int> LDA=LDB=LDC = vel for HPL like memory\n" );
	fprintf(STD_OUT, "\t-T        Allocate Memory using Huge Tables\n" );
	fprintf(STD_OUT, "\t-B        Keep DMA Buffers mapped during kernel execution\n" );
	fprintf(STD_OUT, "\t-x <file> Load Matrix\n" );
	fprintf(STD_OUT, "\t--  <int> Torture Test, n iterations\n" );
	fprintf(STD_OUT, "\t-*        Enable special torture kernel, extreme core stress but no memory stress\n" );
}

void linpack_fake1() {fprintf(STD_OUT, "Linpack fake 1 called\n");}
void linpack_fake2() {fprintf(STD_OUT, "Linpack fake 2 called\n");}
void linpack_fake3() {fprintf(STD_OUT, "Linpack fake 3 called\n");}

CALboolean ParseCommandLine(CALuint argc, CALchar* argv[], caldgemm::SampleInfo* Info)
{
	Info->Quiet = CAL_FALSE;
#ifndef TEST_PARAMETERS
	Info->Verify = CAL_FALSE;
	Info->MemPolicy = CAL_FALSE;
	Info->Disassemble = CAL_FALSE;
	Info->PrintILKernel = CAL_FALSE;
	Info->MultiThread = CAL_FALSE;
	//Info->DeviceNum = 0;
	//Info->Width = 1024;
	//Info->Height = 4096;
	Info->AutoHeight = CAL_FALSE;
	Info->DynamicSched = CAL_FALSE;
	Info->VerboseTiming = CAL_FALSE;
	Info->TabularTiming = CAL_FALSE;
	Info->Debug = CAL_FALSE;
	Info->m = Info->n = 4096;
	Info->Iterations = 1;
	//Info->DstMemory = 'g';
	Info->UseCPU = Info->UseGPU = CAL_FALSE;
	//Info->GPURatio = -1;
	Info->DumpMatrix = CAL_FALSE;
	Info->DivideToGPU = CAL_FALSE;
	Info->AsyncDMA = CAL_FALSE;
	Info->KeepBuffersMapped = CAL_FALSE;

	Info->linpack_factorize_function = linpack_fake1;
	Info->linpack_broadcast_function = linpack_fake2;
	Info->linpack_swap_function = linpack_fake3;
#endif

	for (CALuint x = 1; x < argc; ++x)
	{
		switch(argv[x][1])
		{
		default:
			fprintf(STD_OUT, "Invalid parameter: %s\n", argv[x]);
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
		case '0':
			Info->DivideToGPU = CAL_TRUE;
			break;
		case 'A':
			Info->AsyncDMA = CAL_TRUE;
			break;
		case 'B':
			Info->KeepBuffersMapped = CAL_TRUE;
			break;
		case 'L':
			linpackmemory = true;
			break;
		case 'C':
			linpack_callbacks = true;
			break;
		case 'P':
			if (++x < argc)
			{
				linpackpitch = true;
				sscanf(argv[x], "%lld", &pitch_c);
			}
			else
			{
				return CAL_FALSE;
			}
			break;
		case '*':
			Info->Torture = CAL_TRUE;
			break;
		case '-':
			if (++x < argc)
			{
				Info->AsyncDMA = Info->KeepBuffersMapped = CAL_TRUE;
				Info->m = Info->n = 86016;
				Info->MemPolicy = CAL_TRUE;
				Info->MultiThread = CAL_TRUE;
				Info->UseCPU = CAL_FALSE;
				Info->UseGPU = CAL_TRUE;
				sscanf(argv[x], "%d", &torture);
				iterations = torture;
			}
			else
			{
				return CAL_FALSE;
			}
			break;
		case 'T':
			mem_huge_table = true;
			break;
		case '8':
			initialrun = false;
			break;
		case '7':
			verifylarge = true;
			break;
		case '6':
			fprintf(STD_OUT, "Set m and n to %lld\n", Info->m = Info->n = Info->Height * atoi(argv[++x]));
			break;
		case '4':
			Info->m = atoi(argv[++x]);
			Info->m -= Info->m % Info->Height;
			fprintf(STD_OUT, "Set m and n to %lld\n", Info->n = Info->m);
			break;
		case '5':
			quietbench = true;
			break;
		case '3':
			alphaone = true;
			break;
		case '#':
			betazero = true;
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
					fprintf(STD_OUT, "Invalid destination memory type\n" );
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
		case 'W':
			if (++x < argc)
			{
				sscanf(argv[x], "%d", &reduced_width);
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
		case 'H':
			if (++x < argc)
			{
				sscanf(argv[x], "%d", &reduced_height);
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
		case 'k':
			Info->AsyncTiming = CAL_TRUE;
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
		case 'R':
			if (++x < argc)
			{
				sscanf(argv[x], "%u", &iterations);
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
				fprintf(STD_OUT, "Using GPU Ratio %lf\n", Info->GPURatio);
			}
			else
			{
				return CAL_FALSE;
			}
			break;
		};
	}

	if (!quietbench) fprintf(STD_OUT, "Use -? for help\n");
	if (Info->UseCPU == CAL_FALSE && Info->UseGPU == CAL_FALSE) Info->UseGPU = CAL_TRUE;

	return CAL_TRUE;
}

void SetupUserDataC(caldgemm::SampleInfo &Info)
{
	if (fastinit || torture)
	{
		memset(CC, 0, Info.m * pitch_c * sizeof(double));
	}
	else
	{
		for (size_t i = 0;i < Info.m;i++)
		{
			for (size_t j = 0;j < Info.n;j++)
			{
#ifdef TESTMODE
				CC[i * pitch_c + j] = 0;
#else
				CC[i * pitch_c + j] = (CALdouble) (i + j % 16);
#endif
			}
		}
	}
}

int SetupUserData(caldgemm::SampleInfo &Info)
{
	timespec randtime;
	clock_gettime(CLOCK_REALTIME, &randtime);
	srand((int) (seedused = randtime.tv_nsec));

	if (linpackmemory)
	{
		if (transa || transb) fprintf(STD_OUT, "WARNING: Transposed not supported in linpackmem-mode, disabling !!!\n");
		transa = transb = false;
		if (linpackmem) delete[] linpackmem;

		if (linpackpitch)
		{
			pitch_a = pitch_b = pitch_c;
		}
		else
		{
			pitch_a = pitch_b = pitch_c = Info.n + Info.Width + (Info.n + Info.Width) % 8;
		}
		linpackmem = dgemm.AllocMemory(pitch_c * (Info.m + Info.Width + 1) + 8, mem_page_lock, mem_huge_table);
		if (linpackmem == NULL) {fprintf(STD_OUT, "Memory Allocation Error\n"); return(1);}

		char* linpackmem2 = (char*) linpackmem;
		if ((size_t) linpackmem2 % 64) linpackmem2 += 64 - ((size_t) linpackmem2) % 64;
		double* linpackmem3 = (double*) linpackmem2;


		AA = linpackmem3 + Info.Width * pitch_c;
		BB = linpackmem3 + Info.Width;
		CC = linpackmem3 + Info.Width * (pitch_c + 1);
	}
	else
	{
		if (transa)
		{
			pitch_a = Info.m;
			height_a = Info.Width;
		}
		else
		{
			pitch_a = Info.Width;
			height_a = Info.m;
		}
		if (pitch_a % 8) pitch_a += (8 - pitch_a % 8);
		if (((pitch_a / 8) & 1) == 0)
		{
			pitch_a += 8;
		}
		if (transb)
		{
			pitch_b = Info.Width;
			height_b = Info.n;
		}
		else
		{
			height_b = Info.Width;
			pitch_b = Info.n;
		}
		if (pitch_b % 8) pitch_b += (8 - pitch_b % 8);
		if (((pitch_b / 8) & 1) == 0)
		{
			pitch_b += 8;
		}
		pitch_c = Info.n;
		if (pitch_c % 8) pitch_c += (8 - pitch_c % 8);
		if (Info.n % 8) fprintf(STD_OUT, "Padding 8 bytes for correct alignment of B, n = %lld, pitch = %lld\n", Info.n, pitch_b);
		if (((pitch_c / 8) & 1) == 0)
		{
			pitch_c += 8;
		}

		if (AA) dgemm.FreeMemory(AA);
		if (BB) dgemm.FreeMemory(BB);
		if (CC) dgemm.FreeMemory(CC);
		AA = dgemm.AllocMemory(height_a * pitch_a, mem_page_lock, mem_huge_table);
		BB = dgemm.AllocMemory(height_b * pitch_b, mem_page_lock, mem_huge_table);
		CC = dgemm.AllocMemory(Info.m * pitch_c, mem_page_lock, mem_huge_table);

		if (AA == NULL || BB == NULL || CC == NULL)
		{
			fprintf(STD_OUT, "Memory allocation error allocating matrices\n");
			return(1);
		}
	}

	if (fastinit)
	{
		memset(AA, 0, height_a * pitch_a * sizeof(double));
		memset(BB, 0, height_b * pitch_b * sizeof(double));
	}
	else
	{
		for (CALuint y = 0; y < pitch_a; y++)
		{
			for (CALuint x = 0; x < height_a; x++)
			{
#ifdef TESTMODE
				AA[x * pitch_a + y] = 1;
#else
				AA[x * pitch_a + y] = (x&1? -1.0 : 0) + (rand() / static_cast<CALdouble>(RAND_MAX + 1.0));
#endif
			}
		}
		for (CALuint y = 0; y < height_b; y++)
		{
			for (CALuint x = 0; x < pitch_b; x++)
			{
#ifdef TESTMODE
				BB[y * pitch_b + x] = 1;
#else
				BB[y * pitch_b + x] = (x&1? -1.0 : 0) + (rand() / static_cast<CALdouble>(RAND_MAX + 1.0));
#endif
			}
		}
	}
	if (Info.Debug) fprintf(STD_OUT, "User Data Initialized\n");
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

	if (!ParseCommandLine(argc, argv, &Info))
	{
		return 1;
	}

	if (dgemm.InitCALDGEMM(&Info))
	{
		fprintf(STD_OUT, "Error initializing CALDGEMM\n");
		return(1);
	}
	if (reduced_height != -1)
	{
		fprintf(STD_OUT, "Using partial buffers %d / %lld\n", reduced_height, Info.Height);
		Info.Height = reduced_height;
	}
	if (reduced_width != -1)
	{
		fprintf(STD_OUT, "Using partial buffer width %d / %lld\n", reduced_width, Info.Width);
		Info.Width = reduced_width;
	}

#ifndef TEST_PARAMETERS
	if (loadmatrix)
	{
		FILE* fp;
		double* a, b, c;
		double alpha, beta;
		int tmp_m, tmp_k, tmp_n;
		int Apitch, Bpitch, Cpitch;

		if ((fp = fopen(matrixfile, "rb")) == NULL)
		{
			fprintf(STD_OUT, "Error opening matrix dump\n");
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

		fprintf(STD_OUT, "matrix loaded: m=%d k=%d n=%d lda=%d ldb=%d ldc=%d alpha=%2.4lf beta=%2.4lf\n", tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch, alpha, beta);

		dgemm.RunCALDGEMM(AA, BB, CC, alpha, beta, tmp_m, tmp_k, tmp_n, Apitch, Bpitch, Cpitch);
	}
	else
	{
		if (!quietbench)
		{
			fprintf(stderr, "Initializing Data... ");
		}
		if (SetupUserData(Info))
		{
			return(1);
		}
		if (!quietbench)
		{
			fprintf(stderr, "Done\n");
		}

		//Initial run to negate cache effects
#ifndef TESTMODE
		if (Info.Debug == CAL_FALSE && Info.DumpMatrix == CAL_FALSE && initialrun && !torture)
		{
			if (!quietbench)
			{
				fprintf(stderr, "Doing initial run... ");
			}
			CALboolean tmpquiet = Info.Quiet, tmpverify = Info.Verify;
			CALuint tmpiter = Info.Iterations;
			CALuint tmpm = Info.m, tmpn = Info.n;
			Info.Quiet = CAL_TRUE;
			Info.Verify = CAL_FALSE;
			Info.Iterations = 2;
			if (Info.m > 2 * Info.Height) Info.m = 2 * Info.Height;
			if (Info.n > 2 * Info.Height) Info.n = 2 * Info.Height;
			if (dgemm.RunCALDGEMM(AA, BB, CC, alphaone ? 1.0 : 0.5, 1.0, Info.m, Info.Width, Info.n, pitch_a, pitch_b, pitch_c, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans))
			{
				fprintf(STD_OUT, "Error running CALDGEMM\nexiting\n");
				return(1);
			}
			Info.m = tmpm;
			Info.n = tmpn;
			Info.Quiet = tmpquiet;
			Info.Verify = tmpverify;
			Info.Iterations = tmpiter;
			if (!quietbench)
			{
				fprintf(stderr, "Done\n");
			}
		}
#endif
		if (!quietbench)
		{
			fprintf(stderr, "Initializing Matrix C\n");
		}
		SetupUserDataC(Info);
		dgemm.ResetTimers();
		if (!quietbench)
		{
			fprintf(stderr, "Running Benchmark\n");
		}
		do
		{
			for (int iter = 0;iter < iterations;iter++)
			{
				if (iterations > 1 && !quietbench) fprintf(STD_OUT, "\nDGEMM Call Iteration %d\n\n", iter);
#ifdef TESTMODE
				if (dgemm.RunCALDGEMM(AA, BB, CC, 1.0, 0.0, Info.m, Info.Width, pitch_a, pitch_b, pitch_c, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans))
#else
				size_t tmpn = Info.m > Info.n ? Info.m : Info.n;
				if (linpack_callbacks) Info.LinpackSwapN = &tmpn;
				if (dgemm.RunCALDGEMM(AA, BB, CC, alphaone ? 1.0 : -1.0, betazero ? 0.0 : 1.0, Info.m, Info.Width, Info.n, pitch_a, pitch_b, pitch_c, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans, linpack_callbacks))
#endif
				{
					fprintf(STD_OUT, "Error running CALDGEMM\n");
					return(1);
				}
				if (torture)
				{
					dgemm.RunCALDGEMM(AA, BB, CC, 1.0, 1.0, Info.m, Info.Width, Info.n, pitch_a, pitch_b, pitch_c, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans, linpack_callbacks);
				}
			}
		} while (benchmark && (Info.n += Info.Height) < 70000 && (Info.m += Info.Height) < 70000 && SetupUserData(Info) == 0);
		
	}
	
	if (torture)
	{
		for (size_t i = 0;i < Info.m * pitch_c;i++)
		{
			if (CC[i] > 10E-10)
			{
				fprintf(STD_OUT, "Torture Test FAILED\n");
				if (!quietbench) fprintf(STD_OUT, "Entry %lld is %lf\n", i, CC[i]);
				torture = 0;
				break;
			}
		}
		if (torture) fprintf(STD_OUT, "Torture Test PASSED (%2.3lf gflops)\n", dgemm.avggflops);
	}

	if (verifylarge)
	{
		fprintf(STD_OUT, "Running verification for large matrices\n");
		srand((int) seedused);
		Info.UseGPU = CAL_FALSE;
		Info.UseCPU = CAL_TRUE;
		Info.Verify = CAL_FALSE;
		Info.Quiet = CAL_TRUE;
		dgemm.RunCALDGEMM(AA, BB, CC, alphaone ? -1.0 : -0.5, 1.0, Info.m, Info.Width, Info.n, pitch_a, pitch_b, pitch_c, CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans);
		fprintf(STD_OUT, "CPU DGEMM Comparison run complete, comparing results\n");
		int verifyok = 1;
		for (size_t i = 0;i < Info.m * pitch_c;i++)
		{
			if (!isDoubleEqual(CC[i] * 1.0, (CALdouble) (i % 16)))
			{
				fprintf(STD_OUT, "Verification failed at i = %lld, m = %lld, n = %lld\n", i, i / pitch_c, i % pitch_c);
				verifyok = 0;
				break;
			}
		}
		if (verifyok) fprintf(STD_OUT, "Verification succeeded\n");
	}
#else //TEST_PARAMETERS
	char* mem = new char[(size_t) 40 * 1024 * 1024 * 1024];

	//CALDGEMM_dgemm (ORDER=CblasColMajor, TRANSA=CblasNoTrans, TRANSB=CblasTrans, M=4096, N=4096, K=1024, ALPHA=-1, A=0x2aab136ea040, LDA=4096, B=0x2aab15eec080, LDB=4096, BETA=1, C=0x2aab09495040, LDC=4104)
	//int RunCALDGEMM(double* A, double* B, double* C, double alpha, double beta, size_t m, size_t k, size_t n, size_t Apitch, size_t Bpitch, size_t Cpitch, CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB);
	{
		size_t tmpmem = (size_t) mem;
		fprintf(STD_OUT, "tmpmem = 0x%llx\n", tmpmem);
		tmpmem += (size_t) 1024 * 1024 * 1024;
		fprintf(STD_OUT, "tmpmem = 0x%llx\n", tmpmem);
		tmpmem -= ((size_t) tmpmem) % ((size_t) 1024 * 1024 * 1024);
		fprintf(STD_OUT, "tmpmem = 0x%llx\n", tmpmem);
		AA = (CALdouble*) tmpmem;
		tmpmem += (size_t) 10 * 1024 * 1024 * 1024;
		BB = (CALdouble*) tmpmem;
		tmpmem += (size_t) 10 * 1024 * 1024 * 1024;
		CC = (CALdouble*) tmpmem;

		AA = (CALdouble*) (((size_t) AA) | ((size_t) 0x6ea040));
		BB = (CALdouble*) (((size_t) BB) | ((size_t) 0xeec080));
		CC = (CALdouble*) (((size_t) CC) | ((size_t) 0x495040));
		double ALPHA = -1.0;
		double BETA = 1.0;
		size_t M = 3072, N = 3072, K = 1024;
		size_t APITCH = 4104, BPITCH = 3072, CPITCH = 4104;
		CBLAS_ORDER ORDER = CblasColMajor;
		CBLAS_TRANSPOSE TRANSA = CblasNoTrans, TRANSB = CblasTrans;
		fprintf(STD_OUT, "Filling Source Matrices with random data\n");
		for (int i = 0;i < APITCH * (M > K ? M : K);i++) AA[i] = i % 257;
		for (int i = 0;i < BPITCH * (N > K ? N : K);i++) BB[i] = i % 97;
		for (int i = 0;i < CPITCH * (M > N ? M : N);i++) CC[i] = i % 65537;

		fprintf(STD_OUT, "Running with caldgemm parameters: A=0x%llx, B=0x%llx, C=0x%llx, ALPHA=%2.4lf, BETA=%2.4lf, m=%lld, k=%lld, n=%lld, Apitch=0x%llx, Bpitch=0x%llx, Cpitch=0x%llx, ColMajor=%d, TransA=%d, TransB=%d\n", AA, BB, CC, ALPHA, BETA, M, K, N, APITCH, BPITCH, CPITCH, (int) (ORDER == CblasColMajor), (int) (TRANSA == CblasTrans), (int) (TRANSB == CblasTrans));
		dgemm.RunCALDGEMM(AA, BB, CC, ALPHA, BETA, M, K, N, APITCH, BPITCH, CPITCH, ORDER, TRANSA, TRANSB);
		fprintf(STD_OUT, "Caldgemm run complete\n");

		delete[] mem;
	}
#endif //TEST_PARAMETERS

	dgemm.ExitCALDGEMM();

#ifndef TEST_PARAMETERS
	if (linpackmemory)
	{
		dgemm.FreeMemory(linpackmem);
	}
	else
	{
		dgemm.FreeMemory(AA);
		dgemm.FreeMemory(BB);
		dgemm.FreeMemory(CC);
	}
#endif
	return 0;
}
