#define GET_BOOLEAN(default) ((x + 1 < argc && argv[x + 1][0] != '-') ? (atoi(argv[++x]) > 0) : (default))

for (unsigned int x = 1; x < argc; ++x)
{
	//fprintf(STD_OUT, "Parsing option %s\n", argv[x]);
	switch(argv[x][1])
	{
	case 'q':
		Config->Quiet = GET_BOOLEAN(true);
		break;
	case 'e':
		Config->Verify = GET_BOOLEAN(true);
		break;
	case 'p':
		Config->MemPolicy = GET_BOOLEAN(true);
		break;
	case 'b':
		if (argv[x][2] == 'b')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", (int*) &Config->max_bbuffers);
		}
		else
		{
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			benchmark = GET_BOOLEAN(true);
#else
			fprintf(STD_OUT, "Option %s only supported in DGEMM bench\n", argv[x]);
			return(1);
#endif
		}
		break;
	case 'u':
		Config->DumpMatrix = GET_BOOLEAN(true);
		break;
	case '*':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", (int*) &Config->ParallelDMA);
		break;
	case '[':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", (int*) &Config->GroupParallelDMA);
		break;
	case '@':
	{
		if (++x >= argc) return(1);
		if (Config->ExcludeCPUCores) free(Config->ExcludeCPUCores);
		Config->ExcludeCPUCores = NULL;
		Config->nExcludeCPUCores = 0;
		char* ptr = argv[x];
		int j = 0;
		int a = strlen(ptr);
		for (int i = 0;i <= a;i++)
		{
			if (ptr[i] == ',' || ptr[i] == ';' || ptr[i] == 0)
			{
				if (i > j)
				{
					Config->nExcludeCPUCores++;
					if (Config->nExcludeCPUCores == 1) Config->ExcludeCPUCores = (int*) malloc(sizeof(int));
					else Config->ExcludeCPUCores = (int*) realloc(Config->ExcludeCPUCores, Config->nExcludeCPUCores * sizeof(int));
					ptr[i] = 0;
					sscanf(&ptr[j], "%d", &Config->ExcludeCPUCores[Config->nExcludeCPUCores - 1]);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
					fprintf(STD_OUT, "Excluding CPU Core %d\n", Config->ExcludeCPUCores[Config->nExcludeCPUCores - 1]);
#endif
					j = i + 1;
				}
			}
		}
		break;
	}
	case '/':
	{
		if (++x >= argc) return(1);
		char* ptr = argv[x];
		int j = 0;
		int a = strlen(ptr);
		int devnum = 0;
		for (int i = 0;i <= a;i++)
		{
			if (ptr[i] == ',' || ptr[i] == ';' || ptr[i] == 0)
			{
				if (i > j)
				{
					int tmpval;
					ptr[i] = 0;
					sscanf(&ptr[j], "%d", &tmpval);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
					fprintf(STD_OUT, "GPU device %d ID %d\n", devnum, tmpval);
#endif
					j = i + 1;
					if (devnum >= (signed) max_devices)
					{
						fprintf(STD_OUT, "ERROR, Please increase max_devices\n");
						return(1);
					}
					Config->DeviceNums[devnum] = tmpval;
					devnum++;
				}
			}
		}
		break;
	}
	case 'a':
		Config->Disassemble = GET_BOOLEAN(true);
		break;
	case '9':
		Config->TabularTiming = GET_BOOLEAN(true);
		break;
	case '0':
		Config->DivideToGPU = GET_BOOLEAN(true);
		break;
	case 'X':
		if (argv[x][2] == 'b')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", (int*) &Config->ImprovedSchedulerBalance);
		}
		else
		{
			Config->ImprovedScheduler = GET_BOOLEAN(true);
		}
		break;
	case 'A':
		if (argv[x][2] == 'p')
		{
			Config->PipelinedOperation = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'q')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", (long long int*) &Config->PipelinedMidMarker);
		}		
		else
		{
			Config->AsyncDMA = GET_BOOLEAN(true);
		}
		break;
	case '.':
		Config->RepinDuringActiveWaitForEvent = GET_BOOLEAN(true);
		break;
	case '~':
		Config->RepinMainThreadAlways = GET_BOOLEAN(true);
		break;
	case ':':
		Config->NumaPinning = GET_BOOLEAN(true);
		break;
	case ',':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", (int*) &Config->SleepDuringActiveWait);
		break;
	case 'J':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", (int*) &Config->SmallTiles);
		break;
	case 'B':
		Config->KeepBuffersMapped = GET_BOOLEAN(true);
		break;
	case '%':
		Config->SkipCPUProcessing = GET_BOOLEAN(true);
		break;
	case 'C':
		if (argv[x][2] == 'a')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->AlternateLookahead);
		}
		else if (argv[x][2] == 'm')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->MinimizeCPUPart);
		}
		else
		{
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			linpack_callbacks = 2;
			Config->LinpackNodes = 2;
			Config->linpack_factorize_function = linpack_fake1;
			Config->linpack_broadcast_function = linpack_fake2;
			Config->linpack_swap_function = linpack_fake3;
#else
			fprintf(STD_OUT, "Option %s only supported in DGEMM bench\n", argv[x]);
			return(1);
#endif
		}
		break;
	case '=':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", &Config->OutputThreads);
		break;
	case 'i':
		if (argv[x][2] == 'f')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->ForceKernelVariant);
		}
		else
		{
			Config->PrintILKernel = GET_BOOLEAN(true);
		}
		break;
	case 'c':
		Config->UseCPU = GET_BOOLEAN(true);
		break;
	case 'l':
		Config->AutoHeight = GET_BOOLEAN(true);
		break;
	case 's':
		Config->DynamicSched = GET_BOOLEAN(true);
		break;
	case 'M':
		Config->ThirdPhaseDynamicRuns = !GET_BOOLEAN(true);
		break;
	case 'N':
		Config->SecondPhaseDynamicRuns = !GET_BOOLEAN(true);
		break;
	case 'S':
		Config->SlowCPU = GET_BOOLEAN(true);
		break;
	case 'g':
		Config->UseGPU = GET_BOOLEAN(true);
		break;
	case 'I':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", (int*) &Config->ImplicitDriverSync);
		break;
	case '^':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", (int*) &Config->UseDMAFetchQueue);
		break;
	case 'O':
		if (argv[x][2] == 'c')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->GPU_C);
		}
		else if (argv[x][2] == 'l')
		{
#ifdef CALDGEMM_PARAMETERS_BACKEND
			if (++x >= argc) return(1);
			kernelLib = argv[x];
#else
			if (x + 1>= argc) return(1);
			Config->AddBackendArgv(argv[x++]);
			Config->AddBackendArgv(argv[x]);
#endif
		}
		else if (argv[x][2] == 'e')
		{
			Config->NoConcurrentKernels = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'a')
		{
			Config->AsyncSideQueue = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'b')
		{
			Config->AsyncSideQueueBalance = (int) GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'r')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->AsyncDGEMMThreshold);
		}
		else if (argv[x][2] == 's')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->AsyncDTRSMThreshold);
		}
		else if (argv[x][2] == 'd')
		{
			Config->AsyncDTRSM = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'q')
		{
			Config->SimpleGPUQueuing = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'Q')
		{
			Config->AlternateSimpleQueuing = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'p')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PreallocData);
		}
		else if (argv[x][2] == 'x')
		{
			Config->CPUInContext = !GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 't')
		{
			Config->Use3rdPartyTranspose = GET_BOOLEAN(true);
		}
		else
		{
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &use_opencl_not_cal);
#else
			fprintf(STD_OUT, "Option %s only supported in DGEMM bench\n", argv[x]);
			return(1);
#endif
		}
		break;
	case 'F':
		if (argv[x][2] == 'c')
		{
#ifdef CALDGEMM_PARAMETERS_BACKEND
			allowCPUDevice = GET_BOOLEAN(true);
#else
			Config->AddBackendArgv(argv[x]);
			if (x + 1 < argc && argv[x + 1][0] != '-') Config->AddBackendArgv(argv[++x]);
#endif

		}
		else
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->OpenCLPlatform);
		}
		break;
	case 'o':
		if (++x >= argc) return(1);
		Config->DstMemory = argv[x][0];
		if (Config->DstMemory != 'c' && Config->DstMemory != 'g')
		{
			fprintf(STD_OUT, "ERROR, Invalid destination memory type\n");
			return(1);
		}
		break;
	case 'w':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%lld", (long long int*) &Config->Width);
		break;
	case 't':
		if (argv[x][2] == 's')
		{
			Config->ShowThreadPinning = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'c')
		{
			Config->ShowConfig = GET_BOOLEAN(true);
		}
		else if (argv[x][2] == 'r')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PinDeviceRuntimeThreads);
		}
		else
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PinCPU);
		}
		break;
	case 'K':
		if (argv[x][2] == 'b')
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PinBroadcastThread);
		}
		else
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &Config->PinMainThread);
		}
		break;
	case 'G':
		if (x + 1 >= argc) return(1);
		int gpuid;
		sscanf(&argv[x++][2], "%d", &gpuid);
		if ((unsigned) gpuid >= sizeof(Config->GPUMapping) / sizeof(Config->GPUMapping[0]))
		{
			fprintf(STD_OUT, "ERROR, Invalid GPU ID (%d)\n", gpuid);
			break;
		}
		sscanf(argv[x], "%d", &Config->GPUMapping[gpuid]);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
		printf("Set CPU core for GPU %d to %d\n", gpuid, Config->GPUMapping[gpuid]);
#endif
		break;
	case 'U':
		if (x + 1 >= argc) return(1);
		if (argv[x][2] == 'A')
		{
			sscanf(&argv[x++][3], "%d", &gpuid);
			if ((unsigned) gpuid >= sizeof(Config->AllocMapping) / sizeof(Config->AllocMapping[0]))
			{
				fprintf(STD_OUT, "ERROR; Invalid GPU ID (%d)\n", gpuid);
				break;
			}
			sscanf(argv[x], "%d", &Config->AllocMapping[gpuid]);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			printf("Allocating memory for GPU %d on core %d\n", gpuid, Config->AllocMapping[gpuid]);
#endif
		}
		else if (argv[x][2] == 'B')
		{
			sscanf(&argv[x++][3], "%d", &gpuid);
			if ((unsigned) gpuid >= sizeof(Config->DMAMapping) / sizeof(Config->DMAMapping[0]))
			{
				fprintf(STD_OUT, "ERROR, Invalid GPU ID (%d)\n", gpuid);
				break;
			}
			sscanf(argv[x], "%d", &Config->DMAMapping[gpuid]);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			printf("DMA Mapping for GPU %d: core %d\n", gpuid, Config->DMAMapping[gpuid]);
#endif
		}
		else
		{
			sscanf(&argv[x++][2], "%d", &gpuid);
			if ((unsigned) gpuid >= sizeof(Config->PostprocessMapping) / sizeof(Config->PostprocessMapping[0]))
			{
				fprintf(STD_OUT, "ERROR, Invalid GPU ID (%d)\n", gpuid);
				break;
			}
			sscanf(argv[x], "%d", &Config->PostprocessMapping[gpuid]);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			printf("Set CPU core for postprocessing of GPU %d to %d\n", gpuid, Config->PostprocessMapping[gpuid]);
#endif
		}
		break;
	case 'h':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%lld", (long long int*) &Config->Height);
		break;
	case 'v':
		Config->VerboseTiming = GET_BOOLEAN(true);
		break;
	case 'V':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", &Config->ThreadSaveDriver);
		break;
	case 'k':
		Config->AsyncTiming = GET_BOOLEAN(true);
		break;
	case 'd':
		Config->Debug = GET_BOOLEAN(true);
		break;
	case 'z':
		Config->MultiThread = GET_BOOLEAN(true);
		break;
	case 'Z':
		Config->MultiThreadDivide = GET_BOOLEAN(true);
		break;
	case 'r':
		if (argv[x][2] == 'r')
		{
			Config->RereserveLinpackCPU = GET_BOOLEAN(true);
		}
		else
		{
			if (++x >= argc) return(1);
			sscanf(argv[x], "%u", &Config->Iterations);
		}
		break;
	case 'y':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%u", &Config->DeviceNum);
		break;
	case 'Y':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%u", &Config->NumDevices);
		break;
	case 'j':
		if (x + 1 >= argc) return(1);
		if (argv[x][2] == 'f') sscanf(argv[++x], "%lf", &Config->GPURatioDuringFact);
		else if (argv[x][2] == 'm') sscanf(argv[++x], "%lf", &Config->GPURatioMax);
		else if (argv[x][2] == 't') sscanf(argv[++x], "%lf", &Config->GPURatioMarginTime);
		else if (argv[x][2] == 's') sscanf(argv[++x], "%lf", &Config->GPURatioMarginTimeDuringFact);
		else if (argv[x][2] == 'l') sscanf(argv[++x], "%lf", &Config->GPURatioLookaheadSizeMod);
		else if (argv[x][2] == 'p') sscanf(argv[++x], "%d", &Config->GPURatioPenalties);
		else if (argv[x][2] == 'q') sscanf(argv[++x], "%lf", &Config->GPURatioPenaltyFactor);
		else
		{
			sscanf(argv[++x], "%lf", &Config->GPURatio);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			fprintf(STD_OUT, "Using GPU Ratio %lf\n", Config->GPURatio);
#endif
		}
		break;
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
	case '?':
		PrintUsage();
		return(1);
	case 'Q':
		wait_key = GET_BOOLEAN(true);
		break;
	case '!':
		mem_page_lock = !GET_BOOLEAN(true);
		break;
	case '_':
		mem_gpu_access = GET_BOOLEAN(true);
		break;
	case ']':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", &MaxGPUTemperature);
		break;
	case '1':
		transa = GET_BOOLEAN(true);
		break;
	case '2':
		transb = GET_BOOLEAN(true);
		break;
	case 'L':
		linpackmemory = GET_BOOLEAN(true);
		break;
	case 'P':
		if (++x >= argc) return(1);
		linpackpitch = true;
		sscanf(argv[x], "%lld", (long long int*) &pitch_c);
		break;
	case '-':
		if (argv[x][2])
		{
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
			PrintUsage();
#else
			fprintf(STD_OUT, "Option %s only supported in DGEMM bench\n", argv[x]);
#endif
			return(1);
		}
		if (++x >= argc) return(1);
		Config->AsyncDMA = Config->KeepBuffersMapped = true;
		matrix_m = matrix_n = 86016;
		Config->MemPolicy = true;
		Config->MultiThread = true;
		Config->UseCPU = false;
		Config->UseGPU = false;
		sscanf(argv[x], "%d", &torture);
		iterations = torture;
		break;
	case 'T':
		mem_huge_table = GET_BOOLEAN(true);
		break;
	case '8':
		initialrun = !GET_BOOLEAN(true);
		break;
	case '7':
		verifylarge = GET_BOOLEAN(true);
		break;
	case '6':
		fprintf(STD_OUT, "Set m and n to %lld\n", (long long int) (matrix_m = matrix_n = Config->Height * atoi(argv[++x])));
		break;
	case '4':
		matrix_m = atoi(argv[++x]);
		matrix_m -= matrix_m % Config->Height;
		fprintf(STD_OUT, "Set m and n to %lld\n", (long long int) (matrix_n = matrix_m));
		break;
	case '5':
		quietbench = GET_BOOLEAN(true);
		break;
	case '3':
		alphaone = GET_BOOLEAN(true);
		break;
	case '#':
		betazero = GET_BOOLEAN(true);
		break;
	case 'E':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", &random_seed);
		break;
	case 'f':
		fastinit = GET_BOOLEAN(true);
		break;
	case 'W':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", &reduced_width);
		break;
	case 'H':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%d", &reduced_height);
		break;
	case 'x':
		if (++x >= argc) return(1);
		loadmatrix = true;
		matrixfile = argv[x];
		break;
	case 'R':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%u", &iterations);
		break;
	case 'm':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%lld", (long long int*) &matrix_m);
		break;
	case 'n':
		if (++x >= argc) return(1);
		sscanf(argv[x], "%lld", (long long int*) &matrix_n);
		break;

#endif
	default:
		fprintf(STD_OUT, "Invalid parameter: '%s'\n", argv[x]);
#ifdef CALDGEMM_PARAMETERS_BENCHMARK
		PrintUsage();
#endif
		return(1);
		break;
	};
}