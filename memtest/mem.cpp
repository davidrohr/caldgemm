#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include <sched.h>
#include <syscall.h>
#include <unistd.h>
#include "../cmodules/timer.h"
#include "../cmodules/os_low_level_helper.h"
#include <sys/mman.h>
#include <algorithm>

//#define USE_SVM_ALLOC

#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3

#define quit(...) {fprintf(stderr, __VA_ARGS__);fprintf(stderr, "\n");return(1);}
#define DEFAULT_GPU_DATA_SIZE (2048 * 4096 * 8)
#define DEFAULT_DATA_SIZE (DEFAULT_GPU_DATA_SIZE * 2)
#define ITERATIONS_DEFAULT 16

long long int stride_size = 0;
long long int matrix_rows = 1;
long long int matrix_columns = 0;
bool emulateStrided = false;
bool copyBufferStrided = false;
bool twoSocketMapping = false;
long long int DATA_SIZE = DEFAULT_DATA_SIZE;
long long int GPU_DATA_SIZE = DEFAULT_GPU_DATA_SIZE;
int ITERATIONS = ITERATIONS_DEFAULT;

cl_mem host_pinned_mem;
void* host_ptr = NULL;
cl_kernel ocl_kernel;

static const char* opencl_error_string(int errorcode)
{
	switch (errorcode)
	{
		case CL_SUCCESS: return "Success!";
		case CL_DEVICE_NOT_FOUND: return "Device not found.";
		case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
		case CL_OUT_OF_RESOURCES: return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information not available";
		case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
		case CL_MAP_FAILURE: return "Map failure";
		case CL_INVALID_VALUE: return "Invalid value";
		case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
		case CL_INVALID_PLATFORM: return "Invalid platform";
		case CL_INVALID_DEVICE: return "Invalid device";
		case CL_INVALID_CONTEXT: return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
		case CL_INVALID_HOST_PTR: return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
		case CL_INVALID_SAMPLER: return "Invalid sampler";
		case CL_INVALID_BINARY: return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
		case CL_INVALID_PROGRAM: return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
		case CL_INVALID_KERNEL: return "Invalid kernel";
		case CL_INVALID_ARG_INDEX: return "Invalid argument index";
		case CL_INVALID_ARG_VALUE: return "Invalid argument value";
		case CL_INVALID_ARG_SIZE: return "Invalid argument size";
		case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
		case CL_INVALID_EVENT: return "Invalid event";
		case CL_INVALID_OPERATION: return "Invalid operation";
		case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL: return "Invalid mip-map level";
		default: return "Unknown Errorcode";
	}
}

static const char* sourceCode =
	"__kernel void kernelReadWrite(__global double *dest_a, ulong matrix_rows, ulong matrix_columns, ulong stride_size, ulong offset)"
	"{"
	"	__global double* dest = dest_a + offset;"
	"	for (ulong i = get_global_id(1);i < matrix_rows;i += get_global_size(1))"
	"	{"
	"		for (ulong j = get_global_id(0);j < matrix_columns / sizeof(double);j += get_global_size(0))"
	"		{"
	"			dest[i * stride_size / sizeof(double) + j]++;"
	"		}"
	"	}"
	"}"
;

template <class T, class S, class U> int doTestA (int i_strided, int i_image, int i_zero, cl_command_queue& command_queue, cl_mem& gpu_mem, void* host_ptr, bool to_gpu, T op1, S op2, U op3)
{
	size_t origin[3] = {0, 0, 0};

	if (i_zero)
	{
		if (clSetKernelArg(ocl_kernel, 0, sizeof(cl_mem), &gpu_mem) != CL_SUCCESS) quit("Error setting kernel argument");
		long long int  matrix_rows_a = i_strided ? matrix_rows : 1;
		long long int matrix_columns_a = i_strided ? matrix_columns : GPU_DATA_SIZE;
		if (clSetKernelArg(ocl_kernel, 1, sizeof(long long int), &matrix_rows_a) != CL_SUCCESS) quit("Error setting kernel argument");
		if (clSetKernelArg(ocl_kernel, 2, sizeof(long long int), &matrix_columns_a) != CL_SUCCESS) quit("Error setting kernel argument");
		if (clSetKernelArg(ocl_kernel, 3, sizeof(long long int), &stride_size) != CL_SUCCESS) quit("Error setting kernel argument");
		long long int offset = (long long int) (size_t) ((double*) host_ptr - (double*) ::host_ptr);
		if (clSetKernelArg(ocl_kernel, 4, sizeof(long long int), &offset) != CL_SUCCESS) quit("Error setting kernel argument");
		
		size_t local_size[2];
		size_t global_size[2];
		if (i_strided)
		{
			local_size[0] = local_size[1] = 16;
			global_size[0] = global_size[1] = local_size[0] * 16;
		}
		else
		{
			local_size[0] = 256;
			global_size[0] = local_size[0] * 128;
			local_size[1] = global_size[1] = 1;
		}
		if (clEnqueueNDRangeKernel(command_queue, ocl_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL) != CL_SUCCESS) quit("Error executing kernel");

		return(0);
	}
	else if (i_image)
	{
		size_t region[3] = {(i_strided ? matrix_columns : (GPU_DATA_SIZE / 4096)) / (4 * sizeof(int)), i_strided ? matrix_rows : 4096, 1};
		if (op3(command_queue, gpu_mem, CL_FALSE, origin, region, i_strided ? stride_size : 0, 0, host_ptr, 0, NULL, NULL) != CL_SUCCESS) return(1);
	}
	else if (i_strided)
	{
		size_t region_bytes[3] = {matrix_columns, matrix_rows, 1};
		if (copyBufferStrided)
		{
			cl_mem *mem_from, *mem_to;
			size_t origin_packed[3] = {0, 0, 0};
			size_t offset = ((char*) host_ptr - (char*) ::host_ptr);
			size_t origin[3] = {0, offset / stride_size, 0};
			size_t *origin_from, *origin_to;
			size_t zero_pitch = 0;
			size_t host_pitch = stride_size;
			size_t *pitch_from, *pitch_to;
			if (to_gpu)
			{
				mem_from = &host_pinned_mem;
				mem_to = &gpu_mem;
				origin_from = origin;
				origin_to = origin_packed;
				pitch_from = &host_pitch;
				pitch_to = &zero_pitch;
			}
			else
			{
				mem_from = &gpu_mem;
				mem_to = &host_pinned_mem;
				origin_from = origin_packed;
				origin_to = origin;
				pitch_from = &zero_pitch;
				pitch_to = &host_pitch;
			}
			clEnqueueCopyBufferRect(command_queue, *mem_from, *mem_to, origin_from, origin_to, region_bytes, *pitch_from, zero_pitch, *pitch_to, zero_pitch, 0, NULL, NULL);
			return(0);
		}
		else if (emulateStrided)
		{
			for(int i = 0;i < region_bytes[1];i++)
			{
				if (op2(command_queue, gpu_mem, CL_FALSE, i * region_bytes[0], region_bytes[0], (char*) host_ptr + i * stride_size, 0, NULL, NULL) != CL_SUCCESS) return(1);
			}
			return(0);
		}
		else
		{
			//return(op1(command_queue, gpu_mem, CL_FALSE, origin, origin, region_bytes, 0, 0, stride_size, 0, host_ptr, 0, NULL, NULL) != CL_SUCCESS);
			size_t offset = ((char*) host_ptr - (char*) ::host_ptr);
			size_t host_origin[3] = {0, offset / stride_size, 0};
			return(op1(command_queue, gpu_mem, CL_FALSE, origin, host_origin, region_bytes, 0, 0, stride_size, 0, ::host_ptr, 0, NULL, NULL) != CL_SUCCESS);
		}
	}
	else
	{
		return(op2(command_queue, gpu_mem, CL_FALSE, 0, GPU_DATA_SIZE, host_ptr, 0, NULL, NULL) != CL_SUCCESS);
	}
}

inline int doTest (bool doRead, int i_strided, int i_image, int i_zero, cl_command_queue& command_queue, cl_mem& gpu_mem, void* host_ptr, bool to_gpu)
{
	return(doRead == false ? doTestA(i_strided, i_image, i_zero, command_queue, gpu_mem, host_ptr, to_gpu, clEnqueueWriteBufferRect, clEnqueueWriteBuffer, clEnqueueWriteImage) :
		doTestA(i_strided, i_image, i_zero, command_queue, gpu_mem, host_ptr, to_gpu, clEnqueueReadBufferRect, clEnqueueReadBuffer, clEnqueueReadImage));
}

int InitCheckData(void* host_ptr, size_t rows, size_t columns, bool check, bool zero, bool add_one = false)
{
	size_t k = 0;
	int errors = 0;
	for (size_t i = 0;i < rows;i++)
	{
		for (size_t j = 0;j < columns + 1 - sizeof(double);j += sizeof(double))
		{
			k++;
			double* ptr = (double*) ((char*) host_ptr + stride_size * i + j);
			double val = (zero ? 0. : (double) k) + (add_one ? 1. : 0.);
			if (check)
			{
				if (*ptr != val)
				{
					errors++;
					//if (errors < 20) fprintf(stderr, "Error found: Row %lld Column %lld, Expected %f, Found %f...\n", (long long int) i, (long long int) j, val, *ptr);
				}
			}
			else
			{
				*ptr = val;
			}
		}
	}
	//if (errors) fprintf(stderr, "Errors %d, Good Entries %d\n", errors, (int) k - errors);
	return(errors ? 1 : 0);
}

int main(int argc, char** argv)
{
	HighResTimer Timer[3];
	cl_int ocl_error;
	int map_gpu = -1; //-1 = CPU
	int onlymapped = 0;
	int only_gpu = -1;
	int cpu_core = -2;
	int test_strided = 0;
	int test_image = 0;
	int test_zerocopy = 0;
	int test_aggregate = 0;
	
	int num_cpu_cores = get_number_of_cpu_cores();
	
	bool linpackTiles = false;
	int tile_height = 0, tile_width = 0, tile_num_x = 0, tile_num_y = 1;
	
	if (argc <= 1)
	{
	    fprintf(stderr, "Syntax:\n"
	    "-d [DATA_SIZE]          (Size of mapped host buffer, this can be much larger than the GPU buffer to test the limit of mappable OpenCL buffers)\n"
	    "-t [GPU_DATA_SIZE]      (Size of buffers on GPU, Size of Data transfer\n"
	    "-r [ITERATIONS]         (Number of iterations of transfer\n"
	    "-g [GPU_NUM]            (Number of GPU to map the host buffer, -1: none, -2: all)\n"
	    "-y [GPU_NUM]            (Test only one GPU)\n"
	    "-c [CPU_NUM]            (CPU core to pin process to (and allocate memory on), -1 to use interleaved memory)\n"
	    "-u                      (Allocate GPU related memory (not the data buffers) for two CPU sockets (first half of GPUs on first socket)\n"
	    "-o                      (Benchmark only the GPUs where the host buffer is mapped\n"
	    "-x                      (Test also strided transfers)\n"
	    "-s [STRIDE]             (Stride in bytes between the matrix rows)\n"
	    "-m [MATRIX_ROWS]        (Number of rows of the matrix for strided transfer - Number of columns is GPU_DATA_SIZE / MATRIX_ROWS)\n"
	    "-l                      (Test transfer of matrix tiles for linpack, supply additional parameters:)\n"
	    "                            -lh [TILE_HEIGH]\n"
	    "                            -lw [TILE_WIDTH]\n"
	    "                            -lx [NUMBER_X_TILES]\n"
	    "                            -ly [NUMBER_Y_TILES]\n"
	    "-e                      (Emulate strided transfer using multiple linear transfers)\n"
	    "-p                      (Use CopyBufferRect API for strided transfers)\n"
	    "-i                      (Test transfer to/from images)\n"
	    "-z                      (Test Zero-Copy transfer via kernel)\n"
	    "-a                      (Test aggregate bandwidth of all GPUs)\n"
	    );
	    return(1);
	}
	
	int* setPtr;
	
	for (int x = 1; x < argc; ++x)
	{
		switch(argv[x][1])
		{
		default:
			fprintf(stderr, "Invalid parameter: %s\n", argv[x]);
			return(1);
		case 's':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", &stride_size);
			break;
		case 'm':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", &matrix_rows);
			break;
		case 'y':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%ld", &only_gpu);
			break;
		case 'e':
			emulateStrided = true;
			break;
		case 'u':
			twoSocketMapping = true;
			break;
		case 'p':
			copyBufferStrided = true;
			break;
		case 'a':
			test_aggregate = true;
			break;
		case 'z':
			test_zerocopy = true;
			break;
		case 'i':
			test_image = true;
			break;
		case 'd':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", &DATA_SIZE);
			if (DATA_SIZE % 1024) DATA_SIZE += 1024 - DATA_SIZE % 1024;
			break;
		case 't':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%lld", &GPU_DATA_SIZE);
			if (GPU_DATA_SIZE % 1024) GPU_DATA_SIZE += 1024 - GPU_DATA_SIZE % 1024;
			break;
			
		case 'g':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &map_gpu);
			break;

		case 'l':
			setPtr = NULL;
			if (argv[x][2] == 0) {linpackTiles = true;continue;}
			else if (argv[x][2] == 'h') setPtr = &tile_height;
			else if (argv[x][2] == 'w') setPtr = &tile_width;
			else if (argv[x][2] == 'x') setPtr = &tile_num_x;
			else if (argv[x][2] == 'y') setPtr = &tile_num_y;
			
			if (++x >= argc) return(1);
			if (setPtr == NULL)
			{
				fprintf(stderr, "Invalid Linpack Option\n");
				return(1);
			}
			sscanf(argv[x], "%d", setPtr);
			break;
			
		case 'r':
			if (++x >= argc) return(1);
			sscanf(argv[x], "%d", &ITERATIONS);
			break;
			
		case 'c':
			if (++x >= argc) return(1);
			cpu_core = atoi(argv[x]);
			if (cpu_core == -1)
			{
				unsigned long nodemask = 0xffffff;
				syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
				fprintf(stderr, "Using interleaved memory\n");
			}
			else
			{
				cpu_set_t mask;
				CPU_ZERO(&mask);
				CPU_SET(cpu_core, &mask);
				sched_setaffinity(0, sizeof(mask), &mask);
				fprintf(stderr, "Setting CPU affinity to CPU %d\n", cpu_core);
			}
			break;
			
		case 'o':
			onlymapped = 1;
			fprintf(stderr, "Testing only GPUs where memory has been mapped\n");
			break;
		case 'x':
			test_strided = 1;
			fprintf(stderr, "Running linear and strided tests\n");
			break;
		}
	}
	
	long long int strided_host_size;
	
	if (linpackTiles)
	{
		fprintf(stderr, "Linpack Mode enabled: %d tiles of size %d x %d doubles\n", tile_num_x, tile_height, tile_width);
		if (tile_height <= 0 || tile_width <= 0 || tile_num_x <= 0 || tile_num_y <= 0)
		{
			fprintf(stderr, "Invalid Linpack Tile parameters\n");
			return(1);
		}
		test_strided = 1;
		GPU_DATA_SIZE = tile_height * tile_width * sizeof(double);
		matrix_rows = tile_height;
		matrix_columns = tile_width;
		stride_size = tile_width * tile_num_x * sizeof(double);
		strided_host_size = matrix_rows * stride_size;
		DATA_SIZE = strided_host_size * std::max(2, tile_num_y);
	}
	
	if (test_strided)
	{
		if (GPU_DATA_SIZE % matrix_rows) quit("GPU_DATA_SIZE not divisible by MATRIX_ROWS");
		matrix_columns = GPU_DATA_SIZE / matrix_rows;
		if (matrix_rows <= 0 || matrix_columns <= 0 || stride_size < matrix_columns)
		{
			fprintf(stderr, "Invalid settings for strided transfer: Matrix %lld x %lld - Stride %lld\n", matrix_rows, matrix_columns, stride_size);
			return(1);
		}
		strided_host_size = matrix_rows * stride_size;
	}
	else
	{
		strided_host_size = GPU_DATA_SIZE;
	}
	
	if (test_image)
	{
		if (GPU_DATA_SIZE % (4 * sizeof(int) * 4096) || (test_strided && matrix_columns % (4 * sizeof(int)))) //4 * int for RGBA32 (4096 for 4096 rows in image...)
		{
			fprintf(stderr, "Invalid dimensions for image test, (GPU_DATA_SIZE must be multiple of %d, matrix_columns must be multiple of %d\n", 4 * sizeof(int) * 4096, 4 * sizeof(int));
			return(1);
		}
	}
	if (strided_host_size % 1024) strided_host_size += 1024 - strided_host_size % 1024;
	if (2 * strided_host_size > DATA_SIZE) quit("DATA_SIZE too small");
	if (GPU_DATA_SIZE == 0) quit("Test data size must not be zero");

	fprintf(stderr, "Running dma-mem-bench, settings: Data Size %lld, Data Size GPU %lld, Map GPU %d, CPU Core %d, Use Only Mapped GPUs %d, Iterations %d", DATA_SIZE, GPU_DATA_SIZE, map_gpu, cpu_core, onlymapped, ITERATIONS);
	if (test_strided)
	{
		fprintf(stderr, ", Strided Test: Matrix %lld x %lld - Stride: %lld", matrix_rows, matrix_columns, stride_size);
	}
	fprintf(stderr, "\n");

	//Query platform count
	cl_uint num_platforms;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) quit("Error getting OpenCL Platform Count");
	if (num_platforms == 0) quit("No OpenCL Platform found");
	fprintf(stderr, "%d OpenCL Platforms found\n", num_platforms);
	
	//Query platforms
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	if (platforms == NULL) quit("Memory allocation error");
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) quit("Error getting OpenCL Platforms");

	for (unsigned int i_platform = 0;i_platform < num_platforms;i_platform++)
	{
		cl_platform_id platform = platforms[i_platform];
		char platform_profile[64], platform_version[64], platform_name[64], platform_vendor[64];
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_PROFILE, 64, platform_profile, NULL);
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VERSION, 64, platform_version, NULL);
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_NAME, 64, platform_name, NULL);
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VENDOR, 64, platform_vendor, NULL);

		//Query device count for this platform
		cl_uint num_devices;
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		if (num_devices == 0) quit("No OpenCL device for this platform found");
		//Query devices
		cl_device_id* devices_tmp = new cl_device_id[num_devices];
		if (devices_tmp == NULL) quit("Memory allocation error");
		if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices_tmp, NULL) != CL_SUCCESS) quit("Error getting OpenCL devices");
		
		cl_device_id device_cpu = 0;
		cl_device_id* devices = NULL;
		int num_gpus;
		bool cpu_found = false;

		for (int i = 0;i < 2;i++)
		{
			num_gpus = 0;
			for (unsigned int i_device = 0;i_device < num_devices;i_device++)
			{
				char device_vendor[64], device_name[64];
				cl_device_type device_type;
				cl_uint nbits;
				clGetDeviceInfo(devices_tmp[i_device], CL_DEVICE_NAME, 64, device_name, NULL);
				clGetDeviceInfo(devices_tmp[i_device], CL_DEVICE_VENDOR, 64, device_vendor, NULL);
				clGetDeviceInfo(devices_tmp[i_device], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
				clGetDeviceInfo(devices_tmp[i_device], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, NULL);
				if (i == 0)
				{
					fprintf(stderr, "Platform %d Device %d: %s %s (%d bits)\n", i_platform, i_device, device_vendor, device_name, nbits);
					if (device_type & CL_DEVICE_TYPE_CPU)
					{
						cpu_found = true;
						device_cpu = devices_tmp[i_device];
					}
				}
				if ((device_type & CL_DEVICE_TYPE_GPU) && !(device_type & CL_DEVICE_TYPE_CPU))
				{
					if (i)
					{
						devices[num_gpus] = devices_tmp[i_device];
					}
					num_gpus++;
				}
			}
			if (i == 0)
			{
				if (cpu_found == false) quit("No CPU device found");
				if (num_gpus == 0) quit("No GPU device with 64 bit pointers found");
				devices = new cl_device_id[num_gpus + 1];
				if (devices == NULL) quit("Memory allocation error");
			}
		}
		devices[num_gpus] = device_cpu;
		delete[] devices_tmp;
		
		if (only_gpu >= 0)
		{
			if (only_gpu >= num_gpus)
		        {
				fprintf(stderr, "Invalid device selected\n");
				return(1);
			}
			fprintf(stderr, "Testing only GPU %d\n", only_gpu);
			devices[0] = devices[only_gpu];
			devices[1] = device_cpu;
			num_gpus = 1;
		}
		
		if (test_aggregate)
		{
			if (2 * num_gpus * strided_host_size > DATA_SIZE) quit("DATA_SIZE too small for aggregate test");
		}
		
		if (map_gpu < -2 || map_gpu >= num_gpus) quit("Invalid map_gpu parameter");

		//Create OpenCL context
		cl_context context = clCreateContext(NULL, num_gpus + 1, devices, NULL, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL context");

		cl_command_queue* command_queues_read = new cl_command_queue[num_gpus];
		if (command_queues_read == NULL) quit("Memory allocation error");
		cl_command_queue* command_queues_write = new cl_command_queue[num_gpus];
		if (command_queues_write == NULL) quit("Memory allocation error");
		cl_command_queue command_queue_cpu = clCreateCommandQueue(context, device_cpu, 0, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
		for (int i = 0;i < num_gpus;i++)
		{
			command_queues_read[i] = clCreateCommandQueue(context, devices[i], 0, &ocl_error);
			if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
			command_queues_write[i] = clCreateCommandQueue(context, devices[i], 0, &ocl_error);
			if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
		}

		fprintf(stderr, "Trying to allocate buffer of %lld bytes\n", DATA_SIZE);

#ifndef USE_SVM_ALLOC
		host_pinned_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, DATA_SIZE, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error allocating pinned host memory");
		
		fprintf(stderr, "Mapping buffer using CPU\n");
		host_ptr = clEnqueueMapBuffer(command_queue_cpu, host_pinned_mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, DATA_SIZE, 0, NULL, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS || host_ptr == NULL) quit("Error mapping pinned host memory")
		
		fprintf(stderr, "Locking memory\n");
		mlock(host_ptr, DATA_SIZE);

		for (int i = 0;i < num_gpus;i++)
		{
			if (i == map_gpu || map_gpu == -2)
			{
				if (twoSocketMapping)
				{
					cpu_set_t mask;
					CPU_ZERO(&mask);
					int cpu_map_core = i < num_gpus / 2 ? 0 : num_cpu_cores - 1;
					CPU_SET(cpu_map_core, &mask);
					sched_setaffinity(0, sizeof(mask), &mask);
					fprintf(stderr, "Setting CPU affinity to CPU %d\n", cpu_map_core);
				}
				
				fprintf(stderr, "Mapping buffer using GPU %d\n", i);
				void* host_ptr_tmp = clEnqueueMapBuffer(command_queues_read[i], host_pinned_mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, DATA_SIZE, 0, NULL, NULL, &ocl_error);
				if (ocl_error != CL_SUCCESS || host_ptr_tmp == NULL) quit("Error mapping pinned host memory")
				//if (host_ptr == NULL) host_ptr = host_ptr_tmp;
				if (host_ptr != host_ptr_tmp) quit("Host pointers for GPUs differ, how can this be?");
			}
		}
#else
		host_ptr = clSVMAlloc(context, CL_MEM_READ_WRITE, DATA_SIZE, 0);
		if (host_ptr == NULL) quit("Error running clSVMAlloc");
		ocl_error = clEnqueueSVMMap(command_queue_cpu, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, host_ptr, DATA_SIZE, 0, NULL, NULL);
		if (ocl_error != CL_SUCCESS) quit("Error running clEnqueueSVMMap (%d / %s)", ocl_error, opencl_error_string(ocl_error));

		fprintf(stderr, "Locking memory\n");
		mlock(host_ptr, DATA_SIZE);
#endif


		fprintf(stderr, "Allocating and mapping done, zeroing memory\n");
		memset(host_ptr, 0, DATA_SIZE);
		fprintf(stderr, "Succeeded - Starting transfer tests (to GPU, to Host, Bidirectional, MaxFlop Limit for DGEMM with k = 2048)\n");
		
		void** host_ptr_write = new void*[num_gpus];
		void** host_ptr_read = new void*[num_gpus];
		
		for (int i = 0;i < num_gpus;i++)
		{
			host_ptr_write[i] = (void*) ((char*) host_ptr + strided_host_size * (test_aggregate * i * 2));
			host_ptr_read[i] = (void*) ((char*) host_ptr + strided_host_size * (test_aggregate * i * 2 + 1));
		}
		
		cl_program program;
		if (test_zerocopy)
		{
			//Create OpenCL program object
			program = clCreateProgramWithSource(context, 1, (const char**) &sourceCode, NULL, &ocl_error);
			if (ocl_error != CL_SUCCESS) quit("Error creating program object");
			//Compile program
			ocl_error = clBuildProgram(program, num_gpus, devices, "", NULL, NULL);
		}

		cl_mem* gpu_mem[2];
		for (int i = 0;i < 2;i++) gpu_mem[i] = new cl_mem[num_gpus];

		for (int i_strided = 0;i_strided <= test_strided;i_strided++)
		{
			for (int i_image = 0;i_image <= test_image;i_image++)
			{
				for (int i_zero = 0;i_zero <= (i_image ? 0 : test_zerocopy);i_zero++)
				{
					for (int i_dodevice = 0;i_dodevice < num_gpus + test_aggregate;i_dodevice++)
					{
						//fprintf(stderr, "Testing transfer to GPU %d\n", i_dodevice);
						if (!(onlymapped == 0 || i_dodevice == map_gpu || map_gpu == -2)) continue;
						for (int i_device = 0;i_device < num_gpus;i_device++)
						{
							if (i_device != i_dodevice && i_dodevice != num_gpus) continue;
							for (int i = 0;i < 2;i++)
							{
								if (i_image)
								{
									cl_image_format ocl_image_format;
									ocl_image_format.image_channel_order = CL_RGBA;
									ocl_image_format.image_channel_data_type = CL_UNSIGNED_INT32;

									cl_image_desc ocl_image_desc;
									ocl_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
									ocl_image_desc.image_width = (i_strided ? matrix_columns : (GPU_DATA_SIZE / 4096)) / (4 * sizeof(int));
									ocl_image_desc.image_height = i_strided ? matrix_rows : 4096;
									ocl_image_desc.image_depth = 1;
									ocl_image_desc.image_array_size = 0;
									ocl_image_desc.image_row_pitch = 0;
									ocl_image_desc.image_slice_pitch = 0;
									ocl_image_desc.num_mip_levels = 0;
									ocl_image_desc.num_samples = 0;
									ocl_image_desc.buffer = NULL;
				
									gpu_mem[i][i_device] = clCreateImage(context, CL_MEM_READ_WRITE, &ocl_image_format, &ocl_image_desc, NULL, &ocl_error);
								}
								else
								{
									gpu_mem[i][i_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, GPU_DATA_SIZE, NULL, &ocl_error);
								}
								if (ocl_error != CL_SUCCESS) quit("Error allocating GPU memory");
								if (clEnqueueMigrateMemObjects(command_queues_write[i_device], 1, &gpu_mem[i][i_device], 0, 0, NULL, NULL) != CL_SUCCESS) quit("Error migrating buffer");
							}
							
							if (i_zero)
							{
								//Create kernel
								ocl_kernel = clCreateKernel(program, "kernelReadWrite", &ocl_error);
								if (ocl_error != CL_SUCCESS) quit("Error creating kernel");
							}

							InitCheckData(host_ptr_write[i_device], i_strided ? matrix_rows : 1, i_strided ? matrix_columns : GPU_DATA_SIZE, false, false);
							InitCheckData(host_ptr_read[i_device], i_strided ? matrix_rows : 1, i_strided ? matrix_columns : GPU_DATA_SIZE, false, true);
						}

						for (int i_run = -1;i_run < 3;i_run++)
						{
							if (i_zero && (i_run >= 0 && i_run < 2)) continue;
							if (i_run != -1)
							{
								Timer[i_run].Reset();
								Timer[i_run].Start();
							}
							for (int i = 0;i < ITERATIONS;i++)
							{
								for (int i_device = 0;i_device < num_gpus;i_device++)
								{
									if (i_device != i_dodevice && i_dodevice != num_gpus) continue;
									cl_mem* gpu_mem_write_use = i_zero ? &host_pinned_mem : &gpu_mem[0][i_device];
									cl_mem* gpu_mem_read_use = i_zero ? &host_pinned_mem : &gpu_mem[1][i_device];

									if (i_run != 1) if (doTest(false, i_strided, i_image, i_zero, command_queues_write[i_device], *gpu_mem_write_use, host_ptr_write[i_device], true)) quit("Error copying data to device");

									if (i_run == -1)
									{
										//Finish writing, and use write address for reading, in order to check the results
										clFinish(command_queues_write[i_device]); 
										gpu_mem_read_use = gpu_mem_write_use;
									}

									if (i_run != 0 && i_zero == 0) if (doTest(true, i_strided, i_image, i_zero, command_queues_read[i_device], *gpu_mem_read_use, host_ptr_read[i_device], false)) quit("Error reading data from device");

									if (i_run != 1) clFlush(command_queues_write[i_device]);
									if (i_run != 0) clFlush(command_queues_read[i_device]);
								}
								if (i_run == -1) break;
							}
							for (int i_device = 0;i_device < num_gpus;i_device++)
							{
								if (i_device != i_dodevice && i_dodevice != num_gpus) continue;
								if (i_run != 1) clFinish(command_queues_write[i_device]);
								if (i_run != 0) clFinish(command_queues_read[i_device]);
								if (i_run == -1)
								{
									if (InitCheckData(i_zero ? host_ptr_write[i_device] : host_ptr_read[i_device], i_strided ? matrix_rows : 1, i_strided ? matrix_columns : GPU_DATA_SIZE, true, false, i_zero))
									{
										fprintf(stderr, "Data corruption detected during test of GPU %d\n", i_device);
									}
								}
							}
							if (i_run != -1) Timer[i_run].Stop();
						}
						
						char device[16];
						if (i_dodevice == num_gpus) strcpy(device, "all");
						else sprintf(device, "%3d", i_dodevice);
						fprintf(stderr, "Platform %d Device %s: %s/%s   -  to GPU: %6.3lf GB/s (%2.3lf s)  -  to Host: %6.3lf GB/s (%2.3lf s)  -  bidir: %6.3lf GB/s (%2.3lf s)  -  MaxFlop %9.3lf\n",
							i_platform, device, i_zero ? "ZeroCopy" : (i_image ? "   Image" : "  Buffer"), i_strided ? "Strided" : " Linear",
							i_zero ? 0 : ((double) (i_dodevice == num_gpus ? num_gpus : 1) * (double) ITERATIONS * (double) GPU_DATA_SIZE / Timer[0].GetElapsedTime() / 1e9), i_zero ? 0 : Timer[0].GetElapsedTime(),
							i_zero ? 0 : ((double) (i_dodevice == num_gpus ? num_gpus : 1) * (double) ITERATIONS * (double) GPU_DATA_SIZE / Timer[1].GetElapsedTime() / 1e9), i_zero ? 0 : Timer[1].GetElapsedTime(),
							2. * (double) (i_dodevice == num_gpus ? num_gpus : 1) * (double) ITERATIONS * (double) GPU_DATA_SIZE / Timer[2].GetElapsedTime() / 1e9, Timer[2].GetElapsedTime(),
							(double) (i_dodevice == num_gpus ? num_gpus : 1) * (double) ITERATIONS * (double) GPU_DATA_SIZE / Timer[2].GetElapsedTime() / 1e9 / sizeof(double) * 2048 * 2
						);

						for (int i_device = 0;i_device < num_gpus;i_device++)
						{
							if (i_device != i_dodevice && i_dodevice != num_gpus) continue;
							if (i_zero)
							{
								clReleaseKernel(ocl_kernel);
							}
							for (int i = 0;i < 2;i++) clReleaseMemObject(gpu_mem[i][i_device]);
						}
					}
				}
			}
		}
		
		for (int i = 0;i < 2;i++) delete[] gpu_mem[i];
		delete[] host_ptr_write;
		delete[] host_ptr_read;

		if (test_zerocopy)
		{
			clReleaseProgram(program);
		}
		
#ifndef USE_SVM_ALLOC
		for (int i = 0;i < num_gpus;i++)
		{
			if (i == map_gpu || map_gpu == -2)
			{
				if (clEnqueueUnmapMemObject(command_queues_read[i], host_pinned_mem, host_ptr, 0, NULL, NULL) != CL_SUCCESS) quit("Error unmapping pinned memory");
				clFinish(command_queues_read[i]);
			}
		}
		if (clEnqueueUnmapMemObject(command_queue_cpu, host_pinned_mem, host_ptr, 0, NULL, NULL) != CL_SUCCESS) quit("Error unmapping pinned memory");

		clReleaseMemObject(host_pinned_mem);
#else
		clEnqueueSVMUnmap(command_queue_cpu, host_ptr, 0, NULL, NULL);
		clSVMFree(context, host_ptr);
#endif
		clFinish(command_queue_cpu);
		
		clReleaseCommandQueue(command_queue_cpu);
		for (int i = 0;i < num_gpus;i++) clReleaseCommandQueue(command_queues_read[i]);
		delete[] command_queues_read;
		for (int i = 0;i < num_gpus;i++) clReleaseCommandQueue(command_queues_write[i]);
		delete[] command_queues_write;
		
		clReleaseContext(context);
		delete[] devices;
	}

	delete[] platforms;

	//Exit
	return 0;
}
