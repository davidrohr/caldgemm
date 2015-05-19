#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)

#ifdef OCL_USE_SIMPLE_BUFFERS

#ifdef CALDGEMM_TRANSPOSED_B

const char *caldgemm_opencl::OCLKernelName =
OCL_KERNEL_PRE
"//KERNEL TRANSPOSED B SIMPLE BUFFERS\n"
"__kernel void oclkernel(__global double* C, __global const double* __restrict const A, __global const double* __restrict const B, int height1, int height2, int width, double alpha, double beta, int pitch, ulong offset)\n"
"{\n"
"	int i, j, k;\n"
"	for (i = get_global_id(1);i < height2;i += get_global_size(1))\n"
"	{\n"
"		for (j = get_global_id(0);j < height1;j += get_global_size(0))\n"
"		{\n"
"			double addval = 0.;\n"
#ifdef CALDGEMM_FORCE_K
"			for (k = 0;k < " qon_mxstr(CALDGEMM_FORCE_K) ";k++)\n"
#else
"			for (k = 0;k < width;k++)\n"
#endif
"			{\n"
"				addval += A[i * width + k] * B[j * width + k];\n"
"			}\n"
#ifdef CALDGEMM_ALPHA1
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + addval;\n"
#else
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + alpha * addval;\n"
#endif
"		}\n"
"	}\n"
"}\n"
;

#else

const char *caldgemm_opencl::OCLKernelName =
OCL_KERNEL_PRE
"//KERNEL TRANSPOSED A SIMPLE BUFFERS\n"
"__kernel void oclkernel(__global double* C, __global const double* __restrict const A, __global const double* __restrict const B, int height1, int height2, int width, double alpha, double beta, int pitch, ulong offset)\n"
"{\n"
"	int i, j, k;\n"
"	for (i = get_global_id(1);i < height2;i += get_global_size(1))\n"
"	{\n"
"		for (j = get_global_id(0);j < height1;j += get_global_size(0))\n"
"		{\n"
"			double addval = 0.;\n"
#ifdef CALDGEMM_FORCE_K
"			for (k = 0;k < " qon_mxstr(CALDGEMM_FORCE_K) ";k++)\n"
#else
"			for (k = 0;k < width;k++)\n"
#endif
"			{\n"
"				addval += A[k * height2 + i] * B[k * height1 + j];\n"
"			}\n"
#ifdef CALDGEMM_ALPHA1
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + addval;\n"
#else
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + alpha * addval;\n"
#endif
"		}\n"
"	}\n"
"}\n"
;

#endif


#else //OCL_USE_SIMPLE_BUFFERS


#ifdef CALDGEMM_TRANSPOSED_B

const char *caldgemm_opencl::OCLKernelName =
OCL_KERNEL_PRE
"//KERNEL TRANSPOSED B TEXTURE BUFFERS\n"
"union double_read {uint4 f; double2 d;};\n"
"__kernel void oclkernel(__global double* C, image2d_t A, image2d_t B, int height1, int height2, int width, double alpha, double beta, int pitch, ulong offset)\n"
"{\n"
"	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"	int i, j, k;\n"
"	for (i = get_global_id(1);i < height2;i += get_global_size(1))\n"
"	{\n"
"		for (j = get_global_id(0);j < height1;j += get_global_size(0))\n"
"		{\n"
"			double addval = 0.;\n"
#ifdef CALDGEMM_FORCE_K
"			for (k = 0;k < " qon_mxstr(CALDGEMM_FORCE_K) " / 2;k++)\n"
#else
"			for (k = 0;k < width / 2;k++)\n"
#endif
"			{\n"
"				float2 coord;\n"
"				union double_read tmp, tmp2;\n"
"				coord.x = k;\n"
"				coord.y = i;\n"
"				tmp.f = read_imageui(A, sampler, coord);\n"
"				coord.y = j;\n"
"				tmp2.f = read_imageui(B, sampler, coord);\n"
"				addval += tmp.d.x * tmp2.d.x + tmp.d.y * tmp2.d.y;\n"
"			}\n"
#ifdef CALDGEMM_ALPHA1
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + addval;\n"
#else
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + alpha * addval;\n"
#endif
"		}\n"
"	}\n"
"}\n"
;

#elif defined(CALDGEMM_TRANSPOSED_A)

#ifndef OCL_TILED_KERNEL

const char *caldgemm_opencl::OCLKernelName =
OCL_KERNEL_PRE
"//KERNEL TRANSPOSED A TEXTURE BUFFERS\n"
"union double_read {uint4 f; double2 d;};\n"
"__kernel void oclkernel(__global double* C, image2d_t A, image2d_t B, int height1, int height2, int width, double alpha, double beta, int pitch, ulong offset)\n"
"{\n"
"	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"	int i, j, k;\n"
"	for (i = get_global_id(1);i < height2;i += get_global_size(1))\n"
"	{\n"
"		for (j = get_global_id(0);j < height1;j += get_global_size(0))\n"
"		{\n"
"			double addval = 0.;\n"
#ifdef CALDGEMM_FORCE_K
"			for (k = 0;k < " qon_mxstr(CALDGEMM_FORCE_K) ";k++)\n"
#else
"			for (k = 0;k < width;k++)\n"
#endif
"			{\n"
"				float2 coord;\n"
"				union double_read tmp, tmp2;\n"
"				coord.x = i / 2;\n"
"				coord.y = k;\n"
"				tmp.f = read_imageui(A, sampler, coord);\n"
"				coord.x = j / 2;\n"
"				tmp2.f = read_imageui(B, sampler, coord);\n"
"				double v1 = (i & 1) ? tmp.d.y : tmp.d.x, v2 = (j & 1) ? tmp2.d.y : tmp2.d.x;\n"
"				addval += v1 * v2;\n"
"			}\n"
#ifdef CALDGEMM_ALPHA1
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + addval;\n"
#else
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + alpha * addval;\n"
#endif
"		}\n"
"	}\n"
"}\n"
;

#else

const char *caldgemm_opencl::OCLKernelName =
OCL_KERNEL_PRE
"//KERNEL TRANSPOSED A TEXTURE BUFFERS TILED\n"
"//#pragma OPENCL EXTENSION CP_FP_FMA\n"
"union double_read {uint4 f; double2 d;};\n"
"#define OCL_TILING_X " qon_mxstr(OCL_TILING_X) "\n"
"#define OCL_TILING_Y " qon_mxstr(OCL_TILING_Y) "\n"
"__kernel void oclkernel(__global double* C, image2d_t A, image2d_t B, int height1, int height2, int width, double alpha, double beta, int pitch, ulong offset)\n"
"{\n"
"	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"	int i, j, k, l, m;\n"
"	for (i = get_global_id(1) * OCL_TILING_Y;i < height2;i += get_global_size(1) * OCL_TILING_Y)\n"
"	{\n"
"		for (j = get_global_id(0) * OCL_TILING_X;j < height1;j += get_global_size(0) * OCL_TILING_X)\n"
"		{\n"
"			double addval[OCL_TILING_X][OCL_TILING_Y];\n"
"#pragma unroll\n"
"			for (k = 0;k < OCL_TILING_X;k++) for (l = 0;l < OCL_TILING_Y;l++) addval[k][l] = 0.;\n"
"#pragma unroll 1\n"
#ifdef CALDGEMM_FORCE_K
"			for (k = 0;k < " qon_mxstr(CALDGEMM_FORCE_K) ";k++)\n"
#else
"			for (k = 0;k < width;k++)\n"
#endif
"			{\n"
"				float2 coord;\n"
"				union double_read tmp[OCL_TILING_X / 2], tmp2[OCL_TILING_Y / 2];\n"
"				coord.y = k;\n"
"#pragma unroll\n"
"				for (l = 0;l < OCL_TILING_X / 2;l++)\n"
"				{\n"
"					coord.x = i / 2 + l;\n"
"					tmp[l].f = read_imageui(A, sampler, coord);\n"
"				}\n"
"				for (l = 0;l < OCL_TILING_Y / 2;l++)\n"
"				{\n"
"					coord.x = j / 2 + l;\n"
"					tmp2[l].f = read_imageui(B, sampler, coord);\n"
"				}\n"
"#pragma unroll\n"
"				for (l = 0;l < OCL_TILING_X / 2;l++)\n"
"				{\n"
"#pragma unroll\n"
"					for (m = 0;m < OCL_TILING_Y / 2;m++)\n"
"					{\n"
"						addval[2 * l][2 * m] = mad(tmp[l].d.x, tmp2[m].d.x, addval[2 * l][2 * m]);\n"
"						addval[2 * l + 1][2 * m] = mad(tmp[l].d.y, tmp2[m].d.x, addval[2 * l + 1][2 * m]);\n"
"						addval[2 * l][2 * m + 1] = mad(tmp[l].d.x, tmp2[m].d.y, addval[2 * l][2 * m + 1]);\n"
"						addval[2 * l + 1][2 * m + 1] = mad(tmp[l].d.y, tmp2[m].d.y, addval[2 * l + 1][2 * m + 1]);\n"

"					}\n"
"				}\n"
"			}\n"
"#pragma unroll\n"
"			for (k = 0;k < OCL_TILING_X;k++)\n"
"			{\n"
"#pragma unroll\n"
"				for (l = 0;l < OCL_TILING_Y;l++)\n"
"				{\n"
#ifdef CALDGEMM_ALPHA1
"					C[offset + (i + k) * pitch + j + l] = beta * C[offset + (i + k) * pitch + j + l] + addval[k][l];\n"
#else
"					C[offset + (i + k) * pitch + j + l] = beta * C[offset + (i + k) * pitch + j + l] + alpha * addval[k][l];\n"
#endif
"				}\n"
"			}\n"
"		}\n"
"	}\n"
"}\n"
;

#endif
#endif
#endif
