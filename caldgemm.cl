#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)

const char *caldgemm_opencl::OCLKernelName =
OCL_KERNEL_PRE
"union double_read {uint4 f; double2 d;};\n"
"__kernel void oclkernel(__global double* C, image2d_t A, image2d_t B, int height1, int height2, int width, double alpha, double beta, int pitch, int offset)\n"
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
"			C[offset + i * pitch + j] = beta * C[offset + i * pitch + j] + alpha * addval;\n"
"		}\n"
"	}\n"
"}\n"
;
