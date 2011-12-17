__global__ void CUDAKernelName(double* C, double* A, double* B, size_t height1, size_t height2, size_t width, double Alpha, double Beta, size_t pitch, size_t offset)
{
	for (int j = blockIdx.y * blockDim.y + threadIdx.y;j < height2;j += blockDim.y * gridDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < height1;i += blockDim.x * gridDim.x)
		{
			double addval = 0;
			for (int k = 0;k < width;k++)
			{
				addval += A[i * width + k] * B[i * width + k];
			}
			double* destptr = &C[offset + j * pitch];
			*destptr = Alpha * addval + Beta * *destptr;
		}
	}
}
