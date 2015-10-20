/**
 * This file is part of the CALDGEMM library.
 *
 * Copyright 2015:
 *  - David Rohr (drohr@jwdt.org)
 *  - Matthias Bach (bach@compeng.uni-frankfurt.de)
 *  - Matthias Kretz (kretz@compeng.uni-frankfurt.de)
 *
 * This file is part of CALDGEMM.
 *
 * CALDGEMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CALDGEMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CALDGEMM.  If not, see <http://www.gnu.org/licenses/>.
 */

__global__ void CUDAKernelName(double* C, double* A, double* B, size_t height1, size_t height2, size_t width, double Alpha, double Beta, size_t pitch)
{
	for (int j = blockIdx.y * blockDim.y + threadIdx.y;j < height2;j += blockDim.y * gridDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < height1;i += blockDim.x * gridDim.x)
		{
			double addval = 0;
#ifdef CALDGEMM_FORCE_K
			for (int k = 0;k < CALDGEMM_FORCE_K;k++)
#else
			for (int k = 0;k < width;k++)
#endif
			{
				addval += A[j * width + k] * B[i * width + k];
			}
			double* destptr = &C[j * pitch + i];
			*destptr = Alpha * addval + Beta * *destptr;
		}
	}
}
