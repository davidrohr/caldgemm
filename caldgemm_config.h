/**
 * Compile time configuration of the CALDGEMM library.
 *
 * Copyright 2010:
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

//#define CALDGEMM_TRANSPOSED_A				//Use Kernel for transposed A Matrix
#define CALDGEMM_TRANSPOSED_B				//Use Kernel for transposed B Matrix
//#define CALDGEMM_88					//8x8 tiling (implies memexport)
//#define CALDGEMM_84					//8x4 tiling (implies memexport)
//#define CALDGEMM_48					//4x8 tiling (implies memexport)
#define CALDGEMM_44					//4x4 tiling
//#define CALDGEMM_USE_MEMEXPORT			//Use Memexport for output instead of color buffers
//#define CALDGEMM_DIAGONAL_TEXTURE			//Alternate storage format, only valid for 4x4 kernel, obsolete
#define CALDGEMM_DUAL_ENTRY				//Unroll factor of 2 for 4x4 tiling
//#define TESTMODE					//Activate Test Mode for debugging
//#define TEST_KERNEL
//#define TEST_PARAMETERS
//#define CALDGEMM_UNALIGNED_ADDRESSES
//#define CALDGEMM_UNEQUAL_PINNING			//Do not ensure good CPU core pinning
#define STD_OUT stdout					//Output for all messages
#define CALDGEMM_OUTPUT_THREADS 1			//Number of Output threads
#define CALDGEMM_OUTPUT_THREADS_SLOW 2			//Number of output threads when KeepBuffersMapped = false
#define CALDGEMM_EXTRA_OUTPUT_THREADS_LINPACK 0		//Number of additional output threads when running in linpack mode
#define RERESERVE_LINPACK_CPUS				//Use the Linpack CPU cores for DGEMM after they finished the broadcast
#define REUSE_BBUFFERS					//Allocate many BBuffers on the GPU so B is not necessarily retransferred, used for A as well
//#define NO_ASYNC_LINPACK				
//#define WASTE_MEMORY					//Allocate extra memory before and after every memory segment allocated
//#define CALDGEMM_BENCHMARK_KERNEL 1

//#define DEBUG_MSG_ALLOCATION				//Debug Messages considering GPU buffer allocation when in Debug = true
#define DEBUG_MSG_TIMED				//Add timestamps to all messages

#define CALDGEMM_44_BT_64				//64 bit DMA transfers for 4x4 B transposed kernel
#define CALDGEMM_44_BT_64_CONVERT			//Perform 64 bit DMA transfer but transform to 128 bit for kernel input
