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

//#define CALDGEMM_TRANSPOSED_A
#define CALDGEMM_TRANSPOSED_B
//#define CALDGEMM_88
//#define CALDGEMM_84
//#define CALDGEMM_48
#define CALDGEMM_44
//#define CALDGEMM_USE_MEMEXPORT
//#define CALDGEMM_DIAGONAL_TEXTURE
#define CALDGEMM_DUAL_ENTRY
//#define TESTMODE
//#define TEST_KERNEL
//#define TEST_PARAMETERS
//#define CALDGEMM_UNALIGNED_ADDRESSES
//#define CALDGEMM_UNEQUAL_PINNING
#define STD_OUT stdout
#define CALDGEMM_OUTPUT_THREADS 1
#define CALDGEMM_OUTPUT_THREADS_SLOW 2
#define CALDGEMM_EXTRA_OUTPUT_THREADS_LINPACK 0
#define RERESERVE_LINPACK_CPUS
#define REUSE_BBUFFERS
//#define NO_ASYNC_LINPACK
//#define WASTE_MEMORY
//#define CALDGEMM_BENCHMARK_KERNEL 1

//#define DEBUG_MSG_ALLOCATION