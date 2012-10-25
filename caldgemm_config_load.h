/**
 * Utility header to complete configuration given in caldgemm_config.h
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

#include "caldgemm_config.h"

#ifdef CALDGEMM_COMPUTE_SHADER
#define CALDGEMM_USE_MEMEXPORT
#endif


#ifdef CALDGEMM_44_BT_64
#ifdef CALDGEMM_88
#undef CALDGEMM_88
#endif
#ifdef CALDGEMM_84
#undef CALDGEMM_84
#endif
#ifdef CALDGEMM_48
#undef CALDGEMM_48
#endif
#ifdef CALDGEMM_TRANSPOSED_A
#undef CALDGEMM_TRANSPOSED_A
#endif
#define CALDGEMM_TRANSPOSED_B
#define CALDGEMM_44
#ifndef CALDGEMM_44_BT_64_CONVERT
#define CALDGEMM_44_BT_64_KERNEL
#endif
#endif

#ifdef CALDGEMM_88
#define CALDGEMM_84
#define CALDGEMM_48
#endif

#if defined(CALDGEMM_84) | defined(CALDGEMM_48)
#define CALDGEMM_44
#define CALDGEMM_USE_MEMEXPORT
#ifndef CALDGEMM_TRANSPOSED_A
#define CALDGEMM_TRANSPOSED_A
#warning Setting CALDGEMM_TRANSPOSED_A for 8x?/?x8 CAL tiling
#endif
#ifdef CALDGEMM_TRANSPOSED_B
#warning Unsetting CALDGEMM_TRANSPOSED_B for 8x?/?x8 CAL tiling
#undef CALDGEMM_TRANSPOSED_B
#endif
#endif

#ifdef CALDGEMM_44
#ifdef CALDGEMM_TRANSPOSED_B
#ifdef CALDGEMM_TRANSPOSED_A
#warning Unsetting CALDGEMM_TRANSPOSED_A for != 8x2 CAL tiling
#undef CALDGEMM_TRANSPOSED_A
#endif
#else
#ifndef CALDGEMM_TRANSPOSED_A
#warning Setting CALDGEMM_TRANSPOSED_A for != 8x2 CAL tiling
#define CALDGEMM_TRANSPOSED_A
#endif
#endif
#endif

#if defined(CALDGEMM_DIAGONAL_TEXTURE) & (!defined(CALDGEMM_44) | defined(CALDGEMM_84) | defined(CALDGEMM_48) | !defined(CALDGEMM_TRANSPOSED_A))
#undef CALDGEMM_DIAGONAL_TEXTURE
#endif

#if defined(CALDGEMM_DUAL_ENTRY) & (!defined(CALDGEMM_44) | defined(CALDGEMM_84) | defined(CALDGEMM_48) | !defined(CALDGEMM_TRANSPOSED_A))
#undef CALDGEMM_DUAL_ENTRY
#endif

#if defined(CALDGEMM_SINGLE_BUFFER) | defined(CALDGEMM_DOUBLE_BUFFERS)
#if !defined(CALDGEMM_44) | defined(CALDGEMM_48) | defined(CALDGEMM_84) | !defined(CALDGEMM_DUAL_ENTRY) | defined(CALDGEMM_TRANSPOSED_B)
#error Invalid options for CALDGEMM_SINGLE_BUFFER/CALDGEMM_DOUBLE_BUFFERS
#endif
#endif


#if defined(CALDGEMM_48) | !defined(CALDGEMM_44)
#define TILING_Y 8
#else
#define TILING_Y 4
#endif

#if defined(CALDGEMM_84)
#define TILING_X 8
#elif defined(CALDGEMM_44)
#define TILING_X 4
#else
#define TILING_X 2
#endif

#ifdef CALDGEMM_LDAB_INC
#define CALDGEMM_LDA_INC CALDGEMM_LDAB_INC
#ifndef CALDGEMM_LDB_INC
#define CALDGEMM_LDB_INC CALDGEMM_LDAB_INC
#endif
#endif

#ifdef CALDGEMM_SHIFT_TEXTURE
#if defined(CALDGEMM_LDA_INC) & CALDGEMM_LDA_INC < CALDGEMM_SHIFT_TEXTURE
#undef CALDGEMM_LDA_INC
#endif
#if defined(CALDGEMM_LDB_INC) & CALDGEMM_LDB_INC < CALDGEMM_SHIFT_TEXTURE
#undef CALDGEMM_LDB_INC
#endif
#ifndef CALDGEMM_LDA_INC
#define CALDGEMM_LDA_INC CALDGEMM_SHIFT_TEXTURE
#endif
#ifndef CALDGEMM_LDB_INC
#define CALDGEMM_LDB_INC CALDGEMM_SHIFT_TEXTURE
#endif
#endif

#if defined(CALDGEMM_SINGLE_BUFFER_IMPROVED) & !defined(CALDGEMM_SINGLE_BUFFER)
#undef CALDGEMM_SINGLE_BUFFER
#endif

#ifdef CALDGEMM_DIVIDE_STATIC_BUFFER
#ifdef _WIN32
#define CALDGEMM_DIVBUFA ,double* tmpBuffer
#else
#define CALDGEMM_DIVBUFA ,double* __restrict__ tmpBuffer
#endif
#define CALDGEMM_DIVBUFB , tmpBuffer
#else
#define CALDGEMM_DIVBUFA
#define CALDGEMM_DIVBUFB
#endif
