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

#ifdef CALDGEMM_88
#define CALDGEMM_84
#define CALDGEMM_48
#endif

#if defined(CALDGEMM_84) | defined(CALDGEMM_48)
#define CALDGEMM_44
#define CALDGEMM_USE_MEMEXPORT
#define CALDGEMM_TRANSPOSED_A
#ifdef CALDGEMM_TRANSPOSED_B
#undef CALDGEMM_TRANSPOSED_B
#endif
#endif

#ifdef CALDGEMM_44
#ifdef CALDGEMM_TRANSPOSED_B
#ifdef CALDGEMM_TRANSPOSED_A
#undef CALDGEMM_TRANSPOSED_A
#endif
#else
#define CALDGEMM_TRANSPOSED_A
#endif
#endif

#if defined(CALDGEMM_DIAGONAL_TEXTURE) & (!defined(CALDGEMM_44) | defined(CALDGEMM_84) | defined(CALDGEMM_48) | !defined(CALDGEMM_TRANSPOSED_A))
#undef CALDGEMM_DIAGONAL_TEXTURE
#endif

#if defined(CALDGEMM_DUAL_ENTRY) & (!defined(CALDGEMM_44) | defined(CALDGEMM_84) | defined(CALDGEMM_48) | !defined(CALDGEMM_TRANSPOSED_A))
#undef CALDGEMM_DUAL_ENTRY
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
