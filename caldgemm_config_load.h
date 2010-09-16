/* ============================================================

The source code is property of the Frankfurt Institute for Advanced Studies (FIAS).
None of the material may be copied, reproduced, distributed, republished, downloaded,
displayed, posted or transmitted in any form or by any means, including, but not
limited to, electronic, mechanical, photocopying, recording, or otherwise,
without the prior written permission of FIAS.

Authors:
David Rohr (drohr@jwdt.org)
Matthias Bach (bach@compeng.uni-frankfurt.de)
Matthias Kretz (kretz@compeng.uni-frankfurt.de)

============================================================ */

#include "caldgemm_config.h"

#ifdef CALDGEMM_88
#define CALDGEMM_84
#endif

#if defined(CALDGEMM_84)
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

#if defined(CALDGEMM_DIAGONAL_TEXTURE) & (!defined(CALDGEMM_44) | defined(CALDGEMM_88) | !defined(CALDGEMM_TRANSPOSED_A))
#undef CALDGEMM_DIAGONAL_TEXTURE
#endif

#if defined(CALDGEMM_88) | !defined(CALDGEMM_44)
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
