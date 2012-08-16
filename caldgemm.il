/**
 * GPU code of the CALDGEMM library.
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
 *
 */

#if defined(CALDGEMM_88) | defined(CALDGEMM_DOUBLE_BUFFERS)
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_A_3 "2"
#define CALDGEMM_INPUT_A_4 "3"
#define CALDGEMM_INPUT_B_1 "4"
#define CALDGEMM_INPUT_B_2 "5"
#define CALDGEMM_INPUT_B_3 "6"
#define CALDGEMM_INPUT_B_4 "7"
#elif defined(CALDGEMM_84)
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_B_1 "2"
#define CALDGEMM_INPUT_B_2 "3"
#define CALDGEMM_INPUT_B_3 "4"
#define CALDGEMM_INPUT_B_4 "5"
#elif defined(CALDGEMM_48)
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_A_3 "2"
#define CALDGEMM_INPUT_A_4 "3"
#define CALDGEMM_INPUT_B_1 "4"
#define CALDGEMM_INPUT_B_2 "5"
#elif defined(CALDGEMM_TRANSPOSED_A) | defined(CALDGEMM_44)
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_B_1 "2"
#define CALDGEMM_INPUT_B_2 "3"
#else
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_A_3 "2"
#define CALDGEMM_INPUT_A_4 "3"
#define CALDGEMM_INPUT_A_5 "4"
#define CALDGEMM_INPUT_A_6 "5"
#define CALDGEMM_INPUT_A_7 "6"
#define CALDGEMM_INPUT_A_8 "7"
#define CALDGEMM_INPUT_B_1 "8"
#define CALDGEMM_INPUT_B_2 "9"
#endif

#define mstr(a) #a
#define mxstr(a) mstr(a)

#define MMUL22(o1_12, a1_1, a1_2, b1_1, b1_2, b2_1, b2_2) \
"    dmad r12.xy, " mxstr(a1_2) ", " mxstr(b2_1) ", " mxstr(o1_12.xy) "\n" \
"    dmad r12.zw, " mxstr(a1_2) ", " mxstr(b2_2) ", " mxstr(o1_12.zw) "\n" \
"    dmad " mxstr(o1_12.zw) ", " mxstr(a1_1) ", " mxstr(b1_2) ", r12.zw\n" \
"    dmad " mxstr(o1_12.xy) ", " mxstr(a1_1) ", " mxstr(b1_1) ", r12.xy\n"

#define MMUL82(o1_12, o2_12, o3_12, o4_12, o5_12, o6_12, o7_12, o8_12, a1_1, a1_2, a2_1, a2_2, a3_1, a3_2, a4_1, a4_2, a5_1, a5_2, a6_1, a6_2, a7_1, a7_2, a8_1, a8_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o1_12, a1_1, a1_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o2_12, a2_1, a2_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o3_12, a3_1, a3_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o4_12, a4_1, a4_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o5_12, a5_1, a5_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o6_12, a6_1, a6_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o7_12, a7_1, a7_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o8_12, a8_1, a8_2, b1_1, b1_2, b2_1, b2_2)

#define MMUL441(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"dmad " mxstr(o1_12.xy) ", " mxstr(a1) ", " mxstr(b1) ", " mxstr(o1_12.xy) "\n" \
"dmad " mxstr(o1_12.zw) ", " mxstr(a1) ", " mxstr(b2) ", " mxstr(o1_12.zw) "\n" \
"dmad " mxstr(o1_34.xy) ", " mxstr(a1) ", " mxstr(b3) ", " mxstr(o1_34.xy) "\n" \
"dmad " mxstr(o1_34.zw) ", " mxstr(a1) ", " mxstr(b4) ", " mxstr(o1_34.zw) "\n" \
"dmad " mxstr(o2_12.xy) ", " mxstr(a2) ", " mxstr(b1) ", " mxstr(o2_12.xy) "\n" \
"dmad " mxstr(o2_12.zw) ", " mxstr(a2) ", " mxstr(b2) ", " mxstr(o2_12.zw) "\n" \
"dmad " mxstr(o2_34.xy) ", " mxstr(a2) ", " mxstr(b3) ", " mxstr(o2_34.xy) "\n" \
"dmad " mxstr(o2_34.zw) ", " mxstr(a2) ", " mxstr(b4) ", " mxstr(o2_34.zw) "\n" \
"dmad " mxstr(o3_12.xy) ", " mxstr(a3) ", " mxstr(b1) ", " mxstr(o3_12.xy) "\n" \
"dmad " mxstr(o3_12.zw) ", " mxstr(a3) ", " mxstr(b2) ", " mxstr(o3_12.zw) "\n" \
"dmad " mxstr(o3_34.xy) ", " mxstr(a3) ", " mxstr(b3) ", " mxstr(o3_34.xy) "\n" \
"dmad " mxstr(o3_34.zw) ", " mxstr(a3) ", " mxstr(b4) ", " mxstr(o3_34.zw) "\n" \
"dmad " mxstr(o4_12.xy) ", " mxstr(a4) ", " mxstr(b1) ", " mxstr(o4_12.xy) "\n" \
"dmad " mxstr(o4_12.zw) ", " mxstr(a4) ", " mxstr(b2) ", " mxstr(o4_12.zw) "\n" \
"dmad " mxstr(o4_34.xy) ", " mxstr(a4) ", " mxstr(b3) ", " mxstr(o4_34.xy) "\n" \
"dmad " mxstr(o4_34.zw) ", " mxstr(a4) ", " mxstr(b4) ", " mxstr(o4_34.zw) "\n"

#if defined(CALDGEMM_BGEMM)
#define MMUL441_S1(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"ior " mxstr(r100.x) ", " mxstr(a1) ", " mxstr(b1) "\nixor " mxstr(o1_12.x) ", r100.x, " mxstr(o1_12.x) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a1) ", " mxstr(b2) "\nixor " mxstr(o1_12.z) ", r100.y, " mxstr(o1_12.z) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a1) ", " mxstr(b3) "\nixor " mxstr(o1_34.x) ", r100.z, " mxstr(o1_34.x) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a1) ", " mxstr(b4) "\nixor " mxstr(o1_34.x) ", r100.w, " mxstr(o1_34.z) "\n" \
"ior " mxstr(r100.x) ", " mxstr(a2) ", " mxstr(b1) "\nixor " mxstr(o2_12.x) ", r100.x, " mxstr(o2_12.x) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a2) ", " mxstr(b2) "\nixor " mxstr(o2_12.z) ", r100.y, " mxstr(o2_12.z) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a2) ", " mxstr(b3) "\nixor " mxstr(o2_34.x) ", r100.z, " mxstr(o2_34.x) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a2) ", " mxstr(b4) "\nixor " mxstr(o2_34.z) ", r100.w, " mxstr(o2_34.z) "\n" \
"ior " mxstr(r100.x) ", " mxstr(a3) ", " mxstr(b1) "\nixor " mxstr(o3_12.x) ", r100.x, " mxstr(o3_12.x) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a3) ", " mxstr(b2) "\nixor " mxstr(o3_12.z) ", r100.y, " mxstr(o3_12.z) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a3) ", " mxstr(b3) "\nixor " mxstr(o3_34.x) ", r100.z, " mxstr(o3_34.x) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a3) ", " mxstr(b4) "\nixor " mxstr(o3_34.z) ", r100.w, " mxstr(o3_34.z) "\n" \
"ior " mxstr(r100.x) ", " mxstr(a4) ", " mxstr(b1) "\nixor " mxstr(o4_12.x) ", r100.x, " mxstr(o4_12.x) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a4) ", " mxstr(b2) "\nixor " mxstr(o4_12.z) ", r100.y, " mxstr(o4_12.z) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a4) ", " mxstr(b3) "\nixor " mxstr(o4_34.x) ", r100.z, " mxstr(o4_34.x) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a4) ", " mxstr(b4) "\nixor " mxstr(o4_34.x) ", r100.w, " mxstr(o4_34.z) "\n"

#define MMUL441_S2(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"ior " mxstr(r100.x) ", " mxstr(a1) ", " mxstr(b1) "\nixor " mxstr(o1_12.y) ", r100.x, " mxstr(o1_12.y) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a1) ", " mxstr(b2) "\nixor " mxstr(o1_12.w) ", r100.y, " mxstr(o1_12.w) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a1) ", " mxstr(b3) "\nixor " mxstr(o1_34.y) ", r100.z, " mxstr(o1_34.y) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a1) ", " mxstr(b4) "\nixor " mxstr(o1_34.w) ", r100.w, " mxstr(o1_34.w) "\n" \
"ior " mxstr(r100.x) ", " mxstr(a2) ", " mxstr(b1) "\nixor " mxstr(o2_12.y) ", r100.x, " mxstr(o2_12.y) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a2) ", " mxstr(b2) "\nixor " mxstr(o2_12.w) ", r100.y, " mxstr(o2_12.w) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a2) ", " mxstr(b3) "\nixor " mxstr(o2_34.y) ", r100.z, " mxstr(o2_34.y) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a2) ", " mxstr(b4) "\nixor " mxstr(o2_34.w) ", r100.w, " mxstr(o2_34.w) "\n" \
"ior " mxstr(r100.x) ", " mxstr(a3) ", " mxstr(b1) "\nixor " mxstr(o3_12.y) ", r100.x, " mxstr(o3_12.y) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a3) ", " mxstr(b2) "\nixor " mxstr(o3_12.w) ", r100.y, " mxstr(o3_12.w) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a3) ", " mxstr(b3) "\nixor " mxstr(o3_34.y) ", r100.z, " mxstr(o3_34.y) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a3) ", " mxstr(b4) "\nixor " mxstr(o3_34.w) ", r100.w, " mxstr(o3_34.w) "\n" \
"ior " mxstr(r100.x) ", " mxstr(a4) ", " mxstr(b1) "\nixor " mxstr(o4_12.y) ", r100.x, " mxstr(o4_12.y) "\n" \
"ior " mxstr(r100.y) ", " mxstr(a4) ", " mxstr(b2) "\nixor " mxstr(o4_12.w) ", r100.y, " mxstr(o4_12.w) "\n" \
"ior " mxstr(r100.z) ", " mxstr(a4) ", " mxstr(b3) "\nixor " mxstr(o4_34.y) ", r100.z, " mxstr(o4_34.y) "\n" \
"ior " mxstr(r100.w) ", " mxstr(a4) ", " mxstr(b4) "\nixor " mxstr(o4_34.w) ", r100.w, " mxstr(o4_34.w) "\n"

#elif defined(CALDGEMM_IGEMM)
#define MMUL441_S1(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"imad " mxstr(o1_12.x) ", " mxstr(a1) ", " mxstr(b1) ", " mxstr(o1_12.x) "\n" \
"imad " mxstr(o1_12.z) ", " mxstr(a1) ", " mxstr(b2) ", " mxstr(o1_12.z) "\n" \
"imad " mxstr(o1_34.x) ", " mxstr(a1) ", " mxstr(b3) ", " mxstr(o1_34.x) "\n" \
"imad " mxstr(o1_34.z) ", " mxstr(a1) ", " mxstr(b4) ", " mxstr(o1_34.z) "\n" \
"imad " mxstr(o2_12.x) ", " mxstr(a2) ", " mxstr(b1) ", " mxstr(o2_12.x) "\n" \
"imad " mxstr(o2_12.z) ", " mxstr(a2) ", " mxstr(b2) ", " mxstr(o2_12.z) "\n" \
"imad " mxstr(o2_34.x) ", " mxstr(a2) ", " mxstr(b3) ", " mxstr(o2_34.x) "\n" \
"imad " mxstr(o2_34.z) ", " mxstr(a2) ", " mxstr(b4) ", " mxstr(o2_34.z) "\n" \
"imad " mxstr(o3_12.x) ", " mxstr(a3) ", " mxstr(b1) ", " mxstr(o3_12.x) "\n" \
"imad " mxstr(o3_12.z) ", " mxstr(a3) ", " mxstr(b2) ", " mxstr(o3_12.z) "\n" \
"imad " mxstr(o3_34.x) ", " mxstr(a3) ", " mxstr(b3) ", " mxstr(o3_34.x) "\n" \
"imad " mxstr(o3_34.z) ", " mxstr(a3) ", " mxstr(b4) ", " mxstr(o3_34.z) "\n" \
"imad " mxstr(o4_12.x) ", " mxstr(a4) ", " mxstr(b1) ", " mxstr(o4_12.x) "\n" \
"imad " mxstr(o4_12.z) ", " mxstr(a4) ", " mxstr(b2) ", " mxstr(o4_12.z) "\n" \
"imad " mxstr(o4_34.x) ", " mxstr(a4) ", " mxstr(b3) ", " mxstr(o4_34.x) "\n" \
"imad " mxstr(o4_34.z) ", " mxstr(a4) ", " mxstr(b4) ", " mxstr(o4_34.z) "\n"

#define MMUL441_S2(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"imad " mxstr(o1_12.y) ", " mxstr(a1) ", " mxstr(b1) ", " mxstr(o1_12.y) "\n" \
"imad " mxstr(o1_12.w) ", " mxstr(a1) ", " mxstr(b2) ", " mxstr(o1_12.w) "\n" \
"imad " mxstr(o1_34.y) ", " mxstr(a1) ", " mxstr(b3) ", " mxstr(o1_34.y) "\n" \
"imad " mxstr(o1_34.w) ", " mxstr(a1) ", " mxstr(b4) ", " mxstr(o1_34.w) "\n" \
"imad " mxstr(o2_12.y) ", " mxstr(a2) ", " mxstr(b1) ", " mxstr(o2_12.y) "\n" \
"imad " mxstr(o2_12.w) ", " mxstr(a2) ", " mxstr(b2) ", " mxstr(o2_12.w) "\n" \
"imad " mxstr(o2_34.y) ", " mxstr(a2) ", " mxstr(b3) ", " mxstr(o2_34.y) "\n" \
"imad " mxstr(o2_34.w) ", " mxstr(a2) ", " mxstr(b4) ", " mxstr(o2_34.w) "\n" \
"imad " mxstr(o3_12.y) ", " mxstr(a3) ", " mxstr(b1) ", " mxstr(o3_12.y) "\n" \
"imad " mxstr(o3_12.w) ", " mxstr(a3) ", " mxstr(b2) ", " mxstr(o3_12.w) "\n" \
"imad " mxstr(o3_34.y) ", " mxstr(a3) ", " mxstr(b3) ", " mxstr(o3_34.y) "\n" \
"imad " mxstr(o3_34.w) ", " mxstr(a3) ", " mxstr(b4) ", " mxstr(o3_34.w) "\n" \
"imad " mxstr(o4_12.y) ", " mxstr(a4) ", " mxstr(b1) ", " mxstr(o4_12.y) "\n" \
"imad " mxstr(o4_12.w) ", " mxstr(a4) ", " mxstr(b2) ", " mxstr(o4_12.w) "\n" \
"imad " mxstr(o4_34.y) ", " mxstr(a4) ", " mxstr(b3) ", " mxstr(o4_34.y) "\n" \
"imad " mxstr(o4_34.w) ", " mxstr(a4) ", " mxstr(b4) ", " mxstr(o4_34.w) "\n"
#else //CALDGEMM_IGEMM
#define MMUL441_S1(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"mad " mxstr(o1_12.x) ", " mxstr(a1) ", " mxstr(b1) ", " mxstr(o1_12.x) "\n" \
"mad " mxstr(o1_12.z) ", " mxstr(a1) ", " mxstr(b2) ", " mxstr(o1_12.z) "\n" \
"mad " mxstr(o1_34.x) ", " mxstr(a1) ", " mxstr(b3) ", " mxstr(o1_34.x) "\n" \
"mad " mxstr(o1_34.z) ", " mxstr(a1) ", " mxstr(b4) ", " mxstr(o1_34.z) "\n" \
"mad " mxstr(o2_12.x) ", " mxstr(a2) ", " mxstr(b1) ", " mxstr(o2_12.x) "\n" \
"mad " mxstr(o2_12.z) ", " mxstr(a2) ", " mxstr(b2) ", " mxstr(o2_12.z) "\n" \
"mad " mxstr(o2_34.x) ", " mxstr(a2) ", " mxstr(b3) ", " mxstr(o2_34.x) "\n" \
"mad " mxstr(o2_34.z) ", " mxstr(a2) ", " mxstr(b4) ", " mxstr(o2_34.z) "\n" \
"mad " mxstr(o3_12.x) ", " mxstr(a3) ", " mxstr(b1) ", " mxstr(o3_12.x) "\n" \
"mad " mxstr(o3_12.z) ", " mxstr(a3) ", " mxstr(b2) ", " mxstr(o3_12.z) "\n" \
"mad " mxstr(o3_34.x) ", " mxstr(a3) ", " mxstr(b3) ", " mxstr(o3_34.x) "\n" \
"mad " mxstr(o3_34.z) ", " mxstr(a3) ", " mxstr(b4) ", " mxstr(o3_34.z) "\n" \
"mad " mxstr(o4_12.x) ", " mxstr(a4) ", " mxstr(b1) ", " mxstr(o4_12.x) "\n" \
"mad " mxstr(o4_12.z) ", " mxstr(a4) ", " mxstr(b2) ", " mxstr(o4_12.z) "\n" \
"mad " mxstr(o4_34.x) ", " mxstr(a4) ", " mxstr(b3) ", " mxstr(o4_34.x) "\n" \
"mad " mxstr(o4_34.z) ", " mxstr(a4) ", " mxstr(b4) ", " mxstr(o4_34.z) "\n"

#define MMUL441_S2(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1, a2, a3, a4, b1, b2, b3, b4) \
"mad " mxstr(o1_12.y) ", " mxstr(a1) ", " mxstr(b1) ", " mxstr(o1_12.y) "\n" \
"mad " mxstr(o1_12.w) ", " mxstr(a1) ", " mxstr(b2) ", " mxstr(o1_12.w) "\n" \
"mad " mxstr(o1_34.y) ", " mxstr(a1) ", " mxstr(b3) ", " mxstr(o1_34.y) "\n" \
"mad " mxstr(o1_34.w) ", " mxstr(a1) ", " mxstr(b4) ", " mxstr(o1_34.w) "\n" \
"mad " mxstr(o2_12.y) ", " mxstr(a2) ", " mxstr(b1) ", " mxstr(o2_12.y) "\n" \
"mad " mxstr(o2_12.w) ", " mxstr(a2) ", " mxstr(b2) ", " mxstr(o2_12.w) "\n" \
"mad " mxstr(o2_34.y) ", " mxstr(a2) ", " mxstr(b3) ", " mxstr(o2_34.y) "\n" \
"mad " mxstr(o2_34.w) ", " mxstr(a2) ", " mxstr(b4) ", " mxstr(o2_34.w) "\n" \
"mad " mxstr(o3_12.y) ", " mxstr(a3) ", " mxstr(b1) ", " mxstr(o3_12.y) "\n" \
"mad " mxstr(o3_12.w) ", " mxstr(a3) ", " mxstr(b2) ", " mxstr(o3_12.w) "\n" \
"mad " mxstr(o3_34.y) ", " mxstr(a3) ", " mxstr(b3) ", " mxstr(o3_34.y) "\n" \
"mad " mxstr(o3_34.w) ", " mxstr(a3) ", " mxstr(b4) ", " mxstr(o3_34.w) "\n" \
"mad " mxstr(o4_12.y) ", " mxstr(a4) ", " mxstr(b1) ", " mxstr(o4_12.y) "\n" \
"mad " mxstr(o4_12.w) ", " mxstr(a4) ", " mxstr(b2) ", " mxstr(o4_12.w) "\n" \
"mad " mxstr(o4_34.y) ", " mxstr(a4) ", " mxstr(b3) ", " mxstr(o4_34.y) "\n" \
"mad " mxstr(o4_34.w) ", " mxstr(a4) ", " mxstr(b4) ", " mxstr(o4_34.w) "\n"
#endif //CALDGEMM_IGEMM

#define MUL8(x1, x2, x3, x4, x5, x6, x7, x8, factor) \
"dmul " mxstr(x1) ".xy, " mxstr(x1) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x1) ".zw, " mxstr(x1) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x2) ".xy, " mxstr(x2) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x2) ".zw, " mxstr(x2) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x3) ".xy, " mxstr(x3) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x3) ".zw, " mxstr(x3) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x4) ".xy, " mxstr(x4) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x4) ".zw, " mxstr(x4) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x5) ".xy, " mxstr(x5) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x5) ".zw, " mxstr(x5) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x6) ".xy, " mxstr(x6) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x6) ".zw, " mxstr(x6) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x7) ".xy, " mxstr(x7) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x7) ".zw, " mxstr(x7) ".zw, " mxstr(factor) "\n" \
"dmul " mxstr(x8) ".xy, " mxstr(x8) ".xy, " mxstr(factor) "\n" \
"dmul " mxstr(x8) ".zw, " mxstr(x8) ".zw, " mxstr(factor) "\n"

#ifdef CALDGEMM_TRANSPOSED_B
#define MMUL_N(o1, o2, o3, o4, o5, o6, o7, o8, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2) \
MMUL82(o1, o2, o3, o4, o5, o6, o7, o8, a1.xy, a1.zw, a2.xy, a2.zw, a3.xy, a3.zw, a4.xy, a4.zw, a5.xy, a5.zw, a6.xy, a6.zw, a7.xy, a7.zw, a8.xy, a8.zw, b1.xy, b2.xy, b1.zw, b2.zw)
#define MMUL_T(o1, o2, o3, o4, o5, o6, o7, o8, a1_12, a1_34, a1_56, a1_78, a2_12, a2_34, a2_56, a2_78, b1, b2) \
MMUL82(o1, o2, o3, o4, o5, o6, o7, o8, a1_12.xy, a2_12.xy, a1_12.zw, a2_12.zw, a1_34.xy, a2_34.xy, a1_34.zw, a2_34.zw, a1_56.xy, a2_56.xy, a1_56.zw, a2_56.zw, a1_78.xy, a2_78.xy, a1_78.zw, a2_78.zw, b1.xy, b2.xy, b1.zw, b2.zw)
#else
#define MMUL_N(o1, o2, o3, o4, o5, o6, o7, o8, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2) \
MMUL82(o1, o2, o3, o4, o5, o6, o7, o8, a1.xy, a1.zw, a2.xy, a2.zw, a3.xy, a3.zw, a4.xy, a4.zw, a5.xy, a5.zw, a6.xy, a6.zw, a7.xy, a7.zw, a8.xy, a8.zw, b1.xy, b1.zw, b2.xy, b2.zw)
#define MMUL_T(o1, o2, o3, o4, o5, o6, o7, o8, a1_12, a1_34, a1_56, a1_78, a2_12, a2_34, a2_56, a2_78, b1, b2) \
MMUL82(o1, o2, o3, o4, o5, o6, o7, o8, a1_12.xy, a2_12.xy, a1_12.zw, a2_12.zw, a1_34.xy, a2_34.xy, a1_34.zw, a2_34.zw, a1_56.xy, a2_56.xy, a1_56.zw, a2_56.zw, a1_78.xy, a2_78.xy, a1_78.zw, a2_78.zw, b1.xy, b1.zw, b2.xy, b2.zw)
#endif

#ifdef CALDGEMM_88

#define SAMPLE_A(a1, a2, a3, a4, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_3")_sampler("CALDGEMM_INPUT_A_3") " mxstr(a3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_4")_sampler("CALDGEMM_INPUT_A_4") " mxstr(a4) ", " mxstr(pos) "\n" /* normalized k, y pos */

#define SAMPLE_B(b1, b2, b3, b4, pos) \
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") " mxstr(b1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") " mxstr(b2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_3")_sampler("CALDGEMM_INPUT_B_3") " mxstr(b3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_4")_sampler("CALDGEMM_INPUT_B_4") " mxstr(b4) ", " mxstr(pos) "\n" /* normalized k, y pos */

#elif defined(CALDGEMM_84) //CALDGEMM_88

#define SAMPLE_A(a1, a2, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */

#define SAMPLE_B(b1, b2, b3, b4, pos) \
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") " mxstr(b1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") " mxstr(b2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_3")_sampler("CALDGEMM_INPUT_B_3") " mxstr(b3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_4")_sampler("CALDGEMM_INPUT_B_4") " mxstr(b4) ", " mxstr(pos) "\n" /* normalized k, y pos */

#elif defined(CALDGEMM_48) //CALDGEMM_88

#define SAMPLE_A(a1, a2, a3, a4, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_3")_sampler("CALDGEMM_INPUT_A_3") " mxstr(a3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_4")_sampler("CALDGEMM_INPUT_A_4") " mxstr(a4) ", " mxstr(pos) "\n" /* normalized k, y pos */

#define SAMPLE_B(b1, b2, pos) \
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") " mxstr(b1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") " mxstr(b2) ", " mxstr(pos) "\n" /* normalized k, y pos */

#elif defined(CALDGEMM_44) //CALDGEMM_88

#define SAMPLE_A(a1, a2, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */

#define SAMPLE_B(b1, b2, pos) \
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") " mxstr(b1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") " mxstr(b2) ", " mxstr(pos) "\n" /* normalized k, y pos */

#else //CALDGEMM_44

#define SAMPLE_A_N(a1, a2, a3, a4, a5, a6, a7, a8, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_3")_sampler("CALDGEMM_INPUT_A_3") " mxstr(a3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_4")_sampler("CALDGEMM_INPUT_A_4") " mxstr(a4) ", " mxstr(pos) "\n" /* ... */ \
"    sample_resource("CALDGEMM_INPUT_A_5")_sampler("CALDGEMM_INPUT_A_5") " mxstr(a5) ", " mxstr(pos) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_6")_sampler("CALDGEMM_INPUT_A_6") " mxstr(a6) ", " mxstr(pos) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_7")_sampler("CALDGEMM_INPUT_A_7") " mxstr(a7) ", " mxstr(pos) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_8")_sampler("CALDGEMM_INPUT_A_8") " mxstr(a8) ", " mxstr(pos) "\n"

#define SAMPLE_A_T(a1_12, a1_34, a1_56, a1_78, a2_12, a2_34, a2_56, a2_78, pos12, pos34, pos56, pos78) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1_12) ", " mxstr(pos12) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2_12) ", " mxstr(pos12) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1_34) ", " mxstr(pos34) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2_34) ", " mxstr(pos34) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1_56) ", " mxstr(pos56) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2_56) ", " mxstr(pos56) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1_78) ", " mxstr(pos78) "\n" \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2_78) ", " mxstr(pos78) "\n"

#endif //CALDGEMM_44

#if defined(CALDGEMM_44)
#define UNNORM_TYPE ",unnorm"
#else
#define UNNORM_TYPE
#endif

#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)

const char *caldgemm_cal::ILKernelName =
#ifdef CALDGEMM_COMPUTE_SHADER
"il_cs_2_0\n"
#else
"il_ps_2_0\n"
#endif
"dcl_cb cb0[4]\n"
#ifdef CALDGEMM_COMPUTE_SHADER
"dcl_num_thread_per_group " mxstr(CALDGEMM_COMPUTE_SHADER) "\n"
#else
"dcl_input_position_interp(linear_noperspective) vWinCoord0.xy__\n"
#endif
"dcl_resource_id(0)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(1)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#ifndef CALDGEMM_SINGLE_BUFFER
"dcl_resource_id(2)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(3)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#if (!defined(CALDGEMM_TRANSPOSED_A) & !defined(CALDGEMM_44)) | defined(CALDGEMM_84) | defined(CALDGEMM_48) | defined(CALDGEMM_DOUBLE_BUFFERS)
"dcl_resource_id(4)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(5)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#if (!defined(CALDGEMM_TRANSPOSED_A) & !defined(CALDGEMM_44)) | defined(CALDGEMM_88)  | defined(CALDGEMM_DOUBLE_BUFFERS)
"dcl_resource_id(6)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(7)_type(2d" UNNORM_TYPE ")_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#if !defined(CALDGEMM_44)
"dcl_resource_id(8)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(9)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#endif
#endif
#endif
#endif
"dcl_literal l0, 0x00000000, 0x00000000, 0x00000000, 0x00000000\n"
#ifdef CALDGEMM_44_BT_64_KERNEL
"dcl_literal l1, %d.0, 2.0, -1.5, 4.0\n"
#else
#ifdef CALDGEMM_FORCE_K
"dcl_literal l1, " qon_mxstr(CALDGEMM_FORCE_K) ".0, 1.0, -0.5, 2.0\n"
#else
"dcl_literal l1, %d.0, 1.0, -0.5, 2.0\n"
#ifdef CALDGEMM_SHIFT_TEXTURE
"dcl_literal l2, " mxstr(CALDGEMM_SHIFT_TEXTURE) ".0, 0x00000003, 0x00000002, -1.5\n"
#elif defined(CALDGEMM_SINGLE_BUFFER)
"dcl_literal l2, 0.0, 0x00000003, 0x00000002, -1.5\n"
#endif
#endif
#endif

#if defined(CALDGEMM_44)
#ifdef CALDGEMM_SINGLE_BUFFER_IMPROVED
"ftoi r13.xy, vWinCoord0.xy\n"
"umod r14.xy, r13.xy, l2.yy\n"
"isub r13.xy, r13.xy, r14.xy\n"
"umul r13.xy, r13.xy, l2.zz\n"
"iadd r13.xy, r13.xy, r14.xy\n"
"iadd r13.zw, r13.xy, l2.yy\n"
"itof r0.xy, r13.xz\n"
"itof r100.xy, r13.yw\n"
#elif defined(CALDGEMM_SINGLE_BUFFER)
"mul r0.y, vWinCoord0.x, l1.w\n"
"mul r100.y, vWinCoord0.y, l1.w\n"
"sub r0.x, r0.y, r0.1\n"
"sub r100.x, r100.y, r100.1\n"
#else
#ifdef CALDGEMM_COMPUTE_SHADER
"umod r0.x, vaTid0.x, cb0[0].x\n"
"udiv r0.y, vaTid0.x, cb0[0].x\n"
"itof r0, r0\n"
#else
"mov r0.xy__, vWinCoord0.xy00\n"
#endif
#endif
#ifndef CALDGEMM_LATE_EXIT_CONDITION
#ifdef CALDGEMM_SHIFT_TEXTURE
"mov r100.z, l2.w\n"
#endif
"mov r0.w, l1.z\n"
#endif
#else
"mul r0.xyz0, vWinCoord0.xyyx, cb0[0].zwxz\n"		//r0.x: normalized x pos, r0.y: 0, r0.z: normalized y pos, r0.w: 0
#endif

#ifdef CALDGEMM_SHIFT_TEXTURE
#ifdef CALDGEMM_SINGLE_BUFFER
"add r101.xy__, r0.xy00, l2.xx00\n"
"add r102.xy__, r100.xy00, l2.xx00\n"
#else
"add r100.xy__, r0.xy00, l2.xx00\n"
#endif
#endif

#ifdef CALDGEMM_LATE_EXIT_CONDITION
#if defined(CALDGEMM_SINGLE_BUFFER) & defined(CALDGEMM_SHIFT_TEXTURE)
"mov r0.w, r0.1\n"
"mov r100.w, r100.1\n"
"mov r101.z, r101.0\n"
"mov r102.z, r102.0\n"
#else //SINGLE BUFFER and SHIFT_TEXTURE
#ifdef CALDGEMM_SHIFT_TEXTURE
"mov r100.z, r100.0\n"
#else
"mov r0.z, r0.0\n"
#endif
"mov r0.w, r0.1\n"
#ifdef CALDGEMM_SINGLE_BUFFER
"mov r100.z, r100.0\n"
"mov r100.w, r100.1\n"
#endif
#endif //SINGLE_BUFFER and SHIFT_TEXTURE
#endif

#if defined(CALDGEMM_TRANSPOSED_A) & !defined(CALDGEMM_44)
"sub r0.z, vWinCoord0.y, cb0[2].x\n"
"mul r0.z, r0.z, cb0[0].x\n"
"add r2.x, r0.z, cb0[1].y\n"
"add r2.y, r2.x, cb0[1].y\n"
"add r2.z, r2.y, cb0[1].y\n"
#endif
#ifndef CALDGEMM_44
"mov r1.__zw, r0.zzzx\n"							//r1.z: normalized y pos, r1.w: normalized x pos
"mov r11.w, r0.y\n"									//r11.w: 0, run variable for k
"mov r11.x, l0\n"									//r11.x: 0, run variable for loop
#endif

"mov r3, l0\n"
"mov r4, l0\n"
"mov r5, l0\n"
"mov r6, l0\n"
"mov r7, l0\n"
"mov r8, l0\n"
"mov r9, l0\n"
"mov r10, l0\n"
#if defined(CALDGEMM_84) | defined(CALDGEMM_48) | defined(CALDGEMM_SGEMM)
"mov r37, l0\n"
"mov r38, l0\n"
"mov r39, l0\n"
"mov r40, l0\n"
"mov r41, l0\n"
"mov r42, l0\n"
"mov r43, l0\n"
"mov r44, l0\n"
#if defined(CALDGEMM_88)
"mov r29, l0\n"
"mov r30, l0\n"
"mov r31, l0\n"
"mov r32, l0\n"
"mov r33, l0\n"
"mov r34, l0\n"
"mov r35, l0\n"
"mov r36, l0\n"
"mov r45, l0\n"
"mov r46, l0\n"
"mov r47, l0\n"
"mov r48, l0\n"
"mov r49, l0\n"
"mov r50, l0\n"
"mov r51, l0\n"
"mov r52, l0\n"
#endif
#endif

"whileloop\n"
#ifndef CALDGEMM_LATE_EXIT_CONDITION
#ifndef CALDGEMM_44
"    ge r0.y, r11.x, cb0[1].x\n"				//cmp with width
#else
#if (defined(CALDGEMM_DUAL_ENTRY) | defined(CALDGEMM_TRANSPOSED_B) | (defined(CALDGEMM_48) & !defined(CALDGEMM_88))) & !defined(CALDGEMM_DOUBLE_BUFFERS)
#ifdef CALDGEMM_SHIFT_TEXTURE
"    add r100.z, r100.z, l1.w\n"
#else
"    add r0.z, r0.w, l1.y\n"
#endif
"    add r0.w, r0.w, l1.w\n"
#else
"    add r0.w, r0.w, l1.y\n"
#endif
#ifdef CALDGEMM_SINGLE_BUFFER
"    mov r100.xy__, r0.xy00"
#endif
#ifdef CALDGEMM_44_BT_64_KERNEL
"    add r1.xyzw, r0.xyzw, l1.0011\n"
#endif
#if defined(CALDGEMM_LINPACK_KERNEL) | defined(CALDGEMM_BENCHMARK_KERNEL) | defined(CALDGEMM_FORCE_K)
"    ge r13.z, r0.w, l1.x\n"				//cmp with width
#else
"    ge r13.z, r0.w, cb0[1].x\n"				//cmp with width
#endif
#endif
#if defined(CALDGEMM_44)
"    break_logicalnz r13.z\n"
#else
"    break_logicalnz r0.y\n"
#endif
#endif //CALDGEMM_LATE_EXIT_CONDITION

#ifndef CALDGEMM_44
"    mov r0.___w, r11.w\n"							//r0:w normalized k pos
"    add r0._y__, r0.w, cb0[0].y\n"					//r0.y normalized k pos + 1
"    add r1.x___, r0.y, cb0[0].y\n"					//r1.x normalized k pos + 2
"    add r1._y__, r1.x, cb0[0].y\n"					//r1.y normalized k pos + 3
"    add r11.___w, r1.y, cb0[0].y\n"				//r11.w: normalized k pos + 4 steps (for next iteration in whileloop)
#endif

#ifdef CALDGEMM_88
SAMPLE_A(r13, r14, r15, r16, r0.yw)
SAMPLE_B(r21, r22, r23, r24, r0.xw)
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r29, r30, r31, r32, r33, r34, r35, r36, r15.xy, r15.zw, r16.xy, r16.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r37, r38, r39, r40, r41, r42, r43, r44, r13.xy, r13.zw, r14.xy, r14.zw, r23.xy, r23.zw, r24.xy, r24.zw)
MMUL441(r45, r46, r47, r48, r49, r50, r51, r52, r15.xy, r15.zw, r16.xy, r16.zw, r23.xy, r23.zw, r24.xy, r24.zw)

#elif defined(CALDGEMM_84)
SAMPLE_A(r13, r14, r0.yw)
SAMPLE_B(r21, r22, r23, r24, r0.xw)
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r37, r38, r39, r40, r41, r42, r43, r44, r13.xy, r13.zw, r14.xy, r14.zw, r23.xy, r23.zw, r24.xy, r24.zw)

#elif defined(CALDGEMM_48)
SAMPLE_A(r13, r14, r15, r16, r0.yz)
SAMPLE_B(r21, r22, r0.xz)
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r37, r38, r39, r40, r41, r42, r43, r44, r15.xy, r15.zw, r16.xy, r16.zw, r21.xy, r21.zw, r22.xy, r22.zw)
SAMPLE_A(r13, r14, r15, r16, r0.yw)
SAMPLE_B(r21, r22, r0.xw)
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r37, r38, r39, r40, r41, r42, r43, r44, r15.xy, r15.zw, r16.xy, r16.zw, r21.xy, r21.zw, r22.xy, r22.zw)

#elif defined(CALDGEMM_44)
#if defined(CALDGEMM_TRANSPOSED_A)
#ifdef CALDGEMM_DUAL_ENTRY
#ifdef CALDGEMM_SINGLE_BUFFER
#ifdef CALDGEMM_SHIFT_TEXTURE
"sample_resource(0)_sampler(0) r13, r100.xw\n"
"sample_resource(0)_sampler(0) r14, r100.yw\n"
"sample_resource(1)_sampler(1) r15, r0.xw\n"
"sample_resource(1)_sampler(1) r16, r0.yw\n"
"sample_resource(0)_sampler(0) r17, r102.xz\n"
"sample_resource(0)_sampler(0) r18, r102.yz\n"
"sample_resource(1)_sampler(1) r19, r101.xz\n"
"sample_resource(1)_sampler(1) r20, r101.yz\n"
#else
"sample_resource(0)_sampler(0) r13, r100.xz\n"
"sample_resource(0)_sampler(0) r14, r100.yz\n"
"sample_resource(1)_sampler(1) r15, r0.xz\n"
"sample_resource(1)_sampler(1) r16, r0.yz\n"
"sample_resource(0)_sampler(0) r17, r100.xw\n"
"sample_resource(0)_sampler(0) r18, r100.yw\n"
"sample_resource(1)_sampler(1) r19, r0.xw\n"
"sample_resource(1)_sampler(1) r20, r0.yw\n"
#endif
#elif defined (CALDGEMM_DOUBLE_BUFFERS)
"sample_resource(0)_sampler(0) r13, r0.yw\n"
"sample_resource(1)_sampler(1) r14, r0.yw\n"
"sample_resource(4)_sampler(4) r15, r0.xw\n"
"sample_resource(5)_sampler(5) r16, r0.xw\n"
"sample_resource(2)_sampler(2) r17, r0.yw\n"
"sample_resource(3)_sampler(3) r18, r0.yw\n"
"sample_resource(6)_sampler(6) r19, r0.xw\n"
"sample_resource(7)_sampler(7) r20, r0.xw\n"
#else
#ifdef CALDGEMM_SHIFT_TEXTURE
SAMPLE_A(r13, r14, r100.yz)
SAMPLE_B(r15, r16, r100.xz)
#else
SAMPLE_A(r13, r14, r0.yz)
SAMPLE_B(r15, r16, r0.xz)
#endif
SAMPLE_A(r17, r18, r0.yw)
SAMPLE_B(r19, r20, r0.xw)
#endif //SINGLE_BUFFER
#ifdef CALDGEMM_LATE_EXIT_CONDITION
#ifdef CALDGEMM_SHIFT_TEXTURE
#ifdef CALDGEMM_SINGLE_BUFFER
"add r100.w, r100.w, l1.w\n"
"add r101.z, r101.z, l1.w\n"
"add r102.z, r102.z, l1.w\n"
#else
"add r100.z, r100.z, l1.w\n"
#endif
#else
"add r0.z, r0.z, l1.w\n"
#endif
"add r0.w, r0.w, l1.w\n"
#if defined(CALDGEMM_SINGLE_BUFFER) & !defined(CALDGEMM_SHIFT_TEXTURE)
"mov r100.__zw, r0.00zw\n"
#endif
#endif
#ifdef CALDGEMM_SGEMM
MMUL441_S1(r3, r4, r5, r6, r7, r8, r9, r10, r13.x, r13.z, r14.x, r14.z, r15.x, r15.z, r16.x, r16.z)
MMUL441_S1(r37, r38, r38, r40, r41, r42, r43, r44, r13.x, r13.z, r14.x, r14.z, r15.y, r15.w, r16.y, r16.w)
MMUL441_S2(r3, r4, r5, r6, r7, r8, r9, r10, r13.y, r13.w, r14.y, r14.w, r15.x, r15.z, r16.x, r16.z)
MMUL441_S2(r37, r38, r39, r40, r41, r42, r43, r44, r13.y, r13.w, r14.y, r14.w, r15.y, r15.w, r16.y, r16.w)
#else //CALDGEMM_SGEMM
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r15.xy, r15.zw, r16.xy, r16.zw)
#endif //CALDGEMM_SGEMM
#else
SAMPLE_A(r17, r18, r0.yw)
SAMPLE_B(r19, r20, r0.xw)
#endif //DUAL_ENTRY
#ifdef CALDGEMM_SGEMM
MMUL441_S1(r3, r4, r5, r6, r7, r8, r9, r10, r17.x, r17.z, r18.x, r18.z, r19.x, r19.z, r20.x, r20.z)
MMUL441_S1(r37, r38, r38, r40, r41, r42, r43, r44, r17.x, r17.z, r18.x, r18.z, r19.y, r19.w, r20.y, r20.w)
MMUL441_S2(r3, r4, r5, r6, r7, r8, r9, r10, r17.y, r17.w, r18.y, r18.w, r19.x, r19.z, r20.x, r20.z)
MMUL441_S2(r37, r38, r39, r40, r41, r42, r43, r44, r17.y, r17.w, r18.y, r18.w, r19.y, r19.w, r20.y, r20.w)
#else //CALDGEMM_SGEMM
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r17.xy, r17.zw, r18.xy, r18.zw, r19.xy, r19.zw, r20.xy, r20.zw)
#endif //CALDGEMM_SGEMM

#else

SAMPLE_A(r13, r14, r0.zy)
SAMPLE_B(r15, r16, r0.zx)
SAMPLE_A(r17, r18, r0.wy)
SAMPLE_B(r19, r20, r0.wx)

#ifdef CALDGEMM_44_BT_64_KERNEL
SAMPLE_A(r23, r24, r1.zy)
SAMPLE_B(r25, r26, r1.zx)
SAMPLE_A(r27, r28, r1.wy)
SAMPLE_B(r29, r30, r1.wx)
#endif

#ifdef CALDGEMM_LATE_EXIT_CONDITION
#ifdef CALDGEMM_SHIFT_TEXTURE
"add r100.z, r100.z, l1.w\n"
#else
"add r0.z, r0.z, l1.w\n"
#endif
"add r0.w, r0.w, l1.w\n"
#endif //CALDGEMM_LATE_EXIT_CONDITION

#ifdef CALDGEMM_SGEMM
MMUL441_S1(r3, r4, r5, r6, r7, r8, r9, r10, r13.x, r17.x, r14.x, r18.x, r15.x, r19.x, r16.x, r20.x)
MMUL441_S1(r37, r38, r38, r40, r41, r42, r43, r44, r13.x, r17.x, r14.x, r18.x, r15.y, r19.y, r16.y, r20.y)
MMUL441_S2(r3, r4, r5, r6, r7, r8, r9, r10, r13.y, r17.y, r14.y, r18.y, r15.x, r19.x, r16.x, r20.x)
MMUL441_S2(r37, r38, r39, r40, r41, r42, r43, r44, r13.y, r17.y, r14.y, r18.y, r15.y, r19.y, r16.y, r20.y)

MMUL441_S1(r3, r4, r5, r6, r7, r8, r9, r10, r13.z, r17.z, r14.z, r18.z, r15.z, r19.z, r16.z, r20.z)
MMUL441_S1(r37, r38, r39, r40, r41, r42, r43, r44, r13.z, r17.z, r14.z, r18.z, r15.w, r19.w, r16.w, r20.w)
MMUL441_S2(r3, r4, r5, r6, r7, r8, r9, r10, r13.w, r17.w, r14.w, r18.w, r15.z, r19.z, r16.z, r20.z)
MMUL441_S2(r37, r38, r39, r40, r41, r42, r43, r44, r13.w, r17.w, r14.w, r18.w, r15.w, r19.w, r16.w, r20.w)
#else //CALDGEMM_SGEMM
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r17.xy, r14.xy, r18.xy, r15.xy, r19.xy, r16.xy, r20.xy)
#ifdef CALDGEMM_44_BT_64_KERNEL
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r23.xy, r27.xy, r24.xy, r28.xy, r25.xy, r29.xy, r26.xy, r30.xy)
#else
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.zw, r17.zw, r14.zw, r18.zw, r15.zw, r19.zw, r16.zw, r20.zw)
#endif
#endif //CALDGEMM_SGEMM
#endif

#else
#if defined(CALDGEMM_TRANSPOSED_B)
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r21, r0.wx\n"
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r22, r0.wx\n"
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r23, r0.yx\n"
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r24, r0.yx\n"
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r25, r1.xw\n"
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r26, r1.xw\n"
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r27, r1.yw\n"
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r28, r1.yw\n"
#else
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r21, r0.xw\n"	//normalized x,k pos
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r22, r0.xw\n"  //normalized x,k pos
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r23, r0.xy\n"  //normalized x,k+1 pos
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r24, r0.xy\n"	//normalized x,k+1 pos
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r25, r1.wx\n"  //normalized  ,k+2 pos
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r26, r1.wx\n"
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") r27, r1.wy\n"	//normalized ,k+3 pos
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") r28, r1.wy\n"
#endif //CALDGEMM_TRANSPOSED_B
#if defined(CALDGEMM_TRANSPOSED_A)
"    mov r2.w, r0.w\n"
SAMPLE_A_T(r13, r14, r15, r16, r17, r18, r19, r20, r0.zw, r2.xw, r2.yw, r2.zw)		//y,k ;  y+1,k ; y+2,k ; y+3,k
MMUL_T(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22)
"    mov r2.w, r0.y\n"
SAMPLE_A_T(r13, r14, r15, r16, r17, r18, r19, r20, r0.zy, r2.xw, r2.yw, r2.zw)		//y,k+1 ;  ;  ;
MMUL_T(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r23, r24)
"    mov r2.w, r1.x\n"
SAMPLE_A_T(r13, r14, r15, r16, r17, r18, r19, r20, r1.zx, r2.xw, r2.yw, r2.zw)		//y,k+2 ;  ;  ;
MMUL_T(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r25, r26)
"    mov r2.w, r1.y\n"
SAMPLE_A_T(r13, r14, r15, r16, r17, r18, r19, r20, r1.zy, r2.xw, r2.yw, r2.zw)		//y,k+3 ;  ;  ;
MMUL_T(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r27, r28)
#else
SAMPLE_A_N(r13, r14, r15, r16, r17, r18, r19, r20, r0.wz)	//normalized k, normalized y
MMUL_N(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22)
SAMPLE_A_N(r13, r14, r15, r16, r17, r18, r19, r20, r0.yz)	//normalized k+1, normalized y
MMUL_N(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r23, r24)
SAMPLE_A_N(r13, r14, r15, r16, r17, r18, r19, r20, r1.xz)	//normalized k+2, normalized y
MMUL_N(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r25, r26)
SAMPLE_A_N(r13, r14, r15, r16, r17, r18, r19, r20, r1.yz)	//normalized k+3, normalized y
MMUL_N(r3, r4, r5, r6, r7, r8, r9, r10, r13, r14, r15, r16, r17, r18, r19, r20, r27, r28)
#endif

#endif //CALDGEMM_44

#ifdef CALDGEMM_44
#ifdef CALDGEMM_DIAGONAL_TEXTURE
"    add r0.x, r0.x, cb0[2].w\n"
"    add r0.z, r0.z, cb0[2].w\n"
"    flr r13.x, r0.x\n"
"    flr r13.z, r0.z\n"
"    sub r0.x, r0.x, r13.x\n"
"    sub r0.z, r0.z, r13.z\n"
#endif
#else
"    add r11.x___, r11.x, r11.1\n"
#endif

#ifdef CALDGEMM_LATE_EXIT_CONDITION
"ge r13.z, r0.w, l1.x\n"
"break_logicalnz r13.z\n"
#endif

"endloop\n"
#ifndef CALDGEMM_USE_MEMEXPORT
"dcl_output_generic o0\n"
"dcl_output_generic o1\n"
"dcl_output_generic o2\n"
"dcl_output_generic o3\n"
"dcl_output_generic o4\n"
"dcl_output_generic o5\n"
"dcl_output_generic o6\n"
"dcl_output_generic o7\n"
#endif

#if (!defined(CALDGEMM_BENCHMARK_KERNEL) & !defined(CALDGEMM_ALPHA1)) | (!defined(CALDGEMM_84) & !defined(CALDGEMM_48) & defined(CALDGEMM_44) & defined(CALDGEMM_TRANSPOSED_A) & 0) //For some reason the multiplication makes the compiler create a faster kernel for some GPUs, currently disabled
MUL8(r3, r4, r5, r6, r7, r8, r9, r10, cb0[1].zw)
#if defined(CALDGEMM_84) | defined(CALDGEMM_48)
MUL8(r37, r38, r39, r40, r41, r42, r43, r44, cb0[1].zw)
#if defined(CALDGEMM_88)
MUL8(r29, r30, r31, r32, r33, r34, r35, r36, cb0[1].zw)
MUL8(r45, r46, r47, r48, r49, r50, r51, r52, cb0[1].zw)
#endif
#endif
#endif

/*#ifdef TEST_KERNEL
"dadd r3.xy, r3.xy, r37.xy\n"
"dadd r3.zw, r3.zw, r37.zw\n"
"dadd r4.xy, r4.xy, r38.xy\n"
"dadd r4.zw, r4.zw, r38.zw\n"
"dadd r5.xy, r5.xy, r39.xy\n"
"dadd r5.zw, r5.zw, r39.zw\n"
"dadd r6.xy, r6.xy, r40.xy\n"
"dadd r6.zw, r6.zw, r40.zw\n"
"dadd r7.xy, r7.xy, r41.xy\n"
"dadd r7.zw, r7.zw, r41.zw\n"
"dadd r8.xy, r8.xy, r42.xy\n"
"dadd r8.zw, r8.zw, r42.zw\n"
"dadd r9.xy, r9.xy, r43.xy\n"
"dadd r9.zw, r9.zw, r43.zw\n"
"dadd r10.xy, r10.xy, r44.xy\n"
"dadd r10.zw, r10.zw, r44.zw\n"

"dadd r3.xy, r3.xy, r45.xy\n"
"dadd r3.zw, r3.zw, r45.zw\n"
"dadd r4.xy, r4.xy, r46.xy\n"
"dadd r4.zw, r4.zw, r46.zw\n"
"dadd r5.xy, r5.xy, r47.xy\n"
"dadd r5.zw, r5.zw, r47.zw\n"
"dadd r6.xy, r6.xy, r48.xy\n"
"dadd r6.zw, r6.zw, r48.zw\n"
"dadd r7.xy, r7.xy, r49.xy\n"
"dadd r7.zw, r7.zw, r49.zw\n"
"dadd r8.xy, r8.xy, r50.xy\n"
"dadd r8.zw, r8.zw, r50.zw\n"
"dadd r9.xy, r9.xy, r51.xy\n"
"dadd r9.zw, r9.zw, r51.zw\n"
"dadd r10.xy, r10.xy, r52.xy\n"
"dadd r10.zw, r10.zw, r52.zw\n"

"dadd r3.xy, r3.xy, r29.xy\n"
"dadd r3.zw, r3.zw, r29.zw\n"
"dadd r4.xy, r4.xy, r30.xy\n"
"dadd r4.zw, r4.zw, r30.zw\n"
"dadd r5.xy, r5.xy, r31.xy\n"
"dadd r5.zw, r5.zw, r31.zw\n"
"dadd r6.xy, r6.xy, r32.xy\n"
"dadd r6.zw, r6.zw, r32.zw\n"
"dadd r7.xy, r7.xy, r33.xy\n"
"dadd r7.zw, r7.zw, r33.zw\n"
"dadd r8.xy, r8.xy, r34.xy\n"
"dadd r8.zw, r8.zw, r34.zw\n"
"dadd r9.xy, r9.xy, r35.xy\n"
"dadd r9.zw, r9.zw, r35.zw\n"
"dadd r10.xy, r10.xy, r36.xy\n"
"dadd r10.zw, r10.zw, r36.zw\n"
#endif*/

#ifdef CALDGEMM_USE_MEMEXPORT
#ifndef CALDGEMM_COMPUTE_SHADER
"flr r0.xy, vWinCoord0.xy\n"
#endif
"ftoi r1.xy, r0.xy\n"
"imul r1.x, r1.x, cb0[2].z\n"
"imad r0, r1.yyyy, cb0[2].yyyy, r1.xxxx\n"
"iadd r1, r0, cb0[3]\n"
#if defined(CALDGEMM_84)
"iadd r2, r1, cb0[4]\n"
"mov g[r1.x], r3\n"
"mov g[r1.y], r7\n"
"mov g[r1.z], r37\n"
"mov g[r1.w], r41\n"
"mov g[r2.x], r4\n"
"mov g[r2.y], r8\n"
"mov g[r2.z], r38\n"
"mov g[r2.w], r42\n"
"iadd r1, r1, cb0[5]\n"
"iadd r2, r2, cb0[5]\n"
"mov g[r1.x], r5\n"
"mov g[r1.y], r9\n"
"mov g[r1.z], r39\n"
"mov g[r1.w], r43\n"
"mov g[r2.x], r6\n"
"mov g[r2.y], r10\n"
"mov g[r2.z], r40\n"
"mov g[r2.w], r44\n"
#ifdef CALDGEMM_88
"iadd r1, r1, cb0[5]\n"
"iadd r2, r2, cb0[5]\n"
"mov g[r1.x], r29\n"
"mov g[r1.y], r33\n"
"mov g[r1.z], r45\n"
"mov g[r1.w], r49\n"
"mov g[r2.x], r30\n"
"mov g[r2.y], r34\n"
"mov g[r2.z], r46\n"
"mov g[r2.w], r50\n"
"iadd r1, r1, cb0[5]\n"
"iadd r2, r2, cb0[5]\n"
"mov g[r1.x], r31\n"
"mov g[r1.y], r35\n"
"mov g[r1.z], r47\n"
"mov g[r1.w], r51\n"
"mov g[r2.x], r32\n"
"mov g[r2.y], r36\n"
"mov g[r2.z], r48\n"
"mov g[r2.w], r52\n"
#endif //CALDGEMM_88
#else
"iadd r2, r0, cb0[4]\n"
"mov g[r1.x], r3\n"
"mov g[r1.y], r7\n"
"mov g[r1.z], r4\n"
"mov g[r1.w], r8\n"
"mov g[r2.x], r5\n"
"mov g[r2.y], r9\n"
"mov g[r2.z], r6\n"
"mov g[r2.w], r10\n"
#if defined(CALDGEMM_48) | defined(CALDGEMM_SGEMM)
"iadd r1, r1, cb0[5]\n"
"iadd r2, r2, cb0[5]\n"
"mov g[r1.x], r37\n"
"mov g[r1.y], r41\n"
"mov g[r1.z], r38\n"
"mov g[r1.w], r42\n"
"mov g[r2.x], r39\n"
"mov g[r2.y], r43\n"
"mov g[r2.z], r40\n"
"mov g[r2.w], r44\n"
#endif
#endif
#else
"mov o0, r3\n"
"mov o1, r4\n"
"mov o2, r5\n"
"mov o3, r6\n"
"mov o4, r7\n"
"mov o5, r8\n"
"mov o6, r9\n"
"mov o7, r10\n"
#endif
"end\n";

#ifdef CALDGEMM_44
#undef SAMPLE_A
#undef SAMPLE_B
#else
#undef SAMPLE_A_N
#undef SAMPLE_A_T
#endif
#undef MMUL_22
#undef MMUL_44
#undef MMUL_82
#undef MMUL_N
#undef MMUL_T
#undef UNNORM_TYPE

#ifdef CALDGEMM_44
#undef CALDGEMM_INPUT_A_1
#undef CALDGEMM_INPUT_A_2
#undef CALDGEMM_INPUT_A_3
#undef CALDGEMM_INPUT_A_4
#undef CALDGEMM_INPUT_B_1
#undef CALDGEMM_INPUT_B_2
#undef CALDGEMM_INPUT_B_3
#undef CALDGEMM_INPUT_B_4
#elif defined(CALDGEMM_TRANSPOSED_A)
#undef CALDGEMM_INPUT_A_1
#undef CALDGEMM_INPUT_A_2
#undef CALDGEMM_INPUT_B_1
#undef CALDGEMM_INPUT_B_2
#else
#undef CALDGEMM_INPUT_A_1
#undef CALDGEMM_INPUT_A_2
#undef CALDGEMM_INPUT_A_3
#undef CALDGEMM_INPUT_A_4
#undef CALDGEMM_INPUT_A_5
#undef CALDGEMM_INPUT_A_6
#undef CALDGEMM_INPUT_A_7
#undef CALDGEMM_INPUT_A_8
#undef CALDGEMM_INPUT_B_1
#undef CALDGEMM_INPUT_B_2
#endif
