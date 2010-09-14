/* ============================================================

Copyright (c) 2007 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERATION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
 commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be
modified or waived, and no breach of this license can be excused, unless done
so in a writing signed by all affected parties. Each term of this license is
separately enforceable. If any term of this license is determined to be or
becomes unenforceable or illegal, such term shall be reformed to the minimum
extent necessary in order for this license to remain in effect in accordance
with its terms as modified by such reformation. This license shall be governed
by and construed in accordance with the laws of the State of Texas without
regard to rules on conflicts of law of any state or jurisdiction or the United
Nations Convention on the International Sale of Goods. All disputes arising out
of this license shall be subject to the jurisdiction of the federal and state
courts in Austin, Texas, and all defenses are hereby waived concerning personal
jurisdiction and venue of these courts.

============================================================ */

#if defined(CALDGEMM_88)
#ifdef CALDGEMM_TRANSPOSED_A
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_A_3 "2"
#define CALDGEMM_INPUT_A_4 "3"
#define CALDGEMM_INPUT_B_1 "4"
#define CALDGEMM_INPUT_B_2 "5"
#define CALDGEMM_INPUT_B_3 "6"
#define CALDGEMM_INPUT_B_4 "7"
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
#define CALDGEMM_INPUT_B_3 "10"
#define CALDGEMM_INPUT_B_4 "11"
#define CALDGEMM_INPUT_B_5 "12"
#define CALDGEMM_INPUT_B_6 "13"
#define CALDGEMM_INPUT_B_7 "14"
#define CALDGEMM_INPUT_B_8 "15"
#endif
#elif defined(CALDGEMM_TRANSPOSED_A)
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_B_1 "2"
#define CALDGEMM_INPUT_B_2 "3"
#elif defined(CALDGEMM_44)
#define CALDGEMM_INPUT_A_1 "0"
#define CALDGEMM_INPUT_A_2 "1"
#define CALDGEMM_INPUT_A_3 "2"
#define CALDGEMM_INPUT_A_4 "3"
#define CALDGEMM_INPUT_B_1 "4"
#define CALDGEMM_INPUT_B_2 "5"
#define CALDGEMM_INPUT_B_3 "6"
#define CALDGEMM_INPUT_B_4 "7"
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

#define MMUL44(o1_12, o2_12, o3_12, o4_12, o1_34, o2_34, o3_34, o4_34, a1_1, a1_2, a2_1, a2_2, a3_1, a3_2, a4_1, a4_2, b1_1, b2_1, b1_2, b2_2, b1_3, b2_3, b1_4, b2_4) \
MMUL22(o1_12, a1_1, a1_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o2_12, a2_1, a2_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o3_12, a3_1, a3_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o4_12, a4_1, a4_2, b1_1, b1_2, b2_1, b2_2) \
MMUL22(o1_34, a1_1, a1_2, b1_3, b1_4, b2_3, b2_4) \
MMUL22(o2_34, a2_1, a2_2, b1_3, b1_4, b2_3, b2_4) \
MMUL22(o3_34, a3_1, a3_2, b1_3, b1_4, b2_3, b2_4) \
MMUL22(o4_34, a4_1, a4_2, b1_3, b1_4, b2_3, b2_4)

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

#elif defined(CALDGEMM_44) //CALDGEMM_88

#ifdef CALDGEMM_TRANSPOSED_A
#define SAMPLE_A(a1, a2, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */

#define SAMPLE_B(b1, b2, pos) \
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") " mxstr(b1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") " mxstr(b2) ", " mxstr(pos) "\n" /* normalized k, y pos */

#else
#define SAMPLE_A(a1, a2, a3, a4, pos) \
"    sample_resource("CALDGEMM_INPUT_A_1")_sampler("CALDGEMM_INPUT_A_1") " mxstr(a1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_2")_sampler("CALDGEMM_INPUT_A_2") " mxstr(a2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_3")_sampler("CALDGEMM_INPUT_A_3") " mxstr(a3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_A_4")_sampler("CALDGEMM_INPUT_A_4") " mxstr(a4) ", " mxstr(pos) "\n" /* ... */

#define SAMPLE_B(b1, b2, b3, b4, pos) \
"    sample_resource("CALDGEMM_INPUT_B_1")_sampler("CALDGEMM_INPUT_B_1") " mxstr(b1) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_2")_sampler("CALDGEMM_INPUT_B_2") " mxstr(b2) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_3")_sampler("CALDGEMM_INPUT_B_3") " mxstr(b3) ", " mxstr(pos) "\n" /* normalized k, y pos */ \
"    sample_resource("CALDGEMM_INPUT_B_4")_sampler("CALDGEMM_INPUT_B_4") " mxstr(b4) ", " mxstr(pos) "\n" /* ... */
#endif //CALDGEMM_TRANSPOSED_A

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

const char *caldgemm::ILKernelName =
"il_ps_2_0\n"
"dcl_cb cb0[4]\n"
"dcl_input_position_interp(linear_noperspective) vWinCoord0.xy__\n"
"dcl_resource_id(0)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(1)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(2)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(3)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#if !defined(CALDGEMM_TRANSPOSED_A) | defined(CALDGEMM_88)
"dcl_resource_id(4)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(5)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(6)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(7)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#if !defined(CALDGEMM_44)
"dcl_resource_id(8)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(9)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
#endif
#endif
"dcl_literal l0, 0x00000000, 0x00000000, 0x00000000, 0x00000000\n"
"mov r3, l0\n"
"mov r4, l0\n"
"mov r5, l0\n"
"mov r6, l0\n"
"mov r7, l0\n"
"mov r8, l0\n"
"mov r9, l0\n"
"mov r10, l0\n"

#ifdef CALDGEMM_88
"mov r29, l0\n"
"mov r30, l0\n"
"mov r31, l0\n"
"mov r32, l0\n"
"mov r33, l0\n"
"mov r34, l0\n"
"mov r35, l0\n"
"mov r36, l0\n"
"mov r37, l0\n"
"mov r38, l0\n"
"mov r39, l0\n"
"mov r40, l0\n"
"mov r41, l0\n"
"mov r42, l0\n"
"mov r43, l0\n"
"mov r44, l0\n"
"mov r45, l0\n"
"mov r46, l0\n"
"mov r47, l0\n"
"mov r48, l0\n"
"mov r49, l0\n"
"mov r50, l0\n"
"mov r51, l0\n"
"mov r52, l0\n"
#endif

"mul r0.xyz0, vWinCoord0.xyyx, cb0[0].zwxz\n"		//r0.x: normalized x pos, r0.y: 0, r0.z: normalized y pos, r0.w: 0
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
"whileloop\n"
#ifndef CALDGEMM_44
"    ge r0.y, r11.x, cb0[1].x\n"				//cmp with width
#else
"    ge r0.y, r0.w, cb0[1].1\n"				//cmp with width
#endif

"    break_logicalnz r0.y\n"

#ifndef CALDGEMM_44
"    mov r0.___w, r11.w\n"							//r0:w normalized k pos
"    add r0._y__, r0.w, cb0[0].y\n"					//r0.y normalized k pos + 1
"    add r1.x___, r0.y, cb0[0].y\n"					//r1.x normalized k pos + 2
"    add r1._y__, r1.x, cb0[0].y\n"					//r1.y normalized k pos + 3
"    add r11.___w, r1.y, cb0[0].y\n"				//r11.w: normalized k pos + 4 steps (for next iteration in whileloop)
#endif

#ifdef CALDGEMM_88
SAMPLE_A(r13, r14, r15, r16, r0.zw)
SAMPLE_B(r21, r22, r23, r24, r0.xw)
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r29, r30, r31, r32, r33, r34, r35, r36, r15.xy, r15.zw, r16.xy, r16.zw, r21.xy, r21.zw, r22.xy, r22.zw)
MMUL441(r37, r38, r39, r40, r41, r42, r43, r44, r13.xy, r13.zw, r14.xy, r14.zw, r23.xy, r23.zw, r24.xy, r24.zw)
MMUL441(r45, r46, r47, r48, r49, r50, r51, r52, r15.xy, r15.zw, r16.xy, r16.zw, r23.xy, r23.zw, r24.xy, r24.zw)

#elif defined(CALDGEMM_44)
#ifdef CALDGEMM_TRANSPOSED_A
SAMPLE_A(r13, r14, r0.zw)
SAMPLE_B(r15, r16, r0.xw)
MMUL441(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r15.xy, r15.zw, r16.xy, r16.zw)
#else
SAMPLE_A(r13, r14, r15, r16, r0.wz)
SAMPLE_B(r21, r22, r23, r24, r0.wx)
MMUL44(r3, r4, r5, r6, r7, r8, r9, r10, r13.xy, r13.zw, r14.xy, r14.zw, r15.xy, r15.zw, r16.xy, r16.zw, r21.xy, r21.zw, r22.xy, r22.zw, r23.xy, r23.zw, r24.xy, r24.zw)
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
"    add r0.w, r0.w, cb0[0].y\n"
#else
"    add r11.x___, r11.x, r11.1\n"
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

#ifndef CALDGEMM_ALPHA1
MUL8(r3, r4, r5, r6, r7, r8, r9, r10, cb0[1].zw)
#ifdef CALDGEMM_88
MUL8(r29, r30, r31, r32, r33, r34, r35, r36, cb0[1].zw)
MUL8(r37, r38, r39, r40, r41, r42, r43, r44, cb0[1].zw)
MUL8(r45, r46, r47, r48, r49, r50, r51, r52, cb0[1].zw)
#endif
#endif

#ifdef CALDGEMM_USE_MEMEXPORT
"flr r0.xy, vWinCoord0.xy\n"
"ftoi r1.xy, r0.xy\n"
"imul r1.x, r1.x, cb0[2].z\n"
"imad r0, r1.yyyy, cb0[2].yyyy, r1.xxxx\n"
"iadd r1, r0, cb0[3]\n"
#ifdef CALDGEMM_88
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
