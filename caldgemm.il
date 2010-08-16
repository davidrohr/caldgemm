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

const char caldgemm::ILKernel[] =
"il_ps_2_0\n"
"dcl_cb cb0[2]\n"
"dcl_input_position_interp(linear_noperspective) vWinCoord0.xy__\n"
"dcl_resource_id(0)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(1)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(2)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(3)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(4)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(5)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(6)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(7)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(8)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_resource_id(9)_type(2d)_fmtx(unknown)_fmty(unknown)_fmtz(unknown)_fmtw(unknown)\n"
"dcl_literal l0, 0x00000000, 0x00000000, 0x00000000, 0x00000000\n"
"mov r3, l0\n"
"mov r4, l0\n"
"mov r5, l0\n"
"mov r6, l0\n"
"mov r7, l0\n"
"mov r8, l0\n"
"mov r9, l0\n"
"mov r10, l0\n"
    /*data[bStop].f_data[0] = 8.f / Info->Height;   
    data[bStop].f_data[1] = 2.f / Info->Width;
    data[bStop].f_data[2] = 2.f / Info->Height;   
    data[bStop].f_data[3] = 0.f;
    data[bStop].f_data[4] = static_cast<CALfloat>(Info->Width / (bPartsNum << 2));*/

	//Run: Width: Info->Height / bPartsNum , Heigh: Info->Height / aPartsNum
"mul r0.xyz0, vWinCoord0.xyyx, cb0[0].zwxz\n"		//r0.x: normalized x pos, r0.y: 0, r0.z: normalized y pos, r0.w: 0
"mov r1.__zw, r0.zzzx\n"							//r1.z: normalized y pos, r1.w: normalized x pos
"mov r11.w, r0.y\n"									//r11.w: 0, run variable for k
"mov r11.x, l0\n"									//r11.x: 0, run variable for loop
"whileloop\n"
"    ge r11._y__, r11.x, cb0[1].x\n"				//cmp with width
"    break_logicalnz r11.y\n"
"    mov r0.___w, r11.w\n"							//r0:w normalized k pos
"    add r0._y__, r0.w, cb0[0].y\n"					//r0:y normalized k pos + 1
"    add r1.x___, r0.y, cb0[0].y\n"					//r1.x normalized k pos + 2
"    add r1._y__, r1.x, cb0[0].y\n"					//r1.y normalized k pos + 3
"    add r11.___w, r1.y, cb0[0].y\n"				//r11.w: normalized k pos + 4 steps (for next iteration in whileloop)
"    sample_resource(8)_sampler(8) r21, r0.xwww\n"	//normalized x,k pos
"    sample_resource(9)_sampler(9) r22, r0.xwww\n"  //normalized x,k pos
"    sample_resource(8)_sampler(8) r23, r0.xyww\n"  //normalized x,k+1 pos
"    sample_resource(9)_sampler(9) r24, r0.xyww\n"	//normalized x,k+1 pos
"    sample_resource(8)_sampler(8) r25, r1.wxww\n"  //normalized  ,k+2 pos
"    sample_resource(9)_sampler(9) r26, r1.wxww\n"
"    sample_resource(8)_sampler(8) r27, r1.wyzz\n"	//normalized ,k+3 pos
"    sample_resource(9)_sampler(9) r28, r1.wyzz\n"
"    sample_resource(0)_sampler(0) r13, r0.wz\n"	//normalized k, y pos
"    sample_resource(1)_sampler(1) r14, r0.wz\n"	//normalized k, y pos
"    sample_resource(2)_sampler(2) r15, r0.wz\n"	//normalized k, y pos
"    sample_resource(3)_sampler(3) r16, r0.wz\n"	//normalized k, y pos
"    sample_resource(4)_sampler(4) r17, r0.wz\n"	//...
"    sample_resource(5)_sampler(5) r18, r0.wz\n"
"    sample_resource(6)_sampler(6) r19, r0.wz\n"
"    sample_resource(7)_sampler(7) r20, r0.wz\n"
"    dmad r12.xy, r13.zw, r22.xy, r3.xy\n"
"    dmad r12.zw, r13.zw, r22.zw, r3.zw\n"
"    dmad r3.zw, r13.xy, r21.zw, r12.zw\n"
"    dmad r3.xy, r13.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r14.zw, r22.xy, r4.xy\n"
"    dmad r12.zw, r14.zw, r22.zw, r4.zw\n"
"    dmad r4.zw, r14.xy, r21.zw, r12.zw\n"
"    dmad r4.xy, r14.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r15.zw, r22.xy, r5.xy\n"
"    dmad r12.zw, r15.zw, r22.zw, r5.zw\n"
"    dmad r5.zw, r15.xy, r21.zw, r12.zw\n"
"    dmad r5.xy, r15.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r16.zw, r22.xy, r6.xy\n"
"    dmad r12.zw, r16.zw, r22.zw, r6.zw\n"
"    dmad r6.zw, r16.xy, r21.zw, r12.zw\n"
"    dmad r6.xy, r16.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r17.zw, r22.xy, r7.xy\n"
"    dmad r12.zw, r17.zw, r22.zw, r7.zw\n"
"    dmad r7.zw, r17.xy, r21.zw, r12.zw\n"
"    dmad r7.xy, r17.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r18.zw, r22.xy, r8.xy\n"
"    dmad r12.zw, r18.zw, r22.zw, r8.zw\n"
"    dmad r8.zw, r18.xy, r21.zw, r12.zw\n"
"    dmad r8.xy, r18.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r19.zw, r22.xy, r9.xy\n"
"    dmad r12.zw, r19.zw, r22.zw, r9.zw\n"
"    dmad r9.zw, r19.xy, r21.zw, r12.zw\n"
"    dmad r9.xy, r19.xy, r21.xy, r12.xy\n"
"    dmad r12.xy, r20.zw, r22.xy, r10.xy\n"
"    dmad r12.zw, r20.zw, r22.zw, r10.zw\n"
"    dmad r10.zw, r20.xy, r21.zw, r12.zw\n"
"    dmad r10.xy, r20.xy, r21.xy, r12.xy\n"
"    sample_resource(0)_sampler(0) r13, r0.yz\n"
"    sample_resource(1)_sampler(1) r14, r0.yz\n"
"    sample_resource(2)_sampler(2) r15, r0.yz\n"
"    sample_resource(3)_sampler(3) r16, r0.yz\n"
"    sample_resource(4)_sampler(4) r17, r0.yz\n"
"    sample_resource(5)_sampler(5) r18, r0.yz\n"
"    sample_resource(6)_sampler(6) r19, r0.yz\n"
"    sample_resource(7)_sampler(7) r20, r0.yz\n"
"    dmad r12.xy, r13.zw, r24.xy, r3.xy\n"
"    dmad r12.zw, r13.zw, r24.zw, r3.zw\n"
"    dmad r3.zw, r13.xy, r23.zw, r12.zw\n"
"    dmad r3.xy, r13.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r14.zw, r24.xy, r4.xy\n"
"    dmad r12.zw, r14.zw, r24.zw, r4.zw\n"
"    dmad r4.zw, r14.xy, r23.zw, r12.zw\n"
"    dmad r4.xy, r14.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r15.zw, r24.xy, r5.xy\n"
"    dmad r12.zw, r15.zw, r24.zw, r5.zw\n"
"    dmad r5.zw, r15.xy, r23.zw, r12.zw\n"
"    dmad r5.xy, r15.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r16.zw, r24.xy, r6.xy\n"
"    dmad r12.zw, r16.zw, r24.zw, r6.zw\n"
"    dmad r6.zw, r16.xy, r23.zw, r12.zw\n"
"    dmad r6.xy, r16.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r17.zw, r24.xy, r7.xy\n"
"    dmad r12.zw, r17.zw, r24.zw, r7.zw\n"
"    dmad r7.zw, r17.xy, r23.zw, r12.zw\n"
"    dmad r7.xy, r17.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r18.zw, r24.xy, r8.xy\n"
"    dmad r12.zw, r18.zw, r24.zw, r8.zw\n"
"    dmad r8.zw, r18.xy, r23.zw, r12.zw\n"
"    dmad r8.xy, r18.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r19.zw, r24.xy, r9.xy\n"
"    dmad r12.zw, r19.zw, r24.zw, r9.zw\n"
"    dmad r9.zw, r19.xy, r23.zw, r12.zw\n"
"    dmad r9.xy, r19.xy, r23.xy, r12.xy\n"
"    dmad r12.xy, r20.zw, r24.xy, r10.xy\n"
"    dmad r12.zw, r20.zw, r24.zw, r10.zw\n"
"    dmad r10.zw, r20.xy, r23.zw, r12.zw\n"
"    dmad r10.xy, r20.xy, r23.xy, r12.xy\n"
"    sample_resource(0)_sampler(0) r13, r1.xz\n"
"    sample_resource(1)_sampler(1) r14, r1.xz\n"
"    sample_resource(2)_sampler(2) r15, r1.xz\n"
"    sample_resource(3)_sampler(3) r16, r1.xz\n"
"    sample_resource(4)_sampler(4) r17, r1.xz\n"
"    sample_resource(5)_sampler(5) r18, r1.xz\n"
"    sample_resource(6)_sampler(6) r19, r1.xz\n"
"    sample_resource(7)_sampler(7) r20, r1.xz\n"
"    dmad r12.xy, r13.zw, r26.xy, r3.xy\n"
"    dmad r12.zw, r13.zw, r26.zw, r3.zw\n"
"    dmad r3.zw, r13.xy, r25.zw, r12.zw\n"
"    dmad r3.xy, r13.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r14.zw, r26.xy, r4.xy\n"
"    dmad r12.zw, r14.zw, r26.zw, r4.zw\n"
"    dmad r4.zw, r14.xy, r25.zw, r12.zw\n"
"    dmad r4.xy, r14.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r15.zw, r26.xy, r5.xy\n"
"    dmad r12.zw, r15.zw, r26.zw, r5.zw\n"
"    dmad r5.zw, r15.xy, r25.zw, r12.zw\n"
"    dmad r5.xy, r15.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r16.zw, r26.xy, r6.xy\n"
"    dmad r12.zw, r16.zw, r26.zw, r6.zw\n"
"    dmad r6.zw, r16.xy, r25.zw, r12.zw\n"
"    dmad r6.xy, r16.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r17.zw, r26.xy, r7.xy\n"
"    dmad r12.zw, r17.zw, r26.zw, r7.zw\n"
"    dmad r7.zw, r17.xy, r25.zw, r12.zw\n"
"    dmad r7.xy, r17.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r18.zw, r26.xy, r8.xy\n"
"    dmad r12.zw, r18.zw, r26.zw, r8.zw\n"
"    dmad r8.zw, r18.xy, r25.zw, r12.zw\n"
"    dmad r8.xy, r18.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r19.zw, r26.xy, r9.xy\n"
"    dmad r12.zw, r19.zw, r26.zw, r9.zw\n"
"    dmad r9.zw, r19.xy, r25.zw, r12.zw\n"
"    dmad r9.xy, r19.xy, r25.xy, r12.xy\n"
"    dmad r12.xy, r20.zw, r26.xy, r10.xy\n"
"    dmad r12.zw, r20.zw, r26.zw, r10.zw\n"
"    dmad r10.zw, r20.xy, r25.zw, r12.zw\n"
"    dmad r10.xy, r20.xy, r25.xy, r12.xy\n"
"    sample_resource(0)_sampler(0) r13, r1.yz\n"
"    sample_resource(1)_sampler(1) r14, r1.yz\n"
"    sample_resource(2)_sampler(2) r15, r1.yz\n"
"    sample_resource(3)_sampler(3) r16, r1.yz\n"
"    sample_resource(4)_sampler(4) r17, r1.yz\n"
"    sample_resource(5)_sampler(5) r18, r1.yz\n"
"    sample_resource(6)_sampler(6) r19, r1.yz\n"
"    sample_resource(7)_sampler(7) r20, r1.yz\n"
"    dmad r12.xy, r13.zw, r28.xy, r3.xy\n"
"    dmad r12.zw, r13.zw, r28.zw, r3.zw\n"
"    dmad r3.zw, r13.xy, r27.zw, r12.zw\n"
"    dmad r3.xy, r13.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r14.zw, r28.xy, r4.xy\n"
"    dmad r12.zw, r14.zw, r28.zw, r4.zw\n"
"    dmad r4.zw, r14.xy, r27.zw, r12.zw\n"
"    dmad r4.xy, r14.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r15.zw, r28.xy, r5.xy\n"
"    dmad r12.zw, r15.zw, r28.zw, r5.zw\n"
"    dmad r5.zw, r15.xy, r27.zw, r12.zw\n"
"    dmad r5.xy, r15.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r16.zw, r28.xy, r6.xy\n"
"    dmad r12.zw, r16.zw, r28.zw, r6.zw\n"
"    dmad r6.zw, r16.xy, r27.zw, r12.zw\n"
"    dmad r6.xy, r16.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r17.zw, r28.xy, r7.xy\n"
"    dmad r12.zw, r17.zw, r28.zw, r7.zw\n"
"    dmad r7.zw, r17.xy, r27.zw, r12.zw\n"
"    dmad r7.xy, r17.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r18.zw, r28.xy, r8.xy\n"
"    dmad r12.zw, r18.zw, r28.zw, r8.zw\n"
"    dmad r8.zw, r18.xy, r27.zw, r12.zw\n"
"    dmad r8.xy, r18.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r19.zw, r28.xy, r9.xy\n"
"    dmad r12.zw, r19.zw, r28.zw, r9.zw\n"
"    dmad r9.zw, r19.xy, r27.zw, r12.zw\n"
"    dmad r9.xy, r19.xy, r27.xy, r12.xy\n"
"    dmad r12.xy, r20.zw, r28.xy, r10.xy\n"
"    dmad r12.zw, r20.zw, r28.zw, r10.zw\n"
"    dmad r10.zw, r20.xy, r27.zw, r12.zw\n"
"    dmad r10.xy, r20.xy, r27.xy, r12.xy\n"
"    add r11.x___, r11.x, r11.1\n"
"endloop\n"
"dcl_output_generic o0\n"
"dcl_output_generic o1\n"
"dcl_output_generic o2\n"
"dcl_output_generic o3\n"
"dcl_output_generic o4\n"
"dcl_output_generic o5\n"
"dcl_output_generic o6\n"
"dcl_output_generic o7\n"
"dmul r3.xy, r3.xy, cb0[1].zw\n"
"dmul r3.zw, r3.zw, cb0[1].zw\n"
"dmul r4.xy, r4.xy, cb0[1].zw\n"
"dmul r4.zw, r4.zw, cb0[1].zw\n"
"dmul r5.xy, r5.xy, cb0[1].zw\n"
"dmul r5.zw, r5.zw, cb0[1].zw\n"
"dmul r6.xy, r6.xy, cb0[1].zw\n"
"dmul r6.zw, r6.zw, cb0[1].zw\n"
"dmul r7.xy, r7.xy, cb0[1].zw\n"
"dmul r7.zw, r7.zw, cb0[1].zw\n"
"dmul r8.xy, r8.xy, cb0[1].zw\n"
"dmul r8.zw, r8.zw, cb0[1].zw\n"
"dmul r9.xy, r9.xy, cb0[1].zw\n"
"dmul r9.zw, r9.zw, cb0[1].zw\n"
"dmul r10.xy, r10.xy, cb0[1].zw\n"
"dmul r10.zw, r10.zw, cb0[1].zw\n"
"mov o0, r3\n"
"mov o1, r4\n"
"mov o2, r5\n"
"mov o3, r6\n"
"mov o4, r7\n"
"mov o5, r8\n"
"mov o6, r9\n"
"mov o7, r10\n"
"end\n";
