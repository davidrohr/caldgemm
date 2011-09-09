#Set these Compiler Paths and Variables to your needs! 
VSPATH						= ${VS90COMNTOOLS}../..
VSPATH8						= ${VS80COMNTOOLS}../..
VSPATH6						= c:/Utility/Speeches/Visual Studio 6
ICCPATH						= ${ICPP_COMPILER11}
GCCPATH						= c:/Utility/Speeches/gcc
CYGWINPATH					= /cygdrive/c/Utility/Cygwin
WINPATH						= /cygdrive/c/Windows
CUDAPATH					= c:/Utility/Speeches/cuda/v4.0
AMDPATH						= c:/Utility/Speeches/stream/dev

ICCPATH32					= $(ICCPATH)bin/ia32
ICCPATH64					= $(ICCPATH)bin/intel64

ICC32						= $(HIDEECHO) $(CALLVC) "$(ICCPATH32)/iclvars_ia32.bat" $(HIDEVARS) "$(ICCPATH32)/icl.exe"
ICC64						= $(HIDEECHO) $(CALLVC) "$(ICCPATH64)/iclvars_intel64.bat" $(HIDEVARS) "$(ICCPATH64)/icl.exe"
MSCC32						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/cl.exe"
MSCC64						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvarsamd64.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/amd64/cl.exe"
MASM32						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/ml.exe"
MASM64						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvarsamd64.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/amd64/ml64.exe"
VCC32						= $(HIDEECHO) "c:/Utility/speeches/Codeplay/vectorc86.exe"

MSLINK32GCC					= $(HIDEECHO) $(CALLVC) "$(ICCPATH32)/iclvars_ia32.bat" $(HIDEVARS) "$(VSPATH8)/VC/bin/link.exe"
MSLINK32					= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/VC/bin/link.exe"
MSLINK64					= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvarsamd64.bat" $(HIDEVARS) "$(VSPATH)/VC/bin/amd64/link.exe"
ICCLINK32					= $(HIDEECHO) $(CALLVC) "$(ICCPATH32)/iclvars_ia32.bat" $(HIDEVARS) "$(ICCPATH32)/xilink.exe" -quseenv
ICCLINK64					= $(HIDEECHO) $(CALLVC) "$(ICCPATH64)/iclvars_intel64.bat" $(HIDEVARS) "$(ICCPATH64)/xilink.exe" -quseenv

#Linker Optionss
LINKFLAGSCOMMON				= /fixed:no /nologo /subsystem:console /incremental:no /debug $(MULTITHREADLIBS) $(DNDVERSION) /MANIFEST:NO $(HOARD) /pdb:"$(WORKPATH)/$(TARGET).pdb"
LINKFLAGS32					= $(LINKFLAGSCOMMON) /machine:I386
LINKFLAGS64					= $(LINKFLAGSCOMMON) /machine:X64

#Common Compiler Options
PREHEADER					= /Fp"$@.pch" /Fd"$@.pdb"
CFLAGSCOMMON				= $(PREHEADER) /nologo /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /W3 $(MULTITHREAD)
CFLAGS32					= $(CFLAGSCOMMON)
CFLAGS64					= $(CFLAGSCOMMON) /D "_WIN64" /D "_AMD64_" /D "_X64_" 
DEBUGFLAGS					= /EHs /Zi /Od /D "DEBUG_RUNTIME"

#/Qprof_gen, /Qprof_use
INTELQPROF					= 
INTELOPENMP					= /Qopenmp /Qopenmp-link:static /Qparallel
#INTELOPENMP				= /Qopenmp-stubs
MSOPENMP					= 
#/openmp

#Intel Compiler Options
INTELFLAGSOPT				= /Oa /Ow /Ob2 /Ot /Oi /GA /G7 /O3 /Qip /Qvec_report0 /Qopt-prefetch /Qax$(INTELARCH) /Gs0 /Zd $(INTELOPENMP)
INTELFLAGSDBG				= /Od /Zi /Qopenmp-stubs
INTELFLAGSBASE				= /EHsc /D "INTEL_RUNTIME" /Qvc9 /Qprof_dir$(WORKPATH) $(MULTITHREAD) $(INTELQPROF)
INTELFLAGSCOMMON			= $(INTELFLAGSBASE) $(INTELFLAGSUSE)
INTELFLAGS32				= $(INTELFLAGSCOMMON) /Oy /Og /Gr
INTELFLAGS64				= $(INTELFLAGSCOMMON)
# /Zd /Zi /Qvec_report0 

GCCFLAGSARCHOPT				= -march=opteron

#VectorC Compiler Options
VECTORCOPTIMIZED			= /ssecalls /optimize 10 /max /target p4 /autoinline 4096 /vc /Ob2 /Oi /Ot
VECTORCSTANDARD				= /optimize 0 /novectors /vc /Ob0
VECTORCFLAGS				= /nologo /noprogress /vserror /cpp /mslibs $(VECTORCSTANDARD) /c /D "VECTORC_RUNTIME" $(MULTITHREAD) /I"$(VSPATH6)/VC98/include" $(VC8INCLUDES)

#Visual Studio Compiler Options
VSNETFLAGSOPT				= /EHs /O2 /Ox /Oi /Ot /Oy /GA /Ob2 /Zi /Qfast_transcendentals $(MSOPENMP)
VSNETFLAGSDBG				= /Od /Zi
VSNETFLAGSCOMMON			=  /D "VSNET_RUNTIME" $(VSNETFLAGSUSE) $(EXTRAFLAGSMSCC) /EHsc
VSNETFLAGS32				= $(VSNETFLAGSCOMMON)
VSNETFLAGS64				= $(VSNETFLAGSCOMMON) /favor:INTEL64

GCC32						= "$(GCCPATH)/bin/g++.exe"
GCC64						= x86_64-w64-mingw32-c++.exe

#Compilation Output Control
ifndef HIDEECHO
HIDEECHO					= @
endif
ifeq ($(HIDEECHO), "-")
HIDEECHO					=
endif
ifndef HIDEVARS
HIDEVARS					= 1
endif

CUDASDKPATH					= $(CUDAPATH)/sdk
DIRECTXPATH					= c:/Utility/Speeches/sdk/directx

CALLVC						= $(HIDEECHO) cmd /C "makefiles\callvc.bat"

PATH						= $(CYGWINPATH)/bin:$(CYGWINPATH)/usr/bin:$(WINPATH):$(WINPATH)/system32

ifeq ($(ARCHBITS), 64)
ICC							= $(ICC64) $(INTELFLAGS64) $(CFLAGS64)
CCDBG						= $(ICC64) $(INTELFLAGSBASE) $(INTELFLAGSDBG) $(CFLAGS64) $(DEBUGFLAGS)
ICCLINK						= $(ICCLINK64) $(LINKFLAGS64)
MSCC						= $(MSCC64) $(VSNETFLAGS64) $(CFLAGS64)
MSLINK						= $(MSLINK64) $(LINKFLAGS64)
GCC							= $(GCC64) $(GCCFLAGS64)
MASM						= $(MASM64)
CCCUDA						= $(MSCC) /TP
LIBPATHSUSE					= /LIBPATH:"$(CUDAPATH)/lib/x64" /LIBPATH:"$(AMDPATH)/lib" /LIBPATH:"$(AMDPATH)/lib/x86_64" /LIBPATH:"$(CUDAPATH)/sdk/C/common/lib" /LIBPATH:"$(DIRECTXPATH)/lib/x64" /LIBPATH:"$(ICCPATH)/lib/intel64"
else
ICC							= $(ICC32) $(INTELFLAGS32) $(CFLAGS32)
CCDBG						= $(MSCC32) $(CFLAGS32) $(DEBUGFLAGS)
ICCLINK						= $(ICCLINK32) $(LINKFLAGS32)
MSCC						= $(MSCC32) $(VSNETFLAGS32) $(CFLAGS32) /Gr
MSLINK						= $(MSLINK32) $(LINKFLAGS32)
MSLINKGCC					= $(MSLINK32GCC) $(LINKFLAGS32)
VCC							= $(VCC32) /outfile $@ $(VECTORCFLAGS) $(CFLAGS32)
GCC							= $(GCC32) $(GCCFLAGS32)
MASM						= $(MASM32)
CCCUDA						= $(MSCC32) $(VSNETFLAGS32) $(CFLAGS32) /TP /Gd
LIBPATHSUSE					= /LIBPATH:"$(CUDAPATH)/lib/win32" /LIBPATH:"$(AMDPATH)/lib" /LIBPATH:"$(AMDPATH)/lib/x86" /LIBPATH:"$(DIRECTXPATH)/lib/x86" /LIBPATH:"$(ICCPATH)/lib/ia32"
endif

LIBPATHSUSE					+= $(LIBPATHS:%=/LIBPATH:%)

ifeq ($(CC_i686-pc-cygwin), ICC)
CC							= $(ICC)
ifeq ($(CPPFILES_GCC), )
LINK						= $(ICCLINK)
else
LINK						= $(MSLINKGCC)
endif
else ifeq ($(CC_i686-pc-cygwin), GCC)
CC							= $(GCC)
LINK						= $(GCC)
else
CC							= $(MSCC)
LINK						= $(MSLINK)
endif
GCC3264						= $(GCC)

ASM							= $(MASM)
ASMPRE						= $(MSCC32)
NVCC						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(CUDAPATH)/bin/nvcc"

MULTITHREADGCC				= -mthreads -D_MT

LIBSUSE						= kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib

ifneq ($(CPPFILES_GCC), )
LIBSUSE						+= /LIBPATH:"$(GCCPATH)/lib" /LIBPATH:"`$(GCC) -print-libgcc-file-name | sed -e s/libgcc.a//`"  libgcc.a libstdc++.a libmingw32.a libgcov.a libmingwex.a
endif

ifneq ($(CUFILES), )
LIBSUSE						+= cudart.lib cuda.lib
endif

ifeq ("$(CONFIG_OPENCL)", "1")
LIBSUSE						+= OpenCL.lib
endif
ifeq ("$(CONFIG_CAL)", "1")
ifeq ($(ARCHBITS), 64)
LIBSUSE						+= aticalcl64.lib aticalrt64.lib
else
LIBSUSE						+= aticalcl.lib aticalrt.lib
endif
endif
ifeq ("$(CONFIG_DDRAW)", "1")
LIBSUSE						+= ddraw.lib dxguid.lib
endif
ifeq ("$(CONFIG_VIDE_EDIT)", "1")
LIBSUSE						+= amstrmid.lib msacm32.lib vfw32.lib winmm.lib
endif
ifeq ("$(CONFIG_OPENGL)", "1")
LIBSUSE						+= opengl32.lib glu32.lib
endif

LIBSUSE						+= $(LIBS:%=%.lib)

ifeq ($(TARGETTYPE), LIB)
LINKTARGETTYPE				= /DLL
EXECUTABLE					= $(TARGET).dll
else
LINKTARGETTYPE				=
EXECUTABLE					= $(TARGET).exe
endif

COMMONINCLUDEPATHS			= "$(DIRECTXPATH)/include" "$(CUDAPATH)/include" $(AMDPATH)/include "$(CUDASDKPATH)/C/common/inc"

ifeq ($(CC_i686-pc-cygwin), GCC)
COMPILEOUTPUT				= -o $@
LINKOUTPUT					= -o $@
COMPILEONLY					= -c
PRECOMPILEONLY				= -blakfa
INCLUDEPATHSUSE				= $(GCCINCLUDEPATHS)
DEFINESUSE					= $(GCCDEFINES)
else
INCLUDEPATHSUSE				= $(VSINCLUDEPATHS)
DEFINESUSE					= $(VSDEFINES)
COMPILEOUTPUT				= /Fo"$@"
LINKOUTPUT					= /Out:"$@"
COMPILEONLY					= /c
PRECOMPILEONLY				= /EP 
endif

DEFINESARCH					= "WIN32"

NVCCARCHS					:= `for i in $(CUDAVERSION); do echo -n -gencode arch BAT_SPECIAL_EQ compute_$$i BAT_SPECIAL_KOMMA code BAT_SPECIAL_EQ sm_$$i\ ;done`
NVCC_GREP					= "^#line\|^$$"
DCUDAEMU					= /DCUDA_DEVICE_EMULATION

WORKPATHSUFFIX				=