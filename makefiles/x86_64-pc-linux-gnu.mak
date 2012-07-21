CUDAPATH					= $(CUDA_PATH)
CUDASDKPATH					= $(CUDAPATH)/sdk
AMDPATH						= $(AMDAPPSDKROOT)

GCC3264						= c++
ICC32						= icc
ICC64						= icc

#Intel Compiler Options
INTELOPENMP					= -openmp -openmp-link:static -parallel
#INTELOPENMP				= -openmp-stubs
INTELFLAGSOPT				= -O3 -fno-alias -fno-fnalias -ax$(INTELARCH) -unroll -unroll-aggressive -openmp -g0
INTELFLAGSDBG				= -O0 -openmp-stubs -g
INTELFLAGSCOMMON			= -DINTEL_RUNTIME $(INTELFLAGSUSE) -fasm-blocks
INTELFLAGS32				= $(INTELFLAGSCOMMON) -m32
INTELFLAGS64				= $(INTELFLAGSCOMMON) -m64 -D_AMD64_

ifeq ($(GCCARCH), )
GCCARCH						= -march=native -msse4.2 -m$(ARCHBITS)
endif

ifeq ("$(CONFIG_OPENMP)", "1")
INTELFLAGSOPT				+= -openmp
INTELFLAGSDBG				+= -openmp-stubs
GCCFLAGSARCH				+= -fopenmp
endif

#GCC link flags
LINKFLAGSCOMMON				= -Wall -ggdb
ifeq ($(CONFIG_STATIC), 1)
LINKFLAGSCOMMON				+= -static
endif
LINKFLAGS32					= -m32 $(LINKFLAGSCOMMON)
LINKFLAGS64					= -m64 $(LINKFLAGSCOMMON)


ifeq ($(ARCHBITS), 64)
ICC							= $(ICC64) $(INTELFLAGS64) $(CFLAGS64) $(COMPILETARGETTYPE)
GCC							= $(GCC3264) $(GCCFLAGS64) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE) $(COMPILETARGETTYPE)
CCDBG						= $(GCC3264) $(GCCFLAGS64) $(GCCFLAGSCOMMON) $(GCCFLAGSDBG) $(COMPILETARGETTYPE) -DDEBUG_RUNTIME
GCCLINK						= $(GCC3264) $(LINKFLAGS64) -fopenmp
ICCLINK						= $(ICC64) $(LINKFLAGS64) -openmp
CUDALIBPATH					= $(CUDAPATH)/lib64
AMDLIBPATH					= $(AMDPATH)/lib/x86_64
else
ICC							= $(ICC32) $(INTELFLAGS32) $(CFLAGS32) $(COMPILETARGETTYPE)
GCC							= $(GCC3264) $(GCCFLAGS32) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE) $(COMPILETARGETTYPE)
CCDBG						= $(GCC3264) $(GCCFLAGS32) $(GCCFLAGSCOMMON) $(GCCFLAGSDBG) $(COMPILETARGETTYPE) -DDEBUG_RUNTIME
GCCLINK						= $(GCC3264) $(LINKFLAGS32) -fopenmp
ICCLINK						= $(GCC3264) $(LINKFLAGS32) -openmp
CUDALIBPATH					= $(CUDAPATH)/lib
AMDLIBPATH					= $(AMDPATH)/lib/x86
endif

ifeq ($(TARGETTYPE), LIB)
LINKTARGETTYPE				= -shared
COMPILETARGETTYPE			= -fPIC
EXECUTABLE					= $(TARGET).so
else
LINKTARGETTYPE				=
COMPILETARGETTYPE			=
EXECUTABLE					= $(TARGET)
endif
LIBGLIBC					=

LIBSUSE						= $(LIBGLIBC) -lrt -ldl

ifeq ($(CC_x86_64-pc-linux-gnu), ICC)
CC							= $(ICC)
LINK						= $(ICCLINK)
else
CC							= $(GCC)
LINK						= $(GCCLINK)
ifneq ($(CPPFILES_ICC), )
LIBSUSE						+= -lintlc
endif
endif

CCCUDA						= $(GCC) -x c++
ASM							= yasm
ASMPRE						= $(GCC3264)
NVCC						= $(CUDAPATH)/bin/nvcc

ifneq ($(CUFILES), )
LIBSUSE						+= -lcudart -lcuda
endif
#$(CUDASDKPATH)/C/lib/libcutil.a

ifeq ("$(CONFIG_OPENCL)", "1")
LIBSUSE						+= -lOpenCL
endif
ifeq ("$(CONFIG_CAL)", "1")
LIBSUSE						+= -laticalcl -laticalrt
endif
ifeq ("$(CONFIG_OPENGL)", "1")
LIBSUSE						+= -lGL -lGLU
endif


LIBSUSE						+= $(LIBS:%=-l%)

COMPILEOUTPUT				= -o $@
LINKOUTPUT					= -o $@
COMPILEONLY					= -c
ASMONLY						=
PRECOMPILEONLY				= -x c++ -E

INCLUDEPATHSUSE				= $(GCCINCLUDEPATHS)
DEFINESUSE					= $(GCCDEFINES)

LIBPATHSUSE					= -L$(CUDALIBPATH) -L$(AMDLIBPATH) $(LIBPATHS:%=-L%)

NVCCARCHS					:= `for i in $(CUDAVERSION); do echo -n -gencode arch=compute_$$i,code=sm_$$i\ ;done`
NVCC_GREP					= "^#line\|^$$\|^# [0-9]* "
DCUDAEMU					= -DCUDA_DEVICE_EMULATION

COMMONINCLUDEPATHS			=

ifeq ("$(CONFIG_OPENCL)", "1")
ifeq ("$(CONFIG_OPENCL_VERSION)", "AMD")
COMMONINCLUDEPATHS			+= "$(AMDPATH)/include"
endif
ifeq ("$(CONFIG_OPENCL_VERSION)", "NVIDIA")
COMMONINCLUDEPATHS			+= "$(CUDAPATH)/include"
endif
ifeq ("$(CONFIG_OPENCL_VERSION)", "Intel")
#COMMONINCLUDEPATHS			+= ""
endif
ifeq ("$(CONFIG_OPENCL_VERSION)", "All")
COMMONINCLUDEPATHS			+= "$(AMDPATH)/include"
COMMONINCLUDEPATHS			+= "$(CUDAPATH)/include"
#COMMONINCLUDEPATHS			+= ""
endif
endif

ifeq ("$(CONFIG_CUDA)", "1")
COMMONINCLUDEPATHS			+= "$(CUDASDKPATH)/C/common/inc"
endif

ifeq ("$(CONFIG_CAL)", "1")
COMMONINCLUDEPATHS			+= $(AMDPATH)/include/CAL
endif
