CUDAPATH					= /usr/local/cuda
CUDASDKPATH					= $(CUDAPATH)/sdk
AMDPATH						= $(ATISTREAMSDKROOT)

GCC3264						= c++
ICC32						= icc
ICC64						= icc

#Intel Compiler Options
INTELOPENMP					= -openmp -openmp-link:static -parallel
#INTELOPENMP				= -openmp-stubs
INTELFLAGSOPT				= -O3 -fno-alias -fno-fnalias -ax$(INTELARCH) -unroll -unroll-aggressive -openmp -g0
INTELFLAGSDBG				= -O0 -openmp-stubs -g
INTELFLAGSCOMMON			= -DINTEL_RUNTIME $(INTELFLAGSUSE)
INTELFLAGS32				= $(INTELFLAGSCOMMON) -m32
INTELFLAGS64				= $(INTELFLAGSCOMMON) -m64

GCCFLAGSARCHCOMMON			= -fopenmp
GCCFLAGSARCHOPT				= -march=native

#GCC link flags
LINKFLAGSCOMMON				= -Wall -ggdb
ifeq ($(CONFIG_STATIC), 1)
LINKFLAGSCOMMON				+= -static
endif
LINKFLAGS32					= -m32 $(LINKFLAGSCOMMON)
LINKFLAGS64					= -m64 $(LINKFLAGSCOMMON)


ifeq ($(ARCHBITS), 64)
ICC							= $(ICC64) $(INTELFLAGS64) $(CFLAGS64) $(COMPILETARGETTYPE)
GCC							= $(GCC3264) $(GCCFLAGS64) $(COMPILETARGETTYPE)
CCDBG						= $(GCC3264) $(GCCFLAGS64) $(GCCFLAGSDBG) $(COMPILETARGETTYPE)
GCCLINK						= $(GCC3264) $(LINKFLAGS64) -fopenmp
ICCLINK						= $(ICC64) $(LINKFLAGS64) -openmp
CUDALIBPATH					= $(CUDAPATH)/lib64
AMDLIBPATH					= $(AMDPATH)/lib/x86_64
else
ICC							= $(ICC32) $(INTELFLAGS32) $(CFLAGS32) $(COMPILETARGETTYPE)
GCC							= $(GCC3264) $(GCCFLAGS32) $(COMPILETARGETTYPE)
CCDBG						= $(GCC3264) $(GCCFLAGS32) $(GCCFLAGSDBG) $(COMPILETARGETTYPE)
GCCLINK						= $(GCC3264) $(LINKFLAGS32) -fopenmp
ICCLINK						= $(GCC3264) $(LINKFLAGS32) -openmp
CUDALIBPATH					= $(CUDAPATH)/lib
AMDLIBPATH					= $(AMDPATH)/lib/x86
endif

ifeq ($(CC_x86_64-pc-linux-gnu), ICC)
CC							= $(ICC)
LINK						= $(ICCLINK)
else
CC							= $(GCC)
LINK						= $(GCCLINK)
endif

CCCUDA						= $(GCC) -x c++
ASM							= yasm
ASMPRE						= $(GCC3264)
NVCC						= $(CUDAPATH)/bin/nvcc

ifeq ($(TARGETTYPE), LIB)
LINKTARGETTYPE				= -shared
COMPILETARGETTYPE			= -fPIC
EXECUTABLE					= $(TARGET).so
LIBGLIBC					=
else
LINKTARGETTYPE				=
COMPILETARGETTYPE			=
EXECUTABLE					= $(TARGET)
ifeq ($(ARCHBITS), 64)
LIBGLIBC					= `$(GCC3264) -print-libgcc-file-name` `$(GCC3264) -print-libgcc-file-name | sed -e s/libgcc/libstdc++/`
else
LIBGLIBC					= 
endif
endif

LIBSUSE						= $(LIBGLIBC) -lrt -ldl

ifneq ($(CUFILES), )
LIBSUSE						+= -lcudart -lcuda
endif
#$(CUDASDKPATH)/C/lib/libcutil.a

ifeq ("$(CONFIG_OPENCL)", "1")
LIBSUSE						+= -lOpenCL
endif
ifeq ("$(CONFIG_CAL)", "1")
LIBSUSE						+= -laticalcl.lib -laticalrt.lib
endif
ifeq ("$(CONFIG_OPENGL)", "1")
LIBSUSE						+= -lGL -lGLU
endif


LIBSUSE						+= $(LIBS:%=-l%)

COMPILEOUTPUT				= -o $@
LINKOUTPUT					= -o $@
COMPILEONLY					= -c
PRECOMPILEONLY				= -blakfa

INCLUDEPATHSUSE				= $(GCCINCLUDEPATHS)
DEFINESUSE					= $(GCCDEFINES)

LIBPATHSUSE					= -L$(CUDALIBPATH) -L$(AMDLIBPATH) $(LIBPATHS:%=-L%)

NVCCARCHS					:= `for i in $(CUDAVERSION); do echo -n -gencode arch=compute_$$i,code=sm_$$i\ ;done`
NVCC_GREP					= "^#line\|^$$\|^# [0-9]* "
DCUDAEMU					= -DCUDA_DEVICE_EMULATION

COMMONINCLUDEPATHS			= "$(CUDASDKPATH)/C/common/inc" "$(AMDPATH)/include"
