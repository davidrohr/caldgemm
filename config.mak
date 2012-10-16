include config_options.mak

INTELARCH					= SSE4.2
CUDAVERSION					= 20
CUDAREGS					= 64
ARCHBITS					= 64

HIDEECHO					= @
CC_x86_64-pc-linux-gnu		= GCC
CC_i686-pc-cygwin			= ICC

TARGET						= dgemm_bench

LIBS						= 
LIBPATHS					= 

LIBS						=
EXTRAOBJFILES				=

CONFIG_STATIC				= 0
EXTRAFLAGSGCC				= 

CPPFILES					= caldgemm.cpp benchmark.cpp cmodules/timer.cpp cmodules/qmalloc.cpp caldgemm_cpu.cpp cmodules/affinity.cpp cmodules/threadserver.cpp cmodules/qsem.cpp
CXXFILES					=
ASMFILES					=
CUFILES						=

INTELFLAGSUSE				= $(INTELFLAGSOPT)
VSNETFLAGSUSE				= $(VSNETFLAGSOPT)
GCCFLAGSUSE					= $(GCCFLAGSOPT)
NVCCFLAGSUSE				= $(NVCCFLAGSOPT)

ifeq ($(AMDAPPSDKROOT), )
INCLUDE_CAL					= 0
endif

ifeq ("$(CUDA_PATH)", "")
INCLUDE_CUDA				= 0
endif

ifeq ($(CONFIGURED), 1)

ifeq ($(INCLUDE_CUDA), 1)
CONFIG_CUDA					= 1
CUFILES						+= caldgemm_cuda.cu
DEFINES						+= CALDGEMM_CUDA
endif

ifeq ($(INCLUDE_OPENCL), 1)
CONFIG_OPENCL				= 1
CPPFILES					+= caldgemm_opencl.cpp
DEFINES						+= CALDGEMM_OPENCL
endif

ifeq ($(INCLUDE_CAL), 1)
CONFIG_CAL					= 1
CPPFILES					+= caldgemm_cal.cpp
DEFINES						+= CALDGEMM_CAL
endif

ifeq ($(USE_GOTO_BLAS), 1)
INCLUDEPATHS				+= ../GotoBLAS2
DEFINES						+= USE_GOTO_BLAS
ifeq ($(ARCH), i686-pc-cygwin)
EXTRAOBJFILES				+= ../GotoBLAS2/libgoto2.lib
else
#LIBS						+= gfortran
EXTRAOBJFILES				+= ../GotoBLAS2/libgoto2.a
endif
else
ifeq ($(USE_MKL_NOT_ACML), 1)
INCLUDEPATHS				+= $(MKL_PATH)/include
LIBS						+= iomp5 mkl_intel_lp64 mkl_core mkl_intel_thread
LIBPATHS					+= $(MKL_PATH)/lib/intel64/
ifneq ($(ICC_PATH), )
LIBPATHS					+= $(ICC_PATH)/lib/intel64/
endif
DEFINES						+= USE_MKL
CONFIG_OPENMP				= 1
else
INCLUDEPATHS				+= ../acml-cblas/include
EXTRAOBJFILES				+= ../acml-cblas/lib/cblas_LINUX.a ../acml/gfortran64_fma4_mp/lib/libacml_mp.a
CONFIG_OPENMP				= 1
LIBS						+= gfortran
endif
endif

endif

caldgemm_config.h:			caldgemm_config.sample
							cp caldgemm_config.sample caldgemm_config.h

ALLDEP						+= caldgemm_config.h

config_options.mak:
							cp config_options.sample config_options.mak 

FILEFLAGScaldgemm.cpp			= -Wno-strict-aliasing
FILEFLAGScaldgemm_cal.cpp			= -Wno-strict-aliasing
