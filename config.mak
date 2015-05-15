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

CPPFILES					= caldgemm.cpp benchmark.cpp cmodules/timer.cpp cmodules/qmalloc.cpp caldgemm_cpu.cpp cmodules/affinity.cpp cmodules/threadserver.cpp cmodules/qsem.cpp caldgemm_adl.cpp
CXXFILES					=
ASMFILES					=
CUFILES						=

COMPILER_FLAGS				= OPT

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

ifeq ($(BLAS_BACKEND), GOTOBLAS)
INCLUDEPATHS				+= $(GOTOBLAS_PATH)
DEFINES						+= USE_GOTO_BLAS
ifeq ($(ARCH), i686-pc-cygwin)
EXTRAOBJFILES				+= $(GOTOBLAS_PATH)/libgoto2.lib
else
#LIBS						+= gfortran
EXTRAOBJFILES				+= $(GOTOBLAS_PATH)/libgoto2.a
endif
else
ifeq ($(BLAS_BACKEND), MKL)
INCLUDEPATHS				+= $(MKL_PATH)/include
LIBS						+= iomp5 mkl_intel_lp64 mkl_core mkl_intel_thread
LIBPATHS					+= $(MKL_PATH)/lib/intel64/
ifneq ($(ICC_PATH), )
LIBPATHS					+= $(ICC_PATH)/lib/intel64/
endif
DEFINES						+= USE_MKL
CONFIG_OPENMP				= 1
else
ifeq ($(BLAS_BACKEND), ACML)
INCLUDEPATHS				+= $(CBLAS_PATH)/include
LIBPATHS				+= $(ACML_PATH)/lib $(CBLAS_PATH)/include
LIBS					+= acml_mp
EXTRAOBJFILES				+= $(CBLAS_PATH)/lib/cblas_LINUX.a
CONFIG_OPENMP				= 1
LIBS						+= gfortran
else
error No valid BLAS_BACKEND selected
endif
endif
endif

INCLUDEPATHS				+= $(OPENMPI_PATH)/include/vampirtrace

endif

caldgemm_config.h:
							cp caldgemm_config.sample caldgemm_config.h

ALLDEP						+= caldgemm_config.h

config_options.mak:
							cp config_options.sample config_options.mak 

FILEFLAGSbenchmark.cpp			= -Wno-strict-aliasing
FILEFLAGScaldgemm.cpp			= -Wno-strict-aliasing
FILEFLAGScaldgemm_cal.cpp			= -Wno-strict-aliasing
FILEFLAGScaldgemm_opencl.cpp			= -Wno-strict-aliasing
