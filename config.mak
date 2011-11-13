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

INCLUDE_OPENCL				= 1
INCLUDE_CAL					= 1
DEFINES						= _NO_AMD_CPU
LIBS						=
EXTRAOBJFILES				=

CONFIG_STATIC				= 0
EXTRAFLAGSGCC				= 

INCLUDEPATHS				= ../GotoBLAS2

CPPFILES					= caldgemm.cpp benchmark.cpp cmodules/timer.cpp
CXXFILES					=
ASMFILES					=
CUFILES						=

INTELFLAGSUSE				= $(INTELFLAGSDBG)
VSNETFLAGSUSE				= $(VSNETFLAGSOPT)
GCCFLAGSUSE					= $(GCCFLAGSOPT)
NVCCFLAGSUSE				= $(NVCCFLAGSOPT)

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

ifeq ($(ARCH), i686-pc-cygwin)
EXTRAOBJFILES				+= ../GotoBLAS2/libgoto2.lib
else
LIBS						+= gfortran
EXTRAOBJFILES				+= ../GotoBLAS2/libgoto2.a
endif

caldgemm_config.h:			caldgemm_config.sample
							cp caldgemm_config.sample caldgemm_config.h

ALLDEP						+= caldgemm_config.h
