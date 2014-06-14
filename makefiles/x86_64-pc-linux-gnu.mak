CUDAPATH					= $(CUDA_PATH)
CUDASDKPATH					= $(CUDAPATH)/sdk
AMDPATH						= $(AMDAPPSDKROOT)
INTELPATH					= /opt/intel/composerxe-2013.0.080

GCC3264						= c++
CLANG3264					= clang
ICC32						= $(INTELPATH)/bin/ia32/icc
ICC64						= $(INTELPATH)/bin/intel64/icc

#Intel Compiler Options
INTELOPENMP					= -openmp -openmp-link:static -parallel
#INTELOPENMP				= -openmp-stubs
INTELFLAGSOPT				= -O3 -fno-alias -fno-fnalias -x$(INTELARCH) -unroll -unroll-aggressive -openmp -g0
ifeq ($(CONFIG_LTO), 1)
INTELFLAGSOPT				+= -ipo
INTELLINKIPO				= -ipo-c -ipo-fo
else
INTELFLAGSOPT				+= -ip
endif
INTELFLAGSDBG				= -O0 -openmp-stubs -g
INTELFLAGSCOMMON			= -DINTEL_RUNTIME $(INTELFLAGSUSE) -fasm-blocks
INTELFLAGS32				= $(INTELFLAGSCOMMON) -m32
INTELFLAGS64				= $(INTELFLAGSCOMMON) -m64 -D_AMD64_

ifeq ($(GCCARCH), )
GCCARCHA						= -march=native -msse4.2 -m$(ARCHBITS)
else
GCCARCHA						= -march=$(GCCARCH) -msse4.2 -m$(ARCHBITS)
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
INTELLIBPATH					= $(INTELPATH)/compiler/lib/intel64
CLANG						= $(CLANG3264) $(GCCFLAGS64) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE) $(COMPILETARGETTYPE)
else
ICC							= $(ICC32) $(INTELFLAGS32) $(CFLAGS32) $(COMPILETARGETTYPE)
GCC							= $(GCC3264) $(GCCFLAGS32) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE) $(COMPILETARGETTYPE)
CCDBG						= $(GCC3264) $(GCCFLAGS32) $(GCCFLAGSCOMMON) $(GCCFLAGSDBG) $(COMPILETARGETTYPE) -DDEBUG_RUNTIME
GCCLINK						= $(GCC3264) $(LINKFLAGS32) -fopenmp
ICCLINK						= $(GCC3264) $(LINKFLAGS32) -openmp
CUDALIBPATH					= $(CUDAPATH)/lib
AMDLIBPATH					= $(AMDPATH)/lib/x86
INTELLIBPATH					= $(INTELPATH)/compiler/lib/ia32
CLANG						= $(CLANG3264) $(GCCFLAGS32) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE) $(COMPILETARGETTYPE)
endif
QTUIC						= uic
QTMOC						= moc

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
LIBSUSE						+= -lintlc -lsvml -limf -lirc
endif
endif
CC_SELECTED					= $(CC_x86_64-pc-linux-gnu)

CCCUDA						= $(GCC) -x c++ -Wno-effc++
ASM							= yasm
ASMPRE						= $(GCC3264)
NVCC						= $(CUDAPATH)/bin/nvcc

COMMONINCLUDEPATHS			=
LIBPATHSUSE					=

ifneq ($(CUFILES), )
LIBSUSE						+= -lcudart -lcuda
ifeq ($(CONFIG_CUDA_DC), 1)
LIBSUSE						+= -lcudadevrt
endif
endif
#$(CUDASDKPATH)/C/lib/libcutil.a

OPENCLLIB					= -lOpenCL
ifeq ("$(CONFIG_OPENCL)", "1")
LIBSUSE						+= $(OPENCLLIB)
endif
ifeq ("$(CONFIG_CAL)", "1")
LIBSUSE						+= -laticalcl -laticalrt
COMMONINCLUDEPATHS			+= $(AMDPATH)/include/CAL
LIBPATHSUSE					+= -L$(AMDLIBPATH)
endif
ifeq ("$(CONFIG_OPENGL)", "1")
LIBSUSE						+= -lGL -lGLU -lglut -lGLEW
endif
ifeq ("$(CONFIG_X11)", "1")
LIBSUSE						+= -lX11
endif

ifeq ("$(CONFIG_QT)", "1")
LIBSUSE						+= -lQtGui -lQtCore
COMMONINCLUDEPATHS			+= /usr/include/qt4 $(WORKPATH)/qt
LIBPATHSUSE					+= -L/usr/lib/qt4
endif

LIBSUSE						+= $(LIBS:%=-l%)

ifeq ("$(CONFIG_OPENCL)", "1")
ifeq ("$(CONFIG_OPENCL_VERSION)", "AMD")
COMMONINCLUDEPATHS			+= "$(AMDPATH)/include"
-L$(AMDLIBPATH)
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
LIBPATHSUSE					+= -L$(AMDLIBPATH)
endif
endif

ifeq ("$(CONFIG_CUDA)", "1")
COMMONINCLUDEPATHS			+= "$(CUDAPATH)/include"
COMMONINCLUDEPATHS			+= "$(CUDASDKPATH)/common/inc"
LIBPATHSUSE					+= -L$(CUDALIBPATH)
endif

INCLUDEPATHSUSE				= $(GCCINCLUDEPATHS)
DEFINESUSE					= $(GCCDEFINES)

LIBPATHSUSE					+= -L$(INTELLIBPATH) $(LIBPATHS:%=-L%)

NVCCARCHS					:= `for i in $(CUDAVERSION); do echo -n -gencode arch=compute_$$i,code=sm_$$i\ ;done`
NVCC_GREP					= "^#line\|^$$\|^# [0-9]* "

COMPILEOUTPUTBASE			= -o
COMPILEOUTPUT				= $(COMPILEOUTPUTBASE) $@
LINKOUTPUT					= -o $@
COMPILEONLY					= -c
ASMONLY						=
PRECOMPILEONLY				= -x c++ -E
OPTIONINCLUDEPATH			= -I
OBJ							= o
