all							: all_tmp

ARCH						:= $(shell sort <<< $$MACHTYPE)
ifneq ($(ARCH), i686-pc-cygwin)
ifneq ($(ARCH), i686-pc-linux-gnu)
ifneq ($(ARCH), x86_64-pc-linux-gnu)
ARCH						:= x86_64-pc-linux-gnu
endif
endif
endif
ARCHFILE					= $(ARCH).mak

ifeq ($(CONFIGFILE), )
CONFIGFILE					= config.mak
CLEANRELEASEDIR				= release
endif

ifeq ($(BUILDSCRIPT), )
BUILDSCRIPT					= build.sh
endif

ALLDEP						= makefiles/makefile $(CONFIGFILE) makefiles/$(ARCHFILE) config.mak

#GCC Compiler Options
GCCFLAGSOPT					= -O3 $(GCCFLAGSARCH) -fweb -frename-registers -minline-all-stringops -mfpmath=sse -ftracer -funroll-loops -fpeel-loops -fprefetch-loop-arrays -ffast-math -ggdb
#-fgcse-sm -fgcse-las -fmodulo-sched -fipa-pta -floop-interchange -floop-block 
GCCFLAGSDBG					= -O0 $(GCCFLAGSARCH) -ggdb
GCCFLAGSCOMMON				= $(MULTITHREADGCC) -pipe -DGCC_RUNTIME $(GCCPROF) $(EXTRAFLAGSGCC) -Wall -Wno-write-strings
GCCFLAGS32					= -m32 -mrtd
GCCFLAGS64					= -m64 -D"_AMD64_" -D"_X64_"

GCCPROF						= 
#-fprofile-arcs, -fbranch-probabilities

#Multithread Options
MULTITHREAD					= /MT
MULTITHREADLIBS				= /nodefaultlib:libc.lib

NVCCFLAGSOPT				= --use_fast_math --maxrregcount $(CUDAREGS) -O4 -Xptxas -v -Xptxas -O4 -m$(ARCHBITS) $(NVCCARCHS)
NVCCFLAGSDBG				= --device-emulation --compiler-options $(DCUDAEMU) --maxrregcount $(CUDAREGS) -Xptxas -v -Xptxas -O0 -O0 -m$(ARCHBITS) $(NVCCARCHS)

TARGETTYPE					= EXECUTABLE

include						$(CONFIGFILE)
WORKPATH					= release/$(ARCH)_$(ARCHBITS)$(WORKPATHSUFFIX)
ifeq ($(CONFIG_OPENCL_VERSION), )
CONFIG_OPENCL_VERSION		= All
endif
include						makefiles/$(ARCHFILE)

GCCFLAGSARCH				+= $(GCCARCH)

all_tmp:					$(SUBTARGETS:%=subbuild/%.mak) $(EXECUTABLE)
main:						$(EXECUTABLE)

subbuild/%.mak:
							+$(MAKE) CONFIGFILE=config_`echo $@ | sed s,subbuild/,,` BUILDSCRIPT=config_`echo $@ | sed s,subbuild/,, | sed s,mak,sh,` -f makefile

CUDAINCLUDEPATHS			= $(INCLUDEPATHSUSE:%=--compiler-options %)
CUDADEFINES					= $(DEFINESUSE:%=--compiler-options %)

OBJFILES					= $(CPPFILES:%.cpp=$(WORKPATH)/%.obj) $(CXXFILES:%.cxx=$(WORKPATH)/%.cxobj) $(CUFILES:%.cu=$(WORKPATH)/%.cuobj) $(ASMFILES:%.asm=$(WORKPATH)/%.asmobj) \
								$(CPPFILES_DBG:%.cpp=$(WORKPATH)/%.objdbg) $(CPPFILES_ICC:%.cpp=$(WORKPATH)/%.objicc) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/%.objvcc) $(CPPFILES_GCC:%.cpp=$(WORKPATH)/%.objgcc) \
								$(CPPFILES_MSCC:%.cpp=$(WORKPATH)/%.objmscc)

DEPENDS						= $(CPPFILES:%.cpp=$(WORKPATH)/%.d) $(CXXFILES:%.cxx=$(WORKPATH)/%.cxd) $(CUFILES:%.cu=$(WORKPATH)/%.cud) $(ASMFILES:%.asm=$(WORKPATH)/%.asmd) \
								$(CPPFILES_DBG:%.cpp=$(WORKPATH)/%.dbgd) $(CPPFILES_ICC:%.cpp=$(WORKPATH)/%.iccd) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/%.vccd) $(CPPFILES_GCC:%.cpp=$(WORKPATH)/%.gccd) \
								$(CPPFILES_MSCC:%.cpp=$(WORKPATH)/%.msccd)

ifeq ($(ARCH), i686-pc-cygwin)
GCCINCLUDEPATHSA			:= $(INCLUDEPATHS) $(COMMONINCLUDEPATHS)
GCCINCLUDEPATHSB			:= $(shell cygpath -u $(GCCINCLUDEPATHSA))
GCCINCLUDEPATHS				= $(GCCINCLUDEPATHSB:%=-I%)
else
GCCINCLUDEPATHS				= $(INCLUDEPATHS:%=-I%) $(COMMONINCLUDEPATHS:%=-I%)
endif
VSINCLUDEPATHS				= $(INCLUDEPATHS:%=/I%) $(COMMONINCLUDEPATHS:%=/I%)

ifeq ($(ARCHBITS), 64)
DEFINES						+= _64BIT
endif
GCCDEFINES					= $(DEFINES:%=-D%) $(DEFINESARCH:%=-D%)
VSDEFINES					= $(DEFINES:%=/D%) $(DEFINESARCH:%=/D%)

LIBFILES					= $(LIBSUSE)

.SECONDARY:					$(CUFILES:%.cu=$(WORKPATH)/%.cucpp) $(ASMFILES:%.asm=$(WORKPATH)/%.asmcpp)

$(EXECUTABLE):				$(EXTRADEPS) $(OBJFILES) $(EXTRAOBJFILES) $(ALLDEP)
							$(LINK) $(LIBPATHSUSE) $(OBJFILES) $(EXTRAOBJFILES) $(LIBFILES) $(LINKFLAGSUSE) $(LINKOUTPUT) $(LINKTARGETTYPE)
							$(HIDEECHOA)if [ -e "$(BUILDSCRIPT)" ]; then ./$(BUILDSCRIPT); fi

$(WORKPATH)/%.obj:			%.cpp $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.cxobj:		%.cxx $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.cuobj:		$(WORKPATH)/%.cucpp $(ALLDEP)
							$(CCCUDA) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.asmobj:		$(WORKPATH)/%.asmcpp $(ALLDEP)
							$(ASM) $(COMPILEOUTPUT) $(ASMONLY) $<
							
$(WORKPATH)/%.asmcpp:		%.asm $(ALLDEP)
							$(CC) $(PRECOMPILEONLY) $(FILEFLAGS$<) $(DEFINESUSE) $< > $@

$(WORKPATH)/%.cucpp:		%.cu $(ALLDEP)
							$(NVCC) $(NVCCFLAGSUSE) $(CUDAINCLUDEPATHS) $(CUDADEFINES) --cuda --output-file "$@" $<
							$(HIDEECHOA) cat $@ | grep -v NVCC_GREP > $@.tmp
							$(HIDEECHOA) mv -f $@.tmp $@
							-if [ -e "$<.$(ARCH).patch" ]; then patch -r /dev/null -s --no-backup-if-mismatch -i $<.$(ARCH).patch $@; fi
							
$(WORKPATH)/%.objdbg:		%.cpp $(ALLDEP)
							$(CCDBG) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/%.objicc:		%.cpp $(ALLDEP)
							$(ICC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/%.objvcc:		%.cpp $(ALLDEP)
							$(VCC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $<
$(WORKPATH)/%.objgcc:		%.cpp $(ALLDEP)
							$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) $(FILEFLAGS$<) -c $< -o $@
$(WORKPATH)/%.objmscc:		%.cpp $(ALLDEP)
							$(MSCC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
clean:						$(SUBTARGETS:%=subclean/%.mak)
							rm -Rf *.plg *.dpi *.exp *.lib $(EXECUTABLE) x64/release/* *.cucpp *.cubin *.gpu *.ptx *.linkinfo *.ii cuda.compute_* $(DEPENDS) $(OBJFILES) $(CUFILES:%.cu=$(WORKPATH)/%.cucpp) $(ASMFILES:%.asm=$(WORKPATH)/%.asmcpp) $(CLEANRELEASEDIR)
							
subclean/%.mak:
							+export CONFIGFILE=config_`echo $@ | sed s,subclean/,,` && $(MAKE) -f makefile clean

SAVEDIR						= releases/`date +%F`-BUILD-`cat buildnr`							
backup:						
							$(HIDEECHOA) mkdir $(SAVEDIR)
							cp *.cpp *.h makefile buildnr *.sh *.bat *.conf *.cu $(SAVEDIR)

#Dependancies
$(WORKPATH)/%.cud:			%.cu $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) -I$(GCCPATH)/include $(GCCDEFINES) -D__CUDACC__ -x c++ -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cuobj $(WORKPATH)/$*.cucpp $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.asmd:			%.asm $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -x c -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.asmobj $(WORKPATH)/$*.asmcpp $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.d:			%.cpp $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.obj $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.cxd:			%.cxx $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cxobj $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.dbgd:			%.cpp $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -DDEBUG_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objdbg $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.iccd:			%.cpp $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -DINTEL_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objicc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.vccd:			%.cpp $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -DVECTORC_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objvcc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.gccd:			%.cpp $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -DGCC_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objgcc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.msccd:		%.cpp $(ALLDEP)
							$(HIDEECHOA)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -DVSNET_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objmscc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;																																			

include $(DEPENDS)							
