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
GCCFLAGSOPT					= -O3 $(GCCFLAGSARCHCOMMON) $(GCCFLAGSARCHOPT) -ftree-vectorize -msse3 -fkeep-inline-functions -fweb -frename-registers -minline-all-stringops -funit-at-a-time -mfpmath=sse -ftracer -finline-limit=1200 -funroll-loops -fpeel-loops -fprefetch-loop-arrays -finline-functions
GCCFLAGSDBG					= -O0 $(GCCFLAGSARCHCOMMON)
GCCFLAGSCOMMON				= $(MULTITHREADGCC) -pipe -DGCC_RUNTIME $(GCCPROF) $(GCCFLAGSUSE) $(EXTRAFLAGSGCC) -ggdb -Wall -Wno-write-strings
GCCFLAGS32					= $(GCCFLAGSCOMMON) -m32 -mrtd 
GCCFLAGS64					= $(GCCFLAGSCOMMON) -m64 -D"_AMD64_" -D"_X64_"

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
include						makefiles/$(ARCHFILE)
all_tmp:					$(SUBTARGETS:%=subbuild/%.mak) $(EXECUTABLE)
main:						$(EXECUTABLE)

subbuild/%.mak:
							+$(MAKE) CONFIGFILE=config_`echo $@ | sed s,subbuild/,,` BUILDSCRIPT=config_`echo $@ | sed s,subbuild/,, | sed s,mak,sh,` -f makefile

CUDAINCLUDEPATHS			= $(INCLUDEPATHSUSE:%=--compiler-options %)
CUDADEFINES					= $(DEFINESUSE:%=--compiler-options %)

OBJFILES					= $(CPPFILES:%.cpp=$(WORKPATH)/%.obj) $(CXXFILES:%.cxx=$(WORKPATH)/%.cxobj) $(CUFILES:%.cu=$(WORKPATH)/%.cuobj) $(ASMFILES:%.asm=$(WORKPATH)/%.asmobj) \
								$(CPPFILES_DBG:%.cpp=$(WORKPATH)/%.objdbg) $(CPPFILES_ICC:%.cpp=$(WORKPATH)/%.objicc) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/%.objvcc) $(CPPFILES_GCC:%.cpp=$(WORKPATH)/%.objgcc) \
								$(CPPFILES_MSCC:%.cpp=$(WORKPATH)/%.objmscc) $(EXTRAOBJFILES)

DEPENDS						= $(CPPFILES:%.cpp=$(WORKPATH)/%.d) $(CXXFILES:%.cxx=$(WORKPATH)/%.cxd) $(CUFILES:%.cu=$(WORKPATH)/%.cud) $(ASMFILES:%.asm=$(WORKPATH)/%.asmd) \
								$(CPPFILES_DBG:%.cpp=$(WORKPATH)/%.dbgd) $(CPPFILES_ICC:%.cpp=$(WORKPATH)/%.iccd) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/%.vccd) $(CPPFILES_GCC:%.cpp=$(WORKPATH)/%.gccd) \
								$(CPPFILES_MSCC:%.cpp=$(WORKPATH)/%.msccd)

GCCINCLUDEPATHS				= $(INCLUDEPATHS:%=-I%) $(COMMONINCLUDEPATHS:%=-I%)
VSINCLUDEPATHS				= $(INCLUDEPATHS:%=/I%) $(COMMONINCLUDEPATHS:%=/I%)

GCCDEFINES					= $(DEFINES:%=-D%) $(DEFINESARCH:%=-D%)
VSDEFINES					= $(DEFINES:%=/D%) $(DEFINESARCH:%=/D%)

LIBFILES					= $(LIBSUSE)

.SECONDARY:					$(CUFILES:%.cu=$(WORKPATH)/%.cucpp) $(ASMFILES:%.asm=$(WORKPATH)/%.asmcpp)

$(EXECUTABLE):				$(EXTRADEPS) $(OBJFILES) $(ALLDEP)
							$(LINK) $(LIBPATHSUSE) $(OBJFILES) $(LIBFILES) $(LINKFLAGSUSE) $(LINKOUTPUT) $(LINKTARGETTYPE)
							$(HIDEECHO)if [ -e "$(BUILDSCRIPT)" ]; then ./$(BUILDSCRIPT); fi

$(WORKPATH)/%.obj:			%.cpp $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.cxobj:		%.cxx $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.cuobj:		$(WORKPATH)/%.cucpp $(ALLDEP)
							$(CCCUDA) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.asmobj:		$(WORKPATH)/%.asmcpp $(ALLDEP)
							$(ASM) $(COMPILEOUTPUT) $(COMPILEONLY) $<
							
$(WORKPATH)/%.asmcpp:		%.asm $(ALLDEP)
							$(MSCC32) $(PRECOMPILEONLY) $(DEFINESUSE) /D "_AMD64_" /DINTEL_RUNTIME $< > $@

$(WORKPATH)/%.cucpp:		%.cu $(ALLDEP)
							$(NVCC) $(NVCCFLAGSUSE) $(CUDAINCLUDEPATHS) $(CUDADEFINES) --cuda --output-file "$@" $<
							$(HIDEECHO) cat $@ | grep -v NVCC_GREP > $@.tmp
							$(HIDEECHO) mv -f $@.tmp $@
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
							$(HIDEECHO) mkdir $(SAVEDIR)
							cp *.cpp *.h makefile buildnr *.sh *.bat *.conf *.cu $(SAVEDIR)

#Dependancies
$(WORKPATH)/%.cud:			%.cu $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) -I$(GCCPATH)/include $(GCCDEFINES) -D__CUDACC__ -x c++ -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cuobj $(WORKPATH)/$*.cucpp $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.asmd:			%.asm $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -x c -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.asmobj $(WORKPATH)/$*.asmcpp $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.d:			%.cpp $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.obj $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.cxd:			%.cxx $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cxobj $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.dbgd:			%.cpp $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -DDEBUG_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objdbg $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.iccd:			%.cpp $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -DINTEL_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objicc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.vccd:			%.cpp $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -DVECTORC_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objvcc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.gccd:			%.cpp $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -DGCC_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objgcc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.msccd:		%.cpp $(ALLDEP)
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC3264) -msse4.2 $(GCCINCLUDEPATHS) $(GCCDEFINES) -DVSNET_RUNTIME -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objmscc $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;																																			
							
include $(DEPENDS)							
