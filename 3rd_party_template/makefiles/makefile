all							: all_tmp

run							: all
							./$(EXECUTABLE)

ARCH						:= $(shell sort <<< $$MACHTYPE)
ARCHCHK						:= $(shell if [ -a makefiles/$(ARCH).mak ]; then echo -n 1; else echo -n 0; fi)
ifeq ($(ARCHCHK), 1)
else
$(warning Unknown Architecture: $(ARCH) $(ARCHCHK), defaulting to x86_64-pc-linux-gnu)
ARCH						:= x86_64-pc-linux-gnu
endif
ARCHFILE					= $(ARCH).mak

ifeq ($(CONFIGFILE), )
CONFIGFILE					= config.mak
CLEANRELEASEDIR				= release
endif

ifeq ($(BUILDSCRIPT), )
BUILDSCRIPT					= build.sh
endif

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
NVCCFLAGSDBG				= --maxrregcount $(CUDAREGS) -Xptxas -v -Xptxas -O0 -O0 -m$(ARCHBITS) $(NVCCARCHS)

TARGETTYPE					= EXECUTABLE

WORKPATHSUFFIX				= $(TARGETTYPE)_$(TARGET)
include						$(CONFIGFILE)
WORKPATH					= release/$(ARCH)_$(ARCHBITS)$(WORKPATHSUFFIX)
ifeq ($(CONFIG_OPENCL_VERSION), )
CONFIG_OPENCL_VERSION		= All
endif
ifeq ($(COMPILER_FLAGS), DBG)
INTELFLAGSUSE				= $(INTELFLAGSDBG)
VSNETFLAGSUSE				= $(VSNETFLAGSDBG)
GCCFLAGSUSE					= $(GCCFLAGSDBG)
NVCCFLAGSUSE				= $(NVCCFLAGSDBG)
CONFIG_LTO					= 0
endif
ifeq ($(COMPILER_FLAGS), OPT)
INTELFLAGSUSE				= $(INTELFLAGSOPT)
VSNETFLAGSUSE				= $(VSNETFLAGSOPT)
GCCFLAGSUSE					= $(GCCFLAGSOPT)
NVCCFLAGSUSE				= $(NVCCFLAGSOPT)
endif

ifeq ($(CONFIG_CPP11), 1)
GCCFLAGSUSE				+= -std=c++11
endif

ALLDEP						= makefiles/makefile $(CONFIGFILE) makefiles/$(ARCHFILE) config.mak
include						makefiles/$(ARCHFILE)
GCCFLAGSARCH				+= $(GCCARCHA)

all_tmp:					$(SUBTARGETS:%=subbuild/%.mak) $(EXECUTABLE)
main:						$(EXECUTABLE)

subbuild/%.mak:
							+$(MAKE) CONFIGFILE=config_`echo $@ | sed s,subbuild/,,` BUILDSCRIPT=config_`echo $@ | sed s,subbuild/,, | sed s,mak,sh,` -f makefile

CUDAINCLUDEPATHS			= $(INCLUDEPATHSUSE:%=--compiler-options %)
CUDADEFINES					= $(DEFINESUSE:%=--compiler-options %)

DEPENDS						:= $(CUFILES:%.cu=$(WORKPATH)/cu/%.d) $(CLFILES:%.cl=$(WORKPATH)/cl/%.d) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.d) $(CPPFILES_DBG:%.cpp=$(WORKPATH)/dbg/%.d) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/vcc/%.d) \
								$(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.d) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.d) $(CFILES:%.c=$(WORKPATH)/c/%.d) \
								$(CPPFILES_MSCC:%.cpp=$(WORKPATH)/mscc/%.d) $(CPPFILES_CLANG:%.cpp=$(WORKPATH)/clang/%.d) $(CPPFILES_ICC:%.cpp=$(WORKPATH)/icc/%.d) $(CPPFILES_GCC:%.cpp=$(WORKPATH)/gcc/%.d) $(QTFILES:%.ui=$(WORKPATH)/qt/%.qtd)

CPPFILES					+= $(QTFILES:%.ui=$(WORKPATH)/cpp/%.o)
OBJFILES					:= $(CUFILES:%.cu=$(WORKPATH)/cu/%.o) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.o) $(CPPFILES_DBG:%.cpp=$(WORKPATH)/dbg/%.o) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/vcc/%.o) $(RESOURCEFILES:%=$(WORKPATH)/res/%.o) $(CLFILES:%.cl=$(WORKPATH)/cl/%.o) $(QTFILES:%.ui=$(WORKPATH)/qt/%_moc.o)

CPPFILES_ICC				:= $(CPPFILES_ICC:%.cpp=$(WORKPATH)/icc/%.o)
CPPFILES_GCC				:= $(CPPFILES_GCC:%.cpp=$(WORKPATH)/gcc/%.o)
CPPFILES_MSCC				:= $(CPPFILES_MSCC:%.cpp=$(WORKPATH)/mscc/%.o)
CPPFILES_CLANG				:= $(CPPFILES_CLANG:%.cpp=$(WORKPATH)/clang/%.o)
ifeq ($(CC_SELECTED), ICC)
CPPFILES_ICC				+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.o) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.o) $(CFILES:%.c=$(WORKPATH)/c/%.o) 
CCWITHLINK					= /link
endif
ifeq ($(CC_SELECTED), MSCC)
CPPFILES_MSCC				+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.o) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.o) $(CFILES:%.c=$(WORKPATH)/c/%.o) 
CCWITHLINK					= /link
endif
ifeq ($(CC_SELECTED), GCC)
CPPFILES_GCC				+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.o) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.o) $(CFILES:%.c=$(WORKPATH)/c/%.o) 
endif

ifeq ($(CONFIG_LTO), 1)
ifneq ("0$(CPPFILES_ICC)", "0")
OBJFILES					+= $(WORKPATH)/make_lto_icc/icclto_$(TARGET).$(OBJ)
endif
OBJFILES					+= $(CPPFILES_MSCC) $(CPPFILES_GCC) $(CPPFILES_CLANG)
else
OBJFILES					+= $(CPPFILES_ICC) $(CPPFILES_MSCC) $(CPPFILES_GCC) $(CPPFILES_CLANG)
endif

ifeq ($(CONFIG_CUDA_DC), 1)
OBJFILES					+= $(WORKPATH)/make_cuda_device_link/dl_$(TARGET).o
NVCCFLAGSUSE					+= -dc
endif

ifeq ($(ARCH), i686-pc-cygwin)
ifeq ($(INCLUDEPATHS)$(COMMONINCLUDEPATHS), )
GCCINCLUDEPATHS				=
else
GCCINCLUDEPATHSA			:= $(INCLUDEPATHS) $(COMMONINCLUDEPATHS)
GCCINCLUDEPATHSB			:= $(shell cygpath -u $(GCCINCLUDEPATHSA))
GCCINCLUDEPATHS				= $(GCCINCLUDEPATHSB:%=-I%)
endif
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

.SECONDARY:					$(CUFILES:%.cu=$(WORKPATH)/cu/%.cpp) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.cpp) $(RESOURCEFILES:%=$(WORKPATH)/res/%.objres) $(CLFILES:%.cl=$(WORKPATH)/cl/%.objrescl) $(QTFILES:%.ui=$(WORKPATH)/qt/%_moc.o) $(QTFILES:%.ui=$(WORKPATH)/qt/%_moc.cpp) $(QTFILES:%.ui=$(WORKPATH)/qt/%_ui.h) $(CLFILES:%.cl=$(WORKPATH)/cl/%.objcl)

$(EXECUTABLE):				$(EXTRADEPS) $(OBJFILES) $(EXTRAOBJFILES) $(ALLDEP)
							$(LINK) $(LIBPATHSUSE) $(OBJFILES) $(EXTRAOBJFILES) $(LIBFILES) $(LINKFLAGSUSE) $(LINKOUTPUT) $(LINKTARGETTYPE)
							$(HIDEECHOA)if [ -e "$(BUILDSCRIPT)" ]; then ./$(BUILDSCRIPT); fi

$(WORKPATH)/cpp/%.o:			%.cpp $(ALLDEP)
							@mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'`
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/cxx/%.o:			%.cxx $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/c/%.o:			%.c $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)

$(WORKPATH)/cu/%.o:		$(WORKPATH)/cu/%.cpp $(ALLDEP)
							$(CCCUDA) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/cu/%.cpp:		%.cu $(ALLDEP)
							$(NVCC) $(NVCCFLAGSUSE) $(CUDAINCLUDEPATHS) $(CUDADEFINES) $(FILEFLAGS$<) --cuda --output-file "$@" $<
							$(HIDEECHOA) cat $@ | grep -v NVCC_GREP | sed "s/#pragma detect_mismatch(\"_MSC_VER\", \"1600\")//g" > $@.tmp
							$(HIDEECHOA) mv -f $@.tmp $@
							-if [ -e "$<.$(ARCH).patch" ]; then patch -r /dev/null -s --no-backup-if-mismatch -i $<.$(ARCH).patch $@; fi

$(WORKPATH)/asm/%.o:		$(WORKPATH)/asm/%.cpp $(ALLDEP)
							$(ASM) $(COMPILEOUTPUT) $(ASMONLY) $<
							
$(WORKPATH)/asm/%.cpp:			%.asm $(ALLDEP)
							$(CC) $(PRECOMPILEONLY) $(FILEFLAGS$<) $(DEFINESUSE) $< > $@

$(WORKPATH)/dbg/%.o:			%.cpp $(ALLDEP)
							$(CCDBG) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/icc/%.o:			%.cpp $(ALLDEP)
							$(ICC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/vcc/%.o:			%.cpp $(ALLDEP)
							$(VCC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $<
$(WORKPATH)/gcc/%.o:			%.cpp $(ALLDEP)
							$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) $(FILEFLAGS$<) -c $< -o $@
$(WORKPATH)/mscc/%.o:			%.cpp $(ALLDEP)
							$(MSCC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/clang/%.o:			%.cpp $(ALLDEP)
							$(CLANG) $(GCCINCLUDEPATHS) $(GCCDEFINES) $(FILEFLAGS$<) -c $< -o $@




$(WORKPATH)/qt/%_ui.h:		%.ui
							mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'`
							$(QTUIC) -o $@ $<

$(WORKPATH)/qt/%_moc.cpp:	%.h $(WORKPATH)/qt/%_ui.h
							$(QTMOC) -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -o $@ $<

$(WORKPATH)/qt/%_moc.o:	$(WORKPATH)/qt/%_moc.cpp
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
				
ifeq ($(CC_SELECTED), MSCC)
OCL_COMPILER_MSCC_FIX		= $(COMPILEOUTPUTBASE)$(WORKPATH)/_makefile_support_opencl_compiled.o
else
OCL_COMPILER_MSCC_FIX		=
endif
$(WORKPATH)/_makefiles_support_opencl.exe:	makefiles/makefile_opencl_compiler.cpp $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $< $(CCWITHLINK) $(OCL_COMPILER_MSCC_FIX) $(LINKFLAGSARCH) $(LIBPATHSUSE) $(OPENCLLIB) $(LINKOUTPUT)

$(WORKPATH)/cl/%.objcl:		%.cl $(ALLDEP) $(WORKPATH)/_makefiles_support_opencl.exe
							$(HIDEECHOA)$(OPENCL_ENVIRONMENT) ./$(WORKPATH)/_makefiles_support_opencl.exe -output-file $@ $< -- -I. $(GCCINCLUDEPATHS) $(GCCDEFINES) $(OPENCL_OPTIONS) > /dev/null

$(WORKPATH)/cl/%.o:		$(WORKPATH)/cl/%.objcl $(ALLDEP)
							$(HIDEECHOA) echo $<
							$(HIDEECHOA) sed -e "s&FILENAMENORMAL&$<&g" -e "s&FILENAMEMOD&_makefile_opencl_program_`echo $< | sed -e "s&$(WORKPATH)/&&g" -e "s&\.objcl&\.cl&g" -e "s&\.&_&g" -e "s&/&_&g" -e "s& &_&g"`&g" makefiles/include.S | $(GCC3264) -c -x assembler -o $@ -
$(WORKPATH)/res/%.o:	% $(ALLDEP)
							$(HIDEECHOA) echo $<
							$(HIDEECHOA) sed -e "s&FILENAMENORMAL&$<&g" -e "s&FILENAMEMOD&_resource_`echo $< | sed -e "s&\.&_&g" -e "s&/&_&g" -e "s& &_&g"`&g" makefiles/include.S | $(GCC3264) -c -x assembler -o $@ -



$(WORKPATH)/make_lto_icc/icclto_$(TARGET).$(OBJ):	$(CPPFILES_ICC)
							$(HIDEECHOA)mkdir -p $(WORKPATH)/make_lto_icc
							$(ICCLINK) $(INTELLINKIPO)$@ $(CPPFILES_ICC)
							
$(WORKPATH)/make_cuda_device_link/dl_$(TARGET).o:	$(CUFILES:%.cu=$(WORKPATH)/cu/%.o)
							$(HIDEECHOA)mkdir -p $(WORKPATH)/make_cuda_device_link
							$(NVCC) $(NVCCFLAGSUSE) -dlink $(CUFILES:%.cu=$(WORKPATH)/cu/%.o) -o $@ -lcudadevrt
							
clean:						$(SUBTARGETS:%=subclean/%.mak)
							rm -Rf *.plg *.dpi *.exp *.lib $(EXECUTABLE) x64/release/* *.cucpp *.cubin *.gpu *.ptx *.linkinfo *.ii cuda.compute_* $(DEPENDS) $(OBJFILES) $(CUFILES:%.cu=$(WORKPATH)/%.cucpp) $(ASMFILES:%.asm=$(WORKPATH)/%.asmcpp) $(CLEANRELEASEDIR)
							
subclean/%.mak:
							+export CONFIGFILE=config_`echo $@ | sed s,subclean/,,` && $(MAKE) -f makefile clean

SAVEDIR						= releases/`date +%F`-BUILD-`cat buildnr`							
backup:						
							$(HIDEECHOA) mkdir $(SAVEDIR)
							cp *.cpp *.h makefile buildnr *.sh *.bat *.conf *.cu $(SAVEDIR)

#Dependencies

DEPMKDIR					= mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` &&
DEPGCC						= $(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -x c++ -MM $<
DEPSED1						= sed -e ':a;N;$$!ba' -e
DEPSED2						= -e 's,\\,/,g' \
							-e 's,/\n,\\\n,g' \
							-e 's,[cC]:/,/cygdrive/c/,g' > \
							$@;


$(WORKPATH)/qt/%.qtd:			%.cpp %.ui $(ALLDEP) $(WORKPATH)/qt/%_ui.h
							@$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/cl/%.d:			%.cl $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -I$(GCCPATH)/include -D__OPENCL__ | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.objcl $(WORKPATH)/$*.cl $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/cu/%.d:			%.cu $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -I$(GCCPATH)/include -D_MSC_VER=1700 -D__CUDACC__ | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $(WORKPATH)/$*.cucpp $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/asm/%.d:			%.asm $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $(WORKPATH)/$*.asmcpp $@ : ,g' $(DEPSED2)

$(WORKPATH)/cpp/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/cxx/%.d:			%.cxx $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)

$(WORKPATH)/c/%.d:			%.c $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cobj $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/dbg/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -DDEBUG_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)

$(WORKPATH)/icc/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -DINTEL_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/vcc/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -DVECTORC_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/gcc/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -DGCC_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)
							
$(WORKPATH)/mscc/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -DVSNET_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)

$(WORKPATH)/clang/%.d:			%.cpp $(ALLDEP)
							@$(DEPMKDIR) $(DEPGCC) -DCLANG_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.o $@ : ,g' $(DEPSED2)

include $(DEPENDS)							
