CXX		= g++
CXXOPTS		= -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope -O3 -march=barcelona -ftree-vectorize -msse3 -fkeep-inline-functions -fweb -frename-registers -minline-all-stringops -funit-at-a-time -mfpmath=sse -ftracer -finline-limit=1200 -fpeel-loops
LIBS		= -lpthread -ldl -L/usr/X11R6/lib -laticalrt -laticalcl -lgfortran ../GotoBLAS2/libgoto2.a

INCLUDE		= -I ../GotoBLAS2 -I $(ATISTREAMSDKROOT)/include

all:		dgemm_bench

dgemm_bench:	caldgemm.o benchmark.o
		g++ -o $@ $^ $(LIBS)

caldgemm.o:	caldgemm.cpp caldgemm.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

benchmark.o:	benchmark.cpp caldgemm.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

caldgemm.so: caldgemm.o

clean:
		rm -f caldgemm.o benchmark.o dgemm_bench caldgemm.so
