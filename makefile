# The source code is property of the Frankfurt Institute for Advanced Studies (FIAS).
# None of the material may be copied, reproduced, distributed, republished, downloaded,
# displayed, posted or transmitted in any form or by any means, including, but not
# limited to, electronic, mechanical, photocopying, recording, or otherwise,
# without the prior written permission of FIAS.
# 
# Authors:
# David Rohr (drohr@jwdt.org)
# Matthias Bach (bach@compeng.uni-frankfurt.de)
# Matthias Kretz (kretz@compeng.uni-frankfurt.de)

CXX		= g++
CXXOPTS		= -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope -O3 -march=barcelona -ftree-vectorize -msse3 -fkeep-inline-functions -fweb -frename-registers -minline-all-stringops -funit-at-a-time -mfpmath=sse -ftracer -finline-limit=1200 -fpeel-loops
LIBS		= -lpthread -ldl -L/usr/X11R6/lib -laticalrt -laticalcl -lgfortran ../GotoBLAS2/libgoto2.a

INCLUDE		= -I ../GotoBLAS2 -I /home/fias/ati-stream-sdk-v2.2-lnx64/include

all:		dgemm_bench

dgemm_bench:	caldgemm.o benchmark.o calutil.o
		g++ -o $@ $^ $(LIBS)

calutil.o:	calutil.cpp calutil.h caldgemm_config.h caldgemm_config_load.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

caldgemm.o:	caldgemm.cpp caldgemm.h caldgemm_config.h caldgemm.il calutil.h caldgemm_config_load.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

benchmark.o:	benchmark.cpp caldgemm.h caldgemm_config.h calutil.h caldgemm_config_load.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

caldgemm.so:	caldgemm.o calutil.o

clean:
		rm -f caldgemm.o calutil.o benchmark.o dgemm_bench caldgemm.so
