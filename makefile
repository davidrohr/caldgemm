#
# Build rules for CALDGEMM
#
# Copyright 2010:
#  - David Rohr (drohr@jwdt.org)
#  - Matthias Bach (bach@compeng.uni-frankfurt.de)
#  - Matthias Kretz (kretz@compeng.uni-frankfurt.de)
#
# This file is part of CALDGEMM.
#
# CALDGEMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CALDGEMM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CALDGEMM.  If not, see <http://www.gnu.org/licenses/>.
#

CXX		= g++
CXXOPTS		= -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope -O3 -march=barcelona -ftree-vectorize -msse3 -fkeep-inline-functions -fweb -frename-registers -minline-all-stringops -funit-at-a-time -mfpmath=sse -ftracer -finline-limit=1200 -fpeel-loops
#CXXOPTS		= -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -O0 -ggdb -D_NO_AMD_CPU
LIBS		= -lpthread -ldl -L/usr/X11R6/lib -laticalrt -laticalcl -lgfortran ../GotoBLAS2/libgoto2.a

INCLUDE		= -I ../GotoBLAS2 -I $(ATISTREAMSDKROOT)/include

all:		dgemm_bench

dgemm_bench:	caldgemm.o benchmark.o
		g++ -o $@ $^ $(LIBS)

caldgemm.o:	caldgemm.cpp caldgemm.h caldgemm_config.h caldgemm.il caldgemm_config_load.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

benchmark.o:	benchmark.cpp caldgemm.h caldgemm_config.h caldgemm_config_load.h
		g++ -c $< $(CXXOPTS) $(INCLUDE)

caldgemm.so:	caldgemm.o

clean:
		rm -f caldgemm.o benchmark.o dgemm_bench caldgemm.so
