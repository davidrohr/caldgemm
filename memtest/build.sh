c++ -m64 -o mem -L$AMDAPPSDKROOT/lib/x86_64 -I$AMDAPPSDKROOT/include -lrt -lOpenCL mem.cpp ../cmodules/timer.cpp
