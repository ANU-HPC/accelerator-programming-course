#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

CXX = c++

INC = ../common
CXXFLAGS = -std=c++11 -O3 -I $(INC)
LDFLAGS = -lOpenCL -lrt

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -framework OpenCL
endif

EXES = matmul-c++

all: $(EXES)

matmul-c++: matmul.cpp matmul.hpp matrix_lib.cpp matrix_lib.hpp $(INC)/util.hpp $(INC)/device_picker.hpp
	$(CXX) $(CXXFLAGS) matmul.cpp matrix_lib.cpp $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f $(EXES)

