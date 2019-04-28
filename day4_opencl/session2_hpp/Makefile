#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

CXX = c++

INC = ../common
CXXFLAGS = -std=c++11 -O3
LDFLAGS = -lOpenCL -lrt

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -framework OpenCL
endif

SRC = vadd.cpp
EXES = vadd-c++

all: $(EXES)

vadd-c++: vadd.cpp $(INC)/err_code.h $(INC)/util.hpp
	$(CXX) $(CXXFLAGS) -I $(INC) vadd.cpp  $(LDFLAGS) -o $@

.PHONY: clean

clean:
	rm -f $(EXES)

