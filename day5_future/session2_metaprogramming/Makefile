#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

CC = cc
CXX = c++

CXXFLAGS = -std=c++11 -O3 -I ./common
LDFLAGS  = -lm -lOpenCL -lrt

all: bilateral 

bilateral: bilateral.cpp ./common/*.hpp
	$(CXX) $(CXXFLAGS) bilateral.cpp $(LDFLAGS) -o $@

clean:
	rm -f $(EXES) output.bmp
