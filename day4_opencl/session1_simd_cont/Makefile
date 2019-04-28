UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	#Use Apple's OpenCL runtime
	CXXLIBS += -framework OpenCL
else
	CXXLIBS += -lOpenCL -lm -ldl
endif

CXXLIBS += -llsb -L$(LSB)/lib -I$(LSB)/include
CXXFLAGS += -std=c++11

SOURCE_CPP_FILES = mandelbrot_vectorization.cpp

all:	mandelbrot

mandelbrot: $(SOURCE_CPP_FILES)
	$(CXX) $(SOURCE_CPP_FILES) $(CXXFLAGS) -o mandelbrot $(CXXLIBS) 

clean:
	rm -rf mandelbrot lsb.mandelbrot.r* mandelbrot_set.csv mandelbrot_set.png mandelbrot_set_vectorized.csv mandelbrot_set_vectorized.png runtimes.png

