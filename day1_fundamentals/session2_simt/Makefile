CXXLIBS += -fopenmp
CXXFLAGS += -O3 -std=c++11

all:	dot_product vector_add

%: %.cpp benchmark.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(CXXLIBS) 

clean:
	rm -rf dot_product vector_add

