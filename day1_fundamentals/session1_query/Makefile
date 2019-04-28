UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	#Use Apple's OpenCL runtime
	CCLIBS += -framework OpenCL
	#or use the installed POCL runtime
	#CCLIBS += -L/usr/local/lib -lOpenCL #
else
	CCLIBS += -lOpenCL -lm -ldl
	CFLAGS += -std=gnu99
endif

SOURCE_C_FILES = opencl_query.c

all:	opencl_device_query

opencl_device_query: $(SOURCE_C_FILES)
	$(CC) $(SOURCE_C_FILES) $(CFLAGS) -o opencl_device_query $(CCLIBS) 

clean:
	rm opencl_device_query

