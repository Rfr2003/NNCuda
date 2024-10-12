NVCC = nvcc        

NVFLAGS = -Iinclude          

CU_SRCS = $(wildcard src/*.cu)   

CU_OBJS = $(CU_SRCS:.cu=.o)      

TARGET = bin/nn

all: $(TARGET)

$(TARGET): $(CU_OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET)

.PHONY: all clean
