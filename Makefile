TARGET := CUDACyclone
SRC := CUDACyclone.cu CUDAHash.cu CUDAUtils.cu
OBJ := $(SRC:.cu=.o)
CC := nvcc
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS := 75 86 89 $(GPU_ARCH)
GENCODE := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS := -std=c++17
LDFLAGS := -lcudadevrt -cudart=static

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

CUDACyclone.o: CUDACyclone.cu CUDAMath.h CUDAStructures.h CUDAHash.cuh CUDAUtils.h sha256.h
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

CUDAHash.o: CUDAHash.cu CUDAHash.cuh
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

CUDAUtils.o: CUDAUtils.cu CUDAUtils.h
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ)