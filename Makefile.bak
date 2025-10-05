NVCC = nvcc
NVCCFLAGS = -O3 -rdc=true -use_fast_math --ptxas-options=-O3 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -std=c++17

all: CUDACyclone

CUDACyclone: CUDACyclone.o CUDAHash.o CUDAUtils.o
    $(NVCC) $(NVCCFLAGS) -o CUDACyclone CUDACyclone.o CUDAHash.o CUDAUtils.o

CUDACyclone.o: CUDACyclone.cu CUDAMath.h CUDAStructures.h CUDAHash.cuh CUDAUtils.h sha256.h
    $(NVCC) $(NVCCFLAGS) -c CUDACyclone.cu -o CUDACyclone.o

CUDAHash.o: CUDAHash.cu CUDAHash.cuh
    $(NVCC) $(NVCCFLAGS) -c CUDAHash.cu -o CUDAHash.o

CUDAUtils.o: CUDAUtils.cu CUDAUtils.h
    $(NVCC) $(NVCCFLAGS) -c CUDAUtils.cu -o CUDAUtils.o

clean:
    rm -f *.o CUDACyclone