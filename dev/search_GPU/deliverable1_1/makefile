CUDA_TOOLKIT=/usr/local/cuda-7.5
nvcc=$(CUDA_TOOLKIT)/bin/nvcc

all: gpuKnnLibrary.so

gpuKnnLibrary.so:
	$(nvcc) -Xcompiler -fPIC -shared -arch=sm_52 -o gpuKnnLibrary.so gpuKnnLibrary.cu

clean:
	rm -f gpuKnnLibrary.so
