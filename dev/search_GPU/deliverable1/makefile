CUDA_PATH = /usr/local/cuda-7.5/bin/

gpuKnnLibrary: gpuKnnLibrary.cu
	$(CUDA_PATH)nvcc  -Xcompiler  -fPIC -shared -o gpuKnnLibrary.so gpuKnnLibrary.cu 
