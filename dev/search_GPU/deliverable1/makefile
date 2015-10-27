CUDA_PATH = /usr/local/cuda/bin/

gpuKnnLibrary: gpuKnnLibrary.cu
	$(CUDA_PATH)nvcc  -Xcompiler  -fPIC -shared -o gpuKnnLibrary.so gpuKnnLibrary.cu 
