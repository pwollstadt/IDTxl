


Requirements:
NVIDIA CUDA Toolkit 13 (11+)
gcc


compile on Linux:
nvcc -Xcompiler -fPIC -shared -arch=native -o gpuKnnLibrary.so gpuKnnLibrary.cu


