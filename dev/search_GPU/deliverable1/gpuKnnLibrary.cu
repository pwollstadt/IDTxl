#include "gpuKnnBF_kernel.cu"
#include <stdio.h>
#include "helperfunctions.cu"

extern "C" {
int cudaFindKnn(int* h_bf_indexes, float* h_bf_distances, float* h_pointset, float* h_query, int kth, int thelier, int nchunks, int pointdim, int signallength){
	float *d_bf_pointset, *d_bf_query;
	int *d_bf_indexes;
	float *d_bf_distances;

	unsigned int meminputsignalquerypointset= pointdim * signallength * sizeof(float);
	unsigned int mem_bfcl_outputsignaldistances= kth * signallength * sizeof(float);
	unsigned int mem_bfcl_outputsignalindexes = kth * signallength * sizeof(int);

	//fprintf(stderr,"\nValues:%d %d %d %d %d", kth, thelier, nchunks, pointdim, signallength);

	checkCudaErrors( cudaMalloc( (void**) &(d_bf_query), meminputsignalquerypointset));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_pointset), meminputsignalquerypointset));
	//GPU output
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_distances), mem_bfcl_outputsignaldistances ));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_indexes), mem_bfcl_outputsignalindexes ));

	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}

	//Upload input data
	checkCudaErrors( cudaMemcpy(d_bf_query, h_query, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
	checkCudaErrors( cudaMemcpy(d_bf_pointset, h_pointset, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
	error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	//Kernel launch
	dim3 threads(1,1,1);
	dim3 grid(1,1,1);
	threads.x = 512;
	grid.x = (signallength-1)/threads.x + 1;
	int memkernel = kth*sizeof(float)*threads.x+\
					kth*sizeof(int)*threads.x;
	int triallength = signallength / nchunks;
	kernelKNNshared<<< grid.x, threads.x, memkernel>>>(\
			d_bf_query,d_bf_pointset,d_bf_indexes,d_bf_distances, \
			pointdim,triallength, signallength,kth,thelier);


	//Download result
	checkCudaErrors( cudaMemcpy( h_bf_distances, d_bf_distances, mem_bfcl_outputsignaldistances, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy( h_bf_indexes, d_bf_indexes, mem_bfcl_outputsignalindexes, cudaMemcpyDeviceToHost) );
	error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
//	//Add +1 to indexes this is only necessary for Matlab! 
//	for(unsigned int i=0;i<kth*signallength;i++){
//		h_bf_indexes[i]+=1;
//	}

	//Free resources
	checkCudaErrors(cudaFree(d_bf_query));
	checkCudaErrors(cudaFree(d_bf_pointset));
	checkCudaErrors(cudaFree(d_bf_distances));
	checkCudaErrors(cudaFree(d_bf_indexes));
	cudaDeviceReset();  //Matlab integration segmentation fault from mex files workaround
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	
	return 1;
}
}

extern "C" {
int cudaFindKnnSetGPU(int* h_bf_indexes, float* h_bf_distances, float* h_pointset, float* h_query, int kth, int thelier, int nchunks, int pointdim, int signallength, int deviceid){
	float *d_bf_pointset, *d_bf_query;
	int *d_bf_indexes;
	float *d_bf_distances;

	cudaSetDevice(deviceid);

	unsigned int meminputsignalquerypointset= pointdim * signallength * sizeof(float);
	unsigned int mem_bfcl_outputsignaldistances= kth * signallength * sizeof(float);
	unsigned int mem_bfcl_outputsignalindexes = kth * signallength * sizeof(int);

	//fprintf(stderr,"\nValues:%d %d %d %d %d", kth, thelier, nchunks, pointdim, signallength);

	checkCudaErrors( cudaMalloc( (void**) &(d_bf_query), meminputsignalquerypointset));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_pointset), meminputsignalquerypointset));
	//GPU output
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_distances), mem_bfcl_outputsignaldistances ));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_indexes), mem_bfcl_outputsignalindexes ));

	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}

	//Upload input data
	checkCudaErrors( cudaMemcpy(d_bf_query, h_query, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
	checkCudaErrors( cudaMemcpy(d_bf_pointset, h_pointset, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
	error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	//Kernel launch
	dim3 threads(1,1,1);
	dim3 grid(1,1,1);
	threads.x = 512;
	grid.x = (signallength-1)/threads.x + 1;
	int memkernel = kth*sizeof(float)*threads.x+\
					kth*sizeof(int)*threads.x;
	int triallength = signallength / nchunks;
	kernelKNNshared<<< grid.x, threads.x, memkernel>>>(\
			d_bf_query,d_bf_pointset,d_bf_indexes,d_bf_distances, \
			pointdim,triallength, signallength,kth,thelier);


	//Download result
	checkCudaErrors( cudaMemcpy( h_bf_distances, d_bf_distances, mem_bfcl_outputsignaldistances, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy( h_bf_indexes, d_bf_indexes, mem_bfcl_outputsignalindexes, cudaMemcpyDeviceToHost) );
	error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
//	//Add +1 to indexes
//	for(unsigned int i=0;i<kth*signallength;i++){
//		h_bf_indexes[i]+=1;
//	}

	//Free resources
	checkCudaErrors(cudaFree(d_bf_query));
	checkCudaErrors(cudaFree(d_bf_pointset));
	checkCudaErrors(cudaFree(d_bf_distances));
	checkCudaErrors(cudaFree(d_bf_indexes));
	cudaDeviceReset();  //Matlab integration segmentation fault from mex files workaround
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	
	return 1;
}
}

/*
 * Range search being radius a vector of length number points in queryset/pointset
 */

extern "C" {
int cudaFindRSAll(int* h_bf_npointsrange, float* h_pointset, float* h_query, float* h_vecradius, int thelier, int nchunks, int pointdim, int signallength){
	float *d_bf_pointset, *d_bf_query, *d_bf_vecradius;
	int *d_bf_npointsrange;

	unsigned int meminputsignalquerypointset= pointdim * signallength * sizeof(float);
	unsigned int mem_bfcl_outputsignalnpointsrange= signallength * sizeof(int);
    	unsigned int mem_bfcl_inputvecradius = signallength * sizeof(float);

	checkCudaErrors( cudaMalloc( (void**) &(d_bf_query), meminputsignalquerypointset));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_pointset), meminputsignalquerypointset));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_npointsrange), mem_bfcl_outputsignalnpointsrange ));
    checkCudaErrors( cudaMalloc( (void**) &(d_bf_vecradius), mem_bfcl_inputvecradius ));

    cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	//Upload input data
	checkCudaErrors( cudaMemcpy(d_bf_query, h_query, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
	checkCudaErrors( cudaMemcpy(d_bf_pointset, h_pointset, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
    checkCudaErrors( cudaMemcpy(d_bf_vecradius, h_vecradius, mem_bfcl_inputvecradius, cudaMemcpyHostToDevice ));

    error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	//Kernel launch
	//Kernel launch
	dim3 threads(1,1,1);
	dim3 grid(1,1,1);
	threads.x = 512;
	grid.x = (signallength-1)/threads.x + 1;
	int memkernel = sizeof(int)*threads.x;
	int triallength = signallength / nchunks;

	kernelBFRSAllshared<<< grid.x, threads.x, memkernel>>>(\
					d_bf_query,d_bf_pointset, d_bf_npointsrange, \
					pointdim,triallength, signallength,thelier, d_bf_vecradius);

	checkCudaErrors( cudaMemcpy( h_bf_npointsrange, d_bf_npointsrange,mem_bfcl_outputsignalnpointsrange, cudaMemcpyDeviceToHost) );


	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}


	//Free resources
	checkCudaErrors(cudaFree(d_bf_query));
	checkCudaErrors(cudaFree(d_bf_pointset));
	checkCudaErrors(cudaFree(d_bf_npointsrange));
    checkCudaErrors(cudaFree(d_bf_vecradius));
	cudaDeviceReset();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	
	return 1;
}
}

extern "C" {
int cudaFindRSAllSetGPU(int* h_bf_npointsrange, float* h_pointset, float* h_query, float* h_vecradius, int thelier, int nchunks, int pointdim, int signallength, int deviceid){
	float *d_bf_pointset, *d_bf_query, *d_bf_vecradius;
	int *d_bf_npointsrange;

	unsigned int meminputsignalquerypointset= pointdim * signallength * sizeof(float);
	unsigned int mem_bfcl_outputsignalnpointsrange= signallength * sizeof(int);
    	unsigned int mem_bfcl_inputvecradius = signallength * sizeof(float);

	cudaSetDevice(deviceid);

	checkCudaErrors( cudaMalloc( (void**) &(d_bf_query), meminputsignalquerypointset));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_pointset), meminputsignalquerypointset));
	checkCudaErrors( cudaMalloc( (void**) &(d_bf_npointsrange), mem_bfcl_outputsignalnpointsrange ));
    checkCudaErrors( cudaMalloc( (void**) &(d_bf_vecradius), mem_bfcl_inputvecradius ));

    cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	//Upload input data
	checkCudaErrors( cudaMemcpy(d_bf_query, h_query, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
	checkCudaErrors( cudaMemcpy(d_bf_pointset, h_pointset, meminputsignalquerypointset, cudaMemcpyHostToDevice ));
    checkCudaErrors( cudaMemcpy(d_bf_vecradius, h_vecradius, mem_bfcl_inputvecradius, cudaMemcpyHostToDevice ));

    error = cudaGetLastError();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	//Kernel launch
	//Kernel launch
	dim3 threads(1,1,1);
	dim3 grid(1,1,1);
	threads.x = 512;
	grid.x = (signallength-1)/threads.x + 1;
	int memkernel = sizeof(int)*threads.x;
	int triallength = signallength / nchunks;

	kernelBFRSAllshared<<< grid.x, threads.x, memkernel>>>(\
					d_bf_query,d_bf_pointset, d_bf_npointsrange, \
					pointdim,triallength, signallength,thelier, d_bf_vecradius);

	checkCudaErrors( cudaMemcpy( h_bf_npointsrange, d_bf_npointsrange,mem_bfcl_outputsignalnpointsrange, cudaMemcpyDeviceToHost) );


	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}


	//Free resources
	checkCudaErrors(cudaFree(d_bf_query));
	checkCudaErrors(cudaFree(d_bf_pointset));
	checkCudaErrors(cudaFree(d_bf_npointsrange));
    checkCudaErrors(cudaFree(d_bf_vecradius));
	cudaDeviceReset();
	if(error!=cudaSuccess){
		fprintf(stderr,"%s",cudaGetErrorString(error));
		return 0;
	}
	
	return 1;
}
}
