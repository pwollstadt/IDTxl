#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

//#include <cpugpuKnn_common.h>
#ifndef INFINITY
#define INFINITY 0x7F800000
#endif

__device__ float
insertPointKlist(int kth, float distance, int indexv,float* kdistances, int* kindexes){
	int k=0;
	while( (distance>*(kdistances+k)) && (k<kth-1)){k++;}
	//Move value to the next
	for(int k2=kth-1;k2>k;k2--){
		*(kdistances+k2)=*(kdistances+k2-1);
		*(kindexes+k2)=*(kindexes+k2-1);
	}
	//Replace
	*(kdistances+k)=distance;
	*(kindexes+k)=indexv;

	//printf("\n -> Modificacion pila: %.f %.f. New max distance: %.f", *kdistances, *(kdistances+1), *(kdistances+kth-1));
	return *(kdistances+kth-1);
}

__device__ float
maxMetricPoints(const float* g_uquery, const float* g_vpoint, int pointdim, int signallength){
	float	r_u1;
	float	r_v1;
	float	r_d1,r_dim=0;

	r_dim=0;
	for(int d=0; d<pointdim; d++){
		r_u1 = *(g_uquery+d*signallength);
		r_v1 = *(g_vpoint+d*signallength);
		r_d1 = r_v1 - r_u1;
		r_d1 = r_d1 < 0? -r_d1: r_d1;  //abs
		r_dim= r_dim < r_d1? r_d1: r_dim;
	}
	return r_dim;
}

/*
 * KNN
 */

//extern __shared__ char array[];

__global__ void
kernelKNNshared(const float* g_uquery, const float* g_vpointset, int *g_indexes, float* g_distances, const int pointdim, const int triallength, const int signallength, const int kth, const int exclude)
{

    // shared memory
	extern __shared__ char array[];
	float *kdistances;
	int *kindexes;
	kdistances = (float*)array;
	kindexes = (int*)array+kth*blockDim.x;

	const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const unsigned int itrial = tid / triallength;  //  indextrial

if(tid<signallength){

	for(int k=0;k<kth;k++){
		kdistances[threadIdx.x*kth+k] = INFINITY;
	}

	__syncthreads();

	//int   r_index;
	float r_kdist=INFINITY;
	unsigned int indexi = tid-triallength*itrial;

	for(int t=0; t<triallength; t++){
			int indexu = tid;
			int indexv = (t + itrial*triallength);
			int condition1=indexi-exclude;
			int condition2=indexi+exclude;
			if((t<condition1)||(t>condition2)){
				float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
				if(temp_dist <= r_kdist){
					r_kdist = insertPointKlist(kth,temp_dist,t,kdistances+threadIdx.x*kth,kindexes+threadIdx.x*kth);
				//printf("\nId: %d, Temp_dist: %.f. r_index: %d", tid, temp_dist, r_index);
				}
			}
			//printf("tid:%d indexes: %d, %d distances: %.f %.f\n",tid, *kindexes, *(kindexes+1), *kdistances, *(kdistances+1));
	}

	__syncthreads();
	//COPY TO GLOBAL MEMORY
	for(int k=0;k<kth;k++){
		g_indexes[tid+k*signallength] = kindexes[threadIdx.x*kth+k];
		g_distances[tid+k*signallength]= kdistances[threadIdx.x*kth+k];//*(kdistances+k);
	}
}

}

/*
 * Only valid for 10 nearest neighbors
 */

__global__ void
kernelKNN(const float* g_uquery, const float* g_vpointset, int *g_indexes, float* g_distances, int pointdim, int triallength, int signallength, int kth, int exclude)
{

   	const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
	//const unsigned int tidim = tid*pointdim;
	const unsigned int itrial = tid / triallength;  //  indextrial

	int kindexes[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	float kdistances[]= {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, \
						 INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};

if(tid<signallength){

	//int   r_index;
	float r_kdist=INFINITY;
	int indexi = tid-triallength*itrial;
	for(int t=0; t<triallength; t++){
			int indexu = tid;
			int indexv = (t + itrial*triallength);
			int condition1=indexi-exclude;
			int condition2=indexi+exclude;
			if((t<condition1)||(t>condition2)){
				float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
				if(temp_dist <= r_kdist){
					r_kdist = insertPointKlist(kth,temp_dist,t,kdistances,kindexes);
					//printf("\nId: %d, Temp_dist: %.f. r_index: %d", tid, temp_dist, r_index);
				}
			}
			//printf("tid:%d indexes: %d, %d distances: %.f %.f\n",tid, *kindexes, *(kindexes+1), *kdistances, *(kdistances+1));
	}

	__syncthreads();
	//COPY TO GLOBAL MEMORY
	for(int k=0;k<kth;k++){
		g_indexes[tid+k*signallength] = *(kindexes+k);
		g_distances[tid+k*signallength]= *(kdistances+k);
	}

}

}


/*
 * Range search using bruteforce
 */

__global__ void
kernelBFRSshared(const float* g_uquery, const float* g_vpointset, int *g_npoints, int pointdim, int triallength, int signallength, int exclude, float radius)
{

    // shared memory
	extern __shared__ char array[];
	int *s_npointsrange;
	s_npointsrange = (int*)array;

	const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const unsigned int itrial = tid / triallength;  //  indextrial

if(tid<signallength){

	s_npointsrange[threadIdx.x] = 0;
	__syncthreads();


	unsigned int indexi = tid-triallength*itrial;
	for(int t=0; t<triallength; t++){
			int indexu = tid;
			int indexv = (t + itrial*triallength);
			int condition1=indexi-exclude;
			int condition2=indexi+exclude;
			if((t<condition1)||(t>condition2)){
				float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
				if(temp_dist <= radius){
					s_npointsrange[threadIdx.x]++;
				}
			}

	}

	__syncthreads();
	//printf("\ntid:%d npoints: %d\n",tid, s_npointsrange[threadIdx.x]);
	//COPY TO GLOBAL MEMORY
	g_npoints[tid] = s_npointsrange[threadIdx.x];

}
}


__global__ void
kernelBFRSMultishared(const float* g_uquery, const float* g_vpointset, int *g_npoints, int pointdim, int triallength, int signallength, int exclude, const float* vecradius)
{

    // shared memory
	extern __shared__ char array[];
	int *s_npointsrange;
	s_npointsrange = (int*)array;
    float radius=0;
	const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const unsigned int itrial = tid / triallength;  //  indextrial

if(tid<signallength){

	s_npointsrange[threadIdx.x] = 0;
	__syncthreads();

    radius = *(vecradius+itrial);
	unsigned int indexi = tid-triallength*itrial;
	for(int t=0; t<triallength; t++){
			int indexu = tid;
			int indexv = (t + itrial*triallength);
			int condition1=indexi-exclude;
			int condition2=indexi+exclude;
			if((t<condition1)||(t>condition2)){
				float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
				if(temp_dist <= radius){
					s_npointsrange[threadIdx.x]++;
				}
			}

	}

	__syncthreads();
	//printf("\ntid:%d npoints: %d\n",tid, s_npointsrange[threadIdx.x]);
	//COPY TO GLOBAL MEMORY
	g_npoints[tid] = s_npointsrange[threadIdx.x];

}
}


__global__ void
kernelBFRSAllshared(const float* g_uquery, const float* g_vpointset, int *g_npoints, int pointdim, int triallength, int signallength, int exclude, const float* vecradius)
{

    // shared memory
	extern __shared__ char array[];
	int *s_npointsrange;
	s_npointsrange = (int*)array;
    float radius=0;
	const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const unsigned int itrial = tid / triallength;  //  indextrial

if(tid<signallength){

	s_npointsrange[threadIdx.x] = 0;
	__syncthreads();

    radius = *(vecradius+tid);
	unsigned int indexi = tid-triallength*itrial;
	for(int t=0; t<triallength; t++){
			int indexu = tid;
			int indexv = (t + itrial*triallength);
			int condition1=indexi-exclude;
			int condition2=indexi+exclude;
			if((t<condition1)||(t>condition2)){
				float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
				if(temp_dist <= radius){
					s_npointsrange[threadIdx.x]++;
				}
			}

	}

	__syncthreads();
	//printf("\ntid:%d npoints: %d\n",tid, s_npointsrange[threadIdx.x]);
	//COPY TO GLOBAL MEMORY
	g_npoints[tid] = s_npointsrange[threadIdx.x];

}
}

#endif // #ifndef _TEMPLATE_KERNEL_H_


