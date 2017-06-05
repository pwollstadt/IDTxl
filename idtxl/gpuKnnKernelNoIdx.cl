#ifndef INFINITY
#define INFINITY 0x7F800000
#endif

float insertPointKlist(
    int kth,
    float distance,
    int indexv,
    __local float* kdistances)
{
	int k=0;
	while( (distance>*(kdistances+k)) && (k<kth-1))
        {k++;}
	//Move value to the next
	for(int k2=kth-1;k2>k;k2--)
    {
		*(kdistances+k2)=*(kdistances+k2-1);
	}
	//Replace
	*(kdistances+k)=distance;

	return *(kdistances+kth-1);
}

float maxMetricPoints(
    __global const float* g_uquery,
    __global const float* g_vpoint,
    int pointdim,
    int signallength)
{
	float	r_u1;
	float	r_v1;
	float	r_d1,r_dim=0;

	r_dim=0;
	for(int d=0; d<pointdim; d++)
    {
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

__kernel void kernelKNNshared(
    __global const float* g_uquery,
    __global const float* g_vpointset,
    __global float* g_distances,
    const int pointdim, 
    const int triallength,
    const int signallength,
    const int kth,
    const int exclude,
    __local float* kdistances)
{
	const unsigned int tid = get_global_id(0)+get_global_id(1)*get_global_size(0); //Global identifier
	const unsigned int itrial = tid / triallength; //Chunk index

	if (tid<signallength)
	{
		for (int k=0; k<kth; k++)
		{
			kdistances[get_local_id(0)*kth + k] = INFINITY;
		}

	    barrier(CLK_LOCAL_MEM_FENCE);

	    float r_kdist=INFINITY;
	    unsigned int indexi = tid-triallength*itrial; //Position inside the chunk

	    for(int t=0; t<triallength; t++)
	    {
		    int indexu = tid; //Current position
		    int indexv = (t + itrial*triallength); //Read all chunk members
		    int condition1=indexi-exclude;
		    int condition2=indexi+exclude;
		    //Exclude = thelier. If thelier = 0, analize all points except the actual one
		    if((t<condition1)||(t>condition2))
		    {
			    float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv, pointdim, signallength);
			    if(temp_dist <= r_kdist)
			    {
				    r_kdist = insertPointKlist(kth,temp_dist,t,kdistances+get_local_id(0)*kth);
			    }
		    }
	    }
	
	    barrier(CLK_LOCAL_MEM_FENCE);

	    //Copy to global memory
	    for(int k=0; k<kth; k++)
	    {
		    g_distances[tid+k*signallength] = kdistances[get_local_id(0)*kth+k];
	    }
    }
}

/*
 * Radius shared
 */

__kernel void kernelBFRSAllshared(
    __global const float* g_uquery,
    __global const float* g_vpointset,
    __global const float* vecradius,
    __global int* g_npoints,
    const int pointdim,
    const int triallength,
    const int signallength,
    const int exclude,
    __local int* s_npointsrange2) //Best performance without using shared/local memory)
{

	float radius=0;

	int s_npointsrange;

	const unsigned int tid = get_global_id(0)+get_global_id(1)*get_global_size(0); //Global identifier
	const unsigned int itrial = tid / triallength; //Chunk index

	if(tid<signallength)
	{
		s_npointsrange= 0;

	    radius = *(vecradius+tid);
		unsigned int indexi = tid-triallength*itrial;
		for(int t=0; t<triallength; t++)
		{
			int indexu = tid;
			int indexv = (t + itrial*triallength);
			int condition1=indexi-exclude;
			int condition2=indexi+exclude;
			if((t<condition1)||(t>condition2))
			{
				float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
				if(temp_dist < radius)
				{
					s_npointsrange++;
				}
			}
		}

	    barrier(CLK_LOCAL_MEM_FENCE);
        //COPY TO GLOBAL MEMORY
	    g_npoints[tid] = s_npointsrange;
	}
}


