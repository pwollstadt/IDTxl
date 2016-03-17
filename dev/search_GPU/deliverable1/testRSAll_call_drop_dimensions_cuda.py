#!/usr/bin/env python

'''
    *
    * testRSAll_call.py
    *
    *  Created on: 07/04/2014
    *      Author: cgongut
    *
'''

from python_to_c import *
import numpy as np
import time

# main function
if __name__ == '__main__':
    
    #DATA INITIALIZATION
    gpuid = int(0)
    thelier = int(0)
    nchunkspergpu = int(1)
    kth = int(7)
    reps = 10
    error_signature = []
    # change the pointcount below to provoke an error where the number of
    # neighbours in a higher dimensional space is bigger than in a corresponding
    # lower dimensional one - while it should be lower, because of the lower 
    # dimnesional constraints that apply
    # on my Carrizo iGPU the error appears first at 1625 points in the space
    # on my Bonaire dGPU the error appears first at 1633 points in the space
    #
    # What we know so far:
    # 1. the error is deterministic, i.e. if it occurs in one dataset of that size
    #   it will occur in all of them
    # 2. the error is (slightly?) hardware dependend since the two opencl capable
    #   GPUs in my device choke at slightly different point counts (perhaps because
    #   of sligthly different buffer sizes ??) 
    num_points = 1632
   
    
    for i in range(0, reps):

        pointset = np.random.normal(0, 1.0, [10, num_points]).astype('float32') # ten dimensions
        queryset = pointset
    
        pointsdim = int(pointset.shape[0])
        dim_to_drop = 2
        chunksize=int(pointset.shape[1]);
        signallengthpergpu = int(nchunkspergpu * chunksize)
        print(signallengthpergpu)
        
        #Create an array of zeros to hold indexes and distances
        indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
        distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)
        
        #Create an array of zeros for npointsrange, and for a space with less dimensions
        npointsrange = np.zeros(signallengthpergpu, dtype=np.int32)
        npointsrange_low_dim = np.zeros(signallengthpergpu, dtype=np.int32)
       
        ####################
        # test KNN search
        #DATA INITIALIZATION
        
     
        start = time.time()
        correct = clFindKnn(indexes, distances, pointset, queryset, kth, 
                            thelier, nchunkspergpu, pointsdim,
                            signallengthpergpu, gpuid)
        end = time.time()
        
        if correct == 0:
            print( "GPU execution failed")
        else:
            pass
#            print(( "Execution time: %f" %(end - start)))
#            print( pointset )
#            print( "Array of distances")
#            print( distances )
#            print( "Array of index")
#            print( indexes)
        
    
        vecradius =  distances[-1, :]
    
        ###########################
        # test range search
        start = time.time()
        correct = clFindRSAll(npointsrange, pointset, queryset, vecradius, thelier,
                              nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        end = time.time()
        
        if correct == 0:
            print ("GPU OpenCL execution failed")
        else:
            pass
#            print(("Execution time of OpenCL: %f" %(end - start))) 
#            print("Array of points inside radius")
#            print(npointsrange)
    
        ##########################
        # test range search for lower dimension
    
        poinset_low_dim = pointset[0:pointsdim-dim_to_drop,:]
        correct = clFindRSAll(npointsrange_low_dim, poinset_low_dim, 
                              poinset_low_dim, vecradius, thelier, nchunkspergpu,
                              pointsdim, signallengthpergpu, gpuid)
                                             
        
        if correct == 0:
            print ("GPU OpenCL execution failed")
        else:
            pass
#            print(("Execution time of OpenCL: %f" %(end - start))) 
#            print("Array of points inside radius in lower dimensions")
#            print(npointsrange_low_dim)    
            
        # the quantity computed below should always be positive,nbecause by
        # dropping dimensions we effectively drop constraints, but it isn't
        # for certain input configurations
        error_signature.append(np.min((npointsrange_low_dim - npointsrange)))

    print("error signature was: {0}".format(error_signature))
    print("worst miscount was: {0}".format(np.min(error_signature)))