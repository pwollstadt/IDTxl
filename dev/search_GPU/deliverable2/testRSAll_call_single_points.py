#!/usr/bin/env python

'''
    *
    * testRSAll_call.py
    *
    *  Created on: 07/04/2014
    *      Author: cgongut
    *
'''

from clKnnLibrary import *

import numpy as np
import time

# main function
if __name__ == '__main__':
    
    #DATA INITIALIZATION
    gpuid = int(0)
    thelier = int(0)
    nchunkspergpu = int(1)

    pointset = np.array( [(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.0, -2.0, 0.0), (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 2.0, 0.0, -2.0 ) ], dtype=np.float32)
    queryset = pointset

    #pointset = np.random.random((signallengthpergpu,pointsdim)).astype('float32')
	
	# coding this manually to reflect how I interpret(!) the data
	# i.e. 8 points in 2 dimensions
    pointsdim = int(2)
    chunksize=int(pointset.shape[1]);
    signallengthpergpu = int(nchunkspergpu * chunksize)

    #Create an array of zeros for npointsrange, fill in pointset with a random vector, and start vecradius as an array of ones
    npointsrange = np.zeros((queryset.shape[1]), dtype=np.int32)
    vecradius = 2.01 * np.ones((signallengthpergpu), dtype=np.float32) 
    #GPU Execution
    start = time.time()
    correct = clFindRSAll(npointsrange, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()
    
    if correct == 0:
        print ("GPU OpenCL execution failed")
    else:
        print(("Execution time of OpenCL: %f" %(end - start))) 
        print("Array of points inside radius")
        print(npointsrange)
        
    ####################
    # test KNN search
    #DATA INITIALIZATION
    kth = int(7)
    thelier = 0
 
    #Create an array of zeros for indexes and distances, and random array for pointset 
    indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
    distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)

    #GPU Execution
    start = time.time()
    correct = clFindKnn(indexes, distances, pointset, queryset, kth, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    
    if correct == 0:
        print( "GPU execution failed")
    else:
        print(( "Execution time: %f" %(end - start)))
        print( pointset )
        print( "Array of distances")
        print( distances )
        print( "Array of index")
        print( indexes)
