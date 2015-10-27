#!/usr/bin/env python

'''
    *
    * testKNN_call.py
    *
    *  Created on: 01/04/2014
    *      Author: cgongut
    *
'''

from clKnnLibrary import *
from python_to_c import *

import numpy as np
import time

# main function
if __name__ == '__main__':
    
    #DATA INITIALIZATION
    compareResults = 1 # 1 for Compare
    error = float (10**-5) #Maximun difference of distance between both versions
    gpuid = int(0)
    chunksize=8000
    nchunkspergpu = int(1000)
    kth = int(5)
    pointsdim = int(8)
    thelier = int(0)
    signallengthpergpu = nchunkspergpu * chunksize

    #Create an array of zeros for indexes and distances, and random array for pointset 
    indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
    distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)
    pointset = np.random.random((pointsdim, signallengthpergpu)).astype('float32')
    #pointset = np.array( [(-12.1, 23.4, -20.6, 21.6, -8.5, 23.7, -10.1, 8.5), (5.3, -9.2, 8.2, -15.3, 15.1, -9.2,  5.5, -15.1) ], dtype=np.float32)
    queryset = pointset

    #GPU Execution
    start = time.time()
    correct = clFindKnn(indexes, distances, pointset, queryset, kth, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    
    if correct == 0:
        print "GPU execution failed"
    else:
        print "Execution time in OpenCL: %f" %(end - start)

    #Compare results with CUDA version
    if compareResults == 1:
        indexes2 = np.zeros((kth, signallengthpergpu), dtype=np.int32)
        distances2 = np.zeros((kth, signallengthpergpu), dtype=np.float32)
        results = np.zeros((signallengthpergpu), dtype=np.int32)
        results2 = np.zeros((signallengthpergpu), dtype=np.float32)
        start = time.time()
        correct2 = cudaFindKnnSetGPU(indexes2, distances2, pointset, queryset, kth, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        end = time.time()

        if correct2 == 0:
            print "CUDA execution failed"
        else:
            print "Execution time in CUDA: %f" %(end - start) 
            #Substract results of both executions, and check if any of them it's different than 0.
            results = indexes - indexes2
            if np.any (results != 0) :
                print "Indexes Test Not Passed"
                print "Check distances. Different indexes found (distances probably equal)"
            else:
                print "Indexes Test Passed"

            #Check if distances are similar enough
            results2 = np.absolute(distances - distances2)
            if np.any (results2 > error) :
                print "Distances Test Not Passed"
            else:
                print "Distances Test Passed"

