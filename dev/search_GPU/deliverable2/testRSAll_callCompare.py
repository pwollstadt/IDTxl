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
from python_to_c import *

import numpy as np
import time

# main function
if __name__ == '__main__':
    
    #DATA INITIALIZATION
    compareResults = 1 # 1 for Compare
    gpuid = int(0)
    thelier = int(0)
    nchunkspergpu = int(1000)
    pointsdim = int(2)
    chunksize=8000;
    signallengthpergpu = nchunkspergpu * chunksize;

    #Create an array of zeros for npointsrange, fill in pointset with a random vector, and start vecradius as an array of ones
    npointsrange = np.zeros((signallengthpergpu), dtype=np.int32)
    vecradius = 0.5 * np.ones((signallengthpergpu), dtype=np.float32)
    pointset = np.random.random((signallengthpergpu,pointsdim)).astype('float32')
    #pointset = np.array( [(-12.1, 23.4, -20.6, 21.6, -8.5, 23.7, -10.1, 8.5), (5.3, -9.2, 8.2, -15.3, 15.1, -9.2,  5.5, -15.1) ], dtype=np.float32)
    queryset = pointset

    #GPU Execution
    start = time.time()
    correct = clFindRSAll(npointsrange, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()
    
    if correct == 0:
        print "GPU OpenCL execution failed"
    else:
        print "Execution time in OpenCL: %f" %(end - start) 

    #Compare results with CUDA version
    if compareResults == 1:
        npointsrange2 = np.zeros((signallengthpergpu), dtype=np.int32)
        results = np.zeros((signallengthpergpu), dtype=np.int32)
        start = time.time()
        correct2 = cudaFindRSAllSetGPU(npointsrange2, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        end = time.time()

        if correct2 == 0:
            print "CUDA execution failed"
        else:
            print "Execution time in CUDA: %f" %(end - start) 
            #Substract results of both executions, and check if any of them it's different than 0.
            results = npointsrange - npointsrange2
            if any (results !=0):
                print "Test Not Passed"
            else:
                print "Test Passed"

