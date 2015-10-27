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
        print "Execution time of OpenCL: %f" %(end - start) 
        print "Array of points inside radius"
        print npointsrange
