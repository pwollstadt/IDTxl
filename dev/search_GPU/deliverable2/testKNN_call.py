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

import numpy as np
import time

# main function
if __name__ == '__main__':
    
    #DATA INITIALIZATION
    gpuid = int(0)
    chunksize=40000
    nchunkspergpu = int(100)
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
        print "Execution time: %f" %(end - start)
        print pointset 
        print "Array of distances"
        print distances 
        print "Array of index"
        print indexes
