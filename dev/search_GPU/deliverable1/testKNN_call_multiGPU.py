#!/usr/bin/env python

'''
    *
    * testKNN_call_multiGPU.py
    *
    *  Created on: 15/03/2014
    *      Author: cgongut
    *
'''

from python_to_c import *

import numpy as np
import time


if __name__ == '__main__':
    
    #DATA INITIALIZATION
    gpuid = int(1)
    chunksize=int(8)
    nchunkspergpu = int(1)
    kth = int(5);
    pointsdim = int(2);
    thelier = int(0)
    signallengthpergpu = nchunkspergpu * chunksize

    #Create an array of zeros for indexes and distances, and random array for pointset 
    indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
    distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)
    pointset = np.array( [(-12.1, 23.4, -20.6, 21.6, -8.5, 23.7, -10.1, 8.5), (5.3, -9.2, 8.2, -15.3, 15.1, -9.2,  5.5, -15.1) ], dtype=np.float32)
    #pointset = np.random.random((signallengthpergpu,pointsdim)).astype('float32')
    queryset = pointset

    '''pointset2 = np.array( [(-12.1, 23.4, -20.6, 21.6, -8.5, 23.7, -10.1, 8.5), (5.3, -9.2, 8.2, -15.3, 15.1, -9.2,  5.5, -15.1) ], dtype=np.float32)
    pointset = np.concatenate((pointset2,pointset2), axis=1)
    for i in range(0,1000):
        pointset = np.concatenate((pointset,pointset2), axis=1)'''

    start = time.time()
    bool = testKNN_call_multiGPU(indexes, distances, pointset, queryset, kth, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    if bool == 0:
        print "GPU execution failed"
    else:
        print "Execution time: %f" %(end - start) 
        print "Array of distances"
        print distances 
        print "Array of index"
        print indexes
