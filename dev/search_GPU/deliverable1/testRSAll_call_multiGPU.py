#!/usr/bin/env python

'''
    *
    * testRSAll_call_multiGPU.py
    *
    *  Created on: 02/04/2014
    *      Author: cgongut
    *
'''

from python_to_c import *

import numpy as np
import time

if __name__ == '__main__':
    
    #DATA INITIALIZATION
    gpuid = int(0)
    chunksize=8000
    nchunkspergpu = int(10)
    pointsdim = int(8)
    thelier = int(0)
    datalengthpergpu = nchunkspergpu * chunksize

    #Create an array of zeros for npointsrange, fill in pointset with a random vector, and start vecradius as an array of ones
    npointsrange = np.zeros((datalengthpergpu), dtype=np.int32)
    pointset = np.random.random((datalengthpergpu,pointsdim)).astype('float32')
    queryset = pointset
    vecradius = 0.5 * np.ones((datalengthpergpu), dtype=np.float32)

    '''pointset2 = np.array( [(-12.1, 23.4, -20.6, 21.6, -8.5, 23.7, -10.1, 8.5), (5.3, -9.2, 8.2, -15.3, 15.1, -9.2,  5.5, -15.1) ], dtype=np.float32)
    pointset = np.concatenate((pointset2,pointset2), axis=1)
    for i in range(0,1000):
        pointset = np.concatenate((pointset,pointset2), axis=1)'''    
    
    #GPU Execution
    start = time.time()
    bool = testRSAll_call_multiGPU(npointsrange, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, datalengthpergpu, gpuid)
    end = time.time()
    
    if bool == 0:
        print "GPU execution failed"
    else:
        print "Execution time: %f" %(end - start) 
        print "Array of points inside radius"
        print npointsrange 
