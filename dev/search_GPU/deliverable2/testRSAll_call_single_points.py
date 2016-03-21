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

    # GPU INITIALIZATION
    gpuid = int(0)
    thelier = int(0)
    nchunkspergpu = int(1)

    pointset = np.array(
                        [
                         (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.0, -2.0, 0.0 ),
                         (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 2.0, 0.0, -2.0 )
                          ], dtype=np.float32 )
    queryset = pointset  # important! function does not work without this!

    pointsdim = int(2)
    chunksize   =int(pointset.shape[1]);
    signallengthpergpu = int(nchunkspergpu * chunksize)

    #Create an array of zeros for npointsrange, fill in pointset with a random vector, and start vecradius as an array of ones
    npointsrange = np.zeros((queryset.shape[1]), dtype=np.int32)

    ####################
    # test KNN search
    #DATA INITIALIZATION
    kth = int(6)
    theiler = 0

    #Create an array of zeros for indexes and distances, and random array for pointset
    indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
    distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)

    #GPU Execution
    start = time.time()
    correct = clFindKnn(indexes, distances, pointset.transpose(),
                        queryset.transpose(), kth, thelier, nchunkspergpu,
                        pointsdim, signallengthpergpu, gpuid)
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

    # range searches
    vecradius =  distances[-1, :]
    #G PU Execution
    start = time.time()
    correct = clFindRSAll(npointsrange, pointset, queryset, vecradius, theiler,
                          nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    if correct == 0:
        print ("GPU OpenCL execution failed")
    else:
        print(("Execution time of OpenCL: %f" %(end - start)))
        print("Array of points strictly inside radius")
        print(npointsrange)
