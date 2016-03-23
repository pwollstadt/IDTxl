#!/usr/bin/env python

'''
    *
    * testRSAll_call.py
    *
    *  Created on: 2016-03-23
    *      Author: mwibral
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

    pointset_orig = np.array(
                        [
                         (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.0, -2.0, 0.0, -3.0, -4.0 ),
                         (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0 )
                          ], dtype=np.float32 )
    queryset_orig = pointset_orig  # important! function does not work without this!

    print(pointset_orig.flags)

    pointsdim = int(pointset_orig.shape[0])
    chunksize   =int(pointset_orig.shape[1]);
    signallengthpergpu = int(nchunkspergpu * chunksize)

    #Create an array of zeros for npointsrange, fill in pointset with a random vector, and start vecradius as an array of ones
    npointsrange = np.zeros((queryset_orig.shape[1]), dtype=np.int32)

    ####################
    # test KNN search
    #DATA INITIALIZATION
    kth = int(8)
    theiler = 0

    #Create an array of zeros for indexes and distances, and random array for pointset
    indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
    distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)

    # make deep copies of the original data to avoid any side effects
    pointset = pointset_orig.copy()
    queryset = queryset_orig.copy()
    #GPU Execution
    start = time.time()
    correct = clFindKnn(indexes, distances, pointset.transpose(),
                        queryset.transpose(), kth, thelier, nchunkspergpu,
                        pointsdim, signallengthpergpu, gpuid)
    end = time.time()


    if correct == 0:
        print( "GPU execution failed")
    else:
        print(("Execution time: %f" %(end - start)))
        print(pointset)
        print("Array of distances")
        print(distances)
        print("Array of index")
        print(indexes)

    # range searches
    vecradius =  distances[-1, :]

    # with original pointset
    pointset = pointset_orig.copy()
    queryset = queryset_orig.copy()

    correct = clFindRSAll(npointsrange, pointset, queryset, vecradius, theiler,
                          nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    if correct == 0:
        print ("GPU OpenCL execution failed")
    else:
        print("Points strictly inside radius")
        print(npointsrange)

    # with original pointset.transpose() - this should only yield a view into the data ...
    pointset = pointset_orig.copy()
    queryset = queryset_orig.copy()

    correct = clFindRSAll(npointsrange, pointset.transpose(), queryset.transpose(),
                          vecradius, theiler, nchunkspergpu, pointsdim,
                          signallengthpergpu, gpuid)
    if correct == 0:
        print ("GPU OpenCL execution failed")
        npointsrange_correct = npointsrange
    else:
        print("Points.transpose() strictly inside radius")
        print(npointsrange)

    # with original pointset.transpose().copy() - this should yield a new memory layout
    pointset = pointset_orig.copy()
    queryset = queryset_orig.copy()

    correct = clFindRSAll(npointsrange, pointset.transpose().copy(),
                          queryset.transpose().copy(), vecradius, theiler,
                          nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    if correct == 0:
        print ("GPU OpenCL execution failed")
    else:
        print("Points.transpose().copy() strictly inside radius")
        print(npointsrange)

    # with original pointset.asfortranarray()
    pointset = pointset_orig.copy()
    queryset = queryset_orig.copy()

    correct = clFindRSAll(npointsrange, np.asfortranarray(pointset),
                          np.asfortranarray(queryset), vecradius, theiler,
                          nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    if correct == 0:
        print ("GPU OpenCL execution failed")
    else:
        print("np.asfortranarray(Points) strictly inside radius")
        print(npointsrange)

    # with original pointset.asontiguouscarray()
    pointset = pointset_orig.copy()
    queryset = queryset_orig.copy()

    correct = clFindRSAll(npointsrange, np.ascontiguousarray(pointset),
                          np.ascontiguousarray(queryset), vecradius, theiler,
                          nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    if correct == 0:
        print ("GPU OpenCL execution failed")
    else:
        print("np.ascontiguousarray(Points) strictly inside radius")
        print(npointsrange)

# results:
# pyopencl sees B1 = A.transpose() and B2 = A as the same thing (B1 is only a view)
# pyopencl sees B1 = A.transpose().copy() and B2 = A as different
# pyopencl sees B1  = A.transpose().copy() and B2 = np.asfortranarray (A) as the same thing
# numpy (!) sees B1  = A.transpose().copy() and B2 = np.asfortranarray (A) as different things
#        when printing, and when indexing the array elements