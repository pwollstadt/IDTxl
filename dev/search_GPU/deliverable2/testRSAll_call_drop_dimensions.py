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
    reps = 12  # repetitions for reliability testing

#    extra_radius = 0.0 # for tweaking the search range

    # common paramters for all gpu calculations
    gpuid = int(0)
    thelier = int(0)
    nchunkspergpu = int(1)
    kth = int(7)

    error_signature = []
    half_ndim = 10
    ndim = 2 * half_ndim
    num_points = 1000

    for i in range(0, reps):
        # for testing create a pointset with very different spread along
        # its dimensions:
        contribution_1 = np.random.normal(
                            0, 0.1, [half_ndim, num_points]).astype('float32')
        contribution_2 = np.random.normal(
                            0, 10.0, [half_ndim, num_points]).astype('float32')
        pointset = np.vstack((contribution_1, contribution_2))
        # or a homogenuous one
#        pointset = np.random.normal(
#                            0, 10.0, [ndim, num_points]).astype('float32')
        pointsdim = int(pointset.shape[0])

        # low dimensional pointset
        dim_to_drop = half_ndim
        pointset_low_dim = pointset[0:pointsdim-dim_to_drop, :]
        pointsdim_low = int(pointset_low_dim.shape[0])

        chunksize = int(pointset.shape[1])
        signallengthpergpu = int(nchunkspergpu * chunksize)
        print(signallengthpergpu)

        # Create an array of zeros to hold indexes and distances
        indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
        distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)

        # Create an array of zeros for npointsrange,
        # and for a space with less dimensions
        npointsrange = np.zeros(signallengthpergpu, dtype=np.int32)
        npointsrange_low_dim = np.zeros(signallengthpergpu, dtype=np.int32)

        ####################
        # test KNN search
        start = time.time()
        correct = clFindKnn(indexes, distances, pointset, pointset, kth,
                            thelier, nchunkspergpu, pointsdim,
                            signallengthpergpu, gpuid)
        end = time.time()

        if correct == 0:
            print("GPU execution failed")
        else:
            pass

        # range searches from here on
        vecradius = distances[-1, :]  # + extra_radius

        ##########################
        # test range search for lower dimension

        print("looking for the neighbours of {0} points".format(
                                npointsrange_low_dim.shape[0]))
        correct = clFindRSAll(
                              npointsrange_low_dim,
                              pointset_low_dim.transpose(),
                              pointset_low_dim.transpose(),
                              vecradius, thelier, nchunkspergpu,
                              pointsdim_low, signallengthpergpu, gpuid)

        if correct == 0:
            print("GPU OpenCL execution failed")
        else:
            print(
                "maximum number of neighbours in low dim: {0}, minimimum: {1}".format(
                    np.max(npointsrange_low_dim), np.min(npointsrange_low_dim)))
#            pass
#            print(("Execution time of OpenCL: %f" %(end - start)))
#            print("Array of points inside radius in lower dimensions")
#            print(npointsrange_low_dim)

        ###########################
        # test range search
        start = time.time()
        correct = clFindRSAll(
                              npointsrange, pointset.transpose(),
                              pointset.transpose(), vecradius, thelier,
                              nchunkspergpu, pointsdim, signallengthpergpu,
                              gpuid)
        end = time.time()

        if correct == 0:
            print("GPU OpenCL execution failed")
        else:
            print(
                "maximum number of neighbours: {0}, minimimum: {1}".format(
                    np.max(npointsrange), np.min(npointsrange)))

        if np.min((npointsrange_low_dim - npointsrange)) < 0:
            error_signature.append(
                                   np.min((npointsrange_low_dim - npointsrange)))
        else:
            error_signature.append(0)

    print("error signature was: {0}".format(error_signature))
    print("worst miscount was: {0}".format(np.min(error_signature)))
