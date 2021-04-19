import sys
from . import idtxl_exceptions as ex
import numpy as np
import math
try:
    from numba import njit, prange, cuda, float32, float64, int32, int64, jitclass, types
except ImportError as err:
    ex.package_missing(err, 'Numba is not available on this system. Install'
                            ' it using pip or the package manager to use '
                            'the Numba estimators.')


@njit()
def _insertPointKlistNumbaCPU(kth, dist, kdist):
    # get dist position for kdist
    ik = 0
    while (dist > kdist[ik]) and (ik < kth - 1):
        ik += 1

    for k2 in range(kth - 1, ik, -1):
        kdist[k2] = kdist[k2 - 1]

    # Replace
    kdist[ik] = dist

    return kdist[kth - 1], kdist


@njit()
def _maxPointMetricNumbaCPU(queryvec, pointsvec, pointdim):
    rdim = np.float32(0)
    for d in range(pointdim):
        r_u = queryvec[d]
        r_v = pointsvec[d]
        rd = r_v - r_u
        if rd < 0.0:
            rd = -rd
        if rdim < rd:
            rdim = rd
    return rdim


@njit(parallel=True)
def _knnNumbaCPU(query, pointset, kdistances, pointdim, chunklength, signallength, kth, exclude, datatype):

    # loop over all data points in query
    for tid in prange(signallength):

        # get corresponding chunk number
        ichunk = int(tid / chunklength)

        # initialize new max distance as inf
        if datatype == 32:
            r_kdist = float32(math.inf)
        else:
            r_kdist = math.inf

        # index
        indexi = tid - chunklength * ichunk

        # loop over corresponding chunk samples of data points tid
        for t in range(chunklength):

            # exclude Theiler or at least one
            if t < (indexi - exclude) or t > (indexi + exclude):

                # data points for distance search
                queryvec = query[:, tid]
                pointsvec = pointset[:, ichunk * chunklength + t]

                # brute force knn search
                temp_dist = _maxPointMetricNumbaCPU(queryvec, pointsvec, pointdim)

                if datatype == 32:
                    temp_dist = float32(temp_dist)

                # add new smaller distance to kdistances output
                if temp_dist <= r_kdist:

                    r_kdist, kdist = _insertPointKlistNumbaCPU(kth, temp_dist, kdistances[tid, :])
                    kdistances[tid, :] = kdist

    return kdistances


@njit(parallel=True)
def _rsAllNumbaCPU(uquery, vpointset, vecradius, npoints, pointdim, chunklength, signallength, exclude, datatype):

    # loop over all datapoints in query
    for tid in prange(signallength):

        # get corresponding chunk number
        ichunk = int(tid / chunklength)

        # initialize no of points in range
        npointsrange = np.int(0)

        # index
        indexi = tid - chunklength * ichunk

        # loop over corresponding chunk samples of datapoints tid
        for t in range(chunklength):

            # exclude Theiler or at least one
            if t < (indexi - exclude) or t > (indexi + exclude):

                # data points for distance search
                queryvec = uquery[tid]
                pointsvec = vpointset[ichunk * chunklength + t]

                # brute force knn search
                temp_dist = _maxPointMetricNumbaCPU(queryvec, pointsvec, pointdim)

                if datatype == 32:
                    temp_dist = float32(temp_dist)
                    vecrad = float32(vecradius[tid])
                else:
                    vecrad = vecradius[tid]

                # add new smaller dist to kdistances output
                if temp_dist < vecrad:
                    npointsrange += 1

        npoints[tid] = npointsrange

    return npoints


@cuda.jit
# @cuda.jit('void(float32[:,:], float32[:,:], float32[:], int32, int32, int32, int32, int32, int32, int32)', debug=True)
def _knnNumbaCuda(gquery,
                 gpointset,
                 gdistances,
                 pointdim,
                 chunklength,
                 signallength_padded,
                 signallength_orig,
                 kth,
                 exclude,
                 kdistances):

    # thread indexes
    tx = cuda.threadIdx.x

    # block indexes
    bx = cuda.blockIdx.x

    # block width
    bwx = cuda.blockDim.x

    # actual position
    tid = tx + bx * bwx

    if tid < signallength_orig:

        r_kdist = float32(math.inf)

        for t in range(chunklength):

            if t < (tid - chunklength * np.int(tid / chunklength)) - exclude or \
                    t > (tid - chunklength * np.int(tid / chunklength)) + exclude:

                # brute force knn search

                # maxPointMetric
                temp_dist = 0.0
                for d in range(pointdim):
                    r_u = gquery[d, tid]
                    r_v = gpointset[d, np.int(tid / chunklength) * chunklength + t]
                    rd = r_v - r_u
                    if rd < 0.0:
                        rd = -rd
                    if temp_dist < rd:
                        temp_dist = rd

                # add new smaller dist to kdistances output
                if temp_dist <= r_kdist:

                    # insertPointKlist
                    # get dist position for kdist
                    ik = 0
                    while (temp_dist > kdistances[tid, ik]) and (ik < kth - 1):
                        ik += 1

                    for k2 in range(kth - 1, ik, -1):
                        kdistances[tid, k2] = kdistances[tid, k2 - 1]

                    # Replace
                    kdistances[tid, ik] = temp_dist

                    r_kdist = kdistances[tid, kth - 1]

        cuda.syncthreads()

        # copy to global memory
        for k in range(kth):
            gdistances[tid, k] = kdistances[tid, k]


@cuda.jit
def _rsAllNumbaCuda(gquery, gpointset, vecradius, npoints, pointdim, chunklength, signallength_orig, kth, exclude):

    # thread indexes
    tx = cuda.threadIdx.x

    # block indexes
    bx = cuda.blockIdx.x

    # block width
    bwx = cuda.blockDim.x

    # actual position
    tid = tx + bx * bwx

    # loop over all datapoints in query
    if tid < signallength_orig:

        npointsrange = 0

        # loop over corresponding chunk samples of datapoints tid
        for t in range(chunklength):

            if t < (tid - chunklength * np.int(tid / chunklength)) - exclude or \
                    t > (tid - chunklength * np.int(tid / chunklength)) + exclude:

                # brute force ncount

                # maxPointMetric
                temp_dist = 0.0
                for d in range(pointdim):
                    r_u = gquery[d, tid]
                    r_v = gpointset[d, np.int(tid / chunklength) * chunklength + t]
                    rd = r_v - r_u
                    if rd < 0.0:
                        rd = -rd
                    if temp_dist < rd:
                        temp_dist = rd

                # add point to radius
                if temp_dist < vecradius[tid]:
                    npointsrange += 1

        cuda.syncthreads()

        npoints[tid] = npointsrange


