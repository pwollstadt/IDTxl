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


# @cuda.jit()
# def _insertPointKlistNumbaCuda(kth, dist, kdist):
#     # get dist position for kdist
#     ik = 0
#     while (dist > kdist[ik]) and (ik < kth - 1):
#         ik += 1
#
#     for k2 in range(kth - 1, ik, -1):
#         kdist[k2] = kdist[k2 - 1]
#
#     # Replace
#     kdist[ik] = dist
#
#     return kdist[kth - 1], kdist
#
# @cuda.jit()
# def _maxPointMetricNumbaCuda(queryvec, pointsvec, pointdim):
#     rdim = np.float32(0)
#     for d in range(pointdim):
#         r_u = queryvec[d]
#         r_v = pointsvec[d]
#         rd = r_v - r_u
#         if rd < 0.0:
#             rd = -rd
#         if rdim < rd:
#             rdim = rd
#     return rdim


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


#
# @njit(parallel=True)
# def _rsNumbaCPU(query, pointset, radius, pointdim, chunklength, signallength, exclude):
#
#     # initialize npoints as zero vector
#     npoints = np.zeros(signallength)
#
#     # loop over all datapoints in query
#     for tid in prange(signallength):
#
#         # get corresponding chunk number
#         ichunk = int(tid / chunklength)
#
#         # initiale npointsrange to zero
#         npointsrange = 0
#
#         # index
#         indexi = tid - chunklength * ichunk
#
#         # loop over corresponding chunk samples of datapoints tid
#         for t in range(chunklength):
#
#             # exclude Theiler or at least one
#             if t < indexi - exclude or t > indexi + exclude:
#
#                 # data points for distance search
#                 queryvec = query[:, tid]
#                 pointsvec = pointset[:, ichunk * chunklength + t]
#
#                 # brute force knn search
#                 temp_dist = _maxPointMetricNumbaCPU(queryvec, pointsvec, pointdim)
#
#                 # add new smaller dist to kdistances output
#                 if temp_dist <= radius:
#                     npointsrange += 1
#
#     npoints[tid] = npointsrange
#
#     return npoints



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

    # get grid index
    #tid = cuda.grid(1)

    # thread indexes
    # tid = cuda.threadIdx.x

    tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # block indexes
    bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y

    # block width
    bwx = cuda.blockDim.x
    # bwy = cuda.blockDim.y

    # actual position
    tid = tx + bx * bwx

    # skdistances = cuda.shared.array((N/threadsperblock,KTH), dtype=float32)
    # skdist = cuda.shared.array(shape=(np.int(signallength_orig), np.int(kth)), dtype=float32)

    #gquery_idx = gquery[:, idx]
    #gpointset_idx = gpointset[:, idx]

    # for tid in range(signallength_orig):
    if tid < signallength_orig:

        ##fill kdist with inf
        # for kk in range(kth):
        #    kdistances[tid, kk] = math.inf
        # cuda.syncthreads()

        # r_kdist = np.float32(math.inf)

        r_kdist = float32(math.inf)

        # index
        # indexi = tid - chunklength * ichunk
        # indexi = tid - chunklength * np.int(tid / chunklength)
        # or
        # indexi = tid - chunklength * np.floor(tid / chunklength)

        for t in range(chunklength):

            # if t < indexi - exclude or t > indexi + exclude:
            if t < (tid - chunklength * np.int(tid / chunklength)) - exclude or \
                    t > (tid - chunklength * np.int(tid / chunklength)) + exclude:

                # brute force knn search

                # rd = cuda.shared.array(4, float32)
                # temp_dist = cuda.shared.array(0, float32)

                # maxPointMetric
                # -----------------------------
                temp_dist = 0.0
                for d in range(pointdim):
                    r_u = gquery[d, tid]
                    r_v = gpointset[d, np.int(tid / chunklength) * chunklength + t]
                    #                    r_v = gpointset[d, ichunk * chunklength + t]
                    rd = r_v - r_u
                    if rd < 0.0:
                        rd = -rd
                    if temp_dist < rd:
                        temp_dist = rd

                # add new smaller dist to kdistances output
                if temp_dist <= r_kdist:

                    # insertPointKlist
                    # ---------------------------------
                    # get dist position for kdist
                    ik = 0
                    while (temp_dist > kdistances[tid, ik]) and (ik < kth - 1):
                        ik += 1

                    for k2 in range(kth - 1, ik, -1):
                        kdistances[tid, k2] = kdistances[tid, k2 - 1]

                    # Replace
                    kdistances[tid, ik] = temp_dist

                    r_kdist = kdistances[tx, kth - 1]

        cuda.syncthreads()

        # copy to global memory
        for k in range(kth):
            gdistances[tid, k] = kdistances[
                tx, k]  # ?????????????????????????????????????????????????????????????????????????



@cuda.jit
def _rsAllNumbaCuda(gquery, gpointset, vecradius, npoints, pointdim, chunklength, signallength_orig, kth, exclude):

    # thread indexes
    tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # block indexes
    bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y

    # block width
    bwx = cuda.blockDim.x
    # bwy = cuda.blockDim.y

    # actual position
    tid = tx + bx * bwx

    # loop over all datapoints in query
    if tid < signallength_orig:

        npointsrange = 0

        # loop over corresponding chunk samples of datapoints tid
        for t in range(chunklength):

            # if t < indexi - exclude or t > indexi + exclude:
            if t < (tid - chunklength * np.int(tid / chunklength)) - exclude or \
                    t > (tid - chunklength * np.int(tid / chunklength)) + exclude:

                # brute force ncount

                # maxPointMetric
                temp_dist = float32(0.0)
                for d in range(pointdim):
                    r_u = gquery[d, tid]
                    r_v = gpointset[d, np.int(tid / chunklength) * chunklength + t]
                    #                    r_v = gpointset[d, ichunk * chunklength + t]
                    rd = r_v - r_u
                    if rd < 0.0:
                        rd = -rd
                    if temp_dist < rd:
                        temp_dist = rd

                # add new smaller dist to kdistances output
                if temp_dist <= float32(vecradius[tid, kth]):
                    npointsrange += 1

        cuda.syncthreads()

        npoints[tid] = npointsrange


