"""Provide neighbour searches using CUDA GPU-code."""
import numpy as np
from ctypes import *


def knn_search(pointset, queryset, knn_k, theiler_t, n_chunks=1, gpuid=0):
    """Interface with CUDA knn search from Python/IDTxl."""
    n_points = pointset.shape[0]
    pointdim = pointset.shape[1]

    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    success = cudaFindKnnSetGPU(indexes, distances, pointset, queryset, knn_k,
                                theiler_t, n_chunks, pointdim, n_points, gpuid)
    if success:
        return (indexes, distances)
    else:
        print("Error in CUDA knn search!")
        return 1


def range_search(pointset, queryset, radius, theiler_t, n_chunks=1, gpuid=0):
    """Interface with CUDA range search from Python/IDTxl."""
    n_points = pointset.shape[0]
    pointdim = pointset.shape[1]

    pointcount = np.zeros((1, n_points), dtype=np.float32)

    success = cudaFindRSAllSetGPU(pointcount, pointset, queryset, radius,
                                  theiler_t, n_chunks, pointdim, n_points,
                                  gpuid)
    if success:
        return pointcount
    else:
        print("Error in OpenCl range search!")
        return 1


def get_cudaFindKnnSetGPU():
    """Extract CUDA knn function from the shared library."""
    dll = CDLL('./gpuKnnLibrary.so', mode=RTLD_GLOBAL)
    func = dll.cudaFindKnnSetGPU
    func.argtypes = [POINTER(c_int32), POINTER(c_float), POINTER(c_float),
                     POINTER(c_float), c_int, c_int, c_int, c_int, c_int,
                     c_int]
    return func

# create __cudaFindKnnSetGPU function with get_cudaFindKnnSetGPU()
__cudaFindKnnSetGPU = get_cudaFindKnnSetGPU()


def cudaFindKnnSetGPU(indexes, distances, pointset, queryset, knn_k, theiler_t,
                      nchunks, pointsdim, signallengthpergpu, gpuid):
    """Wrap knn CUDA function for use in Python.

    Do type conversions necessary for calling C/CUDA code from Python.
    """
    assert pointset.flags['C_CONTIGUOUS'], 'Pointset is not C-contiguous.'
    assert queryset.flags['C_CONTIGUOUS'], 'Queryset is not C-contiguous.'

    indexes_p = indexes.ctypes.data_as(POINTER(c_int32))
    distances_p = distances.ctypes.data_as(POINTER(c_float))
    pointset_p = pointset.ctypes.data_as(POINTER(c_float))
    queryset_p = queryset.ctypes.data_as(POINTER(c_float))

    success = __cudaFindKnnSetGPU(indexes_p, distances_p, pointset_p,
                                  queryset_p, knn_k, theiler_t, nchunks,
                                  pointsdim, signallengthpergpu, gpuid)

    return success

'''
    MultiGPU Range Search Function
'''


def get_cudaFindRSAllSetGPU():
    """Extract CUDA range search function from the shared library.

    Extract cudaFindRSAllSetGPU function pointer in the shared object
    gpuKnnLibrary.so.
    """
    dll = CDLL('./gpuKnnLibrary.so', mode=RTLD_GLOBAL)
    func = dll.cudaFindRSAllSetGPU
    func.argtypes = [POINTER(c_int32), POINTER(c_float), POINTER(c_float),
                     POINTER(c_float), c_int, c_int, c_int, c_int, c_int]
    return func

# create __cudaFindRSAllSetGPU function with get_cudaFindRSAllSetGPU()
__cudaFindRSAllSetGPU = get_cudaFindRSAllSetGPU()


def cudaFindRSAllSetGPU(npointsrange, pointset, queryset, vecradius, theiler_t,
                        nchunkspergpu, pointsdim, datalengthpergpu, gpuid):
    """Wrap range search CUDA function for use in Python.

    Do type conversions necessary for calling C/CUDA code from Python.
    """
    npointsrange_p = npointsrange.ctypes.data_as(POINTER(c_int32))
    pointset_p = pointset.ctypes.data_as(POINTER(c_float))
    queryset_p = queryset.ctypes.data_as(POINTER(c_float))
    vecradius_p = vecradius.ctypes.data_as(POINTER(c_float))

    success = __cudaFindRSAllSetGPU(npointsrange_p, pointset_p, queryset_p,
                                    vecradius_p, theiler_t, nchunkspergpu,
                                    pointsdim, datalengthpergpu, gpuid)

    return success
