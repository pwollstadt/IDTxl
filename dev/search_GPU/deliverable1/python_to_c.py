#!/usr/bin/env python

'''
    *
    * testKNN_call.py
    *
    *  Created on: 15/03/2014
    *      Author: cgongut
    *
'''

import numpy as np
import ctypes
from ctypes import *

'''
    MultiGPU Knn Function
'''

# extract cudaFindKnnSetGPU function pointer in the shared object gpuKnnLibrary.so
def get_cudaFindKnnSetGPU():
    dll = ctypes.CDLL('./gpuKnnLibrary.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cudaFindKnnSetGPU
    func.argtypes = [POINTER(c_int32), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int, c_int]
    return func

# create __cudaFindKnnSetGPU function with get_cudaFindKnnSetGPU()
__cudaFindKnnSetGPU = get_cudaFindKnnSetGPU()

# convenient python wrapper for __cudaFindKnnSetGPU
# it does all job with types convertation
# from python ones to C++ ones 
def cudaFindKnnSetGPU(indexes, distances, pointset, queryset, kth, thelier, nchunks, pointsdim, signallengthpergpu, gpuid):
    indexes_p = indexes.ctypes.data_as(POINTER(c_int32))
    distances_p = distances.ctypes.data_as(POINTER(c_float))
    pointset_p = pointset.ctypes.data_as(POINTER(c_float))
    queryset_p = queryset.ctypes.data_as(POINTER(c_float))

    bool = __cudaFindKnnSetGPU(indexes_p, distances_p, pointset_p, queryset_p, kth, thelier, nchunks, pointsdim, signallengthpergpu, gpuid)
    
    return bool

'''
    MultiGPU Range Search Function
'''

# extract cudaFindRSAllSetGPU function pointer in the shared object gpuKnnLibrary.so
def get_cudaFindRSAllSetGPU():
    dll = ctypes.CDLL('./gpuKnnLibrary.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cudaFindRSAllSetGPU
    func.argtypes = [POINTER(c_int32), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int]
    return func

# create __cudaFindRSAllSetGPU function with get_cudaFindRSAllSetGPU()
__cudaFindRSAllSetGPU = get_cudaFindRSAllSetGPU()

# convenient python wrapper for __cudaFindRSAll
# it does all job with types convertation
# from python ones to C++ ones 
def cudaFindRSAllSetGPU(npointsrange, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, datalengthpergpu, gpuid):
    npointsrange_p = npointsrange.ctypes.data_as(POINTER(c_int32))
    pointset_p = pointset.ctypes.data_as(POINTER(c_float))
    queryset_p = queryset.ctypes.data_as(POINTER(c_float))
    vecradius_p = vecradius.ctypes.data_as(POINTER(c_float))

    bool = __cudaFindRSAllSetGPU(npointsrange_p, pointset_p, queryset_p, vecradius_p, thelier, nchunkspergpu, pointsdim, datalengthpergpu, gpuid)
    
    return bool

'''
    KNN function
'''

# KNN function
def testKNN_call_multiGPU(indexes, distances, pointset, queryset, kth, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid):

    #GPU Execution
    bool = cudaFindKnnSetGPU(indexes, distances, pointset, queryset, kth, thelier, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    
    return bool
    
'''
    Range Search function
'''

# Range Search function
def testRSAll_call_multiGPU(npointsrange, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, datalengthpergpu, gpuid):

    #GPU Execution
    bool = cudaFindRSAllSetGPU(npointsrange, pointset, queryset, vecradius, thelier, nchunkspergpu, pointsdim, datalengthpergpu, gpuid)
    
    return bool

