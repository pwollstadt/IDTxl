"""CUDA test function

  testKNN_call_multiGPU.py
  Created on: 15/03/2014, modified 06/03/2019
  Authors: cgongut, pwollstadt

  Test parallel knn- and range-searches on GPU using CUDA. Kernel handles
  multiple problem instances (search spaces) in parallel, where each search
  space is called a 'chunk'. Each chunk is handled in parallel and whithin each
  chunk, searches for individual query points are handled in parallel as well.
  Typically, every point in the search space is also a query point such that
  we obtain the nearest neighbors for each point in the search space.
  The code expects a single pointset and the number of chunks as input, where
  the pointset is a concatenation of all search spaces/chunks.

  Examples:
    python test_cuda_search.py
    python test_cuda_search.py -p 35000000  # 1 chunk, 35000000 points
    python test_cuda_search.py -p 5000 -d 3  # 1 chunk, 5000 points, 3 dimensions
    python test_cuda_search.py -p 35000000 -g 2  # 1 chunk, 35000000 points, GPU 3
    python test_cuda_search.py -p 500 -c 10  # 10 chunks with 500 points each
"""
import logging
import argparse
from sys import getsizeof
from python_to_c import *

import numpy as np
import time

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Debug IDTxl CUDA searches')
    parser.add_argument('--npoints', '-p', default=1000, type=int,
                        help='No. points per chunk.')
    parser.add_argument('--ndims', '-d', default=1, type=int,
                        help='No. dimensions.')
    parser.add_argument('--nchunks', '-c', default=1, type=int,
                        help='No. chunks, i.e., search spaces searched in parallel.')
    parser.add_argument('--gpuid', '-g', default=0, type=int,
                        help='GPU device ID.')

    args = parser.parse_args()

    # DATA INITIALIZATION
    gpuid = int(args.gpuid)
    kth = int(5)
    theiler = int(0)
    chunksize = int(args.npoints)
    pointsdim = int(args.ndims)
    nchunkspergpu = int(args.nchunks)
    signallengthpergpu = nchunkspergpu * chunksize

    # Create an array of zeros for indexes and distances, and random array for
    # pointset
    indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
    distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)

    pointset = np.random.random(
      (signallengthpergpu, pointsdim)).astype('float32')
    queryset = pointset
    print(pointset.shape)
    print(getsizeof(pointset))

    # Print memory requirement in MB
    c = 1024**2
    print('pointset: {0:.2f}, queryset: {1:.2f}, indexes: {2:.2f}, distances: {3:.2f}, TOTAL: {4:.2f} MB'.format(
      getsizeof(pointset) / c, getsizeof(queryset) / c, getsizeof(indexes) / c,
      getsizeof(distances) / c,
      (getsizeof(pointset) + getsizeof(queryset) + getsizeof(indexes) +
       getsizeof(distances))  / c))
    print('pointset shape: {}'.format(pointset.shape))

    # KNN search
    start = time.time()
    res = testKNN_call_multiGPU(indexes, distances, pointset, queryset, kth, theiler, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    if res == 0:
        print("\nGPU execution failed")
    else:
        print("Execution time: {0:.2f} min" .format((end - start) / 60))
        print("Array of distances")
        print(distances)
        print("Array of index")
        print(indexes)

    # Range search
    npointsrange = np.zeros((signallengthpergpu), dtype=np.int32)

    start = time.time()
    res = testRSAll_call_multiGPU(
      npointsrange, pointset, queryset, distances, theiler, nchunkspergpu,
      pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    if res == 0:
        print("\nGPU execution failed")
    else:
        print('Range Search:')
        print("Execution time: {0:.2f} min" .format((end - start) / 60))
