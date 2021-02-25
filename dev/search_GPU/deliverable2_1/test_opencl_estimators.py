"""OpenCL test function - tests IDTxl estimators

  Created on: 06/27/2019
  Authors: pwollstadt
"""
import logging
import argparse
from sys import getsizeof
import pyopencl as cl
import numpy as np
import time
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def knn_mi_est(pointset, kth, theiler, nchunkspergpu, pointsdim,
               signallengthpergpu, gpuid, chunklength, cpu=False, padding=True):
    settings = {
        'gpuid': gpuid,
        'kraskov_k': kth,
        'normalise': False,
        'theiler_t': theiler,
        'noise_level': 0,
        'debug': True,
        'padding': padding,
        'return_counts': True,
        'lag_mi': 0
        }

    est_mi = OpenCLKraskovMI(settings)
    mi_array, distances, count_var1, count_var2 = est_mi.estimate(
        pointset[0, :], pointset[1, :], nchunkspergpu)
    print(distances.shape)
    print(count_var1.shape)
    return distances


if __name__ == '__main__':

    # logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(
            format='%(asctime)s - %(levelname)-4s  [%(filename)s:%(funcName)20s():l %(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Debug IDTxl GPU searches')
    parser.add_argument('--npoints', '-p', default=1000, type=int,
                        help='No. points per chunk')
    parser.add_argument('--ndims', '-d', default=2, type=int)
    parser.add_argument('--nchunks', '-c', default=1, type=int)
    parser.add_argument('--gpuid', '-g', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--old', action='store_true',
                        help='Use Marios old code')
    parser.add_argument('--cpu', action='store_false',
                        help='Run code on CPU')
    parser.add_argument('--padding', action='store_true',
                        help='Pad pointset')

    args = parser.parse_args()

    if args.seed != 0:
        print('Setting random seed to {}'.format(args.seed))
        np.random.seed(args.seed)

    #DATA INITIALIZATION
    gpuid = int(args.gpuid)
    kth = int(2)  # int(5)
    theiler = int(0)
    print("gpuid: {}".format(gpuid))

    # # TEST CASE 1 - various combinations of no. points and dims
    pointsdim = int(args.ndims)
    nchunkspergpu = int(args.nchunks)
    signallengthpergpu = nchunkspergpu * int(args.npoints)

    # if args.padding:
    #     # Pad time series to make GPU memory regions a multiple of 1024
    #     pad_target = 1024  # this is used in the toolbox
    #     print(f'Applying padding to multiple of {pad_target}')
    #     # pad_target = 256
    #     pad_size = (int(np.ceil(signallengthpergpu/pad_target)) * pad_target -
    #                 signallengthpergpu)
    #     signallengthpergpu += pad_size
    #     assert signallengthpergpu % pad_target == 0
    # else:
    #     pad_size = 'n/a'
    pad_size = 'done within IDTxl estimator'

    chunksize = signallengthpergpu / nchunkspergpu

    # Create a random array for pointset, orientation in IDTxl is dim x samples
    # pointset = np.random.random((signallengthpergpu, pointsdim)).astype('float32')
    pointset = np.random.random((pointsdim, signallengthpergpu)).astype('float32')

    # Print memory requirement in MB
    c = 1024**2
    print('\npointset: {0:.2f}, TOTAL: {1:.2f} MB, PADDING: {2}'.format(
      getsizeof(pointset) / c, (getsizeof(pointset) * 2) / c, pad_size))
    print('pointset shape: {}'.format(pointset.shape))
    print('pointset shape % n_chunks: {} (chunkkength: {})\n'.format(
        signallengthpergpu % nchunkspergpu, chunksize))

    start = time.time()
    success = True
    try:
        distances = knn_mi_est(
            pointset, kth, theiler, nchunkspergpu, pointsdim,
            signallengthpergpu, gpuid, chunksize, args.cpu, args.padding)
    except cl._cl.RuntimeError as e:
        print(e)
        success = False
    end = time.time()

    print("Execution time: {0:.2f} min" .format((end - start) / 60))
    if not success:
        print("\n!!! GPU execution failed\n")
    else:
        # print("Array of distances")
        signallength_padded = int(len(distances) / kth)
        distances = distances.reshape(kth, signallength_padded)
        print("Array of distances shape: {}".format(distances.shape))
        i = 0
        for c in range(1, nchunkspergpu + 1):
            for k in range(kth):
                print(f'k  = {k}')
                print(distances[k, int(i):int(chunksize*c)])
            i += chunksize
