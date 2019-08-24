"""OpenCL test function

  Created on: 06/27/2019
  Authors: pwollstadt
"""
import logging
import argparse
from sys import getsizeof
import pyopencl as cl
import numpy as np
import time

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def _get_device(gpuid):
    """Return GPU devices, context, and queue."""
    all_platforms = cl.get_platforms()
    platform = next((p for p in all_platforms if p.get_devices(device_type=cl.device_type.GPU) != []),
                    None)
    if platform is None:
        raise RuntimeError('No OpenCL GPU device found.')
    my_gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=my_gpu_devices)
    if gpuid > len(my_gpu_devices)-1:
        raise RuntimeError(
            'No device with gpuid {0} (available device IDs: {1}).'.format(
                gpuid, np.arange(len(my_gpu_devices))))
    queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
    print("Selected Device: ", my_gpu_devices[gpuid].name)
    return my_gpu_devices, context, queue


def _get_kernels(kernel_location, context):
    """Return KNN and range search OpenCL kernels."""
    kernel_source = open(kernel_location).read()
    program = cl.Program(context, kernel_source).build()
    kNN_kernel = program.kernelKNNshared
    kNN_kernel.set_scalar_arg_dtypes([None, None, None, np.int32,
                                      np.int32, np.int32, np.int32,
                                      np.int32, None])

    RS_kernel = program.kernelBFRSAllshared
    RS_kernel.set_scalar_arg_dtypes([None, None, None, None,
                                     np.int32, np.int32, np.int32,
                                     np.int32, None])
    return (kNN_kernel, RS_kernel)


def clFindKnn(pointset, queryset, kth, theiler,
              nchunkspergpu, pointsdim, signallengthpergpu, gpuid):

    devices, context, queue = _get_device(gpuid)
    kNN_kernel, RS_kernel = _get_kernels('gpuKnnKernelNoIdx.cl', context)

    # Set OpenCL kernel launch parameters
    chunklength = signallengthpergpu / nchunkspergpu
    if chunklength < devices[gpuid].max_work_group_size:
        workitems_x = 8
    elif devices[gpuid].max_work_group_size < 256:
        workitems_x = devices[
                            gpuid].max_work_group_size
    else:
        workitems_x = 256
    NDRange_x = (workitems_x *
                 (int((signallengthpergpu - 1)/workitems_x) + 1))
    sizeof_float = int(np.dtype(np.float32).itemsize)
    # Allocate and copy memory to device
    d_pointset = cl.Buffer(
                    context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=pointset)
    d_distances = cl.Buffer(
                    context, cl.mem_flags.READ_WRITE,
                    sizeof_float * kth * signallengthpergpu)
    # Neighbour search
    localmem = cl.LocalMemory(sizeof_float * kth * workitems_x)
    print ("workitems_x: {}".format(workitems_x))
    kNN_kernel(queue, (NDRange_x,), (workitems_x,), d_pointset,
               d_pointset, d_distances, np.int32(pointsdim),
               np.int32(chunklength), np.int32(signallengthpergpu),
               np.int32(kth), theiler, localmem)
    distances = np.zeros(signallengthpergpu * kth, dtype=np.float32)
    cl.enqueue_copy(queue, distances, d_distances)
    queue.finish()
    return distances.reshape(signallengthpergpu, kth)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Debug IDTxl GPU searches')
    parser.add_argument('--npoints', '-p', default=1000, type=int)
    parser.add_argument('--ndims', '-d', default=1, type=int)
    parser.add_argument('--nchunks', '-c', default=1, type=int)
    parser.add_argument('--gpuid', '-g', default=0, type=int)

    args = parser.parse_args()

    #DATA INITIALIZATION
    gpuid = int(args.gpuid)
    kth = int(5)
    theiler = int(0)
    print("gpuid: {}".format(gpuid))

    # # TEST CASE 1 - various combinations of no. points and dims
    chunksize = int(args.npoints)  # works for 4 Mio, chrashes for 9 Mio
    pointsdim = int(args.ndims)
    # Always assume a single chunk
    nchunkspergpu = int(args.nchunks)
    signallengthpergpu = nchunkspergpu * chunksize

    # Pad time series to make GPU memory regions a multiple of 1024
    # pad_target = 1024
    pad_target = 256
    pad_size = (int(np.ceil(signallengthpergpu/pad_target)) * pad_target -
                signallengthpergpu)
    signallengthpergpu += pad_size

    #Create an array of zeros for indexes and distances, and random array for pointset
    pointset = np.random.random((signallengthpergpu, pointsdim)).astype('float32')
    queryset = pointset

    # Print memory requirement in MB
    c = 1024**2
    print('pointset: {0:.2f}, queryset: {1:.2f}, TOTAL: {2:.2f} MB'.format(
      getsizeof(pointset) / c, getsizeof(queryset) / c, 
      (getsizeof(pointset) + getsizeof(queryset)) / c))
    print('pointset shape: {}'.format(pointset.shape))


    start = time.time()
    distances = clFindKnn(pointset, queryset, kth, theiler, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    end = time.time()

    if bool == 0:
        print("GPU execution failed")
    else:
        print("Execution time: {0:.2f} min" .format((end - start) / 60))
        print("Array of distances")
        print(distances)
        print("Array of distances shape:")
        print(distances.shape)

