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


def _get_device(gpuid, run_on_gpu=True):
    """Return GPU or CPU devices, context, and queue."""
    all_platforms = cl.get_platforms()
    if run_on_gpu:
        platform = next((p for p in all_platforms if p.get_devices(device_type=cl.device_type.GPU) != []),
                        None)
        if platform is None:
            raise RuntimeError('No OpenCL GPU device found.')
        my_devices = platform.get_devices(device_type=cl.device_type.GPU)
    else:
        print('RUNNING CODE ON CPU!!')
        platform = next((p for p in all_platforms if p.get_devices(device_type=cl.device_type.CPU) != []),
                        None)
        my_devices = platform.get_devices(device_type=cl.device_type.CPU)

    context = cl.Context(devices=my_devices)
    if gpuid > len(my_devices)-1:
        raise RuntimeError(
            'No device with gpuid {0} (available device IDs: {1}).'.format(
                gpuid, np.arange(len(my_devices))))
    queue = cl.CommandQueue(context, my_devices[gpuid])
    print("Selected Device: ", my_devices[gpuid].name)
    return my_devices, context, queue


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


def clFindKnn(pointset, kth, theiler, nchunkspergpu, pointsdim,
              signallengthpergpu, gpuid, chunklength, cpu=False):

    devices, context, queue = _get_device(gpuid, cpu)
    kNN_kernel, RS_kernel = _get_kernels('gpuKnnKernelNoIdx.cl', context)

    print('Pointset: {0} elements, dim {1}x{2}, {3} chunks (chunklength: {4}).'.format(
        pointset.size, pointset.shape[0], pointset.shape[1], nchunkspergpu, chunklength))

    # Set OpenCL kernel launch parameters
    # chunklength = signallengthpergpu / nchunkspergpu
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
    print("workitems_x: {}, NDrange_x: {}".format(workitems_x, NDRange_x))
    # KNN Kernel Args:
    # __global const float* g_uquery,
    # __global const float* g_vpointset,
    # __global float* g_distances,
    # const int pointdim,
    # const int triallength,
    # const int signallength,
    # const int kth,
    # const int exclude,
    # __local float* kdistances)
    kNN_kernel(
        queue,
        (NDRange_x,),
        (workitems_x,),
        d_pointset,
        d_pointset,
        d_distances,
        np.int32(pointsdim),
        np.int32(chunklength),
        np.int32(signallengthpergpu),
        np.int32(kth),
        theiler,
        localmem)
    distances = np.zeros(signallengthpergpu * kth, dtype=np.float32)
    # distances = np.zeros(nchunkspergpu * chunklength * kth, dtype=np.float32)
    print('device distances size: {}'.format(d_distances.size))
    print('host distances size: {}'.format(distances.size))
    cl.enqueue_copy(queue, distances, d_distances)
    queue.finish()
    return distances.reshape(signallengthpergpu, kth)


def clFindKnn_old(h_bf_indexes, h_bf_distances, h_pointset, h_query, kth, thelier, nchunks, pointdim, signallength, gpuid):

    triallength = int(signallength / nchunks)
#    print 'Values:', pointdim, triallength, signallength, kth, thelier

    '''for platform in cl.get_platforms():
        for device in platform.get_devices():
            print("===============================================================")
            print("Platform name:", platform.name)
            print("Platform profile:", platform.profile)
            print("Platform vendor:", platform.vendor)
            print("Platform version:", platform.version)
            print("---------------------------------------------------------------")
            print("Device name:", device.name)
            print("Device type:", cl.device_type.to_string(device.type))
            print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
            print("Device max clock speed:", device.max_clock_frequency, 'MHz')
            print("Device compute units:", device.max_compute_units)
            print("Device max work group size:", device.max_work_group_size)
            print("Device max work item sizes:", device.max_work_item_sizes)'''


    # Set up OpenCL
    #context = cl.create_some_context()
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
    print(("Selected Device: ", my_gpu_devices[gpuid].name))

    #Check memory resources.
    usedmem =int((h_query.nbytes + h_pointset.nbytes + h_bf_distances.nbytes + h_bf_indexes.nbytes)//1024//1024)
    totalmem = int(my_gpu_devices[gpuid].global_mem_size//1024//1024)

    if (totalmem*0.90) < usedmem:
        print(("WARNING:", usedmem, "Mb used out of", totalmem, "Mb. The GPU could run out of memory."))


    # Create OpenCL buffers
    d_bf_query = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_query)
    d_bf_pointset = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_pointset)
    d_bf_distances = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_bf_distances.nbytes)
    d_bf_indexes = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_bf_indexes.nbytes)

    # Kernel Launch
    kernelsource = open("gpuKnnBF_kernel.cl").read()
    # kernelsource = open('gpuKnnKernelNoIdx.cl').read()
    program = cl.Program(context, kernelsource).build()
    kernelKNNshared = program.kernelKNNshared
    kernelKNNshared.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32, np.int32, None, None])

    #Size of workitems and NDRange
    if signallength/nchunks < my_gpu_devices[gpuid].max_work_group_size:
        workitems_x = 8
    elif my_gpu_devices[gpuid].max_work_group_size < 256:
        workitems_x = my_gpu_devices[gpuid].max_work_group_size
    else:
        workitems_x = 256

    if signallength%workitems_x != 0:
        temp = int(round(((signallength)/workitems_x), 0) + 1)
    else:
        temp = int(signallength/workitems_x)

    NDRange_x = workitems_x * temp

    #Local memory for distances and indexes
    localmem = (np.dtype(np.float32).itemsize*kth*workitems_x + np.dtype(np.int32).itemsize*kth*workitems_x)/1024
    if localmem > my_gpu_devices[gpuid].local_mem_size/1024:
        print(( "Localmem alocation will fail.", my_gpu_devices[gpuid].local_mem_size/1024, "kb available, and it needs", localmem, "kb."))
    localmem1 = cl.LocalMemory(np.dtype(np.float32).itemsize*kth*workitems_x)
    localmem2 = cl.LocalMemory(np.dtype(np.int32).itemsize*kth*workitems_x)

    kernelKNNshared(queue, (NDRange_x,), (workitems_x,), d_bf_query, d_bf_pointset, d_bf_indexes, d_bf_distances, pointdim, triallength, signallength,kth,thelier, localmem1, localmem2)


    # Download results
    cl.enqueue_copy(queue, h_bf_distances, d_bf_distances)
    cl.enqueue_copy(queue, h_bf_indexes, d_bf_indexes)

    queue.finish()

    # Free buffers
    d_bf_distances.release()
    d_bf_indexes.release()
    d_bf_query.release()
    d_bf_pointset.release()

    return 1


if __name__ == '__main__':

    # logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(
            format='%(asctime)s - %(levelname)-4s  [%(filename)s:%(funcName)20s():l %(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Debug IDTxl GPU searches')
    parser.add_argument('--npoints', '-p', default=1000, type=int,
                        help='No. points per chunk')
    parser.add_argument('--ndims', '-d', default=1, type=int)
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

    if args.padding:
        # Pad time series to make GPU memory regions a multiple of 1024
        pad_target = 1024  # this is used in the toolbox
        print(f'Applying padding to multiple of {pad_target}')
        # pad_target = 256
        pad_size = (int(np.ceil(signallengthpergpu/pad_target)) * pad_target -
                    signallengthpergpu)
        signallengthpergpu += pad_size
        assert signallengthpergpu % pad_target == 0
    else:
        pad_size = 'n/a'

    chunksize = int(args.npoints)  # old version
    # chunksize = signallengthpergpu / nchunkspergpu

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
    if args.old:  # run Mario's original code
        distances = np.zeros(signallengthpergpu * kth, dtype=np.float32)
        indexes = np.zeros(signallengthpergpu * kth, dtype=np.float32)
        queryset = pointset
        success = clFindKnn_old(
            indexes, distances, pointset, queryset, kth, theiler,
            nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
    else:  # run Pedro's modified code
        try:
            distances = clFindKnn(
                pointset, kth, theiler, nchunkspergpu, pointsdim,
                signallengthpergpu, gpuid, chunksize, args.cpu)
        except cl._cl.RuntimeError as e:
            print(e)
            success = False

    end = time.time()

    print("Execution time: {0:.2f} min" .format((end - start) / 60))
    if not success:
        print("\n!!! GPU execution failed\n")
    else:
        # print("Array of distances")
        # print(distances)
        print("Array of distances shape: {}".format(distances.shape))
        i = 0
        for c in range(1, nchunkspergpu + 1):
            print(distances[int(i):int(chunksize*c), :])
            i += chunksize
