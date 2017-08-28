"""Provide neighbour searches using OpenCl GPU-code."""
from pkg_resources import resource_filename
import numpy as np
from . import idtxl_exceptions as ex
try:
    import pyopencl as cl
except ImportError as err:
    ex.package_missing(err, 'PyOpenCl is not available on this system. Install'
                            ' it using pip or the package manager to use '
                            'OpenCL-powered CMI estimation.')    


def knn_search(pointset, n_dim, knn_k, theiler_t, n_chunks=1, gpuid=0):
    """Interface with OpenCL knn search from Python/IDTxl."""
    # check for a data layout in memory as expected by the low level functions
    # ndim * [n_points * n_chunks]
    if n_dim != pointset.shape[0]:
        assert n_dim == pointset.shape[1], ('Given dimension does not match '
                                            'data.')
        pointset = pointset.transpose().copy()

        print('>>>search GPU: fixed shape of input data')
    if pointset.flags['C_CONTIGUOUS'] is not True:
        pointset = np.ascontiguousarray(pointset)
        print('>>>search GPU: fixed memory layout of input data')

    pointdim = pointset.shape[0]
    n_points = pointset.shape[1]

    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    success = clFindKnn(indexes, distances, pointset.astype('float32'),
                        pointset.astype('float32'), int(knn_k), int(theiler_t),
                        int(n_chunks), int(pointdim), int(n_points),
                        int(gpuid))
    if success:
        return (indexes, distances)
    else:
        print("Error in OpenCL knn search!")
        return 1


def range_search(pointset, n_dim, radius, theiler_t, n_chunks=1, gpuid=0):
    """Interface with OpenCL range search from Python/IDTxl."""
    # check for a data layout in memory as expected by the low level functions
    # ndim * [n_points * n_chunks]
    if n_dim != pointset.shape[0]:
        assert n_dim == pointset.shape[1], ('Given dimension does not match '
                                            'data axis.')
        pointset = pointset.transpose().copy()
        print('>>>search GPU: fixed shape input data')
    if pointset.flags['C_CONTIGUOUS'] is not True:
        pointset = np.ascontiguousarray(pointset)
        print('>>>search GPU: fixed memory layout of input data')

    pointdim = pointset.shape[0]
    n_points = pointset.shape[1]

    pointcount = np.zeros((n_points), dtype=np.int32)

    success = clFindRSAll(pointcount, pointset.astype('float32'),
                          pointset.astype('float32'), radius, theiler_t,
                          n_chunks, pointdim, n_points, gpuid)
    if success:
        return pointcount
    else:
        print("Error in OpenCL range search!")
        return 1


def clFindKnn(h_bf_indexes, h_bf_distances, h_pointset, h_query, kth, thelier,
              nchunks, pointdim, signallength, gpuid):

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
    # context = cl.create_some_context()
    platform = cl.get_platforms()
    platf_idx = find_nonempty(platform)
    print('platform index chosen is: {0}'.format(platf_idx))
    my_gpu_devices = platform[platf_idx].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
    print(("Selected Device: ", my_gpu_devices[gpuid].name))

    # Check memory resources.
    usedmem =int( (h_query.nbytes + h_pointset.nbytes + h_bf_distances.nbytes + h_bf_indexes.nbytes)//1024//1024)
    totalmem = int(my_gpu_devices[gpuid].global_mem_size//1024//1024)

    if (totalmem*0.90) < usedmem:
        print(("WARNING:", usedmem, "Mb used out of", totalmem,
               "Mb. The GPU could run out of memory."))


    # Create OpenCL buffers
    d_bf_query = cl.Buffer(context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=h_query)
    d_bf_pointset = cl.Buffer(context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=h_pointset)
    d_bf_distances = cl.Buffer(context,
                               cl.mem_flags.READ_WRITE,
                               h_bf_distances.nbytes)
    d_bf_indexes = cl.Buffer(context,
                             cl.mem_flags.READ_WRITE,
                             h_bf_indexes.nbytes)

    # Kernel Launch
    kernelLocation = resource_filename(__name__, 'gpuKnnBF_kernel.cl')
    kernelsource = open(kernelLocation).read()
    program = cl.Program(context, kernelsource).build()
    kernelKNNshared = program.kernelKNNshared
    kernelKNNshared.set_scalar_arg_dtypes([None, None, None, None, np.int32,
                                           np.int32, np.int32, np.int32,
                                           np.int32, None, None])

    # Size of workitems and NDRange
    if signallength/nchunks < my_gpu_devices[gpuid].max_work_group_size:
        workitems_x = 8
    elif my_gpu_devices[gpuid].max_work_group_size < 256:
        workitems_x = my_gpu_devices[gpuid].max_work_group_size
    else:
        workitems_x = 256

    if signallength % workitems_x != 0:
        temp = int(round(((signallength)/workitems_x), 0) + 1)
    else:
        temp = int(signallength/workitems_x)

    NDRange_x = workitems_x * temp

    # Local memory for distances and indexes
    localmem = (np.dtype(np.float32).itemsize*kth*workitems_x +
                np.dtype(np.int32).itemsize*kth*workitems_x) / 1024
    if localmem > my_gpu_devices[gpuid].local_mem_size / 1024:
        print('Localmem alocation will fail. {0} kb available, and it needs '
              '{1} kb.'.format(my_gpu_devices[gpuid].local_mem_size / 1024,
                               localmem))
    localmem1 = cl.LocalMemory(np.dtype(np.float32).itemsize*kth*workitems_x)
    localmem2 = cl.LocalMemory(np.dtype(np.int32).itemsize*kth*workitems_x)

    kernelKNNshared(queue, (NDRange_x,), (workitems_x,), d_bf_query,
                    d_bf_pointset, d_bf_indexes, d_bf_distances, pointdim,
                    triallength, signallength, kth, thelier, localmem1,
                    localmem2)

    queue.finish()

    # Download results
    cl.enqueue_copy(queue, h_bf_distances, d_bf_distances)
    cl.enqueue_copy(queue, h_bf_indexes, d_bf_indexes)

    # Free buffers
    d_bf_distances.release()
    d_bf_indexes.release()
    d_bf_query.release()
    d_bf_pointset.release()

    return 1

'''
 * Range search being radius a vector of length number points in queryset/pointset
'''

def clFindRSAll(h_bf_npointsrange, h_pointset, h_query, h_vecradius, thelier,
                nchunks, pointdim, signallength, gpuid):

    triallength = int(signallength / nchunks)
    # print 'Values:', pointdim, triallength, signallength, kth, thelier

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
    # context = cl.create_some_context()
    platform = cl.get_platforms()
    platf_idx = find_nonempty(platform)
    my_gpu_devices = platform[platf_idx].get_devices(
                                                device_type=cl.device_type.GPU)
    context = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
    print(("Selected Device: ", my_gpu_devices[gpuid].name))

    # Check memory resources.
    usedmem = int((h_query.nbytes + h_pointset.nbytes + h_vecradius.nbytes +
                   h_bf_npointsrange.nbytes) // 1024 // 1024)
    totalmem = int(my_gpu_devices[gpuid].global_mem_size // 1024 // 1024)

    if (totalmem * 0.90) < usedmem:
        print('WARNING: {0} Mb used from a total of {1} Mb. GPU could get '
              'without memory'.format(usedmem, totalmem))

    # Create OpenCL buffers
    d_bf_query = cl.Buffer(context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=h_query)
    d_bf_pointset = cl.Buffer(context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=h_pointset)
    d_bf_vecradius = cl.Buffer(context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=h_vecradius)
    d_bf_npointsrange = cl.Buffer(context,
                                  cl.mem_flags.READ_WRITE,
                                  h_bf_npointsrange.nbytes)

    # Kernel Launch
    kernelLocation = resource_filename(__name__, 'gpuKnnBF_kernel.cl')
    kernelsource = open(kernelLocation).read()
    program = cl.Program(context, kernelsource).build()
    kernelBFRSAllshared = program.kernelBFRSAllshared
    kernelBFRSAllshared.set_scalar_arg_dtypes([None, None, None, None,
                                               np.int32, np.int32, np.int32,
                                               np.int32, None])

    # Size of workitems and NDRange
    if signallength/nchunks < my_gpu_devices[gpuid].max_work_group_size:
        workitems_x = 8
    elif my_gpu_devices[gpuid].max_work_group_size < 256:
        workitems_x = my_gpu_devices[gpuid].max_work_group_size
    else:
        workitems_x = 256

    if signallength % workitems_x != 0:
        temp = int(round(((signallength)/workitems_x), 0) + 1)
    else:
        temp = int(signallength/workitems_x)

    NDRange_x = workitems_x * temp

    # Local memory for rangesearch. Actually not used, better results with
    # private memory
    localmem = cl.LocalMemory(np.dtype(np.int32).itemsize * workitems_x)

    kernelBFRSAllshared(queue, (NDRange_x,), (workitems_x,), d_bf_query,
                        d_bf_pointset, d_bf_vecradius, d_bf_npointsrange,
                        pointdim, triallength, signallength, thelier, localmem)

    queue.finish()

    # Download results
    cl.enqueue_copy(queue, h_bf_npointsrange, d_bf_npointsrange)

    # Free buffers
    d_bf_npointsrange.release()
    d_bf_vecradius.release()
    d_bf_query.release()
    d_bf_pointset.release()

    return 1


def find_nonempty(a_list):
    """Find non-empty device in list."""
    for idx in range(0, len(a_list)):
        if a_list[idx].get_devices(device_type=cl.device_type.GPU) != []:
            break
        else:
            print('found empty platform')
    if a_list[idx] == []:
        print('all platforms empty')
    else:
        return idx
