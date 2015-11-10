import numpy as np
import pyopencl as cl

def knn_search(pointset, queryset, knn_k, theiler_t, n_chunks=1, gpuid=0):
    """Interface with OpenCL knn search from Python/IDTxl."""
    
    n_points = pointset.shape[0]
    pointdim = pointset.shape[1]
    
    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)
    
    success = clFindKnn(indexes, distances, pointset.astype('float32'), 
                        queryset.astype('float32'), int(knn_k), int(theiler_t), 
                        int(n_chunks), int(pointdim), int(n_points), int(gpuid))
    if success:
        return (indexes, distances)
    else:
        print("Error in OpenCL knn search!")
        return 1

def range_search(pointset, queryset, radius, theiler_t, n_chunks=1, gpuid=0):
    """Interface with OpenCL range search from Python/IDTxl."""
    
    n_points = pointset.shape[0]
    pointdim = pointset.shape[1]
    
    pointcount = np.zeros((1, n_points), dtype=np.float32)
    
    success = cclFindRSAll(pointcount, pointset, queryset, radius, theiler_t, 
                           n_chunks, pointdim, n_points, gpuid)
    if success:
        return pointcount
    else:
        print("Error in OpenCL range search!")
        return 1


def clFindKnn(h_bf_indexes, h_bf_distances, h_pointset, h_query, kth, theiler_t, 
              n_chunks, pointdim, signallength, gpuid):
    """Wrap knn OpenCL function for use in Python.
    
    Do type conversions necessary for calling OpenCL code from Python.
    """
    
    triallength = signallength // n_chunks
    
    # Set up OpenCL
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
    print("Selected Device: " + my_gpu_devices[gpuid].name)

    #Check memory resources.
    usedmem = (h_query.nbytes + h_pointset.nbytes + h_bf_distances.nbytes + h_bf_indexes.nbytes)//1024//1024
    totalmem = my_gpu_devices[gpuid].global_mem_size//1024//1024

    if (totalmem*0.90) < usedmem:
        print("WARNING:", usedmem, "Mb used out of", totalmem, "Mb. The GPU could run out of memory.")
    
    # Create OpenCL buffers
    d_bf_query = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_query)
    d_bf_pointset = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_pointset)
    d_bf_distances = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_bf_distances.nbytes)
    d_bf_indexes = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_bf_indexes.nbytes)
    
    # Kernel Launch
    kernelsource = open("gpuKnnBF_kernel.cl").read()
    program = cl.Program(context, kernelsource).build()
    kernelKNNshared = program.kernelKNNshared
    kernelKNNshared.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32, np.int32, None, None])
    
    #Size of workitems and NDRange
    if signallength/n_chunks < my_gpu_devices[gpuid].max_work_group_size:    
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
        print("Localmem alocation will fail.", my_gpu_devices[gpuid].local_mem_size/1024, "kb available, and it needs", localmem, "kb.")
    localmem1 = cl.LocalMemory(np.dtype(np.float32).itemsize*kth*workitems_x)
    localmem2 = cl.LocalMemory(np.dtype(np.int32).itemsize*kth*workitems_x)

    kernelKNNshared(queue, (NDRange_x,), (workitems_x,), d_bf_query, d_bf_pointset, d_bf_indexes, d_bf_distances, pointdim, triallength, signallength,kth,theiler_t, localmem1, localmem2)

    queue.finish()

    # Download results
    cl.enqueue_copy(queue, h_bf_distances, d_bf_distances)
    cl.enqueue_copy(queue, h_bf_indexes, d_bf_indexes)

    ## Increase indexes so it starts at index 1 instead of index 0
    #for n in range(kth):
    #    h_bf_indexes[n] += 1
    
    # Free buffers
    d_bf_distances.release()
    d_bf_indexes.release()
    d_bf_query.release()
    d_bf_pointset.release()
    
    return 1


def clFindRSAll(h_bf_npointsrange, h_pointset, h_query, h_vecradius, theiler_t, 
                n_chunks, pointdim, signallength, gpuid):
    """Wrap range search OpenCL function for use in Python.
    
    Do type conversions necessary for calling OpenCL code from Python.
    """
    
    triallength = signallength // n_chunks
    
    # Set up OpenCL
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
    print("Selected Device: ", my_gpu_devices[gpuid].name)

    #Check memory resources.
    usedmem = (h_query.nbytes + h_pointset.nbytes + h_vecradius.nbytes + h_bf_npointsrange.nbytes)//1024//1024
    totalmem = my_gpu_devices[gpuid].global_mem_size//1024//1024

    if (totalmem*0.90) < usedmem:
        print("WARNING:", usedmem, "Mb used from a total of", totalmem, "Mb. GPU could get without memory.")

    # Create OpenCL buffers
    d_bf_query = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_query)
    d_bf_pointset = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_pointset)
    d_bf_vecradius = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_vecradius)
    d_bf_npointsrange = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_bf_npointsrange.nbytes)

    # Kernel Launch
    kernelsource = open("gpuKnnBF_kernel.cl").read()
    program = cl.Program(context, kernelsource).build()
    kernelBFRSAllshared = program.kernelBFRSAllshared
    kernelBFRSAllshared.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32, None])

    #Size of workitems and NDRange
    if signallength/n_chunks < my_gpu_devices[gpuid].max_work_group_size:    
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

    #Local memory for rangesearch. Actually not used, better results with private memory
    localmem = cl.LocalMemory(np.dtype(np.int32).itemsize*workitems_x)

    kernelBFRSAllshared(queue, (NDRange_x,), (workitems_x,), d_bf_query, d_bf_pointset, d_bf_vecradius, d_bf_npointsrange, pointdim, triallength, signallength, theiler_t, localmem)

    queue.finish()

    # Download results
    cl.enqueue_copy(queue, h_bf_npointsrange, d_bf_npointsrange)

    # Free buffers
    d_bf_npointsrange.release()
    d_bf_vecradius.release()
    d_bf_query.release()
    d_bf_pointset.release()

    return 1
