
import pyopencl as cl
import numpy
from time import time

def clFindKnn(h_bf_indexes, h_bf_distances, h_pointset, h_query, kth, thelier, nchunks, pointdim, signallength, gpuid):

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
    usedmem =int( (h_query.nbytes + h_pointset.nbytes + h_bf_distances.nbytes + h_bf_indexes.nbytes)//1024//1024)
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
    program = cl.Program(context, kernelsource).build()
    kernelKNNshared = program.kernelKNNshared
    kernelKNNshared.set_scalar_arg_dtypes([None, None, None, None, numpy.int32, numpy.int32, numpy.int32, numpy.int32, numpy.int32, None, None])

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
    localmem = (numpy.dtype(numpy.float32).itemsize*kth*workitems_x + numpy.dtype(numpy.int32).itemsize*kth*workitems_x)/1024
    if localmem > my_gpu_devices[gpuid].local_mem_size/1024:
        print(( "Localmem alocation will fail.", my_gpu_devices[gpuid].local_mem_size/1024, "kb available, and it needs", localmem, "kb."))
    localmem1 = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize*kth*workitems_x)
    localmem2 = cl.LocalMemory(numpy.dtype(numpy.int32).itemsize*kth*workitems_x)

    kernelKNNshared(queue, (NDRange_x,), (workitems_x,), d_bf_query, d_bf_pointset, d_bf_indexes, d_bf_distances, pointdim, triallength, signallength,kth,thelier, localmem1, localmem2)

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

def clFindRSAll(h_bf_npointsrange, h_pointset, h_query, h_vecradius, thelier, nchunks, pointdim, signallength, gpuid):

    triallength = int(signallength / nchunks)
    #print 'Values:', pointdim, triallength, signallength, kth, thelier

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
    usedmem = int((h_query.nbytes + h_pointset.nbytes + h_vecradius.nbytes + h_bf_npointsrange.nbytes)//1024//1024)
    totalmem = int(my_gpu_devices[gpuid].global_mem_size//1024//1024)

    if (totalmem*0.90) < usedmem:
        print(("WARNING:", usedmem, "Mb used from a total of", totalmem, "Mb. GPU could get without memory"))

    # Create OpenCL buffers
    d_bf_query = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_query)
    d_bf_pointset = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_pointset)
    d_bf_vecradius = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_vecradius)
    d_bf_npointsrange = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_bf_npointsrange.nbytes)

    # Kernel Launch
    kernelsource = open("gpuKnnBF_kernel.cl").read()
    program = cl.Program(context, kernelsource).build()
    kernelBFRSAllshared = program.kernelBFRSAllshared
    kernelBFRSAllshared.set_scalar_arg_dtypes([None, None, None, None, numpy.int32, numpy.int32, numpy.int32, numpy.int32, None])

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

    #Local memory for rangesearch. Actually not used, better results with private memory
    localmem = cl.LocalMemory(numpy.dtype(numpy.int32).itemsize*workitems_x)

    kernelBFRSAllshared(queue, (NDRange_x,), (workitems_x,), d_bf_query, d_bf_pointset, d_bf_vecradius, d_bf_npointsrange, pointdim, triallength, signallength, thelier, localmem)

    queue.finish()

    # Download results
    cl.enqueue_copy(queue, h_bf_npointsrange, d_bf_npointsrange)

    # Free buffers
    d_bf_npointsrange.release()
    d_bf_vecradius.release()
    d_bf_query.release()
    d_bf_pointset.release()

    return 1

