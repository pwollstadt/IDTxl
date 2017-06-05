from pkg_resources import resource_filename
from scipy.special import digamma
import numpy as np
from idtxl.estimator import Estimator
from . import idtxl_exceptions as ex
try:
    import pyopencl as cl
except ImportError as err:
    ex.package_missing(err, 'PyOpenCl is not available on this system. Install'
                            ' it using pip or the package manager to use '
                            'OpenCL-powered CMI estimation.')

VERBOSE = True


class OpenCLKraskovMI(Estimator):

    def __init__(self, opts=None):

        if opts is None:
            opts = {}
        elif type(opts) is not dict:
            raise TypeError('Opts should be a dictionary.')

        # Set default estimator options.
        opts.setdefault('gpuid', int(0))
        opts.setdefault('kraskov_k', int(4))
        opts.setdefault('theiler_t', int(0))
        opts.setdefault('noise_level', np.float32(1e-8))
        opts.setdefault('debug', False)
        opts.setdefault('lag', 0)
        self.opts = opts

        self.devices, self.context, self.queue = self._get_device(
                                                            self.opts['gpuid'])
        self.kernel_location = resource_filename(__name__,
                                                 'gpuKnnKernelNoIdx.cl')
        self.kNN_kernel, self.RS_kernel = self._get_kernels()

    def estimate(self, var1, var2, n_chunks=1):
        """
        var1 and var2 are expected to be 2D np.arrays with chunklength*n_chunks
        rows and var1dim (or var2dim) columns.
        """
        sizeof_float = int(np.dtype(np.float32).itemsize)
        sizeof_int = int(np.dtype(np.int32).itemsize)

        # Prepare data and add noise
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] % n_chunks == 0
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim
        pointset = np.hstack((var1, var2)).T.copy()
        pointset += np.random.normal(scale=self.opts['noise_level'],
                                     size=pointset.shape)
        if not pointset.dtype == np.float32:
            pointset = pointset.astype(np.float32)

        # Set OpenCL kernel launch parameters
        if chunklength < self.devices[self.opts['gpuid']].max_work_group_size:
            workitems_x = 8
        elif self.devices[self.opts['gpuid']].max_work_group_size < 256:
            workitems_x = self.devices[self.opts['gpuid']].max_work_group_size
        else:
            workitems_x = 256
        NDRange_x = workitems_x * (int((signallength-1)/workitems_x) + 1)

        # Allocate and copy memory to device
        kraskov_k = self.opts['kraskov_k']
        d_pointset = cl.Buffer(
                        self.context,
                        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=pointset)
        d_var1 = d_pointset.get_sub_region(
                                        0,
                                        sizeof_float * signallength * var1dim,
                                        cl.mem_flags.READ_ONLY)
        d_var2 = d_pointset.get_sub_region(sizeof_float*signallength*var1dim,
                                           sizeof_float*signallength*var2dim,
                                           cl.mem_flags.READ_ONLY)
        d_distances = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                sizeof_float * kraskov_k * signallength)
        d_vecradius = d_distances.get_sub_region(
                                signallength * (kraskov_k - 1) * sizeof_float,
                                signallength * sizeof_float)
        d_npointsrange_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     sizeof_int*signallength)
        d_npointsrange_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     sizeof_int*signallength)

        # Neighbour search
        localmem = cl.LocalMemory(sizeof_float*kraskov_k*workitems_x)
        self.kNN_kernel(self.queue, (NDRange_x,), (workitems_x,), d_pointset,
                        d_pointset, d_distances, np.int32(pointdim),
                        np.int32(chunklength), np.int32(signallength),
                        np.int32(kraskov_k), np.int32(self.opts['theiler_t']),
                        localmem)
        self.queue.finish()

        # Range search in var1
        localmem = cl.LocalMemory(sizeof_int*workitems_x)
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_var1,
                       d_var1, d_vecradius, d_npointsrange_x,
                       var1dim, chunklength, signallength,
                       np.int32(self.opts['theiler_t']), localmem)
        count_var1 = np.zeros(signallength, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_var1, d_npointsrange_x)

        # Range search in var2
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_var2,
                       d_var2, d_vecradius, d_npointsrange_y,
                       var2dim, chunklength, signallength,
                       np.int32(self.opts['theiler_t']), localmem)
        count_var2 = np.zeros(signallength, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_var2, d_npointsrange_y)

        d_pointset.release()
        d_distances.release()
        d_npointsrange_x.release()
        d_npointsrange_y.release()

        # Calculate and sum digammas
        mi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
        for c in range(n_chunks):
            mi = (digamma(kraskov_k) + digamma(chunklength) -
                  np.mean(
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) +
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1)))
            mi_array[c] = mi

        if self.opts['debug']:
            return mi_array, d_distances, d_npointsrange_x, d_npointsrange_y
        else:
            return mi_array

    def is_parallel(self):
        return True

    def _get_kernels(self):
        kernel_source = open(self.kernel_location).read()
        program = cl.Program(self.context, kernel_source).build()
        kNN_kernel = program.kernelKNNshared
        kNN_kernel.set_scalar_arg_dtypes([None, None, None, np.int32,
                                          np.int32, np.int32, np.int32,
                                          np.int32, None])

        RS_kernel = program.kernelBFRSAllshared
        RS_kernel.set_scalar_arg_dtypes([None, None, None, None,
                                         np.int32, np.int32, np.int32,
                                         np.int32, None])
        return (kNN_kernel, RS_kernel)

    def _get_device(self, gpuid):
        """Return GPU devices, context, and queue."""
        all_platforms = cl.get_platforms()
        platform = next(p for p in all_platforms
                        if p.get_devices(device_type=cl.device_type.GPU) != [])
        my_gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=my_gpu_devices)
        queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
        if VERBOSE:
            print(("Selected Device: ", my_gpu_devices[gpuid].name))
        return my_gpu_devices, context, queue
