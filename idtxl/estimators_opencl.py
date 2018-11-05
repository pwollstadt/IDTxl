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


class OpenCLKraskov(Estimator):
    """Abstract class for implementation of OpenCL estimators.

    Abstract class for implementation of OpenCL estimators, child classes
    implement estimators for mutual information (MI) and conditional mutual
    information (CMI) using the Kraskov-Grassberger-Stoegbauer estimator for
    continuous data.

    References:

    - Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
      information. Phys Rev E, 69(6), 066138.
    - Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012).
      Local measures of information storage in complex distributed computation.
      Inform Sci, 208, 39-54.
    - Schreiber, T. (2000). Measuring information transfer. Phys Rev Lett,
      85(2), 461.

    Estimators can be used to perform multiple, independent searches in
    parallel. Each of these parallel searches is called a 'chunk'. To search
    multiple chunks, provide point sets as 2D arrays, where the first
    dimension represents samples or points, and the second dimension
    represents the points' dimensions. Concatenate chunk data in the first
    dimension and pass the number of chunks to the estimators. Chunks must be
    of equal size.

    Set common estimation parameters for OpenCL estimators. For usage of these
    estimators see documentation for the child classes.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - debug : bool [optional] - calculate intermediate results, i.e.
              neighbour counts from range searches and KNN distances, print
              debug output to console (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
    """

    def __init__(self, settings=None):
        # Get defaults for estimator settings
        settings = self._check_settings(settings)
        self.settings = settings.copy()
        self.settings.setdefault('gpuid', int(0))
        self.settings.setdefault('kraskov_k', int(4))
        self.settings.setdefault('theiler_t', int(0))
        self.settings.setdefault('noise_level', np.float32(1e-8))
        self.settings.setdefault('local_values', False)
        self.settings.setdefault('debug', False)
        self.settings.setdefault('return_counts', False)
        self.settings.setdefault('verbose', True)
        self.sizeof_float = int(np.dtype(np.float32).itemsize)
        self.sizeof_int = int(np.dtype(np.int32).itemsize)

        if self.settings['return_counts'] and not self.settings['debug']:
            raise RuntimeError(
                'Set debug option to True to return neighbor counts.')

        # Get kernel and devices.
        self.devices, self.context, self.queue = self._get_device(
                                                        self.settings['gpuid'])
        self.kernel_location = resource_filename(__name__,
                                                 'gpuKnnKernelNoIdx.cl')
        self.kNN_kernel, self.RS_kernel = self._get_kernels()

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        return False

    def _get_device(self, gpuid):
        """Return GPU devices, context, and queue."""
        all_platforms = cl.get_platforms()
        platform = next((p for p in all_platforms if
                         p.get_devices(device_type=cl.device_type.GPU) != []),
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
        if self.settings['debug']:
            print("Selected Device: ", my_gpu_devices[gpuid].name)
        return my_gpu_devices, context, queue

    def _get_kernels(self):
        """Return KNN and range search OpenCL kernels."""
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

    def _get_max_mem(self):
        """Return max. GPU main memory available for computation."""
        if 'max_mem' in self.settings:
            return self.settings['max_mem']
        elif 'max_mem_frac' in self.settings:
            return self.settings['max_mem_frac'] * self.devices[
                                    self.settings['gpuid']].global_mem_size
        else:
            return 0.9 * self.devices[self.settings['gpuid']].global_mem_size


class OpenCLKraskovMI(OpenCLKraskov):
    """Calculate mutual information with OpenCL Kraskov implementation.

    Calculate the mutual information (MI) between two variables using OpenCL
    GPU-code. See parent class for references.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - debug : bool [optional] - return intermediate results, i.e.
              neighbour counts from range searches and KNN distances
              (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', 0)

    def estimate(self, var1, var2, n_chunks=1):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
            numpy arrays
                distances and neighborhood counts for var1 and var2 if
                debug=True and return_counts=True
        """
        # Prepare data: check if variable realisations are passed as 1D or 2D
        # arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] % n_chunks == 0
        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim
        kraskov_k = self.settings['kraskov_k']

        mem_data = self.sizeof_float * chunklength * pointdim
        mem_dist = self.sizeof_float * chunklength * kraskov_k
        mem_ncnt = 2 * self.sizeof_int * chunklength
        mem_chunk = mem_data + mem_dist + mem_ncnt
        max_mem = self._get_max_mem()

        max_chunks_per_run = np.floor(max_mem/mem_chunk).astype(int)
        chunks_per_run = min(max_chunks_per_run, n_chunks)

        if self.settings['debug']:
            print('Memory per chunk: {0:.5f} MB, GPU global memory: {1} MB, '
                  'chunks per run: {2}.'.format(mem_chunk / 1024 / 1024,
                                                max_mem / 1024 / 1024,
                                                chunks_per_run))

        if mem_chunk > max_mem:
            raise RuntimeError('Size of single chunk exceeds GPU global '
                               'memory.')

        mi_array = np.array([])
        if self.settings['debug']:
            distances = np.array([])
            count_var1 = np.array([])
            count_var2 = np.array([])

        for r in range(0, n_chunks, chunks_per_run):
            startidx = r*chunklength
            stopidx = min(r+chunks_per_run, n_chunks)*chunklength
            subset1 = var1[startidx:stopidx, :]
            subset2 = var2[startidx:stopidx, :]
            n_chunks_current_run = subset1.shape[0] // chunklength
            results = self._estimate_single_run(subset1, subset2,
                                                n_chunks_current_run)
            if self.settings['debug']:
                mi_array = np.concatenate((mi_array,   results[0]))
                distances = np.concatenate((distances,  results[1]))
                count_var1 = np.concatenate((count_var1, results[2]))
                count_var2 = np.concatenate((count_var2, results[3]))
            else:
                mi_array = np.concatenate((mi_array, results))

        if self.settings['return_counts']:
            return mi_array, distances, count_var1, count_var2
        else:
            return mi_array

    def _estimate_single_run(self, var1, var2, n_chunks=1):
        """Estimate mutual information in a single GPU run.

        This method should not be called directly, only inside estimate()
        after memory bounds have been checked.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        # Prepare data and add noise: check if variable realisations are passed
        # as 1D or 2D arrays and have equal no. observations.
        if self.settings['debug']:
            print('var1 shape: {0}, {1}, n_chunks: {2}'.format(
                             var1.shape[0], var1.shape[1], n_chunks))
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] % n_chunks == 0
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim

        # Pad time series to make GPU memory regions a multiple of 1024
        pad_target = 1024
        pad_size = (int(np.ceil(signallength/pad_target)) * pad_target -
                    signallength)
        pad_var1 = np.vstack(
            [var1, 999999 + 0.1 * np.random.rand(pad_size, var1dim)])
        pad_var2 = np.vstack(
            [var2, 999999 + 0.1 * np.random.rand(pad_size, var2dim)])
        pointset = np.hstack((pad_var1, pad_var2)).T.copy()
        signallength_padded = signallength + pad_size
        if self.settings['noise_level'] > 0:
            pointset += np.random.normal(scale=self.settings['noise_level'],
                                         size=pointset.shape)
        if not pointset.dtype == np.float32:
            pointset = pointset.astype(np.float32)

        if self.settings['debug']:
            # Print memory requirements after padding
            mem_data_pad = (self.sizeof_float *
                            pointset.shape[0] * pointset.shape[1])
            mem_dist = (self.sizeof_float * signallength_padded *
                        self.settings['kraskov_k'])
            mem_ncnt = 2 * self.sizeof_int * signallength_padded
            mem_total = mem_data_pad + mem_dist + mem_ncnt
            print('Memory req. after padding: {0:.5f} MB.'.format(
                      mem_total / 1024 / 1024))

        # Set OpenCL kernel launch parameters
        if chunklength < self.devices[
                                self.settings['gpuid']].max_work_group_size:
            workitems_x = 8
        elif self.devices[self.settings['gpuid']].max_work_group_size < 256:
            workitems_x = self.devices[
                                self.settings['gpuid']].max_work_group_size
        else:
            workitems_x = 256
        NDRange_x = (workitems_x *
                     (int((signallength_padded - 1)/workitems_x) + 1))

        # Allocate and copy memory to device
        kraskov_k = self.settings['kraskov_k']
        d_pointset = cl.Buffer(
                        self.context,
                        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=pointset)
        d_var1 = d_pointset.get_sub_region(
                        0,
                        self.sizeof_float * signallength_padded * var1dim,
                        cl.mem_flags.READ_ONLY)
        d_var2 = d_pointset.get_sub_region(
                        self.sizeof_float * signallength_padded * var1dim,
                        self.sizeof_float * signallength_padded * var2dim,
                        cl.mem_flags.READ_ONLY)
        d_distances = cl.Buffer(
                        self.context, cl.mem_flags.READ_WRITE,
                        self.sizeof_float * kraskov_k * signallength_padded)
        d_vecradius = d_distances.get_sub_region(
                    signallength_padded * (kraskov_k - 1) * self.sizeof_float,
                    signallength_padded * self.sizeof_float)
        d_npointsrange_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)
        d_npointsrange_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)

        # Neighbour search
        theiler_t = np.int32(self.settings['theiler_t'])
        localmem = cl.LocalMemory(self.sizeof_float * kraskov_k * workitems_x)
        self.kNN_kernel(self.queue, (NDRange_x,), (workitems_x,), d_pointset,
                        d_pointset, d_distances, np.int32(pointdim),
                        np.int32(chunklength), np.int32(signallength_padded),
                        np.int32(kraskov_k), theiler_t, localmem)
        distances = np.zeros(signallength_padded * kraskov_k, dtype=np.float32)
        cl.enqueue_copy(self.queue, distances, d_distances)
        self.queue.finish()

        # Range search in var1
        localmem = cl.LocalMemory(self.sizeof_int * workitems_x)
        self.RS_kernel(
            self.queue, (NDRange_x,), (workitems_x,), d_var1,
            d_var1, d_vecradius, d_npointsrange_x,
            var1dim, chunklength, signallength_padded, theiler_t, localmem)
        count_var1 = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_var1, d_npointsrange_x)

        # Range search in var2
        self.RS_kernel(
            self.queue, (NDRange_x,), (workitems_x,), d_var2,
            d_var2, d_vecradius, d_npointsrange_y,
            var2dim, chunklength, signallength_padded, theiler_t, localmem)
        count_var2 = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_var2, d_npointsrange_y)

        d_pointset.release()
        d_distances.release()
        d_npointsrange_x.release()
        d_npointsrange_y.release()

        # Calculate and sum digammas
        if self.settings['local_values']:
            mi_array = -np.inf * np.ones(chunklength * n_chunks,
                                         dtype=np.float64)
            idx = 0
            for c in range(n_chunks):
                mi = (digamma(kraskov_k) + digamma(chunklength) -
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) -
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1))
                mi_array[idx:idx+chunklength] = mi
                idx += chunklength

        else:
            mi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
            for c in range(n_chunks):
                mi = (digamma(kraskov_k) + digamma(chunklength) - np.mean(
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) +
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1)))
                mi_array[c] = mi

        if self.settings['debug']:
            return mi_array, distances, count_var1, count_var2
        else:
            return mi_array


class OpenCLKraskovCMI(OpenCLKraskov):
    """Calculate conditional mutual inform with OpenCL Kraskov implementation.

    Calculate the conditional mutual information (CMI) between three variables
    using OpenCL GPU-code. If no conditional is given (is None), the function
    returns the mutual information between var1 and var2. See parent class for
    references.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - debug : bool [optional] - return intermediate results, i.e.
              neighbour counts from range searches and KNN distances
              (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
    """

    def __init__(self, settings=None):
        super().__init__(settings)

    def estimate(self, var1, var2, conditional=None, n_chunks=1):
        """Estimate conditional mutual information.

        If conditional is None, the mutual information between var1 and var2 is
        calculated.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array
                realisations of conditioning variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
            numpy arrays
                distances and neighborhood counts for var1 and var2 if
                debug=True and return_counts=True
        """
        # Return MI if no conditional is provided
        if conditional is None:
            est_mi = OpenCLKraskovMI(self.settings)
            return est_mi.estimate(var1, var2, n_chunks)

        # Prepare data: check if variable realisations are passed as 1D or 2D
        # arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] == conditional.shape[0]
        assert var1.shape[0] % n_chunks == 0
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        conddim = conditional.shape[1]
        pointdim = var1dim + var2dim + conddim
        kraskov_k = self.settings['kraskov_k']

        mem_data = self.sizeof_float * chunklength * pointdim
        mem_dist = self.sizeof_float * chunklength * kraskov_k
        mem_ncnt = 2 * self.sizeof_int * chunklength
        mem_chunk = mem_data + mem_dist + mem_ncnt
        max_mem = self._get_max_mem()

        max_chunks_per_run = np.floor(max_mem/mem_chunk).astype(int)
        chunks_per_run = min(max_chunks_per_run, n_chunks)

        if self.settings['debug']:
            print('Memory per chunk: {0:.5f} MB, GPU global memory: {1} MB, '
                  'chunks per run: {2}.'.format(mem_chunk / 1024 / 1024,
                                                max_mem / 1024 / 1024,
                                                chunks_per_run))
        if mem_chunk > max_mem:
            raise RuntimeError('Size of single chunk exceeds GPU global '
                               'memory.')

        cmi_array = np.array([])
        if self.settings['debug']:
            distances = np.array([])
            count_var1 = np.array([])
            count_var2 = np.array([])
            count_cond = np.array([])

        for r in range(0, n_chunks, chunks_per_run):
            startidx = r*chunklength
            stopidx = min(r+chunks_per_run, n_chunks)*chunklength
            subset1 = var1[startidx:stopidx, :]
            subset2 = var2[startidx:stopidx, :]
            subset3 = conditional[startidx:stopidx, :]
            n_chunks_current_run = subset1.shape[0] // chunklength
            results = self._estimate_single_run(subset1, subset2, subset3,
                                                n_chunks_current_run)
            if self.settings['debug']:
                cmi_array = np.concatenate((cmi_array,  results[0]))
                distances = np.concatenate((distances,  results[1]))
                count_var1 = np.concatenate((count_var1, results[2]))
                count_var2 = np.concatenate((count_var2, results[3]))
                count_cond = np.concatenate((count_cond, results[4]))
            else:
                cmi_array = np.concatenate((cmi_array, results))

        if self.settings['return_counts']:
            return cmi_array, distances, count_var1, count_var2, count_cond
        else:
            return cmi_array

    def _estimate_single_run(self, var1, var2, conditional=None, n_chunks=1):
        """Estimate conditional mutual information in a single GPU run.

        This method should not be called directly, only inside estimate()
        after memory bounds have been checked.

        If conditional is None, the mutual information between var1 and var2 is
        calculated.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array
                realisations of conditioning variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
        """
        # Return MI if no conditional is provided
        if conditional is None:
            est_mi = OpenCLKraskovMI(self.settings)
            return est_mi.estimate(var1, var2, n_chunks)

        # Prepare data and add noise: check if variable realisations are passed
        # as 1D or 2D arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] == conditional.shape[0]
        assert var1.shape[0] % n_chunks == 0
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        conddim = conditional.shape[1]
        pointdim = var1dim + var2dim + conddim

        # Pad time series to make GPU memory regions a multiple of 1024
        pad_target = 1024
        pad_size = (int(np.ceil(signallength/pad_target)) * pad_target -
                    signallength)
        pad_var1 = np.vstack(
            [var1, 999999 + 0.1 * np.random.rand(pad_size, var1dim)])
        pad_var2 = np.vstack(
            [var2, 999999 + 0.1 * np.random.rand(pad_size, var2dim)])
        pad_conditional = np.vstack(
            [conditional, 999999 + 0.1 * np.random.rand(pad_size, conddim)])
        pointset = np.hstack((pad_var1, pad_conditional, pad_var2)).T.copy()
        signallength_padded = signallength + pad_size
        if self.settings['noise_level'] > 0:
            pointset += np.random.normal(scale=self.settings['noise_level'],
                                         size=pointset.shape)
        if not pointset.dtype == np.float32:
            pointset = pointset.astype(np.float32)

        if self.settings['debug']:
            # Print memory requirements after padding
            mem_data_pad = (self.sizeof_float *
                            pointset.shape[0] * pointset.shape[1])
            mem_dist = (self.sizeof_float * signallength_padded *
                        self.settings['kraskov_k'])
            mem_ncnt = 2 * self.sizeof_int * signallength_padded
            mem_total = mem_data_pad + mem_dist + mem_ncnt
            print('Memory req. after padding: {0:.5f} MB.'.format(
                      mem_total / 1024 / 1024))

        # Set OpenCL kernel launch parameters
        if chunklength < self.devices[
                                self.settings['gpuid']].max_work_group_size:
            workitems_x = 8
        elif self.devices[self.settings['gpuid']].max_work_group_size < 256:
            workitems_x = self.devices[
                                self.settings['gpuid']].max_work_group_size
        else:
            workitems_x = 256
        NDRange_x = (workitems_x *
                     (int((signallength_padded - 1)/workitems_x) + 1))

        # Allocate and copy memory to device
        kraskov_k = self.settings['kraskov_k']
        d_pointset = cl.Buffer(
                    self.context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=pointset)
        d_src = d_pointset.get_sub_region(
                    0,
                    self.sizeof_float * signallength_padded * var1dim,
                    cl.mem_flags.READ_ONLY)
        d_cnd = d_pointset.get_sub_region(
                    self.sizeof_float * signallength_padded * var1dim,
                    self.sizeof_float * signallength_padded * conddim,
                    cl.mem_flags.READ_ONLY)
        d_distances = cl.Buffer(
                    self.context, cl.mem_flags.READ_WRITE,
                    self.sizeof_float * kraskov_k * signallength_padded)
        d_vecradius = d_distances.get_sub_region(
                    signallength_padded * (kraskov_k - 1) * self.sizeof_float,
                    signallength_padded * self.sizeof_float)
        d_npointsrange_x = cl.Buffer(self.context,
                                     cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)
        d_npointsrange_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)
        d_npointsrange_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)

        # Neighbour search in full space
        theiler_t = np.int32(self.settings['theiler_t'])
        localmem = cl.LocalMemory(self.sizeof_float * kraskov_k * workitems_x)
        self.kNN_kernel(self.queue, (NDRange_x,), (workitems_x,), d_pointset,
                        d_pointset, d_distances, np.int32(pointdim),
                        np.int32(chunklength), np.int32(signallength_padded),
                        np.int32(kraskov_k), theiler_t, localmem)
        distances = np.zeros(signallength_padded * kraskov_k, dtype=np.float32)
        cl.enqueue_copy(self.queue, distances, d_distances)
        self.queue.finish()

        # Range search in source and conditional
        localmem = cl.LocalMemory(self.sizeof_int * workitems_x)
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_src, d_src,
                       d_vecradius, d_npointsrange_x, var1dim + conddim,
                       chunklength, signallength_padded, theiler_t, localmem)
        count_src = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_src, d_npointsrange_x)

        # Range search in target and conditional
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_cnd, d_cnd,
                       d_vecradius, d_npointsrange_y, var2dim + conddim,
                       chunklength, signallength_padded, theiler_t, localmem)
        count_tgt = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_tgt, d_npointsrange_y)

        # Range search in conditional
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_cnd, d_cnd,
                       d_vecradius, d_npointsrange_z, conddim, chunklength,
                       signallength_padded, theiler_t, localmem)
        count_cnd = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_cnd, d_npointsrange_z)

        d_pointset.release()
        d_distances.release()
        d_npointsrange_x.release()
        d_npointsrange_y.release()
        d_npointsrange_z.release()

        # Calculate and sum digammas
        if self.settings['local_values']:
            cmi_array = -np.inf * np.ones(n_chunks * chunklength,
                                          dtype=np.float64)
            idx = 0
            for c in range(n_chunks):
                cmi = (digamma(kraskov_k) +
                       digamma(count_cnd[c*chunklength:(c+1)*chunklength]+1) -
                       digamma(count_src[c*chunklength:(c+1)*chunklength]+1) -
                       digamma(count_tgt[c*chunklength:(c+1)*chunklength]+1))
                cmi_array[idx:idx+chunklength] = cmi
                idx += chunklength

        else:
            cmi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
            for c in range(n_chunks):
                cmi = (digamma(kraskov_k) + np.mean(
                        digamma(count_cnd[c*chunklength:(c+1)*chunklength]+1) -
                        digamma(count_src[c*chunklength:(c+1)*chunklength]+1) -
                        digamma(count_tgt[c*chunklength:(c+1)*chunklength]+1)))
                cmi_array[c] = cmi

        if self.settings['debug']:
            return cmi_array, distances, count_src, count_tgt, count_cnd
        else:
            return cmi_array
