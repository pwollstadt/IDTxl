import logging
from pkg_resources import resource_filename
import ctypes
import numpy as np
from idtxl.estimators_gpu import GPUKraskov
from idtxl.idtxl_utils import DotDict, get_cuda_lib

logger = logging.getLogger(__name__)


class CudaKraskov(GPUKraskov):
    """Abstract class for implementation of CUDA estimators.

    Abstract class for implementation of CUDA estimators, child classes
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

    Set common estimation parameters for CUDA estimators. For usage of these
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
        super().__init__(settings)  # set defaults

        self.devices = self._get_device(self.settings['gpuid'])
        self.shared_library_path = resource_filename(
            __name__, 'gpuKnnLibrary.so')
        # create __cudaFindKnnSetGPU function with get_cudaFindKnnSetGPU()
        self.__cudaFindKnnSetGPU = self._get_cudaFindKnnSetGPU()
        # create __cudaFindRSAllSetGPU function with get_cudaFindRSAllSetGPU()
        self.__cudaFindRSAllSetGPU = self._get_cudaFindRSAllSetGPU()

    def _get_device(self, gpuid):
        """Return GPU devices, test requested GPU id."""
        self.cuda = get_cuda_lib()  # load CUDA library

        success = self.cuda.cuInit(0)
        if success != 0:
            raise RuntimeError('cuInit failed with error code {0}: {1}'.format(
                success, self._get_error_str(success)))

        # Test if requested GPU ID is available
        n_devices = ctypes.c_int()
        success = self.cuda.cuDeviceGetCount(ctypes.byref(n_devices))
        if gpuid > n_devices.value:
            raise RuntimeError(
                'No device with gpuid {0} (available device IDs: {1}).'.format(
                    gpuid, np.arange(n_devices.value)))

        # Get global memory for available devices
        my_gpu_devices = {}
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        name = b' ' * 100
        free_mem = ctypes.c_size_t()
        total_mem = ctypes.c_size_t()
        for i in range(n_devices.value):
            self.cuda.cuDeviceGet(ctypes.byref(device), i)
            self.cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
            device_name = name.split(b'\0', 1)[0].decode()
            success = self.cuda.cuCtxCreate(ctypes.byref(context), 0, device)
            if success != 0:
                raise RuntimeError(
                    'Couldn''t create context for device: {} - {}, failed with error code {}: {}'.format(
                        i, device_name, success, self._get_error_str(success)))
            self.cuda.cuMemGetInfo(
                ctypes.byref(free_mem), ctypes.byref(total_mem))
            self.cuda.cuCtxDetach(context)

            my_gpu_devices[i] = DotDict()
            my_gpu_devices[i].name = device_name
            my_gpu_devices[i].global_mem_size = total_mem.value
            my_gpu_devices[i].free_mem_size = free_mem.value

        logger.debug("Selected Device: {}".format(my_gpu_devices[gpuid].name))

        return my_gpu_devices

    def _get_error_str(self, error_code):
        error_str = ctypes.c_char_p()
        self.cuda.cuGetErrorString(error_code, ctypes.byref(error_str))
        return error_str.value.decode()

    def _get_cudaFindKnnSetGPU(self):
        """Get knn search function pointer from shared library."""
        dll = ctypes.CDLL(self.shared_library_path, mode=ctypes.RTLD_GLOBAL)
        func = dll.cudaFindKnnSetGPU
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int32),  # indexes
            ctypes.POINTER(ctypes.c_float),  # distances
            ctypes.POINTER(ctypes.c_float),  # pointset
            ctypes.POINTER(ctypes.c_float),  # queryset
            ctypes.c_int,  # kraskov_k
            ctypes.c_int,  # theiler
            ctypes.c_int,  # n_chunks
            ctypes.c_int,  # pointdim
            ctypes.c_int,  # signallength
            ctypes.c_int]  # gpuid
        return func

    def cudaFindKnnSetGPU(self, indexes, distances, pointset, queryset, kth,
                          theiler_t, nchunks, pointsdim, signallengthpergpu,
                          gpuid):
        """Wrapper for CUDA knn search to handle type conversions."""
        indexes_p = indexes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        distances_p = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pointset_p = pointset.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        queryset_p = queryset.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        bool = self.__cudaFindKnnSetGPU(
            indexes_p, distances_p, pointset_p, queryset_p, kth, theiler_t,
            nchunks, pointsdim, signallengthpergpu, gpuid)
        return bool

    def knn_search(self, pointset, queryset, knn_k, theiler_t, n_chunks=1,
                   gpuid=0):
        """Interface with CUDA knn search from Python/IDTxl.

        Note: pointset and queryset are expected to have orientation
        [variable dim x points] and to be of type float32.
        """
        n_points = pointset.shape[1]
        n_dims = pointset.shape[0]
        logger.debug('pointset shape (dim x n_points): {}, type: {}'.format(
            pointset.shape, pointset.dtype))
        assert pointset.dtype == np.float32, (
            'GPU input data is not of type float32.')

        indexes = np.zeros((knn_k, n_points), dtype=np.int32)
        distances = np.zeros((knn_k, n_points), dtype=np.float32)

        success = self.cudaFindKnnSetGPU(
            indexes, distances, pointset, queryset, knn_k, theiler_t, n_chunks,
            n_dims, n_points, gpuid)

        distances = distances[knn_k-1, :]
        assert distances.shape[0] == n_points
        logger.debug('distances shape: {}, {}'.format(
            distances.shape, distances[:4]))

        if success:
            return (indexes, distances)
        else:
            raise RuntimeError('Error in CUDA kNN-search.')

    def _get_cudaFindRSAllSetGPU(self):
        """Get range search function pointer from shared library."""
        dll = ctypes.CDLL(self.shared_library_path, mode=ctypes.RTLD_GLOBAL)
        func = dll.cudaFindRSAllSetGPU
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int32),  # n_neighbors
            ctypes.POINTER(ctypes.c_float),  # pointset
            ctypes.POINTER(ctypes.c_float),  # queryset
            ctypes.POINTER(ctypes.c_float),  # distances
            ctypes.c_int,  # theiler
            ctypes.c_int,  # n_chunks
            ctypes.c_int,  # pointdim
            ctypes.c_int,  # signallength
            ctypes.c_int]  # gpuid
        return func

    def cudaFindRSAllSetGPU(self, npointsrange, pointset, queryset, vecradius,
                            theiler_t, nchunkspergpu, pointsdim,
                            datalengthpergpu, gpuid):
        """Wrapper for CUDA range search to handle type conversions."""
        npointsrange_p = npointsrange.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32))
        pointset_p = pointset.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        queryset_p = queryset.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vecradius_p = vecradius.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        bool = self.__cudaFindRSAllSetGPU(
            npointsrange_p, pointset_p, queryset_p, vecradius_p, theiler_t,
            nchunkspergpu, pointsdim, datalengthpergpu, gpuid)
        return bool

    def range_search(self, pointset, queryset, radius, theiler_t, n_chunks=1,
                     gpuid=0):
        """Interface with CUDA range search from Python/IDTxl.

        Note: pointset and queryset are expected to have orientation
        [variable dim x points] and to be of type float32.
        """
        n_points = pointset.shape[1]
        n_dims = pointset.shape[0]
        logger.debug('pointset shape (dim x n_points): {}, type: {}'.format(
            pointset.shape, pointset.dtype))
        assert pointset.dtype == np.float32, (
            'GPU input data is not of type float32.')

        pointcount = np.zeros((1, n_points), dtype=np.int32)

        success = self.cudaFindRSAllSetGPU(
            pointcount, pointset, queryset, radius, theiler_t, n_chunks,
            n_dims, n_points, gpuid)
        pointcount = np.squeeze(pointcount)
        if success:
            return pointcount
        else:
            raise RuntimeError('Error in CUDA kNN-search.')


class CudaKraskovMI(CudaKraskov):
    """Calculate mutual information with CUDA Kraskov implementation.

    Calculate the mutual information (MI) between two variables using CUDA
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
        super().__init__(settings)  # set defaults
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
        data_checked = self._prepare_data(n_chunks, var1=var1, var2=var2)
        var1 = data_checked['var1']
        var2 = data_checked['var2']
        var1, var2 = self._add_mi_lag(var1, var2)

        # Check memory requirements and calculate no. chunks that fit into GPU
        # main memory for a single run.
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        chunks_per_run = self._get_chunks_per_run(
            n_chunks=n_chunks,
            dim_pointset=var1dim + var2dim,
            chunklength=chunklength)

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
                logger.debug(
                    'MI estimation results - MI: {} - Distances: {}'.format(
                        results[0][:4], results[1][:4]))
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
        assert var1.shape[0] == var2.shape[0], 'Unequal no. realisations.'
        assert var1.shape[0] % n_chunks == 0, (
            'No. samples not divisible by no. chunks')

        pointset = np.hstack((var1, var2)).T.copy()
        pointset_var1 = var1.T.copy()
        pointset_var2 = var2.T.copy()
        if self.settings['noise_level'] > 0:
            pointset += np.random.normal(
                scale=self.settings['noise_level'], size=pointset.shape)
            pointset_var1 += np.random.normal(
                scale=self.settings['noise_level'], size=pointset_var1.shape)
            pointset_var2 += np.random.normal(
                scale=self.settings['noise_level'], size=pointset_var2.shape)

        # Perform kNN- and range-search
        indexes, distances = self.knn_search(
            pointset=pointset,
            queryset=pointset,
            knn_k=self.settings['kraskov_k'],
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])
        count_var1 = self.range_search(
            pointset=pointset_var1,
            queryset=pointset_var1,
            radius=distances,
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])
        count_var2 = self.range_search(
            pointset=pointset_var2,
            queryset=pointset_var2,
            radius=distances,
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])

        # Calculate and sum digammas
        mi_array = self._calculate_mi(
            n_chunks=n_chunks,
            chunklength=var1.shape[0] // n_chunks,
            count_var1=count_var1,
            count_var2=count_var2)

        if self.settings['debug']:
            return mi_array, distances, count_var1, count_var2
        else:
            return mi_array


class CudaKraskovCMI(CudaKraskov):
    """Calculate conditional mutual inform with CUDA Kraskov implementation.

    Calculate the conditional mutual information (CMI) between three variables
    using CUDA GPU-code. If no conditional is given (is None), the function
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
        super().__init__(settings)  # set defaults

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
        if conditional is None:
            est_mi = CudaKraskovMI(self.settings)
            return est_mi.estimate(var1, var2, n_chunks)

        # Prepare data: check if variable realisations are passed as 1D or 2D
        # arrays and have equal no. observations.
        data_checked = self._prepare_data(
            n_chunks, var1=var1, var2=var2, conditional=conditional)
        var1 = data_checked['var1']
        var2 = data_checked['var2']
        conditional = data_checked['conditional']

        # Check memory requirements and calculate no. chunks that fit into GPU
        # main memory for a single run.
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        conddim = conditional.shape[1]
        chunks_per_run = self._get_chunks_per_run(
            n_chunks=n_chunks,
            dim_pointset=var1dim + var2dim + conddim,
            chunklength=chunklength)

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
                logger.debug(
                    'CMI estimation results - CMI: {} - Distances: {}'.format(
                        results[0][:4], results[1][:4]))
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
            return self.mi_est._estimate_mi(var1, var2, n_chunks)

        assert var1.shape[0] == var2.shape[0], 'Unequal no. realisations.'
        assert var1.shape[0] == conditional.shape[0], (
            'Unequal no. realisations.')
        assert var1.shape[0] % n_chunks == 0, (
            'No. samples not divisible by no. chunks')

        pointset = np.hstack((var1, conditional, var2)).T.copy()
        pointset_var1cond = np.hstack((var1, conditional)).T.copy()
        pointset_condvar2 = np.hstack((conditional, var2)).T.copy()
        pointset_cond = conditional.T.copy()
        logger.debug('shape pointset: {}'.format(pointset.shape))
        if self.settings['noise_level'] > 0:
            pointset += np.random.normal(
                scale=self.settings['noise_level'],
                size=pointset.shape)
            pointset_var1cond += np.random.normal(
                scale=self.settings['noise_level'],
                size=pointset_var1cond.shape)
            pointset_condvar2 += np.random.normal(
                scale=self.settings['noise_level'],
                size=pointset_condvar2.shape)

        # Perform kNN- and range-search
        indexes, distances = self.knn_search(
            pointset=pointset,
            queryset=pointset,
            knn_k=self.settings['kraskov_k'],
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])
        count_var1cond = self.range_search(
            pointset=pointset_var1cond,
            queryset=pointset_var1cond,
            radius=distances,
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])
        count_condvar2 = self.range_search(
            pointset=pointset_condvar2,
            queryset=pointset_condvar2,
            radius=distances,
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])
        count_cond = self.range_search(
            pointset=pointset_cond,
            queryset=pointset_cond,
            radius=distances,
            theiler_t=self.settings['theiler_t'],
            n_chunks=n_chunks,
            gpuid=self.settings['gpuid'])

        # Calculate and sum digammas
        cmi_array = self._calculate_cmi(
            n_chunks, var1.shape[0] // n_chunks, count_cond, count_var1cond, count_condvar2)

        if self.settings['debug']:
            return cmi_array, distances, count_cond, count_var1cond, count_condvar2
        else:
            return cmi_array
