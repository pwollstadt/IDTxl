import logging
from pkg_resources import resource_filename
import numpy as np
from idtxl.estimators_gpu import GPUKraskov
from idtxl.idtxl_utils import DotDict, get_cuda_lib
import ctypes
import math
import idtxl.numba_kernels as nk
from . import idtxl_exceptions as ex
try:
    from numba import float32, float64, int32, int64, cuda
except ImportError as err:
    ex.package_missing(err, 'Numba is not available on this system. Install'
                            ' it using pip or the package manager to use '
                            'the Numba estimators.')

logger = logging.getLogger(__name__)



class NumbaKraskov(GPUKraskov):
    """Abstract class for implementation of Numba estimators.

    Abstract class for implementation of Numba estimators, child classes
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

    Set common estimation parameters for Numba estimators. For usage of these
    estimators see documentation for the child classes.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - padding : bool [optional] - pad data to a length that is a
              multiple of 1024, workaround for a
            - debug : bool [optional] - calculate intermediate results, i.e.
              neighbour counts from range searches and KNN distances, print
              debug output to console (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
    """

    def __init__(self, settings=None):
        super().__init__(settings)  # set defaults
        self.settings.setdefault('lag_mi', 0)

    # def __init__(self, settings=None):
    #     # Get defaults for estimator settings
    #     settings = self._check_settings(settings)
    #     self.settings = settings.copy()
    #     self.settings.setdefault('kraskov_k', int(4))
    #     self.settings.setdefault('theiler_t', int(0))
    #     self.settings.setdefault('floattype', int(64))
    #     if self.settings['floattype'] == 32:
    #         self.settings.setdefault('noise_level', np.float32(1e-8))
    #     else:
    #         self.settings.setdefault('noise_level', np.float64(1e-8))
    #     self.settings.setdefault('local_values', False)
    #     self.settings.setdefault('debug', False)
    #     self.settings.setdefault('return_counts', False)
    #     self.settings.setdefault('verbose', True)
    #
    #     if self.settings['return_counts'] and not self.settings['debug']:
    #         raise RuntimeError(
    #             'Set debug option to True to return neighbor counts.')

    def _get_numba_device(self, gpuid):
        """Tests availablility of CUDA driver and supported hardware, test requested GPU id and returns
         list of supported GPU devices"""
        # check if cuda driver is available
        if not cuda.is_available():
            raise RuntimeError('No cuda driver available!')

        # detect if supported CUDA device are available
        if not cuda.detect():
            raise RuntimeError('No cuda devices available!')

        nr_devices = len(cuda.gpus.lst)
        if gpuid > nr_devices:
            raise RuntimeError(
                'No device with gpuid {0} (available device IDs: {1}).'.format(
                    gpuid, np.arange(nr_devices)))

        # list of cuda devices
        gpus = cuda.list_devices()

        my_gpu_devices = {}
        for i in range(nr_devices):
            my_gpu_devices[i] = DotDict()
            name = gpus[0]._device.name
            my_gpu_devices[i].name = name.decode('utf-8')
            my_gpu_devices[i].global_mem_size = cuda.cudadrv.devices.get_context(gpuid).get_memory_info().total
            my_gpu_devices[i].free_mem_size = cuda.cudadrv.devices.get_context(gpuid).get_memory_info().free

        return my_gpu_devices

    '''
    def _get_device(self, gpuid):
        """Return GPU devices, test requested GPU id."""
        self.cuda = get_cuda_lib()  # load CUDA library

        success = self.cuda.cuInit(0)
        if success != 0:
            #raise RuntimeError('cuInit failed with error code {0}: {1}'.format(
            #    success, self._get_error_str(success)))
            raise RuntimeError('cuInit failed with error code {0}'.format(
                success))

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
                #raise RuntimeError(
                #    'Couldn''t create context for device: {} - {}, failed with error code {}: {}'.format(
                #        i, device_name, success, self._get_error_str(success)))
                raise RuntimeError(
                    'Couldn''t create context for device: {} - {}, failed with error code {}'.format(
                        i, device_name, success))

            self.cuda.cuMemGetInfo(
                ctypes.byref(free_mem), ctypes.byref(total_mem))
            self.cuda.cuCtxDetach(context)

            my_gpu_devices[i] = DotDict()
            my_gpu_devices[i].name = device_name
            my_gpu_devices[i].global_mem_size = total_mem.value
            my_gpu_devices[i].free_mem_size = free_mem.value

        logger.debug("Selected Device: {}".format(my_gpu_devices[gpuid].name))

        return my_gpu_devices

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        return False
    '''

class NumbaCPUKraskovMI(NumbaKraskov):
    """Calculate mutual information with Numba Kraskov implementation.

    Calculate the mutual information (MI) between two variables using Numba
    for CPU. See parent class for references.

    Args:
        settings : dict [optional]
            set estimator parameters:

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
            - floattype : int [optional] - 32 or 64 - type of input data float32
              or float64  (default=32)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', 0)

    def knnNumbaCPU(self, vpointset, distances, pointdim, chunklength, signallength):
        return nk._knnNumbaCPU(vpointset, vpointset, distances, pointdim, chunklength, signallength,
                            np.int32(self.settings['kraskov_k']), np.int32(self.settings['theiler_t']),
                            np.int32(self.settings['floattype']))

    def rsAllNumbaCPU(self, rsvar, vecradius, npoints, pointdim, chunklength, signallength):
        return nk._rsAllNumbaCPU(rsvar, rsvar, vecradius, npoints, pointdim, chunklength, signallength,
                              np.int32(self.settings['theiler_t']), np.int32(self.settings['floattype']))

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

        # Shift variables to calculate a lagged MI.
        var1, var2 = self._add_mi_lag(var1, var2)

        # get values
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim
        kraskov_k = np.int32(self.settings['kraskov_k'])

        # initialize distances as zero vector with float32
        dist = np.zeros([signallength, kraskov_k])
        dist.fill(math.inf)
        if self.settings['floattype'] == 32:
            if not dist.dtype == np.float32:
                dist = dist.astype(np.float32)

        # concatenate vars to pointset
        pointset = np.hstack((var1, var2)).T.copy()
        # change pointset to float32
        if self.settings['floattype'] == 32:
            if not pointset.dtype == np.float32:
                pointset = pointset.astype(np.float32)

        # add noise
        if self.settings['noise_level'] > 0:
            # pointset, var1, var2 = self._add_noise_all(pointset, var1, var2)
            pointset = self._add_noise(pointset)

        # Neighbour search
        distances = self.knnNumbaCPU(pointset, dist, np.int32(pointdim), np.int32(chunklength), np.int32(signallength))

        # Range search var1
        if self.settings['floattype'] == 32:
            vecradius = float32(distances[:, kraskov_k - 1])
            var1 = float32(var1)
            var2 = float32(var2)
        else:
            vecradius = distances[:, kraskov_k - 1]
        count_var1 = self.rsAllNumbaCPU(var1, vecradius, np.zeros([signallength]),
                                        var1dim, chunklength, signallength)

        # Range search var2
        count_var2 = self.rsAllNumbaCPU(var2, vecradius, np.zeros([signallength]),
                                        var2dim, chunklength, signallength)

        mi_array = self._calculate_mi(
            n_chunks=n_chunks,
            chunklength=var1.shape[0] // n_chunks,
            count_var1=count_var1,
            count_var2=count_var2,
            signallength=signallength)

        # return values
        if self.settings['return_counts']:
            return mi_array, distances, count_var1, count_var2
        else:
            return mi_array


class NumbaCudaKraskovMI(NumbaKraskov):
    """Calculate mutual information with Kraskov implementation using Numba for CUDA.

    Calculate the mutual information (MI) between two variables using Numba
    for Cuda. See parent class for references.

    Args:
        settings : dict [optional]
            set estimator parameters:

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
            - floattype : int [optional] - 32 or 64 - type of input data float32
              or float64  (default=32)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', 0)

        self.devices = self._get_numba_device(self.settings['gpuid'])

        # select device
        cuda.select_device(self.settings['gpuid'])

        #if self.settings['debug']:
        #    cuda.profile_start()

        self.shared_library_path = resource_filename(
            __name__, 'gpuKnnLibrary.so')

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

        # Shift variables to calculate a lagged MI.
        var1, var2 = self._add_mi_lag(var1, var2)

        # get values
        self.signallength = var1.shape[0]
        self.chunklength = self.signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        chunks_per_run = self._get_chunks_per_run(
            n_chunks=n_chunks,
            dim_pointset=var1dim + var2dim,
            chunklength=self.chunklength)

        mi_array = np.array([])
        if self.settings['debug']:
            distances = np.array([])
            count_var1 = np.array([])
            count_var2 = np.array([])

        # loop over chunks
        for r in range(0, n_chunks, chunks_per_run):
            startidx = r * self.chunklength
            stopidx = min(r + chunks_per_run, n_chunks) * self.chunklength
            subset1 = var1[startidx:stopidx, :]
            subset2 = var2[startidx:stopidx, :]
            n_chunks_current_run = subset1.shape[0] // self.chunklength
            results = self._estimate_single_run(subset1, subset2, n_chunks_current_run)

            ### debugging distances
            #distances = np.concatenate((distances, results[:, 0]))
            if self.settings['debug']:
                logger.debug(
                    'MI estimation results - MI: {} - Distances: {}'.format(
                        results[0][:4], results[1][:4]))
                mi_array = np.concatenate((mi_array, results[0]))
                distances = np.concatenate((distances, results[1]))
                count_var1 = np.concatenate((count_var1, results[2]))
                count_var2 = np.concatenate((count_var2, results[3]))
            else:
                mi_array = np.concatenate((mi_array, results))

        #if self.settings['debug']:
        #    cuda.profile_stop()

        #device = cuda.get_current_device()
        #device.reset()
        #cuda.close()

        # return distances
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
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        assert var1.shape[0] == var2.shape[0], 'Unequal no. realisations.'
        assert var1.shape[0] % n_chunks == 0, (
            'No. samples not divisible by no. chunks')

        pointset = np.hstack((var1, var2)).T.copy()
        pointset_var1 = var1.T.copy()
        pointset_var2 = var2.T.copy()
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim

        # initialize distances as zero vector with float32
        distances = np.zeros([self.signallength, self.settings['kraskov_k']])
        distances.fill(math.inf)
        kdistances = np.zeros([self.signallength, self.settings['kraskov_k']])
        kdistances.fill(math.inf)

        # if self.settings['floattype'] == 32:
        #     if not dist.dtype == np.float32:
        #         dist = dist.astype(np.float32)

        # add noise
        if self.settings['noise_level'] > 0:
            #pointset, var1, var2 = self._add_noise_all(pointset, var1, var2)
            pointset = self._add_noise(pointset)


        # copy data to device
        d_pointset = cuda.to_device(pointset)
        d_distances = cuda.to_device(distances)
        d_kdistances = cuda.to_device(kdistances)

        # get number of sm
        device = cuda.get_current_device()
        my_sms = getattr(cuda.gpus[self.settings['gpuid']]._device, 'MULTIPROCESSOR_COUNT')
        max_mem = self._get_max_mem()

        tpb = device.WARP_SIZE
        bpg = int(np.ceil(float(self.signallength) / tpb))

        # knn search on device
        nk._knnNumbaCuda[bpg, tpb](d_pointset,
                                   d_pointset,
                                   d_distances,
                                   pointdim,
                                   self.chunklength,
                                   self.signallength,
                                   self.signallength,
                                   self.settings['kraskov_k'],
                                   self.settings['theiler_t'],
                                   d_kdistances)

        d_distances.copy_to_host(distances)
        vecradius = distances[:, self.settings['kraskov_k']-1].copy()

        # initialize data for ncount 1
        npoints1 = np.zeros([self.signallength])

        # copy data to device
        d_var1 = cuda.to_device(pointset_var1)
        d_npoints1 = cuda.to_device(npoints1)
        d_vecradius = cuda.to_device(vecradius)

        # ncount 1
        nk._rsAllNumbaCuda[bpg, tpb](
            d_var1,
            d_var1,
            d_vecradius,
            d_npoints1,
            var1dim,
            self.chunklength,
            self.signallength,
            self.settings['kraskov_k'],
            self.settings['theiler_t'])


        # copy ncounts from device to host
        ncount_var1 = d_npoints1.copy_to_host()

        # initialize data for count 2
        npoints2 = np.zeros([self.signallength])

        # copy data to device
        d_var2 = cuda.to_device(pointset_var2)
        d_npoints2 = cuda.to_device(npoints2)

        # n_count2
        nk._rsAllNumbaCuda[bpg, tpb](
            d_var2,
            d_var2,
            d_vecradius,
            d_npoints2,
            var2dim,
            self.chunklength,
            self.signallength,
            self.settings['kraskov_k'],
            self.settings['theiler_t'])

        # copy variables from device to host
        #result_distances = d_distances.copy_to_host()
        ncount_var2 = d_npoints2.copy_to_host()

        # Calculate and sum digammas
        mi_array = self._calculate_mi(
            n_chunks=n_chunks,
            chunklength=var1.shape[0] // n_chunks,
            count_var1=ncount_var1,
            count_var2=ncount_var2,
            signallength=self.signallength)


        if self.settings['debug']:
            return mi_array, distances[:, 0], ncount_var1, ncount_var2
        else:
            return mi_array