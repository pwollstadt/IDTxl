"""Provides numba CUDA estimators
by Michael Lindner, Uni Göttingen, 2021
"""

import logging
from pkg_resources import resource_filename
import numpy as np
from idtxl.estimators_gpu import GPUKraskov
from idtxl.idtxl_utils import DotDict
import ctypes
import math
import idtxl.numba_kernels as nk
import time
import sys
from . import idtxl_exceptions as ex
try:
    from numba import float32, float64, int32, int64, cuda
except ImportError as err:
    ex.package_missing(err, 'Numba is not available on this system. Install '
                            'it using pip or the package manager to use '
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

    by Michael Lindner, Uni Göttingen, 2021
    """

    def __init__(self, settings=None):
        super().__init__(settings)  # set defaults
        self.settings.setdefault('lag_mi', 0)
        self.settings.setdefault('threadsperblock', 0)

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
        std_ref = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        if not cuda.detect():
            raise RuntimeError('No cuda devices available!')
        sys.stdout = std_ref
        
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
            - threadsperblock [optional] - threads per block for CUDA calculation -
              should be round multiple of warp size (default is the warp size of
              graphic card)

    by Michael Lindner, Uni Göttingen, 2021
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', 0)

        # get CUDA devices
        self.devices = self._get_numba_device(self.settings['gpuid'])

        # select device
        cuda.select_device(self.settings['gpuid'])

        #if self.settings['debug']:
        #    cuda.profile_start()

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
        if self.settings['lag_mi'] > 0:
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

        # add noise
        if self.settings['noise_level'] > 0:
            pointset, pointset_var1, pointset_var2 = self._add_noise_all(pointset, pointset_var1, pointset_var2)


        # copy data to device
        d_pointset = cuda.to_device(pointset)
        d_distances = cuda.to_device(distances)

        # get current device
        device = cuda.get_current_device()

        if self.settings['threadsperblock'] == 0:
            tpb = device.WARP_SIZE
        else:
            tpb = self.settings['threadsperblock']

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
                                   self.settings['theiler_t'])

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


class NumbaCudaKraskovCMI(NumbaKraskov):
    """Calculate conditional mutual information with Kraskov implementation using Numba for CUDA.

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
            - threadsperblock [optional] - threads per block for CUDA calculation -
              should be round multiple of warp size (default is the warp size of
              graphic card)

    by Michael Lindner, Uni Göttingen, 2021
    """

    def __init__(self, settings=None):
        super().__init__(settings)  # set defaults

        # get CUDA devices
        self.devices = self._get_numba_device(self.settings['gpuid'])

        # select device
        cuda.select_device(self.settings['gpuid'])

        #if self.settings['debug']:
        #    cuda.profile_start()

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
            est_mi = NumbaCudaKraskovMI(self.settings)
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
        self.signallength = var1.shape[0]
        self.chunklength = self.signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        conddim = conditional.shape[1]
        chunks_per_run = self._get_chunks_per_run(
            n_chunks=n_chunks,
            dim_pointset=var1dim + var2dim + conddim,
            chunklength=self.chunklength)

        cmi_array = np.array([])
        if self.settings['debug']:
            distances = np.array([])
            count_var1 = np.array([])
            count_var2 = np.array([])
            count_cond = np.array([])

        for r in range(0, n_chunks, chunks_per_run):
            startidx = r * self.chunklength
            stopidx = min(r+chunks_per_run, n_chunks) * self.chunklength
            subset1 = var1[startidx:stopidx, :]
            subset2 = var2[startidx:stopidx, :]
            subset3 = conditional[startidx:stopidx, :]
            n_chunks_current_run = subset1.shape[0] // self.chunklength
            results = self._estimate_single_run(subset1, subset2, subset3,
                                                n_chunks_current_run)
            if self.settings['debug']:
                logger.debug(
                    'CMI estimation results - CMI: {} - Distances: {}'.format(
                        results[0][:4], results[1][:4]))
                cmi_array = np.concatenate((cmi_array,  results[0]))
                distances = np.concatenate((distances,  results[1][:, self.settings['kraskov_k']-1]))
                count_var1 = np.concatenate((count_var1, results[2]))
                count_var2 = np.concatenate((count_var2, results[3]))
                count_cond = np.concatenate((count_cond, results[4]))
            else:
                cmi_array = np.concatenate((cmi_array, results))

        # if self.settings['debug']:
        #    cuda.profile_stop()

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
        var1conddim = pointset_var1cond.shape[0]
        condvar2dim = pointset_condvar2.shape[0]
        conddim = pointset_cond.shape[0]
        pointdim = pointset.shape[0]

        # initialize distances as zero vector
        distances = np.zeros([self.signallength, self.settings['kraskov_k']])
        distances.fill(math.inf)
        kdistances = np.zeros([self.signallength, self.settings['kraskov_k']])
        kdistances.fill(math.inf)

        # add noise
        if self.settings['noise_level'] > 0:
            pointset, pointset_var1cond, pointset_condvar2 = self._add_noise_all(pointset, pointset_var1cond, pointset_condvar2)

        # copy data to device
        d_pointset = cuda.to_device(pointset)
        d_distances = cuda.to_device(distances)

        # get number of sm
        device = cuda.get_current_device()

        if self.settings['threadsperblock'] == 0:
            tpb = device.WARP_SIZE
        else:
            tpb = self.settings['threadsperblock']

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
                                   self.settings['theiler_t'])

        d_distances.copy_to_host(distances)
        vecradius = distances[:, self.settings['kraskov_k'] - 1].copy()

        # initialize data for ncount var1cond
        npointsvar1cond = np.zeros([self.signallength])
        npointscondvar2 = np.zeros([self.signallength])
        npointscond = np.zeros([self.signallength])

        # -----------  range search var1cond
        # copy data to device for ncount var1cond
        d_var1cond = cuda.to_device(pointset_var1cond)
        d_npointsvar1cond = cuda.to_device(npointsvar1cond)
        d_vecradius = cuda.to_device(vecradius)

        # range search
        nk._rsAllNumbaCuda[bpg, tpb](
            d_var1cond,
            d_var1cond,
            d_vecradius,
            d_npointsvar1cond,
            var1conddim,
            self.chunklength,
            self.signallength,
            self.settings['theiler_t'])

        # copy variables from device to host
        d_npointsvar1cond.copy_to_host(npointsvar1cond)

        # -----------  range search condvar2
        # copy data to device for ncount condvar2
        d_condvar2 = cuda.to_device(pointset_condvar2)
        d_npointscondvar2 = cuda.to_device(npointscondvar2)

        # range search
        nk._rsAllNumbaCuda[bpg, tpb](
            d_condvar2,
            d_condvar2,
            d_vecradius,
            d_npointscondvar2,
            condvar2dim,
            self.chunklength,
            self.signallength,
            self.settings['theiler_t'])

        # copy variables from device to host
        d_npointscondvar2.copy_to_host(npointscondvar2)

        # -----------  range search cond
        # copy data to device for ncount cond
        d_cond = cuda.to_device(pointset_cond)
        d_npointscond = cuda.to_device(npointscond)

        # range search cond
        nk._rsAllNumbaCuda[bpg, tpb](
            d_cond,
            d_cond,
            d_vecradius,
            d_npointscond,
            conddim,
            self.chunklength,
            self.signallength,
            self.settings['theiler_t'])

        # copy variables from device to host
        d_npointscond.copy_to_host(npointscond)

        # Calculate and sum digammas
        cmi_array = self._calculate_cmi(
            n_chunks, var1.shape[0] // n_chunks, npointscond, npointsvar1cond, npointscondvar2)

        if self.settings['debug']:
            return cmi_array, distances, npointsvar1cond, npointscondvar2, npointscond
        else:
            return cmi_array
