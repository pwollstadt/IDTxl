"""Provides numba CPU and CUDA estimators
by Michael Lindner, Uni Göttingen, 2021
"""

import logging
import numpy as np
from idtxl.estimators_numba import NumbaKraskov
import math
import idtxl.numba_kernels as nk
from idtxl import idtxl_exceptions as ex

try:
    from numba import float32, float64, int32, int64
except ImportError as err:
    ex.package_missing(err, 'Numba is not available on this system. Install '
                            'it using pip or the package manager to use '
                            'the Numba estimators.')

logger = logging.getLogger(__name__)


class NumbaCPUKraskovMI(NumbaKraskov):
    """Calculate mutual information with Kraskov implementation using Numba on CPU.

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

    by Michael Lindner, Uni Göttingen, 2021
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
        if self.settings['lag_mi'] > 0:
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
        pointset_var1 = var1.T.copy()
        pointset_var2 = var2.T.copy()


        # add noise
        if self.settings['noise_level'] > 0:
            # pointset, var1, var2 = self._add_noise_all(pointset, var1, var2)
            pointset = self._add_noise(pointset)

        # Neighbour search
        distances = self.knnNumbaCPU(pointset, dist, np.int32(pointdim), np.int32(chunklength), np.int32(signallength))

        # Range search var1
        if self.settings['floattype'] == 32:
            vecradius = float32(distances[:, kraskov_k - 1])
            pointset_var1 = float32(pointset_var1)
            pointset_var2 = float32(pointset_var2)
        else:
            vecradius = distances[:, kraskov_k - 1]
        count_var1 = self.rsAllNumbaCPU(pointset_var1, vecradius, np.zeros([signallength]),
                                        var1dim, chunklength, signallength)

        # Range search var2
        count_var2 = self.rsAllNumbaCPU(pointset_var2, vecradius, np.zeros([signallength]),
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


class NumbaCPUKraskovCMI(NumbaKraskov):
    """Calculate conditional mutual information with Kraskov implementation using Numba on CPU.

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

    by Michael Lindner, Uni Göttingen, 2021
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)

    def knnNumbaCPU(self, vpointset, distances, pointdim, chunklength, signallength):
        return nk._knnNumbaCPU(vpointset, vpointset, distances, pointdim, chunklength, signallength,
                               np.int32(self.settings['kraskov_k']), np.int32(self.settings['theiler_t']),
                               np.int32(self.settings['floattype']))

    def rsAllNumbaCPU(self, rsvar, vecradius, npoints, pointdim, chunklength, signallength):
        return nk._rsAllNumbaCPU(rsvar, rsvar, vecradius, npoints, pointdim, chunklength, signallength,
                                 np.int32(self.settings['theiler_t']), np.int32(self.settings['floattype']))

    def estimate(self, var1, var2, conditional=None, n_chunks=1):
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

        if conditional is None:
            est_mi = NumbaCPUKraskovMI(self.settings)
            return est_mi.estimate(var1, var2, n_chunks)

        # Prepare data: check if variable realisations are passed as 1D or 2D
        # arrays and have equal no. observations.
        data_checked = self._prepare_data(
            n_chunks, var1=var1, var2=var2, conditional=conditional)
        var1 = data_checked['var1']
        var2 = data_checked['var2']
        conditional = data_checked['conditional']

        # get values
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        kraskov_k = np.int32(self.settings['kraskov_k'])

        # initialize distances as zero vector with float32
        dist = np.zeros([signallength, kraskov_k])
        dist.fill(math.inf)
        if self.settings['floattype'] == 32:
            if not dist.dtype == np.float32:
                dist = dist.astype(np.float32)

        # concatenate vars to pointset
        pointset = np.hstack((var1, conditional, var2)).T.copy()
        # change pointset to float32
        if self.settings['floattype'] == 32:
            if not pointset.dtype == np.float32:
                pointset = pointset.astype(np.float32)
        pointset_var1cond = np.hstack((var1, conditional)).T.copy()
        pointset_condvar2 = np.hstack((conditional, var2)).T.copy()
        pointset_cond = conditional.T.copy()
        var1conddim = pointset_var1cond.shape[0]
        condvar2dim = pointset_condvar2.shape[0]
        conddim = pointset_cond.shape[0]
        pointdim = pointset.shape[0]

        # add noise
        if self.settings['noise_level'] > 0:
            # pointset, var1, var2 = self._add_noise_all(pointset, var1, var2)
            pointset, pointset_var1cond, pointset_condvar2 = self._add_noise_all(pointset,
                                                                                 pointset_var1cond, pointset_condvar2)

        # Neighbour search
        distances = self.knnNumbaCPU(pointset, dist, np.int32(pointdim), np.int32(chunklength), np.int32(signallength))
        if self.settings['floattype'] == 32:
            vecradius = float32(distances[:, kraskov_k - 1])
            pointset_var1cond = float32(pointset_var1cond)
            pointset_condvar2 = float32(pointset_condvar2)
            pointset_cond = float32(pointset_cond)
        else:
            vecradius = distances[:, kraskov_k - 1]

        # Range search var1cond
        count_var1cond = self.rsAllNumbaCPU(pointset_var1cond, vecradius, np.zeros([signallength]),
                                        var1conddim, chunklength, signallength)

        # Range search condvar2
        count_condvar2 = self.rsAllNumbaCPU(pointset_condvar2, vecradius, np.zeros([signallength]),
                                        condvar2dim, chunklength, signallength)

        # Range search cond
        count_cond = self.rsAllNumbaCPU(pointset_cond, vecradius, np.zeros([signallength]),
                                        conddim, chunklength, signallength)

        # Calculate cmi
        cmi_array = self._calculate_cmi(
            n_chunks, var1.shape[0] // n_chunks, count_cond, count_var1cond, count_condvar2)

        # return values
        if self.settings['return_counts']:
            return cmi_array, distances, count_var1cond, count_condvar2, count_cond
        else:
            return cmi_array


