import logging
from scipy.special import digamma
import numpy as np
from idtxl.estimator import Estimator

logger = logging.getLogger(__name__)


class GPUKraskov(Estimator):
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

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        return False

    def _get_max_mem(self):
        """Return max. GPU main memory available for computation."""
        if 'max_mem' in self.settings:
            return self.settings['max_mem']
        elif 'max_mem_frac' in self.settings:
            return self.settings['max_mem_frac'] * self.devices[
                                    self.settings['gpuid']].global_mem_size
        else:
            return 0.9 * self.devices[self.settings['gpuid']].global_mem_size

    def _prepare_data(self, n_chunks, **data):
        # Check for equal and sufficient no points and make input arrays 2D.
        n_points = data[list(data.keys())[0]].shape[0]
        n_dims = data[list(data.keys())[0]].shape[0]
        logger.debug(
            'var1 shape (points, dims): {0}, {1}, n_chunks: {2}'.format(
                n_points, n_dims, n_chunks))
        # Assert identical no. points in all variables
        for var in data.keys():
            data[var] = self._ensure_two_dim_input(data[var])
            if not data[var].dtype == np.float32:
                data[var] = data[var].astype(np.float32)
            assert data[var].shape[0] == n_points
        self._check_number_of_points(n_points)
        assert data[var].shape[0] % n_chunks == 0, (
            'Can''t split data of length {} into {} chunks'.format(
                data[var].shape[0], n_chunks))
        return data

    def _add_mi_lag(self, var1, var2):
        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]
        self._check_number_of_points(var1.shape[0])
        return var1, var2

    def _get_chunks_per_run(self, n_chunks, dim_pointset, chunklength):
        # Calculate no. chunks per call to GPU
        mem_data = self.sizeof_float * chunklength * dim_pointset
        mem_dist = self.sizeof_float * chunklength * self.settings['kraskov_k']
        mem_ncnt = 2 * self.sizeof_int * chunklength
        mem_chunk = mem_data + mem_dist + mem_ncnt
        max_mem = self._get_max_mem()

        max_chunks_per_run = np.floor(max_mem/mem_chunk).astype(int)
        chunks_per_run = min(max_chunks_per_run, n_chunks)

        logger.debug(
            'Memory per chunk: {0:.5f} MB, GPU global memory: {1} MB, '
            'chunks per run: {2}.'.format(mem_chunk / 1024 / 1024,
                                          max_mem / 1024 / 1024,
                                          chunks_per_run))
        if mem_chunk > max_mem:
            raise RuntimeError('Size of single chunk exceeds GPU global '
                               'memory.')
        return chunks_per_run

    def _calculate_mi(self, n_chunks, chunklength, count_var1, count_var2):
        logger.debug('counts var1: {}'.format(count_var1[:4]))
        logger.debug('counts var2: {}'.format(count_var2[:4]))
        if self.settings['local_values']:
            mi_array = -np.inf * np.ones(
                chunklength * n_chunks, dtype=np.float64)
            idx = 0
            for c in range(n_chunks):
                mi = (digamma(self.settings['kraskov_k']) + digamma(chunklength) -
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) -
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1))
                mi_array[idx:idx+chunklength] = mi
                idx += chunklength

        else:
            mi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
            for c in range(n_chunks):
                mi = (digamma(self.settings['kraskov_k']) + digamma(chunklength) - np.mean(
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) +
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1)))
                mi_array[c] = mi

        return mi_array

    def _calculate_cmi(self, n_chunks, chunklength, count_cond,
                       count_var1cond, count_condvar2):
        logger.debug('counts cond: {}'.format(count_cond[:4]))
        logger.debug('counts var1cond: {}'.format(count_var1cond[:4]))
        logger.debug('counts condvar2: {}'.format(count_condvar2[:4]))
        if self.settings['local_values']:
            cmi_array = -np.inf * np.ones(
                n_chunks * chunklength, dtype=np.float64)
            idx = 0
            for c in range(n_chunks):
                cmi = (digamma(self.settings['kraskov_k']) +
                       digamma(count_cond[c*chunklength:(c+1)*chunklength]+1) -
                       digamma(count_var1cond[c*chunklength:(c+1)*chunklength]+1) -
                       digamma(count_condvar2[c*chunklength:(c+1)*chunklength]+1))
                cmi_array[idx:idx+chunklength] = cmi
                idx += chunklength

        else:
            cmi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
            for c in range(n_chunks):
                cmi = (digamma(self.settings['kraskov_k']) + np.mean(
                        digamma(count_cond[c*chunklength:(c+1)*chunklength]+1) -
                        digamma(count_var1cond[c*chunklength:(c+1)*chunklength]+1) -
                        digamma(count_condvar2[c*chunklength:(c+1)*chunklength]+1)))
                cmi_array[c] = cmi
            logging.debug('Stopping at index: {}'.format((c+1)*chunklength))

        return cmi_array
