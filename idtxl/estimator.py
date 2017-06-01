"""Provide estimator base class for information theoretic measures."""
from abc import ABCMeta, abstractmethod
import numpy as np


# TODO use Estimator_mi = getattr(estimator, 'JidtKraskov') to


class Estimator(metaclass=ABCMeta):

    def __init__(self, opts=None):
        pass

    @abstractmethod
    def estimate(self, **vars):
        pass

    @abstractmethod
    def is_parallel(self):
        pass

    def estimate_mult(self, n_chunks=1, re_use=None, **data):
        """Estimate measure for multiple data sets (chunks).

        Test if the estimator used provides parallel capabilities; if so,
        estimate measure for multiple data sets ('chunks') in parallel.
        Otherwise, iterate over individual chunks.

        The number of variables in data depends on the measure to be estimated,
        e.g., 2 for mutual information and 3 for TE.

        Each entry in data should be a numpy array with realisations, where the
        first axis is assumed to represent realisations (over chunks), while
        the second axis is the variable dimension.

        Each numpy array with realisations can hold either the realisations for
        multiple chunks or can hold the realisation for a single chunk, which
        gets replicated for parallel estimation and gets re-used for iterative
        estimation, in order to save memory. The variables for re-use are
        provided in re-use as list of dictionary keys indicating entries in
        data for re-use.

        Args:
            self : instance of Estimator_cmi
            n_chunks : int [optional]
                number of data chunks (default=1)
            options : dict [optional]
                sets estimation parameters (default=None)
            re_use : list of keys [optional}
                realisatins to be re-used (default=None)
            data: dict of numpy arrays
                realisations of random random variables

        Returns:
            numpy array of estimated values for each data set/chunk
        """
        assert n_chunks > 0, 'n_chunks must be positive.'
        if re_use is None:
            re_use = []

        # If the estimator supports parallel estimation, pass the variables
        # and number of chunks on to the estimator.
        if self.is_parallel:
            for k in re_use:  # multiply data for re-use
                if data[k] is not None:
                    data[k] = np.tile(data[k], (n_chunks, 1))
            return self.estimate(n_chunks=n_chunks, **data)

        # If estimator does not support parallel estimation, loop over chunks
        # and estimate iteratively for individual chunks.
        else:
            # Find arrays that have to be cut up into chunks because they are
            # not re-used.
            slice_vars = list(set(data.keys()).difference(set(re_use)))
            if not slice_vars:
                # If there is nothing to slice, we only have one chunk and can
                # return the estimate directly.
                return [self.estimate(**data)]

            n_samples_total = data[slice_vars[0]].shape[0]
            assert n_samples_total % n_chunks == 0, (
                    'No. chunks ({0}) does not match data length ({1}). '
                    'Remainder: {2}.'.format(
                                    n_chunks,
                                    data[slice_vars[0]].shape[0],
                                    data[slice_vars[0]].shape[0] % n_chunks))
            chunk_size = int(n_samples_total / n_chunks)
            idx_1 = 0
            idx_2 = chunk_size
            res = np.empty((n_chunks))
            i = 0
            # Cut data into chunks and call estimator serially.
            for c in range(n_chunks):
                chunk_data = {}
                for v in slice_vars:  # NOTE: I am consciously not creating a deep copy here to save memory
                    if data[v] is not None:
                        chunk_data[v] = data[v][idx_1:idx_2, :]
                    else:
                        chunk_data[v] = data[v]
                for v in re_use:
                    chunk_data[v] = data[v]
                res[i] = self.estimate(**chunk_data)
                idx_1 = idx_2
                idx_2 += chunk_size
                i += 1

            return res
