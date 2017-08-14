"""Provide estimator base class for information theoretic measures."""
import sys
import imp
import os
import importlib
import inspect
from pprint import pprint
from abc import ABCMeta, abstractmethod
import numpy as np

MODULE_EXTENSIONS = ('.py')  # ('.py', '.pyc', '.pyo')
ESTIMATOR_PREFIX = ('estimators_')


def _package_contents():
    # Return list of IDTxl modules containing estimators.
    file, pathname, description = imp.find_module(__package__)
    if file:
        raise ImportError('Not a package: %r', __package__)
    return [os.path.splitext(module)[0]
            for module in os.listdir(pathname)
            if (module.endswith(MODULE_EXTENSIONS) and
                module.startswith(ESTIMATOR_PREFIX))]


def list_estimators():
    """List all estimators available in IDTxl."""
    module_list = _package_contents()
    for m in module_list:
        module = importlib.import_module('.' + m, __package__)
        class_list = inspect.getmembers(module, inspect.isclass)
        if class_list:
            pprint(class_list)


def find_estimator(est):
    """Return estimator class.

    Return an estimator class. If input is a class, check if it implements
    methods 'estimate' and 'is_parallel' necessary for network analysis
    (see abstract class 'Estimator' for documentation). If input is a string,
    search for class with that name in IDTxl and return it.

    Args:
        est : int | Class
            name of an estimator class implemented in IDTxl or custom estimator
            class

    Returns
        Class
            Estimator class
    """
    if inspect.isclass(est):
        # Test if provided class implements the Estimator class. This
        # constraint may be relaxed in the future.
        if not np.issubclass_(est, Estimator):
            raise RuntimeError('Provided class should implement abstract class'
                               ' Estimator.')
        return est
    elif type(est) is str:
        module_list = _package_contents()
        estimator = None
        for m in module_list:
            try:
                module = importlib.import_module('.' + m, __package__)
                return getattr(module, est)
            except AttributeError:
                pass
        if not estimator:
            raise RuntimeError('Estimator {0} not found.'.format(est))
    else:
        raise TypeError('Please provide an estimator class or the name of an '
                        'estimator as string.')


class Estimator(metaclass=ABCMeta):
    """Abstract class for implementation of IDTxl estimators.

    Abstract class for implementation of IDTxl estimators. Child classes
    implement various estimators for information-theoretic measures, e.g.,
    mutual information (MI), conditional mutual information (CMI),
    active information storage (AIS), or transfer entropy (TE).

    Estimator classes implement a method 'estimate()' for the estimation from
    single or multiple data sets in parallel. Whether 'estimate()' provides
    parallel computing capabilities is indicated by the 'is_parallel()'
    method. The 'estimate_mult()' method of this abstract class provides a
    common interface to parallel and serial methods (see docstring for
    details).

    The method 'is_analytic_null_estimator()' indicates whether the implemented
    estimator supports the generation of analytic surrogates (see docstring for
    details).
    """

    def __init__(self, settings=None):
        pass

    @abstractmethod
    def estimate(self, **vars):
        """Estimate measure for a single data set.

        The number of variables in data depends on the measure to be estimated,
        e.g., 2 for mutual information and 3 for a conditional mutual
        information.

        Each entry in data should be a numpy array with realisations, where the
        first axis is assumed to represent realisations over samples and
        replications, while the second axis represents the variable dimension
        ([(samples * replications) x variable dimension]).

        Args:
            self : instance of Estimator class
            data: dict of numpy arrays
                realisations of random variables

        Returns:
            float
                estimated value
        """
        pass

    @abstractmethod
    def is_parallel(self):
        """Indicate if estimator supports parallel estimation over chunks.

        Return true if the supports parallel estimation over chunks, where a
        chunk is one independent data set.

        Returns:
            bool
        """
        pass

    @abstractmethod
    def is_analytic_null_estimator(self):
        """Indicate if estimator supports analytic surrogates.

        Return true if the estimator implements
        estimate_surrogates_analytic(self, n_perm, **data) where data is
        formatted as per the estimate method for this estimator.

        Returns:
            bool
        """
        pass

    def estimate_mult(self, n_chunks=1, re_use=None, **data):
        """Estimate measure for multiple data sets (chunks).

        Test if the estimator used provides parallel capabilities; if so,
        estimate measure for multiple data sets ('chunks') in parallel.
        Otherwise, iterate over individual chunks.

        The number of variables in data depends on the measure to be estimated,
        e.g., 2 for mutual information and 3 for a conditional mutual
        information.

        Each entry in data should be a numpy array with realisations, where the
        first axis is assumed to represent realisations over samples and
        replications and chunks, while the second axis represents the variable
        dimension ([(samples * replications) * chunks x variable dimension]).

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
            self : instance of Estimator class
            n_chunks : int [optional]
                number of data chunks (default=1)
            re_use : list of keys [optional}
                realisatins to be re-used (default=None)
            data: dict of numpy arrays
                realisations of random variables

        Returns:
            numpy array
                estimated values for each chunk
        """
        assert n_chunks > 0, 'n_chunks must be positive.'
        if re_use is None:
            re_use = []

        # If the estimator supports parallel estimation, pass the variables
        # and number of chunks on to the estimator.
        if self.is_parallel():
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
