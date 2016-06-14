"""Manage different estimators for information theoretic measures."""
import types
import numpy as np
from . import estimators_te
from . import estimators_ais
from . import estimators_cmi
from . import estimators_mi


class Estimator(object):
    """Set the estimator requested by the user.

    Estimator object that provides functionality for the estimation of
    different information theoretic measuers. Child classes add the requested
    estimator dynamically as the class method 'estimate' at runtime.

    Attributes:
        estimator_name : str
            name of the estimator currently set for estimation
        is_parallel : bool
            True if the estimator handles multiple estimations in parallel
    """

    def __init__(self):
        self.estimator_name = None
        self.is_parallel = None

    def estimate(self):
        """Stub for the estimator method."""
        print('No estimator set. Use "add_estimator".')

    def add_estimator(self, func, is_parallel, name=None):
        """Set the estimator for the 'estimate' method.

        Args:
            func : function
                function to be set as 'estimate' method
            is_parallel : bool
                True if the estimator handles multiple estimations in parallel
            name : str [optional]
                name of the estimator set for the 'estimate' method; if no name
                is given, func.__name__ is used
        """
        if name is None:
            name = func.__name__
        self.estimate = types.MethodType(func, self)
        self.estimator_name = name
        self.is_parallel = is_parallel

    @property
    def estimator_name(self):
        """Name of the estimator set for the 'estimate' method."""
        return self._estimator_name

    @estimator_name.setter
    def estimator_name(self, estimator_name):
        self._estimator_name = estimator_name

    def estimate_mult(self, n_chunks=1, options=None, re_use=None, **data):
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
            n_chunks : int
                number of data chunks
            options : dict
                sets estimation parameters
            re_use : list of keys
                realisatins to be re-used
            data: dict of numpy arrays
                realisations of random random variables

        Returns:
            numpy array of estimated values for each data set
        """
        if re_use is None:
            re_use = []

        if self.is_parallel:
            for k in re_use:
                data[k] = np.tile(data[k], (n_chunks, 1))
            return self.estimate(n_chunks=n_chunks, opts=options, **data)

        else:  # cut data into chunks and estimate iteratively
            chunk_size = data[data.keys()[0]].shape[0] / n_chunks
            idx_1 = 0
            idx_2 = chunk_size
            res = np.empty((n_chunks))
            slice_vars = list(set(data.keys()).difference(set(re_use)))
            i = 0
            for c in range(n_chunks):
                chunk_data = {}
                for k in slice_vars:  # NOTE: I AM NOT CREATING A DEEP COPY HERE!
                    chunk_data[k] = data[k][idx_1:idx_2, :]
                for k in re_use:
                    chunk_data[k] = data[k]
                res[i] = self.estimate(opts=options, **chunk_data)
                idx_1 = idx_2
                idx_2 += chunk_size
                i += 1

            return res


class Estimator_te(Estimator):
    """Set the requested transfer entropy estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_te, estimator_name)
        except AttributeError:
            raise AttributeError('The requested TE estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.add_estimator(estimator,
                               estimators_te.is_parallel(estimator_name),
                               estimator_name)


class Estimator_ais(Estimator):
    """Set the requested transfer entropy estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_ais, estimator_name)
        except AttributeError:
            raise AttributeError('The requested AIS estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.add_estimator(estimator,
                               estimators_ais.is_parallel(estimator_name),
                               estimator_name)


class Estimator_cmi(Estimator):
    """Set the requested conditional mutual information estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_cmi, estimator_name)
        except AttributeError:
            raise AttributeError('The requested CMI estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.add_estimator(estimator,
                               estimators_cmi.is_parallel(estimator_name),
                               estimator_name)


class Estimator_mi(Estimator):
    """Set the requested mutual information estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_mi, estimator_name)
        except AttributeError:
            raise AttributeError('The requested MI estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.add_estimator(estimator,
                               estimators_mi.is_parallel(estimator_name),
                               estimator_name)
