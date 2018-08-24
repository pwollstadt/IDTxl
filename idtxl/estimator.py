"""Provide estimator base class for information theoretic measures."""
import imp
import os
import importlib
import inspect
from pprint import pprint
from abc import ABCMeta, abstractmethod
import numpy as np
from . import idtxl_exceptions as ex

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
        est : str | Class
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
    method. The 'estimate_parallel()' method of this abstract class provides a
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

        Each variable in vars should be a numpy array of realisations, where
        the first axis is assumed to represent realisations over samples and
        replications, while the second axis represents the variable dimension
        ([(n_samples * n_replications) x variable dimension]).

        For parallel estimators, the first axis of each variable is assumed to
        represent realisations for multiple chunks of realisations that are
        then handled in parallel. Each array has size
        [(n_samples * n_replications * n_chunks) x variable dimension]. The
        number of chunks has to be passed as an additional parameter (see for
        example the OpenCLKraskov() estimator).

        Args:
            self : Estimator class instance
                estimator
            vars: numpy arrays
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

        Return true if the estimator implements estimate_surrogates_analytic()
        where data is formatted as per the estimate method for this estimator.

        Returns:
            bool
        """
        pass

    def _check_settings(self, settings=None):
        """Set default for settings dictionary.

        Check if settings dictionary is None. If None, initialise an empty
        dictionary. If not None check if type is dictionary. Function should be
        called before setting default values.
        """
        if settings is None:
            return {}
        elif type(settings) is not dict:
            raise TypeError('settings should be a dictionary.')
        else:
            return settings

    def _check_number_of_points(self, n_points):
        """Sanity check for number of points going into the estimator."""
        if (n_points - 1) <= int(self.settings['kraskov_k']):
            raise RuntimeError('Insufficient number of points ({0}) for the '
                               'requested number of nearest neighbours '
                               '(kraskov_k: {1}).'.format(
                                        n_points, self.settings['kraskov_k']))
        if (n_points - 1) <= (int(self.settings['kraskov_k']) +
                              int(self.settings['theiler_t'])):
            raise RuntimeError('Insufficient number of points ({0}) for the '
                               'requested number of nearest neighbours '
                               '(kraskov_k: {1}) and Theiler-correction '
                               '(theiler_t: {2}).'.format(
                                                n_points,
                                                self.settings['kraskov_k'],
                                                self.settings['theiler_t']))

    def _ensure_one_dim_input(self, var):
        """Make sure input arrays have one dimension.

        Check dimensions of input to AIS and TE estimators. JIDT expects one-
        dimensional arrays for these estimators, while it expects two-
        dimensional arrays for MI and CMI estimators. To make usage of all
        estimator types easier, allow both 1D- and 2D inputs for all
        estimators. Squeeze 2D-arrays if their second dimension is 1 when
        calling AIS and TE estimators (assuming that this array dimension
        represents the variable dimension).
        """
        if len(var.shape) == 2:
            if var.shape[1] == 1:
                var = np.squeeze(var)
            else:
                raise TypeError('2D input arrays must have shape[1] == 1.')
        elif len(var.shape) > 2:
            raise TypeError('Input arrays must be 1D or 2D with shape[1] == '
                            '1.')
        return var

    def _ensure_two_dim_input(self, var):
        """Make sure input arrays have two dimension.

        Check dimensions of input to MI and CMI estimators. JIDT expects two-
        dimensional arrays for these estimators, while it expects one-
        dimensional arrays for MI and CMI estimators. To make usage of all
        estimator types easier allow both 1D- and 2D inputs for all estimators.
        Add an extra dimension to 1D-arrays when calling MI and CMI estimators
        (assuming that this array dimension represents the variable dimension).
        """
        if len(var.shape) == 1:
            var = np.expand_dims(var, axis=1)
        elif len(var.shape) > 2:
            raise TypeError('Input arrays must be 1D or 2D')
        return var

    def estimate_parallel(self, n_chunks=1, re_use=None, **data):
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
            self : Estimator class instance
                estimator
            n_chunks : int [optional]
                number of data chunks (default=1)
            re_use : list of keys [optional}
                realisatins to be re-used (default=None)
            data: numpy arrays
                realisations of random variables

        Returns:
            numpy array
                estimated values for each chunk
                
        Raises:
            ex.AlgorithmExhaustedError
                Raised from self.estimate() when calculation cannot be made
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

            # Find the number of samples, check that all data to be sliced into
            # chunks has the same number of samples.
            n_samples_total = []
            for v in slice_vars:
                if data[v] is not None:
                    n_samples_total.append(data[v].shape[0])
            assert (np.array(n_samples_total) == n_samples_total[0]).all(), (
                'No. realisations should be the same for all variables: '
                '{0}'.format(n_samples_total))
            n_samples_total = n_samples_total[0]
            assert n_samples_total is not None, (
                'All variables provided for estimation are empty.')
            assert n_samples_total % n_chunks == 0, (
                'No. chunks ({0}) does not match data length ({1}). Remainder:'
                ' {2}.'.format(n_chunks,
                               data[slice_vars[0]].shape[0],
                               data[slice_vars[0]].shape[0] % n_chunks))

            # Cut data into chunks and call estimator serially on each chunk.
            chunk_size = int(n_samples_total / n_chunks)
            idx_1 = 0
            idx_2 = chunk_size
            results = np.empty((n_chunks))
            for i in range(n_chunks):
                chunk_data = {}
                # Slice data into single chunks
                for v in slice_vars:  # NOTE: I am consciously not creating a deep copy here to save memory
                    if data[v] is not None:
                        chunk_data[v] = data[v][idx_1:idx_2, :]
                    else:
                        chunk_data[v] = data[v]
                # Collect data that is reused over chunks.
                for v in re_use:
                    if data[v] is not None:
                        assert data[v].shape[0] == chunk_size, (
                            'No. samples in variable {0} ({1}) is not equal '
                            'to chunk size ({2}).'.format(
                                v, data[v].shape[0], chunk_size))
                    chunk_data[v] = data[v]
                results[i] = self.estimate(**chunk_data)
                idx_1 = idx_2
                idx_2 += chunk_size

            return results
