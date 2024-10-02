"""Provide estimator base class for information theoretic measures."""

import importlib
import inspect
import os
from abc import ABCMeta, abstractmethod
from pprint import pprint

import numpy as np

MODULE_EXTENSIONS = ".py"  # ('.py', '.pyc', '.pyo')
ESTIMATOR_PREFIX = "estimators_"


def _package_contents():
    # Return list of IDTxl modules containing estimators.
    pkg_spec = importlib.util.find_spec(__package__)
    module_path = pkg_spec.submodule_search_locations[0]
    return [
        os.path.splitext(module)[0]
        for module in os.listdir(module_path)
        if (module.endswith(MODULE_EXTENSIONS) and module.startswith(ESTIMATOR_PREFIX))
    ]


def list_estimators():
    """List all estimators available in IDTxl."""
    module_list = _package_contents()
    for m in module_list:
        module = importlib.import_module("." + m, __package__)
        class_list = inspect.getmembers(module, inspect.isclass)
        if class_list:
            pprint(class_list)


def _find_estimator(est):
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
            raise RuntimeError(
                "Provided class should implement abstract class Estimator."
            )
        return est
    if isinstance(est, str):
        module_list = _package_contents()
        estimator = None
        for m in module_list:
            try:
                module = importlib.import_module("." + m, __package__)
                return getattr(module, est)
            except AttributeError:
                pass
        if not estimator:
            raise RuntimeError(f"Estimator {est} not found.")
    else:
        raise TypeError(
            "Please provide an estimator class or the name of an estimator as string."
        )


def get_estimator(est, settings):
    """Factory method that creates an Estimator instance with the given settings.

    If the MPI flag is set to True, return an MPIEstimator instead.

    Args:
        est : str | Class
            name of an estimator class implemented in IDTxl or custom estimator
            class

    Returns
        Estimator
            Instance of the requestet estimator or MPIEstimator
    """

    # Check if MPI flag is set to True
    if settings.get("MPI", False):
        settings_mpi = settings.copy()

        # Remove MPI flag to avoid infinite recursion
        del settings_mpi["MPI"]

        # Import just in time to avoid cyclic import
        from .estimators_mpi import MPIEstimator

        return MPIEstimator(est, settings_mpi)

    # Otherwise find Estimator and return instance
    EstimatorClass = _find_estimator(est)

    return EstimatorClass(settings)


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
        self.settings = settings
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
        elif not isinstance(settings, dict):
            raise TypeError("settings should be a dictionary.")
        else:
            return settings

    def _check_number_of_points(self, n_points):
        """Sanity check for number of points going into the estimator."""
        if (n_points - 1) <= int(self.settings["kraskov_k"]):
            raise RuntimeError(
                f"Insufficient number of points ({n_points}) for the requested number of nearest neighbours "
                f"(kraskov_k: {self.settings['kraskov_k']})."
            )
        if (n_points - 1) <= (
            int(self.settings["kraskov_k"]) + int(self.settings["theiler_t"])
        ):
            raise RuntimeError(
                f"Insufficient number of points ({n_points}) for the requested number of nearest neighbours (kraskov_k:"
                f" {self.settings['kraskov_k']}) and Theiler-correction (theiler_t: {self.settings['theiler_t']})."
            )

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
                raise TypeError("2D input arrays must have shape[1] == 1.")
        elif len(var.shape) > 2:
            raise TypeError("Input arrays must be 1D or 2D with shape[1] == 1.")
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
            raise TypeError("Input arrays must be 1D or 2D")
        return var

    def estimate_parallel(self, **data):
        """Estimate measure for multiple data sets (chunks).

        Test if the estimator used provides parallel capabilities; if so,
        estimate measure for multiple data sets ('chunks') in parallel.
        Otherwise, iterate over individual chunks.

        The number of variables in data depends on the measure to be estimated,
        e.g., 2 for mutual information and 3 for a conditional mutual
        information.

        Each entry in data should be a list of numpy array with realisations, where the
        the list entries represent chunks, axis 0 of the arrays represent realisations 
        over samples, while the axis 1 represents the variable
        dimension ([chunks x samples x variable dimension]).

        Each numpy array with realisations can hold either the realisations for
        multiple chunks or can hold the realisation for a single chunk, which
        gets replicated for parallel estimation and gets re-used for iterative
        estimation, in order to save memory. The variables for re-use are
        provided in re-use as list of dictionary keys indicating entries in
        data for re-use.

        Args:
            data: numpy arrays
                realisations of random variables

        Returns:
            numpy array
                estimated values for each chunk

        Raises:
            ex.AlgorithmExhaustedError
                Raised from self.estimate() when calculation cannot be made
        """

        # Check the number of chunks
        lengths = [len(v) for v in data.values()]
        n_chunks = lengths[0]
        assert all(l == n_chunks for l in lengths), 'All variables must have the same number of chunks.'
        
        # If the estimator supports parallel estimation, pass the variables
        # and number of chunks on to the estimator.
        if self.is_parallel():
            n_chunks = len(data[list(data.keys())[0]])
            # Concatenate all chunks into a single array for each variable
            data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
            return self.estimate(n_chunks=n_chunks, **data)

        # If estimator does not support parallel estimation, loop over chunks
        results = np.empty(n_chunks)
        for chunk in range(n_chunks):
            chunk_data = {k: v[chunk] if v is not None else None for k, v in data.items()}
            results[chunk] = self.estimate(**chunk_data)

        return results