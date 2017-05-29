"""Provide estimator classes for information theoretic measures.

Example:
    To use a class based on a string provided by the user, call

    >>> from . import estimator
    >>> Estimator = getattr(estimator, 'Class_name')
    >>> estimator = Estimator(options)
    >>> estimator.estimate(var1, var2)

"""
from abc import ABCMeta, abstractmethod
from pkg_resources import resource_filename
import numpy as np
from . import idtxl_exceptions as ex

try:
    import jpype as jp
except ImportError as err:
    ex.package_missing(err, 'Jpype is not available on this system. Install it'
                            ' from https://pypi.python.org/pypi/JPype1 to use '
                            'JAVA/JIDT-powered CMI estimation.')

# TODO use Estimator_mi = getattr(estimator, 'Jidt_kraskov') to


class Estimator(metaclass=ABCMeta):

    def __init__(self, opts=None):
        pass

    @abstractmethod
    def estimate(self, var1, var2):
        raise NotImplementedError('Define estimate to use this class.')

    @property
    def is_parallel(self):
        try:
            return self._is_parallel
        except AttributeError:
            raise NotImplementedError('Define is_parallel to use this class.')

    @is_parallel.setter
    def is_parallel(self, p):
        self._is_parallel = p

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


class Test_estimator(Estimator):

    def __init__(self, opts={}):
        self.is_parallel = 'thisworked'
        pass

    def estimate(self, var1, var2):
        pass


class Jidt_kraskov(Estimator):

    def __init__(self, Calc_class, opts=None):

        self.is_parallel = False

        if opts is None:
            opts = {}
        elif type(opts) is not dict:
            raise TypeError('Opts should be a dictionary.')

        # Get defaults for estimator options
        opts.setdefault('kraskov_k', str(4))
        opts.setdefault('normalise', 'false')
        opts.setdefault('theiler_t', str(0))
        opts.setdefault('noise_level', 1e-8)
        opts.setdefault('num_threads', 'USE_ALL')
        opts.setdefault('local_values', False)
        # debug = opts.get('debug', False).lower()
        self.opts = opts

        # Set properties of JIDT's calculator object.
        self.calc = Calc_class()
        self.calc.setProperty('NORMALISE', str(self.opts['normalise']).lower())
        self.calc.setProperty('k', str(self.opts['kraskov_k']))
        self.calc.setProperty('DYN_CORR_EXCL', str(self.opts['theiler_t']))
        self.calc.setProperty('NOISE_LEVEL_TO_ADD', str(
                                                    self.opts['noise_level']))
        self.calc.setProperty('NUM_THREADS', str(self.opts['num_threads']))
        # calc.setDebug(debug)

    def _start_jvm(self):
        jar_location = resource_filename(__name__, 'infodynamics.jar')
        if not jp.isJVMStarted():
            jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                        jar_location))

    def estimate(self, var1, var2):
        pass


class Jidt_kraskov_cmi(Jidt_kraskov):
    """Calculate conditional mutual infor with JIDT's Kraskov implementation.

    Calculate the conditional mutual information between three variables. Call
    JIDT via jpype and use the Kraskov 1 estimator. If no conditional is given
    (is None), the function returns the mutual information between var1 and
    var2. References:

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
        var1 : numpy array
            realisations of the first random variable, where dimensions are
            realisations x variable dimension
        var2 : numpy array
            realisations of the second random variable
        conditional : numpy array [optional]
            realisations of the random variable for conditioning, if no
            conditional is provided, return MI between var1 and var2
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'num_threads' - no. CPU threads used for estimation
              (default='USE_ALL', this uses all available cores on the
              machine!)

    Returns:
        float
            conditional mutual information

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI calculator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, opts=None):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        Calc_class = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                      ConditionalMutualInfoCalculatorMultiVariateKraskov1)
        super().__init__(Calc_class, opts)

    def estimate(self, var1, var2, conditional=None):
        if conditional is None:
            cond_dim = 0
        else:
            cond_dim = conditional.shape[1]
            assert(conditional.size != 0), 'Conditional Array is empty.'
        assert(var1.shape[0] == var2.shape[0]), (
            'Unequal number of observations (var1: {0}, var2: {1}).'.format(
                                                            var1.shape[0],
                                                            var2.shape[0]))
        self.calc.initialise(var1.shape[1], var2.shape[1], cond_dim)
        self.calc.setObservations(var1, var2, conditional)
        return self.calc.computeAverageLocalOfObservations()


class Jidt_kraskov_mi(Jidt_kraskov):
    """Calculate mutual information with JIDT's Kraskov implementation.

    Calculate the mutual information between two variables. Call JIDT via jpype
    and use the Kraskov 1 estimator. References:

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_mi class.

    Args:
        self : instance of Estimator_mi
            function is supposed to be used as part of the Estimator_mi class
        var1 : numpy array
            realisations of the first random variable, where dimensions are
            realisations x variable dimension
        var2 : numpy array
            realisations of the second random variable
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'num_threads' - no. CPU threads used for estimation
              (default='USE_ALL', this uses all available cores on the
              machine!)

    Returns:
        float
            mutual information

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the MI calculator to save
        computation time. The Theiler window ignores trial boundaries. The
        MI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, opts=None):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        Calc_class = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                      MutualInfoCalculatorMultiVariateKraskov1)
        super().__init__(Calc_class, opts)

        # Get lag and shift second variable to account for a lag if requested
        self.opts.setdefault('lag', 0)
        if self.opts['lag'] > 0:
            var1 = var1[:-lag]
            var2 = var2[lag:]

    def estimate(self, var1, var2):
        calc.initialise(var1.shape[1], var2.shape[1])
        calc.setObservations(var1, var2)
        if local_values:
            return np.array(calc.computeLocalOfPreviousObservations())
        else:
            return calc.computeAverageLocalOfObservations()
