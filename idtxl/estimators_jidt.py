"""Provide JIDT estimators."""
from pkg_resources import resource_filename
import numpy as np
from abc import abstractmethod
from idtxl.estimator import Estimator
from . import idtxl_exceptions as ex
from . import idtxl_utils as utils
try:
    import jpype as jp
except ImportError as err:
    ex.package_missing(err, 'Jpype is not available on this system. Install it'
                            ' from https://pypi.python.org/pypi/JPype1 to use '
                            'JAVA/JIDT-powered CMI estimation.')

# TODO check IDTxl nomenclature (variable > process, estimate vs. calculate)


class JidtEstimator(Estimator):
    """Abstract class for implementation of JIDT estimators.

    Abstract class for implementation of JIDT estimators, child classes
    implement estimators for mutual information (MI), conditional mutual
    information (CMI), actice information storage (AIS), transfer entropy (TE)
    using the Kraskov-Grassberger-Stoegbauer estimator for continuous data,
    plug-in estimators for discrete data, and Gaussian estimators for
    continuous Gaussian data. References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Set common estimation parameters for JIDT estimators. For usage of these
    estimators see documentation for the child classes.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)

    """

    def __init__(self, settings=None):
        """Set default estimator settings."""
        settings = self._check_settings(settings)
        settings.setdefault('local_values', False)
        settings.setdefault('debug', False)
        self.settings = settings

    def _start_jvm(self):
        """Start JAVA virtual machine if it is not running."""
        jar_location = resource_filename(__name__, 'infodynamics.jar')
        if not jp.isJVMStarted():
            jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                                                        jar_location))

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

    def _set_te_defaults(self):
        """Set defaults for transfer entropy estimation."""
        try:
            history_target = self.settings['history_target']
        except KeyError:
            raise RuntimeError('No target history was provided for TE '
                               'estimation.')
        self.settings.setdefault('history_source', history_target)
        self.settings.setdefault('tau_target', 1)
        self.settings.setdefault('tau_source', 1)
        self.settings.setdefault('source_target_delay', 1)

        assert type(self.settings['tau_target']) is int, (
            'Target tau has to be an integer.')
        assert type(self.settings['tau_source']) is int, (
            'Source tau has to be an integer.')
        assert type(self.settings['history_target']) is int, (
            'Target history has to be an integer.')
        assert type(self.settings['history_source']) is int, (
            'Source history has to be an integer.')

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

    def is_parallel(self):
        return False


class JidtKraskov(JidtEstimator):
    """Abstract class for implementation of JIDT Kraskov-estimators.

    Abstract class for implementation of JIDT Kraskov-estimators, child classes
    implement estimators for mutual information (MI), conditional mutual
    information (CMI), actice information storage (AIS), transfer entropy (TE)
    using the Kraskov-Grassberger-Stoegbauer estimator for continuous data.
    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Set common estimation parameters for JIDT Kraskov-estimators. For usage of
    these estimators see documentation for the child classes.

    Args:
        CalcClass : JAVA class
            JAVA class returned by jpype.JPackage
        settings : dict [optional]
            set estimator parameters:

            - 'debug' - return debug information when calling JIDT.
              (boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
    """

    def __init__(self, CalcClass, settings=None):

        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('kraskov_k', str(4))
        self.settings.setdefault('normalise', 'false')
        self.settings.setdefault('theiler_t', str(0))
        self.settings.setdefault('noise_level', 1e-8)
        self.settings.setdefault('num_threads', 'USE_ALL')

        # Set properties of JIDT's estimator object.
        self.calc = CalcClass()
        self.calc.setProperty('PROP_KRASKOV_ALG_NUM', str(1))
        self.calc.setProperty('NORMALISE',
                              str(self.settings['normalise']).lower())
        self.calc.setProperty('k', str(self.settings['kraskov_k']))
        self.calc.setProperty('DYN_CORR_EXCL', str(self.settings['theiler_t']))
        self.calc.setProperty('NOISE_LEVEL_TO_ADD',
                              str(self.settings['noise_level']))
        self.calc.setProperty('NUM_THREADS', str(self.settings['num_threads']))
        self.calc.setDebug(self.settings['debug'])

    def is_analytic_null_estimator(self):
        return False


class JidtDiscrete(JidtEstimator):
    """Abstract class for implementation of discrete JIDT-estimators.

    Abstract class for implementation of plug-in JIDT-estimators for discrete
    data. Child classes implement estimators for mutual information (MI),
    conditional mutual information (CMI), actice information storage (AIS), and
    transfer entropy (TE). References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Set common estimation parameters for discrete JIDT-estimators. For usage of
    these estimators see documentation for the child classes.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'discretise_method' - if and how to discretise incoming
              continuous variables to discrete values, can be 'max_ent' for
              maximum entropy binning, 'equal' for equal size bins, and 'none'
              if no binning is required (default='none')

    Note:
        Discrete JIDT estimators require the data's alphabet size for
        instantiation. Hence, opposed to the Kraskov and Gaussian estimators,
        the JAVA class is added to the object instance, while for Kraskov/
        Gaussian estimators an instance of that class is added (because for the
        latter, objects can be instantiated independent of data properties).
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.settings.setdefault('discretise_method', 'none')

    def _discretise_vars(self, var1, var2, conditional=None):
        # Discretise variables if requested. Otherwise assert data are discrete
        # and provided alphabet sizes are correct.
        if self.settings['discretise_method'] == 'equal':
            var1 = utils.discretise(var1, self.settings['alph1'])
            var2 = utils.discretise(var2, self.settings['alph2'])
            if not (conditional is None):
                conditional = utils.discretise(conditional,
                                               self.settings['alphc'])

        elif self.settings['discretise_method'] == 'max_ent':
            var1 = utils.discretise_max_ent(var1, self.settings['alph1'])
            var2 = utils.discretise_max_ent(var2, self.settings['alph2'])
            if not (conditional is None):
                conditional = utils.discretise_max_ent(conditional,
                                                       self.settings['alphc'])

        elif self.settings['discretise_method'] == 'none':
            assert issubclass(var1.dtype.type, np.integer), (
                'Var1 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert issubclass(var2.dtype.type, np.integer), (
                'Var2 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert min(var1) >= 0, 'Minimum of var1 is smaller than 0.'
            assert min(var2) >= 0, 'Minimum of var2 is smaller than 0.'
            assert max(var1) < self.settings['alph1'], (
                        'Maximum of var1 is larger than the alphabet size.')
            assert max(var2) < self.settings['alph2'], (
                        'Maximum of var2 is larger than the alphabet size.')
            if not (conditional is None):
                assert min(conditional) >= 0, ('Minimum of conditional is '
                                               'smaller than 0.')
                assert issubclass(conditional.dtype.type, np.integer), (
                    'Conditional is not an integer numpy array. '
                    'Discretise data to use this estimator.')
                assert max(conditional) < self.settings['alphc'], (
                    'Maximum of conditional is larger than the alphabet size.')
        else:
            raise ValueError('Unkown discretisation method.')

        if not (conditional is None):
            return var1, var2, conditional
        else:
            return var1, var2

    def is_analytic_null_estimator(self):
        return True

    @abstractmethod
    def get_analytic_distribution(self, **data):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            data : dict of numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per the estimate method for this estimator.

        Returns:
            Java object
                JIDT calculator that was used here
        """
        pass

    def estimate_surrogates_analytic(self, n_perm=200, **data):
        """Return estimate of the analytical surrogate distribution.

        This method must be implemented because this class'
        is_analytic_null_estimator() method returns true.

        Args:
            n_perms : int [optional]
                number of permutations (default=200)
            data : dict of numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per the estimate method for this estimator.

        Returns:
            float | numpy array
                n_perm surrogates of the average MI/CMI/TE over all samples
                under the null hypothesis of no relationship between var1 and
                var2 (in the context of conditional)
        """
        return common_estimate_surrogates_analytic(self, n_perm, **data)


class JidtGaussian(JidtEstimator):
    """Abstract class for implementation of JIDT Gaussian-estimators.

    Abstract class for implementation of JIDT Gaussian-estimators, child
    classes implement estimators for mutual information (MI), conditional
    mutual information (CMI), actice information storage (AIS), transfer
    entropy (TE) using JIDT's Gaussian estimator for continuous data.
    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Set common estimation parameters for JIDT Kraskov-estimators. For usage of
    these estimators see documentation for the child classes.

    Args:
        CalcClass : JAVA class
            JAVA class returned by jpype.JPackage
        settings : dict [optional]
            set estimator parameters:

            - 'debug' - return debug information when calling JIDT.
              (boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
    """

    def __init__(self, CalcClass, settings):
        super().__init__(settings)
        self.calc = CalcClass()
        self.calc.setDebug(self.settings['debug'])

    def is_analytic_null_estimator(self):
        return True

    def get_analytic_distribution(self, **data):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            data : dict of numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per the estimate method for this estimator.

        Returns:
            Java object
                JIDT calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        self.estimate(**data)
        return self.calc.computeSignificance()

    def estimate_surrogates_analytic(self, n_perm=200, **data):
        """Estimate the surrogate distribution analytically.
        This method must be implemented because this class'
        is_analytic_null_estimator() method returns true

        Args:
            n_perms : number of permutations (default 200)
            data : array of numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per estimate_mult for this estimator.

        Returns:
            float | numpy array
                n_perm surrogates of the average MI/CMI/TE over all samples
                under the null hypothesis of no relationship between var1 and
                var2 (in the context of conditional)
        """
        return common_estimate_surrogates_analytic(self, n_perm, **data)


class JidtKraskovCMI(JidtKraskov):
    """Calculate conditional mutual inform with JIDT's Kraskov implementation.

    Calculate the conditional mutual information (CMI) between three variables.
    Call JIDT via jpype and use the Kraskov 1 estimator. If no conditional is
    given (is None), the function returns the mutual information between var1
    and var2. References:

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict [optional]
            set estimator parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings=None):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                     ConditionalMutualInfoCalculatorMultiVariateKraskov1)
        super().__init__(CalcClass, settings)

    def estimate(self, var1, var2, conditional=None):
        """Estimate conditional mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
        """
        # Return MI if no conditional was provided.
        if conditional is None:
            est_mi = JidtKraskovMI(self.settings)
            return est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Check if variable realisations are passed as 1D or 2D arrays and have
        # equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        cond = self._ensure_two_dim_input(conditional)

        assert(var1.shape[0] == var2.shape[0]), (
            'Unequal number of observations (var1: {0}, var2: {1}).'.format(
                var1.shape[0], var2.shape[0]))
        assert(var1.shape[0] == cond.shape[0]), (
            'Unequal number of observations (var1: {0}, cond: {1}).'.format(
                var1.shape[0], cond.shape[0]))

        self.calc.initialise(var1.shape[1], var2.shape[1], cond.shape[1])
        self.calc.setObservations(var1, var2, cond)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


class JidtDiscreteCMI(JidtDiscrete):
    """Calculate CMI with JIDT's implementation for discrete variables.

    Calculate the conditional mutual information between two variables given
    the third. Call JIDT via jpype and use the discrete estimator.

    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default='false')
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'discretise_method' - if and how to discretise incoming
              continuous variables to discrete values, can be 'max_ent' for
              maximum entropy binning, 'equal' for equal size bins, and 'none'
              if no binning is required (default='none')
            - 'num_discrete_bins' - number of discrete bins/levels or the base
              of each dimension of the discrete variables (default=2). If set,
              this parameter overwrites/sets 'alph1', 'alph2' and 'alphc'
            - 'alph1' - number of discrete bins/levels for var1 (default=2, or
              the value set for 'num_discrete_bins')
            - 'alph2' - number of discrete bins/levels for var2 (default=2, or
              the value set for 'num_discrete_bins')
            - 'alphc' - number of discrete bins/levels for conditional
              (default=2, or the value set for 'num_discrete_bins')
    """

    def __init__(self, settings=None):
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretisation if provided, otherwise assume
        # binary variables.
        super().__init__(settings)
        try:
            num_discrete_bins = int(self.settings['num_discrete_bins'])
            self.settings['alph1'] = num_discrete_bins
            self.settings['alph2'] = num_discrete_bins
            self.settings['alphc'] = num_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        self.settings.setdefault('alph1', int(2))
        self.settings.setdefault('alph2', int(2))
        self.settings.setdefault('alphc', int(2))

        # Start JAVA virtual machine and create JAVA object. Add JAVA object to
        # instance, the discrete estimator requires the variable dimensions
        # upon instantiation.
        self._start_jvm()
        self.CalcClass = (jp.JPackage('infodynamics.measures.discrete').
                          ConditionalMutualInformationCalculatorDiscrete)

    def estimate(self, var1, var2, conditional=None, return_calc=False):
        """Estimate conditional mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2
            return_calc : boolean
                return the calculator used here as well as the numeric
                calculated value(s)

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
            Java object
                JIDT calculator that was used here. Only returned if
                return_calc was set.

        """
        # Calculate an MI if no conditional was provided
        if (conditional is None) or (self.settings['alphc'] == 0):
            est = JidtDiscreteMI(self.settings)
            # Return value will be just the estimate if return_calc is False,
            #  or estimate plus the JIDT MI calculator if return_calc is True:
            return est.estimate(var1, var2, return_calc)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Check and remember the no. dimensions for each variable before
        # collapsing them into univariate arrays later.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)
        var1_dim = var1.shape[1]
        var2_dim = var2.shape[1]
        cond_dim = conditional.shape[1]

        # Discretise if requested.
        var1, var2, conditional = self._discretise_vars(var1, var2,
                                                        conditional)

        # Then collapse any mulitvariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])
        conditional = utils.combine_discrete_dimensions(conditional,
                                                        self.settings['alphc'])

        # We have a non-trivial conditional, so make a proper conditional MI
        # calculation
        calc = self.CalcClass(int(np.power(self.settings['alph1'], var1_dim)),
                              int(np.power(self.settings['alph2'], var2_dim)),
                              int(np.power(self.settings['alphc'], cond_dim)))
        calc.setDebug(self.settings['debug'])
        calc.initialise()
        # Unfortunately no faster way to pass numpy arrays in than this list
        # conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(var1.tolist()),
                             jp.JArray(jp.JInt, 1)(var2.tolist()),
                             jp.JArray(jp.JInt, 1)(conditional.tolist()))
        if self.settings['local_values']:
            result = np.array(calc.computeLocalFromPreviousObservations(
                jp.JArray(jp.JInt, 1)(var1.tolist()),
                jp.JArray(jp.JInt, 1)(var2.tolist()),
                jp.JArray(jp.JInt, 1)(conditional.tolist())
                ))
        else:
            result = calc.computeAverageLocalOfObservations()
        if return_calc:
            return (result, calc)
        else:
            return result

    def get_analytic_distribution(self, var1, var2, conditional=None):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            Java object
                JIDT calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        (est, jidt_calc) = self.estimate(var1, var2, conditional, True)
        return jidt_calc.computeSignificance()


class JidtDiscreteMI(JidtDiscrete):
    """Calculate MI with JIDT's discrete-variable implementation.

    Calculate the mutual information (MI) between two variables. Call JIDT via
    jpype and use the discrete estimator.

    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'discretise_method' - if and how to discretise incoming
              continuous variables to discrete values, can be 'max_ent' for
              maximum entropy binning, 'equal' for equal size bins, and 'none'
              if no binning is required (default='none')
            - 'num_discrete_bins' - number of discrete bins/levels or the base
              of each dimension of the discrete variables (default=2). If set,
              this parameter overwrites/sets 'alph1' and 'alph2'
            - 'alph1' - number of discrete bins/levels for var1 (default=2, or
              the value set for 'num_discrete_bins')
            - 'alph2' - number of discrete bins/levels for var2 (default=2, or
              the value set for 'num_discrete_bins')
            - 'lag' - time difference in samples to calculate the lagged MI
              between processes (default=0)
    """

    def __init__(self, settings=None):
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretisation if provided, otherwise assume
        # binary variables.
        super().__init__(settings)
        self.settings.setdefault('lag', int(0))
        try:
            num_discrete_bins = int(self.settings['num_discrete_bins'])
            self.settings['alph1'] = num_discrete_bins
            self.settings['alph2'] = num_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        self.settings.setdefault('alph1', int(2))
        self.settings.setdefault('alph2', int(2))

        # Start JAVA virtual machine and create JAVA object. Add JAVA object to
        # instance, the discrete estimator requires the variable dimensions
        # upon instantiation.
        self._start_jvm()
        self.CalcClass = (jp.JPackage('infodynamics.measures.discrete').
                          MutualInformationCalculatorDiscrete)

    def estimate(self, var1, var2, return_calc=False):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            return_calc : boolean
                return the calculator used here as well as the numeric
                calculated value(s)

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
            Java object
                JIDT calculator that was used here. Only returned if
                return_calc was set.
        """
        # Check and remember the no. dimensions for each variable before
        # collapsing them into univariate arrays later.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        var1_dim = var1.shape[1]
        var2_dim = var2.shape[1]

        # Discretise variables if requested.
        var1, var2 = self._discretise_vars(var1, var2)

        # Then collapse any mulitvariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])

        # Initialise estimator
        max_base = int(max(np.power(self.settings['alph1'], var1_dim),
                           np.power(self.settings['alph2'], var2_dim)))
        calc = self.CalcClass(max_base, self.settings['lag'])
        calc.setDebug(self.settings['debug'])
        calc.initialise()

        # Unfortunately no faster way to pass numpy arrays in than this list
        # conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(var1.tolist()),
                             jp.JArray(jp.JInt, 1)(var2.tolist()))
        if self.settings['local_values']:
            result = np.array(calc.computeLocalFromPreviousObservations(
                jp.JArray(jp.JInt, 1)(var1.tolist()),
                jp.JArray(jp.JInt, 1)(var2.tolist())))
        else:
            result = calc.computeAverageLocalOfObservations()
        if return_calc:
            return (result, calc)
        else:
            return result

    def get_analytic_distribution(self, var1, var2):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            Java object
                JIDT calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        (est, jidt_calc) = self.estimate(var1, var2, True)
        return jidt_calc.computeSignificance()


class JidtKraskovMI(JidtKraskov):
    """Calculate mutual information with JIDT's Kraskov implementation.

    Calculate the mutual information between two variables. Call JIDT via jpype
    and use the Kraskov 1 estimator. References:

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
            - 'lag' - time difference in samples to calculate the lagged MI
              between processes (default=0)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the MI estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        MI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings=None):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                     MutualInfoCalculatorMultiVariateKraskov1)
        super().__init__(CalcClass, settings)

        # Get lag and shift second variable to account for a lag if requested
        self.settings.setdefault('lag', 0)

    def estimate(self, var1, var2):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        # Check if variable realisations are passed as 1D or 2D arrays
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)

        # Shift variables to calculate a lagged MI.
        if self.settings['lag'] > 0:
            var1 = var1[:-self.settings['lag'], :]
            var2 = var2[self.settings['lag']:, :]

        self.calc.initialise(var1.shape[1], var2.shape[1])
        self.calc.setObservations(var1, var2)

        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


class JidtKraskovAIS(JidtKraskov):
    """Calculate active information storage with JIDT's Kraskov implementation.

    Calculate active information storage (AIS) for some process using JIDT's
    implementation of the Kraskov type 1 estimator. AIS is defined as the
    mutual information between the processes' past state and current value.

    The past state needs to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a processes' past,
    tau describes the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    References:

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
            - 'history' - number of samples in the processes' past used as
              embedding
            - 'tau' - the processes' embedding delay (default=1)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the AIS estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        AIS estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings):
        # Check for history for AIS estimation.
        if type(settings) is not dict:
            raise TypeError('settings should be a dictionary.')
        try:
            settings['history']
        except KeyError:
            raise RuntimeError('No history was provided for AIS estimation.')
        settings.setdefault('tau', 1)
        assert type(settings['history']) is int, (
                                            'History has to be an integer.')
        assert type(settings['tau']) is int, ('Tau has to be an integer.')

        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                     ActiveInfoStorageCalculatorKraskov)
        super().__init__(CalcClass, settings)

    def estimate(self, process):
        """Estimate active information storage.

        Args:
            process : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]

        Returns:
            float | numpy array
                average AIS over all samples or local AIS for individual
                samples if 'local_values'=True
        """
        process = self._ensure_one_dim_input(process)

        self.calc.initialise(self.settings['history'], self.settings['tau'])
        self.calc.setObservations(process)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


class JidtDiscreteAIS(JidtDiscrete):
    """Calculate AIS with JIDT's discrete-variable implementation.

    Calculate the active information storage (AIS) for one process. Call JIDT
    via jpype and use the discrete estimator.

    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Args:
        settings : dict
            set estimator parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'discretise_method' - if and how to discretise incoming
              continuous variables to discrete values, can be 'max_ent' for
              maximum entropy binning, 'equal' for equal size bins, and 'none'
              if no binning is required (default='none')
            - 'num_discrete_bins' - number of discrete bins/levels or the base
              of each dimension of the discrete variables (default=2). If set,
              this parameter overwrites/sets 'alph'
            - 'history' - number of samples in the target's past used as
              embedding
            - 'alph' - number of discrete bins/levels for var1 (default=2 , or
              the value set for 'num_discrete_bins')
    """

    def __init__(self, settings):
        if type(settings) is not dict:
            raise TypeError('settings should be a dictionary.')
        try:
            settings['history']
        except KeyError:
            raise RuntimeError('No history was provided for AIS estimation.')
        assert type(settings['history']) is int, (
                                            'History has to be an integer.')

        # Get alphabet sizes and check if discretisation is requested
        try:
            num_discrete_bins = int(settings['num_discrete_bins'])
            settings['alph'] = num_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        settings.setdefault('alph', int(2))

        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        self.CalcClass = (jp.JPackage('infodynamics.measures.discrete').
                          ActiveInformationCalculatorDiscrete)
        super().__init__(settings)

    def estimate(self, process, return_calc=False):
        """Estimate active information storage.

        Args:
            process : numpy array
                realisations as either a 2D numpy array where array dimensions
                represent [realisations x variable dimension] or a 1D array
                representing [realisations], array type can be float (requires
                discretisation) or int
            return_calc : boolean
                return the calculator used here as well as the numeric
                calculated value(s)

        Returns:
            float | numpy array
                average AIS over all samples or local AIS for individual
                samples if 'local_values'=True
            Java object
                JIDT calculator that was used here. Only returned if
                return_calc was set.
        """
        process = self._ensure_one_dim_input(process)

        # Now discretise if required
        if self.settings['discretise_method'] == 'none':
            assert issubclass(process.dtype.type, np.integer), (
                'Process is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert min(process) >= 0, 'Minimum of process is smaller than 0.'
            assert max(process) < self.settings['alph'], (
                'Maximum of process is larger than the alphabet size.')
            if self.settings['alph'] < np.unique(process).shape[0]:
                raise RuntimeError('The process'' alphabet size does not match'
                                   ' the no. unique elements in the process.')
        elif self.settings['discretise_method'] == 'equal':
            process = utils.discretise(process, self.settings['alph'])
        elif self.settings['discretise_method'] == 'max_ent':
            process = utils.discretise_max_ent(process, self.settings['alph'])
        else:
            pass  # don't discretise at all, assume data to be discrete

        # And finally make the TE calculation:
        calc = self.CalcClass(self.settings['alph'], self.settings['history'])
        calc.initialise()
        # Unfortunately no faster way to pass numpy arrays in than this list
        # conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(process.tolist()))
        if self.settings['local_values']:
            result = np.array(calc.computeLocalFromPreviousObservations(
                                    jp.JArray(jp.JInt, 1)(process.tolist())))
        else:
            result = calc.computeAverageLocalOfObservations()
        if return_calc:
            return (result, calc)
        else:
            return result

    def get_analytic_distribution(self, process):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            process : numpy array
                realisations as either a 2D numpy array where array dimensions
                represent [realisations x variable dimension] or a 1D array
                representing [realisations], array type can be float (requires
                discretisation) or int

        Returns:
            Java object
                JIDT calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        (est, jidt_calc) = self.estimate(process, True)
        return jidt_calc.computeSignificance()


class JidtGaussianAIS(JidtGaussian):
    """Calculate active information storage with JIDT's Gaussian implementation.

    Calculate active information storage (AIS) for some process using JIDT's
    implementation of the Gaussian estimator. AIS is defined as the
    mutual information between the processes' past state and current value.

    The past state needs to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a processes' past,
    tau describes the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    References:

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
            - 'history' - number of samples in the processes' past used as
              embedding
            - 'tau' - the processes' embedding delay (default=1)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the AIS estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        AIS estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings):
        # Check for history for AIS estimation.
        if type(settings) is not dict:
            raise TypeError('settings should be a dictionary.')
        try:
            settings['history']
        except KeyError:
            raise RuntimeError('No history was provided for AIS estimation.')
        settings.setdefault('tau', 1)
        assert type(settings['history']) is int, (
                                            'History has to be an integer.')
        assert type(settings['tau']) is int, ('Tau has to be an integer.')

        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.gaussian').
                     ActiveInfoStorageCalculatorGaussian)
        super().__init__(CalcClass, settings)

    def estimate(self, process):
        """Estimate active information storage.

        Args:
            process : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]

        Returns:
            float | numpy array
                average AIS over all samples or local AIS for individual
                samples if 'local_values'=True
        """
        process = self._ensure_one_dim_input(process)
        self.calc.initialise(self.settings['history'], self.settings['tau'])
        self.calc.setObservations(process)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


class JidtGaussianMI(JidtGaussian):
    """Calculate mutual information with JIDT's Gaussian implementation.

    Calculate the mutual information between two variables. Call JIDT via jpype
    and use the Gaussian estimator. References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
            - 'lag' - time difference in samples to calculate the lagged MI
              between processes (default=0)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the MI estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        MI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings=None):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.gaussian').
                     MutualInfoCalculatorMultiVariateGaussian)
        super().__init__(CalcClass, settings)

        # Add lag between input variables. Setting the lag in JIDT didn't work,
        # shift variables when calling the estimate method instead.
        self.settings.setdefault('lag', int(0))
        # self.calc.setProperty('PROP_TIME_DIFF', str(self.settings['lag']))

    def estimate(self, var1, var2):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)

        # Shift variables to calculate a lagged MI.
        if self.settings['lag'] > 0:
            var1 = var1[:-self.settings['lag'], :]
            var2 = var2[self.settings['lag']:, :]

        self.calc.initialise(var1.shape[1], var2.shape[1])
        self.calc.setObservations(var1, var2)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


class JidtGaussianCMI(JidtGaussian):
    """Calculate conditional mutual infor with JIDT's Gaussian implementation.

    Computes the differential conditional mutual information of two
    multivariate sets of observations, conditioned on another, assuming that
    the probability distribution function for these observations is a
    multivariate Gaussian distribution.
    Call JIDT via jpype and use
    ConditionalMutualInfoCalculatorMultiVariateGaussian estimator.
    If no conditional is given (is None), the function returns the mutual
    information between var1 and var2.

    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings=None):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.gaussian').
                     ConditionalMutualInfoCalculatorMultiVariateGaussian)
        super().__init__(CalcClass, settings)
        self.est_mi = None

    def estimate(self, var1, var2, conditional=None):
        """Estimate conditional mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
        """
        # Return MI if no conditioning variable was provided.
        if conditional is None:
            if (self.est_mi is None):
                self.est_mi = JidtGaussianMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        cond = self._ensure_two_dim_input(conditional)

        assert(var1.shape[0] == var2.shape[0]), (
            'Unequal number of observations (var1: {0}, var2: {1}).'.format(
                var1.shape[0], var2.shape[0]))
        assert(var1.shape[0] == cond.shape[0]), (
            'Unequal number of observations (var1: {0}, cond: {1}).'.format(
                var1.shape[0], cond.shape[0]))

        self.calc.initialise(var1.shape[1], var2.shape[1], cond.shape[1])
        self.calc.setObservations(var1, var2, conditional)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()

    def get_analytic_distribution(self, var1, var2, conditional=None):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            Java object
                JIDT calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        self.estimate(var1, var2, conditional)
        if (conditional is None):
            return self.est_mi.calc.computeSignificance()
        else:
            return self.calc.computeSignificance()


class JidtKraskovTE(JidtKraskov):
    """Calculate transfer entropy with JIDT's Kraskov implementation.

    Calculate transfer entropy between a source and a target variable using
    JIDT's implementation of the Kraskov type 1 estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    Past states need to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a variable's past,
    tau descrices the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    References:

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
            - 'history_target' - number of samples in the target's past used as
              embedding
            - 'history_source' - number of samples in the source's past used as
              embedding (default=same as the target history)
            - 'tau_source' - source's embedding delay (default=1)
            - 'tau_target' - target's embedding delay (default=1)
            - 'source_target_delay' - information transfer delay between source
              and target (default=1)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings):
        # Start JAVA virtual machine.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                     TransferEntropyCalculatorKraskov)
        super().__init__(CalcClass, settings)

        # Get embedding and delay parameters.
        self._set_te_defaults()

    def estimate(self, source, target):
        """Estimate transfer entropy from a source to a target variable.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of target variable (similar to var1)

        Returns:
            float | numpy array
                average TE over all samples or local TE for individual
                samples if 'local_values'=True
        """
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        self.calc.initialise(self.settings['history_target'],
                             self.settings['tau_target'],
                             self.settings['history_source'],
                             self.settings['tau_source'],
                             self.settings['source_target_delay'])
        self.calc.setObservations(source, target)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


class JidtDiscreteTE(JidtDiscrete):
    """Calculate TE with JIDT's implementation for discrete variables.

    Calculate the transfer entropy between two time series processes.
    Call JIDT via jpype and use the discrete estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    References:

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'discretise_method' - if and how to discretise incoming
              continuous variables to discrete values, can be 'max_ent' for
              maximum entropy binning, 'equal' for equal size bins, and 'none'
              if no binning is required (default='none')
            - 'num_discrete_bins' - number of discrete bins/levels or the base
              of each dimension of the discrete variables (default=2). If set,
              this parameter overwrites/sets 'alph1' and 'alph2'
            - 'alph1' - number of discrete bins/levels for source
              (default=2, or the value set for 'num_discrete_bins')
            - 'alph2' - number of discrete bins/levels for target
              (default=2, or the value set for 'num_discrete_bins')
            - 'history_target' - number of samples in the target's past used as
              embedding
            - 'history_source' - number of samples in the source's past used as
              embedding (default=same as the target history)
            - 'tau_source' - source's embedding delay (default=1)
            - 'tau_target' - target's embedding delay (default=1)
            - 'source_target_delay' - information transfer delay between source
              and target (default=1)
    """

    def __init__(self, settings):
        super().__init__(settings)

        # Get embedding and delay parameters.
        self._set_te_defaults()

        # Get alphabet sizes and check if discretisation is requested. Try to
        # overwrite alphabet sizes with number of bins.
        try:
            num_discrete_bins = int(settings['num_discrete_bins'])
            settings['alph1'] = num_discrete_bins
            settings['alph2'] = num_discrete_bins
        except KeyError:
            # do nothing and set alphabet sizes to default below
            pass
        self.settings.setdefault('alph1', int(2))
        self.settings.setdefault('alph2', int(2))

        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        self.CalcClass = (jp.JPackage('infodynamics.measures.discrete').
                          TransferEntropyCalculatorDiscrete)

    def estimate(self, source, target, return_calc=False):
        """Estimate transfer entropy from a source to a target variable.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            target : numpy array
                realisations of target variable (similar to var1)
            return_calc : boolean
                return the calculator used here as well as the numeric
                calculated value(s)

        Returns:
            float | numpy array
                average TE over all samples or local TE for individual
                samples if 'local_values'=True
            Java object
                JIDT calculator that was used here. Only returned if
                return_calc was set.
        """
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        # Discretise variables if requested.
        source, target = self._discretise_vars(source, target)

        # And finally make the TE calculation:
        max_base = max(self.settings['alph1'], self.settings['alph2'])
        calc = self.CalcClass(max_base,
                              self.settings['history_target'],
                              self.settings['tau_target'],
                              self.settings['history_source'],
                              self.settings['tau_source'],
                              self.settings['source_target_delay'])
        calc.initialise()
        # Unfortunately no faster way to pass numpy arrays in than this list
        # conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(source.tolist()),
                             jp.JArray(jp.JInt, 1)(target.tolist()))
        if self.settings['local_values']:
            result = np.array(calc.computeLocalFromPreviousObservations(
                jp.JArray(jp.JInt, 1)(source.tolist()),
                jp.JArray(jp.JInt, 1)(target.tolist())))
        else:
            result = calc.computeAverageLocalOfObservations()
        if return_calc:
            return (result, calc)
        else:
            return result

    def get_analytic_distribution(self, source, target):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            target : numpy array
                realisations of target variable (similar to var1)

        Returns:
            Java object
                JIDT calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        (est, jidt_calc) = self.estimate(source, target, True)
        return jidt_calc.computeSignificance()


class JidtGaussianTE(JidtGaussian):
    """Calculate transfer entropy with JIDT's Gaussian implementation.

    Calculate transfer entropy between a source and a target variable using
    JIDT's implementation of the Gaussian estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    Past states need to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a variable's past,
    tau descrices the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    References:

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    Args:
        settings : dict
            sets estimation parameters:

            - 'debug' - return debug information when calling JIDT.
              (Boolean, default=False)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='0')
            - 'noise_level' - random noise added to the data (default='1e-8')
            - 'num_threads' - number of threads used for estimation
              (default='USE_ALL', not that this uses *all* available threads
              on the current machine)
            - 'history_target' - number of samples in the target's past used as
              embedding
            - 'history_source' - number of samples in the source's past used as
              embedding (default=same as the target history)
            - 'tau_source' - source's embedding delay (default=1)
            - 'tau_target' - target's embedding delay (default=1)
            - 'source_target_delay' - information transfer delay between source
              and target (default=1)

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI estimator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """

    def __init__(self, settings):
        # Start JAVA virtual machine and create JAVA object.
        self._start_jvm()
        CalcClass = (jp.JPackage('infodynamics.measures.continuous.gaussian').
                     TransferEntropyCalculatorGaussian)
        super().__init__(CalcClass, settings)

        # Get embedding and delay parameters.
        self._set_te_defaults()

    def estimate(self, source, target):
        """Estimate transfer entropy from a source to a target variable.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of target variable (similar to var1)

        Returns:
            float | numpy array
                average TE over all samples or local TE for individual
                samples if 'local_values'=True
        """
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)
        self.calc.initialise(self.settings['history_target'],
                             self.settings['tau_target'],
                             self.settings['history_source'],
                             self.settings['tau_source'],
                             self.settings['source_target_delay'])
        self.calc.setObservations(source, target)
        if self.settings['local_values']:
            return np.array(self.calc.computeLocalOfPreviousObservations())
        else:
            return self.calc.computeAverageLocalOfObservations()


def common_estimate_surrogates_analytic(estimator, n_perm=200, **data):
    """Estimate the surrogate distribution analytically for JidtEstimator.

    Estimate the surrogate distribution analytically for a JidtEstimator
    which is_analytic_null_estimator(), by sampling estimates at random
    p-values in the analytic distribution.

    Args:
        estimator : a JidtEstimator object, which returns True to a call to
            its is_analytic_null_estimator() method
        n_perms : number of permutations (default 200)
        data : array of numpy arrays
            realisations of random variables required for the calculation
            (varies between estimators, e.g. 2 variables for MI, 3 for CMI)

    Returns:
        float | numpy array
            n_perm surrogates of the average MI/CMI/TE over all samples
            under the null hypothesis of no relationship between var1 and
            var2 (in the context of conditional)
    """
    # Compute the statistical significance of the estimate to get an
    #  AnalyticMeasurementDistribution object:
    analytic_distribution = estimator.get_analytic_distribution(**data)
    # Then compute surrogates at n_perm random p-values
    surrogate_estimates = np.empty(n_perm)
    for perm in range(n_perm):
        surrogate_estimates[perm] = \
            analytic_distribution.computeEstimateForGivenPValue(
                np.random.random())
    return surrogate_estimates
