"""Provide transfer entropy estimators for the Estimator_te class.

This module exports methods for transfer entropy (TE) estimation in the
Estimator_te class.

"""
from pkg_resources import resource_filename
import math
import numpy as np
from . import idtxl_exceptions as ex
from . import idtxl_utils as utils
try:
    import jpype as jp
except ImportError as err:
    ex.package_missing(err, 'Jpype is not available on this system. Install it'
                            ' from https://pypi.python.org/pypi/JPype1 to use '
                            'JAVA/JIDT-powered CMI estimation.')


def is_parallel(estimator_name):
    """Check if estimator can estimate CMI for multiple chunks in parallel."""
    parallel_estimators = {'opencl_kraskov': True,
                           'jidt_kraskov': False,
                           'jidt_discrete': False}
    try:
        return parallel_estimators[estimator_name]
    except KeyError:
        print('Unknown estimator name, assuming estimator to be serial.')
        return False


def jidt_kraskov(self, source, target, opts):
    """Calculate transfer entropy with JIDT's Kraskov implementation.

    Calculate transfer entropy between a source and a target variable using
    JIDT's implementation of the Kraskov type 1 estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    Past states need to be defined in the opts dictionary, where a past state
    is defined as a uniform embedding with parameters history and tau. The
    history describes the number of samples taken from a variable's past, tau
    descrices the embedding delay, i.e., the spacing between every two samples
    from the processes' past.

    References:

    Schreiber, T. (2000). Measuring information transfer. Physical Review
    Letters, 85(2), 461.

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
        source : numpy array
            realisations of the source variable
        target : numpy array
            realisations of the target variable
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'local_values' - return local TE instead of average TE
              (default=False)
            - 'history_target' - number of samples in the target's past to
              consider (mandatory to provide)
            - 'history_source' - number of samples in the source's past to
              consider (default=same as the target history)
            - 'tau_source' - source's embedding delay (default=1)
            - 'tau_target' - target's embedding delay (default=1)
            - 'source_target_delay' - information transfer delay between source
              and target (default=1)
            - 'debug' - set debug prints from the calculator on (default=False)

    Returns:
        float
            transfer entropy from source to target
        OR
        numpy array of floats
            local transfer entropy if local_values is set

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI calculator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """
    if type(opts) is not dict:
        raise TypeError('Opts should be a dictionary.')

    # Get histories.
    try:
        history_target = opts['history_target']
    except KeyError:
        raise RuntimeError('No history was provided for TE estimation.')
    history_source = opts.get('history_source', history_target)
    tau_target = opts.get('tau_target', 1)
    tau_source = opts.get('tau_source', 1)
    delay = opts.get('source_target_delay', 1)
    debug = opts.get('debug', False)

    # Get defaults for estimator options.
    kraskov_k = str(opts.get('kraskov_k', 4))
    normalise = str(opts.get('normalise', False)).lower()
    theiler_t = str(opts.get('theiler_t', 0))  # TODO necessary?
    noise_level = str(opts.get('noise_level', 1e-8))
    local_values = opts.get('local_values', False)

    # Start JAVA virtual machine.
    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                                                    jarLocation))
    # Estimate TE.
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 TransferEntropyCalculatorKraskov)
    calc = calcClass()
    calc.setDebug(debug)
    calc.setProperty('NORMALISE', normalise)
    calc.setProperty('k', kraskov_k)
    calc.setProperty('PROP_KRASKOV_ALG_NUM', str(1))
    calc.setProperty('NOISE_LEVEL_TO_ADD', noise_level)
    calc.setProperty('DYN_CORR_EXCL', theiler_t)
    calc.initialise(history_target, tau_target,
                    history_source, tau_source,
                    delay)
    calc.setObservations(source, target)
    if local_values:
        return np.array(calc.computeLocalOfPreviousObservations())
    else:
        return calc.computeAverageLocalOfObservations()


def jidt_discrete(self, source, target, opts):
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

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_te class.

    Args:
        self : instance of Estimator_te
            function is supposed to be used as part of the Estimator_te class
        source : numpy array (either of integers or doubles to be discretised)
            time series realisations of the first random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        target : numpy array (either of integers or doubles to be discretised)
            time series realisations of the second random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        opts : dict
            sets estimation parameters:

            - 'discretise_method' - if and how to discretise incoming
              continuous variables to discrete values.
              'max_ent' means to use a maximum entropy binning
              'equal' means to use equal size bins
              'none' means variables are already discrete (default='none')
            - 'num_discrete_bins' - number of discrete bins for discretisation
               if requested (default=2 for binary). If this is set, parameters
               'alph_source', 'alph_target' and 'alphc' are all set to
               'num_discrete_bins'
            - 'alph_source' - number of discrete bins/levels for source
              (default=2 for binary, or the value set for 'num_discrete_bins')
            - 'alph_target' - number of discrete bins/levels for target
              (default=2 for binary, or the value set for 'num_discrete_bins')
            - 'history_target' - number of samples in the target's past to
              consider (mandatory)
            - 'history_source' - number of samples in the source's past to
              consider (default=same as the target history)
            - 'tau_source' - source's embedding delay (default=1)
            - 'tau_target' - target's embedding delay (default=1)
            - 'source_target_delay' - information transfer delay between source
              and target (default=1)
            - 'debug' - set debug prints from the calculator on (default=False)

    Returns:
        float
            transfer entropy
    """
    if type(opts) is not dict:
        raise TypeError('Opts should be a dictionary.')

    # Get alphabet sizes and check if discretisation is requested
    discretise_method = opts.get('discretise_method', 'none')
    alph_source = int(opts.get('alph_source', 2))
    alph_target = int(opts.get('alph_target', 2))

    if (discretise_method == 'none'):
        if alph_source < np.unique(source).shape[0]:
            raise RuntimeError('The source''s alphabet size does not match the'
                               ' no. unique elements in the source array.')
        if alph_target < np.unique(target).shape[0]:
            raise RuntimeError('The target''s alphabet size does not match the'
                               ' no. unique elements in the target array.')
    else:  # get the number of bins if discretisation was requested
        try:
            num_discrete_bins = int(opts['num_discrete_bins'])
            alph_source = num_discrete_bins
            alph_target = num_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set above

    # Get embedding and delay parameters.
    try:
        history_target = opts['history_target']
    except KeyError:
        raise RuntimeError('No history was provided for TE estimation.')
    try:
        history_source = opts['history_source']
    except KeyError:
        history_source = history_target
    tau_target = opts.get('tau_target', 1)
    tau_source = opts.get('tau_source', 1)
    delay = opts.get('source_target_delay', 1)
    debug = opts.get('debug', False)

    # Work out the number of samples and dimensions for each variable, before
    #  we collapse all dimensions down:
    if len(source.shape) > 1:
        # source is is multidimensional
        source_dimensions = source.shape[1]
    else:
        # source is unidimensional
        source_dimensions = 1
    if len(target.shape) > 1:
        # target is is multidimensional
        target_dimensions = target.shape[1]
    else:
        # target is unidimensional
        target_dimensions = 1

    # Now discretise if required
    if (discretise_method == 'equal'):
        source = utils.discretise(source, alph_source)
        target = utils.discretise(target, alph_target)
    elif (discretise_method == 'max_ent'):
        source = utils.discretise_max_ent(source, alph_source)
        target = utils.discretise_max_ent(target, alph_target)
    else:
        pass  # don't discretise at all, assume data to be discrete

    # Then collapse any mulitvariates into univariate arrays:
    source = utils.combine_discrete_dimensions(source, alph_source)
    target = utils.combine_discrete_dimensions(target, alph_target)

    # And finally make the TE calculation:
    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.discrete').
                 TransferEntropyCalculatorDiscrete)
    calc = calcClass(int(max(math.pow(alph_source, source_dimensions),
                             math.pow(alph_target, target_dimensions))),
                     history_target, tau_target,
                     history_source, tau_source,
                     delay)
    calc.setDebug(debug)
    calc.initialise()
    # Unfortunately no faster way to pass numpy arrays in than this list conversion
    calc.addObservations(jp.JArray(jp.JInt, 1)(source.tolist()),
                         jp.JArray(jp.JInt, 1)(target.tolist()))
    return calc.computeAverageLocalOfObservations()
