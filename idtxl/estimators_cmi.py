"""Provide CMI estimators for the Estimator_cmi class.

This module exports methods for conditional mutual information (CMI) estimation
in the Estimator_cmi class.

"""
from pkg_resources import resource_filename
import numpy as np
import math
from scipy.special import digamma
from . import neighbour_search_opencl as nsocl
from . import idtxl_exceptions as ex
from . import idtxl_utils as utils
try:
    import jpype as jp
except ImportError:
    ex.jpype_missing('Jpype is not available on this system. To use '
                     'JAVA/JIDT-powered CMI estimation install it from '
                     'https://pypi.python.org/pypi/JPype1')

VERBOSE = False


def is_parallel(estimator_name):
    """Check if estimator can estimate CMI for multiple chunks in parallel."""
    # To add a new parallel estimator, add estimator name to the following
    # dictionary and set the value to True.
    parallel_estimators = {'opencl_kraskov': True,
                           'jidt_kraskov': False,
                           'jidt_discrete': False}
    try:
        return parallel_estimators[estimator_name]
    except KeyError:
        print('Unknown estimator name, assuming estimator to be serial.')
        return False


def opencl_kraskov(self, var1, var2, conditional=None, n_chunks=1, opts=None):
    """Calculate conditional mutual infor using opencl Kraskov implementation.

    Calculate the conditional mutual information between three variables using
    an opencl-based Kraskov type 1 estimator. References:

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
        var1 : numpy array
            realisations of the first random variable, where dimensions are
            realisations x variable dimension
        var2: numpy array
            realisations of the second random variable
        conditional : numpy array [optional]
            realisations of the random variable for conditioning, if no
            conditional is provided, return MI between var1 and var2
        n_chunks : int [optional]
            number of data sets or chunks (default=1)
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'gpuid' - device ID (default=0)

    Returns:
        float
            conditional mutual information

    Note:
        The Theiler window ignores trial boundaries. The CMI estimator does add
        noise to the data as a default. To make analysis runs replicable set
        noise_level to 0.
    """
    if opts is None:
        opts = {}
    elif type(opts) is not dict:
        raise TypeError('Opts should be a dictionary.')

    # Get defaults for estimator options
    kraskov_k = int(opts.get('kraskov_k', 4))
    theiler_t = int(opts.get('theiler_t', 0))  # TODO necessary?
    noise_level = np.float32(opts.get('noise_level', 1e-8))
    gpuid = int(opts.get('gpuid', 0))

    nchunkspergpu = n_chunks  # TODO is there a case where it makes sense to
                              # have these two distinct parameters?
    assert type(nchunkspergpu) is int, 'No chunks per GPU must be an int.'

    # If no conditional is passed, compute and return the mi:
    # this code is a copy of the one in estimatos_mi look there for comments
    if conditional is None:
        if VERBOSE:
            print('no conditional variable - falling back to MI estimation')
        var1 += np.random.normal(scale=noise_level, size=var1.shape)
        var2 += np.random.normal(scale=noise_level, size=var2.shape)
        pointset_full_space = np.hstack((var1, var2))
        pointset_full_space = pointset_full_space.astype('float32')
        n_dim_full = pointset_full_space.shape[1]
        var1 = var1.astype('float32')
        n_dim_var1 = var1.shape[1]
        var2 = var2.astype('float32')
        n_dim_var2 = var2.shape[1]
        signallengthpergpu = pointset_full_space.shape[0]
        assert signallengthpergpu % nchunkspergpu == 0, (
                'signal length {0} can not be divided by no. chunks {1}'
                .format(signallengthpergpu, nchunkspergpu))
        chunksize = int(signallengthpergpu / nchunkspergpu)  # TODO check for integer result
        if VERBOSE:
            print('no. points: {0}, chunksize: {1}, nchunks: {2}'.format(
                                signallengthpergpu, chunksize, nchunkspergpu))
        indexes, distances = nsocl.knn_search(pointset_full_space,
                                              n_dim_full,
                                              kraskov_k,
                                              theiler_t,
                                              nchunkspergpu,
                                              gpuid)
        radii = distances[distances.shape[0] - 1, :]
        count_var1 = nsocl.range_search(var1, n_dim_var1, radii, theiler_t,
                                        nchunkspergpu, gpuid)
        count_var2 = nsocl.range_search(var2, n_dim_var2, radii, theiler_t,
                                        nchunkspergpu, gpuid)
        cmi_array = -np.inf * np.ones(nchunkspergpu).astype('float64')
        for chunknum in range(0, nchunkspergpu):
            mi = (digamma(kraskov_k) + digamma(chunksize) -
                  np.mean(digamma(count_var1[chunknum * chunksize:
                                             (chunknum + 1) * chunksize] + 1) +
                          digamma(count_var2[chunknum * chunksize:
                                             (chunknum + 1) * chunksize] + 1)))
            cmi_array[chunknum] = mi
    else:
        var1 += np.random.normal(size=var1.shape) * noise_level
        var2 += np.random.normal(size=var2.shape) * noise_level
        conditional += np.random.normal(size=conditional.shape) * noise_level

        # Build pointsets (note that we assume that pointsets are given in
        # IDTxl convention.
        # 1. full space
        pointset_full_space = np.hstack((var1, var2, conditional))
        pointset_full_space = pointset_full_space.astype('float32')
        n_dim_full = pointset_full_space.shape[1]
        # 2. conditional variable only
        pointset_conditional = np.array(conditional)
        pointset_conditional = pointset_conditional.astype('float32')
        n_dim_conditional = pointset_conditional.shape[1]
        if VERBOSE:
            print("n_dim_conditional is: {0}".format(n_dim_conditional))
        # 3. pointset variable 1 and conditional
        pointset_var1_conditional = np.hstack((var1, conditional))
        pointset_var1_conditional = pointset_var1_conditional.astype('float32')
        n_dim_var1_conditional = pointset_var1_conditional.shape[1]
        # 4. pointset variable 2 and conditional
        pointset_var2_conditional = np.hstack((var2, conditional))
        pointset_var2_conditional = pointset_var2_conditional.astype('float32')
        n_dim_var2_conditional = pointset_var2_conditional.shape[1]

        signallengthpergpu = pointset_full_space.shape[0]

        if VERBOSE:
            print('working with signallength: {0}'.format(signallengthpergpu))
        chunksize = int(signallengthpergpu / nchunkspergpu)
        assert(signallengthpergpu % nchunkspergpu == 0), ('Chunksize is not an'
                                                          'interger value.')
#        assert(type(chunksize) is int), 'Chunksize has to be an integer.'
        indexes, distances = nsocl.knn_search(pointset_full_space,
                                              n_dim_full,
                                              kraskov_k,
                                              theiler_t,
                                              nchunkspergpu,
                                              gpuid)
        if VERBOSE:
            print('indexes: {0}\n\ndistances: {1}\n\nshape of distance matrix:'
                  ' {2}'.format(indexes, distances, distances.shape))

        # Define search radii as the distances to the kth (=last) neighbours
        radii = distances[distances.shape[0]-1, :]
        if VERBOSE:
            print('Radii: {0}'.format(radii))

        # Get neighbour counts in ranges.
        count_cond = nsocl.range_search(pointset_conditional,
                                        n_dim_conditional,
                                        radii, theiler_t,
                                        nchunkspergpu,
                                        gpuid)
        count_var1_cond = nsocl.range_search(pointset_var1_conditional,
                                             n_dim_var1_conditional,
                                             radii,
                                             theiler_t,
                                             nchunkspergpu,
                                             gpuid)
        count_var2_cond = nsocl.range_search(pointset_var2_conditional,
                                             n_dim_var2_conditional,
                                             radii,
                                             theiler_t,
                                             nchunkspergpu,
                                             gpuid)

        # Return the results, one cmi per chunk of data.
        cmi_array = -np.inf * np.ones(nchunkspergpu).astype('float64')
        for chunknum in range(0, nchunkspergpu):
            cmi = (digamma(kraskov_k) +
                   np.mean(digamma(count_cond[chunknum * chunksize:
                                              (chunknum + 1) * chunksize] + 1) -
                           digamma(count_var1_cond[chunknum * chunksize:
                                                   (chunknum + 1) * chunksize] +
                                   1) -
                           digamma(count_var2_cond[chunknum * chunksize:
                                                   (chunknum + 1) * chunksize] + 1)
                           ))
            cmi_array[chunknum] = cmi
    if VERBOSE:
        print('cmi array reads: {0} (n_chunks = {1})'.format(cmi_array,
                                                             nchunkspergpu))
    return cmi_array


def jidt_kraskov(self, var1, var2, conditional=None, opts=None):
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
    if opts is None:
        opts = {}
    elif type(opts) is not dict:
        raise TypeError('Opts should be a dictionary.')

    # Get defaults for estimator options
    kraskov_k = str(opts.get('kraskov_k', str(4)))
    normalise = str(opts.get('normalise', 'false'))
    theiler_t = str(opts.get('theiler_t', 0))  # TODO necessary?
    noise_level = str(opts.get('noise_level', 1e-8))
    num_threads = str(opts.get('num_threads', 'USE_ALL'))
    # debug = opts.get('debug', 'false')

    # Start JAVA virtual machine.
    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))

    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 ConditionalMutualInfoCalculatorMultiVariateKraskov1)
    calc = calcClass()
    calc.setProperty('NORMALISE', normalise)
    calc.setProperty('k', kraskov_k)
    calc.setProperty('DYN_CORR_EXCL', theiler_t)
    calc.setProperty('NOISE_LEVEL_TO_ADD', noise_level)
    calc.setProperty('NUM_THREADS', num_threads)
    # calc.setDebug(debug)

    if conditional is None:
        cond_dim = 0
    else:
        cond_dim = conditional.shape[1]
        assert(conditional.size != 0), 'Conditional Array is empty.'
    assert(var1.shape[0] == var2.shape[0]), 'Unequal number of observations.'
    calc.initialise(var1.shape[1], var2.shape[1], cond_dim)
    calc.setObservations(var1, var2, conditional)
    return calc.computeAverageLocalOfObservations()

def jidt_discrete(self, var1, var2, conditional, opts=None):
    """Calculate conditional mutual infor with JIDT's implementation for discrete
    variables"

    Calculate the conditional mutual information between two variables given
    the third. Call JIDT via jpype and use the discrete estimator.

    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
        var1 : numpy array (either of integers or doubles to be discretised)
            realisations of the first random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        var2 : numpy array (either of integers or doubles to be discretised)
            realisations of the second random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        conditional : numpy array (either of integers or doubles to be discretised)
            realisations of the conditional random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        opts : dict [optional]
            sets estimation parameters:
            - 'num_discrete_bins' - number of discrete bins/levels or the base of
                            each dimension of the discrete variables (default=2 for binary).
                            If this is set, then parameters 'alph1', 'alph2' and
                            'alphc' are all set to this value.
            - 'alph1' - number of discrete bins/levels for var1
                        (default=2 for binary, or the value set for 'num_discrete_bins').
            - 'alph2' - number of discrete bins/levels for var2
                        (default=2 for binary, or the value set for 'num_discrete_bins').
            - 'alphc' - number of discrete bins/levels for conditional
                        (default=2 for binary, or the value set for 'num_discrete_bins').
            - 'discretise_method' - if and how to discretise incoming continuous variables
                            to discrete values.
                            'max_ent' means to use a maximum entropy binning
                            'equal' means to use equal size bins
                            'none' means variables are already discrete (default='none')
            - 'debug' - set debug prints from the calculator on

    Returns:
        float
            conditional mutual information

    Note:
    """
    # Parse parameters:
    if opts is None:
        opts = {}
    try:
        alph1 = int(opts['alph1'])
    except KeyError:
        alph1 = int(2)
    try:
        alph2 = int(opts['alph2'])
    except KeyError:
        alph2 = int(2)
    try:
        alphc = int(opts['alphc'])
    except KeyError:
        alphc = int(2)
    try:
        num_discrete_bins = int(opts['num_discrete_bins'])
        alph1 = num_discrete_bins
        alph2 = num_discrete_bins
        alphc = num_discrete_bins
    except KeyError:
        # Do nothing, we don't need the parameter if it is not here
        pass
    try:
        discretise_method = opts['discretise_method']
    except KeyError:
        discretise_method = 'none'
    try:
        debug = opts['debug']
    except KeyError:
        debug = False

    # Work out the number of samples and dimensions for each variable, before
    #  we collapse all dimensions down:
    if len(var1.shape) > 1:
        # var1 is is multidimensional
        var1_dimensions = var1.shape[1]
    else:
        # var1 is unidimensional
        var1_dimensions = 1
    if len(var2.shape) > 1:
        # var2 is is multidimensional
        var2_dimensions = var2.shape[1]
    else:
        # var2 is unidimensional
        var2_dimensions = 1

    # Treat the conditional variable separately, as we're allowing
    #  this to be null and then calculating a MI instead:
    if (conditional is None) or (alphc == 0):
        alphc = 0
        varc_dimensions = 0
        # Then we will make a MI call here
    else:
        # Else we have a non-trivial conditional variable:
        assert(conditional.size != 0), 'Conditional Array is empty.'
        if len(conditional.shape) > 1:
            # conditional is is multidimensional
            varc_dimensions = conditional.shape[1]
        else:
            # conditional is unidimensional
            varc_dimensions = 1

    # Now discretise if required
    if (discretise_method == 'equal'):
        var1 = utils.discretise(var1, alph1)
        var2 = utils.discretise(var2, alph2)
        if (alphc > 0):
            conditional = utils.discretise(conditional, alphc)
    elif (discretise_method == 'max_ent'):
        var1 = utils.discretise_max_ent(var1, alph1)
        var2 = utils.discretise_max_ent(var2, alph2)
        if (alphc > 0):
            conditional = utils.discretise_max_ent(conditional, alphc)
    # Else don't discretise at all, assume it is already done

    # Then collapse any mulitvariates into univariate arrays:
    var1 = utils.combine_discrete_dimensions(var1, alph1)
    var2 = utils.combine_discrete_dimensions(var2, alph2)
    if (alphc > 0):
        conditional = utils.combine_discrete_dimensions(conditional, alphc)

    # And finally make the CMI calculation:
    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))
    if (alphc > 0):
        # We have a non-trivial conditional, so make a proper conditional MI calculation
        calcClass = (jp.JPackage('infodynamics.measures.discrete').
                 ConditionalMutualInformationCalculatorDiscrete)
        calc = calcClass(int(math.pow(alph1, var1_dimensions)),
                         int(math.pow(alph2, var2_dimensions)),
                         int(math.pow(alphc, varc_dimensions)))
        calc.setDebug(debug)
        calc.initialise()
        # Unfortunately no faster way to pass numpy arrays in than this list conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(var1.tolist()),
                         jp.JArray(jp.JInt, 1)(var2.tolist()),
                         jp.JArray(jp.JInt, 1)(conditional.tolist()))
        return calc.computeAverageLocalOfObservations()
    else:
        # We have no conditional, so make an MI calculation
        calcClass = (jp.JPackage('infodynamics.measures.discrete').
                 MutualInformationCalculatorDiscrete)
        calc = calcClass(int(max(math.pow(alph1, var1_dimensions),
                                 math.pow(alph2, var2_dimensions))), 0)
        calc.setDebug(debug)
        calc.initialise()
        # Unfortunately no faster way to pass numpy arrays in than this list conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(var1.tolist()),
                         jp.JArray(jp.JInt, 1)(var2.tolist()))
        return calc.computeAverageLocalOfObservations()
