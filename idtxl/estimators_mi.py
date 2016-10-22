"""Provide MI estimators for the Estimator_mi class.

This module exports methods for MI estimation in the Estimator_mi class.

"""
from pkg_resources import resource_filename
import jpype as jp
import numpy as np
import math
from scipy.special import digamma
from . import neighbour_search_opencl as nsocl
from idtxl import idtxl_utils as utils


def opencl_kraskov(self, var1, var2, opts=None):
    """Calculate mutual information using an opencl Kraskov implementation.

    Calculate the mutual information between two variables using an
    opencl-based Kraskov type 1 estimator. Multiple MIs can be estimated in
    parallel, where each instance is called a 'chunk'. References:

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

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
        conditional : numpy array
            realisations of the random variable for conditioning
        opts : dict [optional]
            sets estimation parameters:
            'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            'theiler_t' - no. next temporal neighbours ignored in KNN and
            range searches (default='ACT', the autocorr. time of the target)
            'noise_level' - random noise added to the data (default=1e-8)
            'gpuid' - ID of the GPU device to be used (default=0)
            'nchunkspergpu' - number of chunks passed in the data (default=1)

    Returns:
        float
            conditional mutual information

    Note:
        The Theiler window ignores trial boundaries. The MI estimator does add
        noise to the data as a default. To make analysis runs replicable set
        noise_level to 0.
    """
    if opts is None:
        opts = {}
    try:
        kraskov_k = int(opts['kraskov_k'])
    except KeyError:
        kraskov_k = int(4)
    try:
        theiler_t = int(opts['theiler_t']) # TODO neccessary?
    except KeyError:
        theiler_t = int(0)
    try:
        noise_level = int(opts['noise_level'])
    except KeyError:
        noise_level = np.float32(1e-8)
    try:
        gpuid = int(opts['gpuid'])
    except KeyError:
        gpuid = int(0)
    try:
        nchunkspergpu = int(opts['nchunkspergpu'])
    except KeyError:
        nchunkspergpu = int(1)

    var1 += np.random.normal(scale=noise_level, size=var1.shape)
    var2 += np.random.normal(scale=noise_level, size=var2.shape)

    # build pointsets - Note we assume that pointsets are given in IDTxl conv.
    # also cast to single precision as required by opencl neighbour search
    # 1. full space
    pointset_full_space = np.hstack((var1, var2))
    pointset_full_space = pointset_full_space.astype('float32')
    n_dim_full = pointset_full_space.shape[1]
    var1 = var1.astype('float32')
    n_dim_var1 = var1.shape[1]
    var2 = var2.astype('float32')
    n_dim_var2 = var2.shape[1]

    signallengthpergpu = pointset_full_space.shape[0]
#    print("working with signallength: %i" %signallengthpergpu)
    chunksize = signallengthpergpu / nchunkspergpu # TODO check for integer result

    # KNN search in highest dimensional space.
    indexes, distances = nsocl.knn_search(pointset_full_space, n_dim_full,
                                          kraskov_k, theiler_t, nchunkspergpu,
                                          gpuid)
    radii = distances[distances.shape[0] - 1, :]

    # Get neighbour counts in the ranges defined by the k-th nearest
    # neighbour in the KNN search.
    count_var1 = nsocl.range_search(var1, n_dim_var1, radii, theiler_t,
                                    nchunkspergpu, gpuid)
    count_var2 = nsocl.range_search(var2, n_dim_var2, radii, theiler_t,
                                    nchunkspergpu, gpuid)

    # Return the results, one mi per chunk of data.
    mi_array = -np.inf * np.ones(nchunkspergpu).astype('float64')
    for chunknum in range(0, nchunkspergpu):
        mi = (digamma(kraskov_k) + digamma(chunksize) -
              np.mean(digamma(count_var1[chunknum * chunksize:(chunknum + 1) *
                                         chunksize] + 1) +
              digamma(count_var2[chunknum * chunksize:(chunknum + 1) *
                                 chunksize] + 1)))
        mi_array[chunknum] = mi
    return mi_array


def jidt_kraskov(self, var1, var2, opts=None):
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

    if opts is None:
        opts = {}
    try:
        kraskov_k = str(opts['kraskov_k'])
    except KeyError:
        kraskov_k = str(4)
    try:
        if opts['normalise']:
            normalise = 'true'
        else:
            normalise = 'false'
    except KeyError:
        normalise = 'false'

    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 MutualInfoCalculatorMultiVariateKraskov1)
    calc = calcClass()
    calc.setProperty('NORMALISE', normalise)
    calc.setProperty('k', kraskov_k)
    calc.initialise(var1.shape[1], var2.shape[1])
    calc.setObservations(var1, var2)
    return calc.computeAverageLocalOfObservations()


def jidt_discrete(self, var1, var2, opts=None):
    """Calculate mutual information with JIDT's discrete-variable implementation.

    Calculate the mutual information between two variables. Call JIDT via jpype
    and use the discrete estimator.
    
    References:

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is meant to be imported into the set_estimator module and used
    as a method in the Estimator_mi class.

    Args:
        self : instance of Estimator_mi
            function is supposed to be used as part of the Estimator_mi class
        var1 : numpy array (either of integers or doubles to be discretised)
            realisations of the first random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        var2 : numpy array (either of integers or doubles to be discretised)
            realisations of the second random variable.
            Can be multidimensional (i.e. multivariate) where dimensions of the
            array are realisations x variable dimension
        opts : dict [optional]
            sets estimation parameters:
            - 'num_discrete_bins' - number of discrete bins/levels or the base of 
                            each dimension of the discrete variables (default=2 for binary)
            - 'time_diff' - time difference across which to take MI from variable 1
                            to variable 2, i.e. lag from variable 1 to 2 (default=0)
            - 'discretise_method' - if and how to discretise incoming continuous variables
                            to discrete values.
                            'max_ent' means to use a maximum entropy binning
                            'equal' means to use equal size bins
                            'none' means variables are already discrete (default='none')
            - 'debug' - set debug prints from the calculator on

    Returns:
        float
            mutual information

    """
    
    # Parse parameters:
    if opts is None:
        opts = {}
    try:
        num_discrete_bins = int(opts['num_discrete_bins'])
    except KeyError:
        num_discrete_bins = int(2)
    try:
        time_diff = int(opts['time_diff'])
    except KeyError:
        time_diff = int(0)
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
    var1_samples = var1.shape[0]
    if len(var1.shape) > 1:
        # var1 is is multidimensional
        var1_dimensions = var1.shape[1]
    else:
        # var1 is unidimensional
        var1_dimensions = 1
    var2_samples = var2.shape[0]
    if len(var2.shape) > 1:
        # var2 is is multidimensional
        var2_dimensions = var2.shape[1]
    else:
        # var2 is unidimensional
        var2_dimensions = 1
    # Work out the base we need to use, after combining across all dimensions:
    max_base = int(math.pow(num_discrete_bins, max(var1_dimensions, var2_dimensions)))
    
    # Now discretise if required
    if (discretise_method == 'equal'):
        var1 = utils.discretise(var1, num_discrete_bins)
        var2 = utils.discretise(var2, num_discrete_bins)
    elif (discretise_method == 'max_ent'):
        var1 = utils.discretise_max_ent(var1, num_discrete_bins)
        var2 = utils.discretise_max_ent(var2, num_discrete_bins)
    # Else don't discretise at all, assume it is already done
    
    # Then collapse any mulitvariates into univariate arrays:
    var1 = utils.combine_discrete_dimensions(var1, num_discrete_bins)
    var2 = utils.combine_discrete_dimensions(var2, num_discrete_bins)
    
    # And finally make the MI calculation:
    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.discrete').
                 MutualInformationCalculatorDiscrete)
    calc = calcClass(max_base, time_diff)
    calc.setDebug(debug)
    calc.initialise()
    # Unfortunately no faster way to pass numpy arrays in than this list conversion
    calc.addObservations(jp.JArray(jp.JInt, 1)(var1.tolist()),
                         jp.JArray(jp.JInt, 1)(var2.tolist()))
    return calc.computeAverageLocalOfObservations()

