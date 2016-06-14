"""Provide CMI estimators for the Estimator_cmi class.

This module exports methods for CMI estimation in the Estimator_cmi class.

"""
from pkg_resources import resource_filename
import jpype as jp
import numpy as np
from scipy.special import digamma
from . import idtxl_utils as utils
from . import neighbour_search_opencl as nsocl
VERBOSE = False


def opencl_kraskov(self, var1, var2, conditional=None, opts=None):
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
        conditional : numpy array
            realisations of the random variable for conditioning
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)

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
    kraskov_k = int(opts.get('kraskov_k', default=4))
    theiler_t = int(opts.get('theiler_t', default=0)) # TODO necessary?
    noise_level = np.float32(opts.get('noise_level', default=1e-8))
    gpuid = int(opts.get('gpuid', default=0))
    nchunkspergpu = int(opts.get('nchunkspergpu', default=1))

# If no conditional is passed, compute and return the mi:
# this code is a copy of the one in estimatos_mi look there for comments
    if conditional == None:
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
        chunksize = signallengthpergpu / nchunkspergpu # TODO check for integer result
        indexes, distances = nsocl.knn_search(pointset_full_space, n_dim_full,
                                              kraskov_k, theiler_t, nchunkspergpu,
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
                                              (chunknum + 1) * chunksize] + 1)
                                              + digamma(count_var2[chunknum * chunksize:
                                              (chunknum + 1) * chunksize] + 1)))
            cmi_array[chunknum] = mi
    else:
        var1 += np.random.normal(size=var1.shape) * noise_level
        var2 += np.random.normal(size=var2.shape) * noise_level
        conditional += np.random.normal(size=conditional.shape) * noise_level

        # Build pointsets (note that we assume that pointsets are given in IDTxl
        # convention.
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
        chunksize = signallengthpergpu / nchunkspergpu
        assert(int(chunksize)  == chunksize), 'Chunksize is not an interger value.'
#        assert(type(chunksize) is int), 'Chunksize has to be an integer.'
        indexes, distances = nsocl.knn_search(pointset_full_space, n_dim_full,
                                              kraskov_k, theiler_t, nchunkspergpu,
                                              gpuid)
        if VERBOSE:
            print('indexes: {0}\n\ndistances: {1}\n\nshape of distance matrix: '
                  '{2}'.format(indexes, distances, distances.shape))

        # Define the search radii as the distances to the kth (=last) neighbours
        radii = distances[distances.shape[0]-1, :]
        if VERBOSE:
            print('Radii: {0}'.format(radii))

        # Get neighbour counts in ranges.
        count_cond = nsocl.range_search(pointset_conditional, n_dim_conditional,
                                        radii, theiler_t, nchunkspergpu, gpuid)
        count_var1_cond = nsocl.range_search(pointset_var1_conditional,
                                             n_dim_var1_conditional, radii,
                                             theiler_t, nchunkspergpu, gpuid)
        count_var2_cond = nsocl.range_search(pointset_var2_conditional,
                                             n_dim_var2_conditional, radii,
                                             theiler_t, nchunkspergpu, gpuid)

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
        print('cmi array reads: {0}'.format(cmi_array))
    return cmi_array


def jidt_kraskov(self, var1, var2, conditional, opts=None):
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
        conditional : numpy array
            realisations of the random variable for conditioning
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'num_threads' - no. CPU threads used for estimation
            (default='USE_ALL', this uses all available cores on the machine!)

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
    kraskov_k = str(opts.get('kraskov_k', default=4))
    normalise = str(opts.get('normalise', default='false'))
    theiler_t = int(opts.get('theiler_t', default=0)) # TODO necessary?
    noise_level = np.float32(opts.get('noise_level', default=1e-8))
    num_threads = str(opts.get('num_threads', default='USE_ALL'))
    debug = str(opts.get('debug', default= 'false'))
    
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
    calc.setDebug(debug)

    if conditional is None:
        cond_dim = 0
    else:
        cond_dim = conditional.shape[1]
        assert(conditional.size != 0), 'Conditional Array is empty.'
    calc.initialise(var1.shape[1], var2.shape[1], cond_dim)
    calc.setObservations(var1, var2, conditional)
    return calc.computeAverageLocalOfObservations()

#def jidt_discrete(self, var1, var2, conditional, alph1, alph2, alphc, opts=None):
#    """Calculate conditional mutual infor with JIDT's implementation for discrete
#    variables"
#
#    Calculate the conditional mutual information between three variables. Call
#    JIDT via jpype and use the discrete estimator.
#
#    References
#    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
#    studying the dynamics of complex systems. Front. Robot. AI, 1(11).
#
#    This function is ment to be imported into the set_estimator module and used
#    as a method in the Estimator_cmi class.
#
#    Args:
#        self : instance of Estimator_cmi
#            function is supposed to be used as part of the Estimator_cmi class
#        var1 : numpy array of integers
#            realisations of the first random variable, where dimensions are
#            realisations x variable dimension
#        var2 : numpy array of integers
#            realisations of the second random variable
#        conditional : numpy array on integers
#            realisations of the random variable for conditioning
#        alph1 : alphabet size of var1
#        alph2 : alphabet size of var2
#        alphc : alphabet size of conditional variable
#        opts : dict [optional]
#            sets estimation parameters:
#
#    Returns:
#        float
#            conditional mutual information
#
#    Note:
#    """
#    if opts is None:
#        opts = {}
#    try:
#        num_threads = str(opts['num_threads'])
#    except KeyError:
#        num_threads = 'USE_ALL'
#    try:
#        debug = opts['debug']
#    except KeyError:
#        debug = False
#    assert(var1.type = 'int....'), 'data received by discrete calculator are not integers'
#
#    jarLocation = resource_filename(__name__, 'infodynamics.jar')
#    if not jp.isJVMStarted():
#        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
#                    jarLocation))
#
#    calcClass = (jp.JPackage('infodynamics.measures.discrete.....').
#                 .....)
#    calc = calcClass()
#    calc.setProperty('NUM_THREADS', num_threads)
#    calc.setDebug(debug)
#
#    if conditional is None:
#        cond_dim = 0
#    else:
#        cond_dim = conditional.shape[1]
#        assert(conditional.size != 0), 'Conditional Array is empty.'
#    calc.initialise(var1.shape[1], var2.shape[1], cond_dim)
#    calc.setObservations(var1, var2, conditional)
#    return calc.computeAverageLocalOfObservations()
