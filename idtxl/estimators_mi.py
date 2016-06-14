from pkg_resources import resource_filename
import jpype as jp
import numpy as np
from scipy.special import digamma
from . import neighbour_search_opencl as nsocl


def opencl_kraskov(self, var1, var2, opts=None):
    """Calculate mutual information using an opencl Kraskov implementation.

    Calculate the mutual information between two variables using an
    opencl-based Kraskov type 1 estimator. Multiple CMIs can be estimated in
    parallel, where each instance is called a 'chunk'. References:

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

    # Return the results, one cmi per chunk of data.
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
    as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
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
    elif type(opts) is not dict:
        raise TypeError('Opts should be a dictionary.')

    # Get defaults for estimator options
    kraskov_k = str(opts.get('kraskov_k', default=4))
    normalise = str(opts.get('normalise', default='false'))
    
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
