import jpype as jp
import numpy as np
import pyinfo
import neighbour_search_opencl as nsocl
from scipy.special import digamma


def opencl_kraskov(self, var1, var2, opts=None):
    """Calculate conditional mutual information with an opencl-based Kraskov
        type 1 estimator.

    Calculate the conditional mutual information between three variables.
    Uses the Kraskov type1 estimator.
    References:

        Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
        information. Physical review E, 69(6), 066138.

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self (Estimator_cmi): instance of Estimator_cmi
        var1: numpy array with realisations of the first random variable, where
            dimensions are realisations x variable dimension
        var2: numpy array with realisations of the second random variable
        conditional: numpy array with realisations of the random variable for
            conditioning
        opts : dict [optional]
            sets estimation parameters:
            'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            'theiler_t' - no. next temporal neighbours ignored in KNN and
            range searches (default='ACT', the autocorr. time of the target)
            'noise_level' - random noise added to the data (default=1e-8)

    Returns:
        conditional mutual information

    Note:
        The Theiler window ignores
        trial boundaries. The CMI estimator does add noise to the data as a
        default. To make analysis runs replicable set noise_level to 0.
    """
    if opts is None:
        opts = {}
    try:
        kraskov_k = int(opts['kraskov_k'])
    except KeyError:
        kraskov_k = int(4)
    try:
        theiler_t = int(opts['theiler_t']) # neccessary?
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
    chunksize =signallengthpergpu / nchunkspergpu # TODO check for integer result

    indexes, distances = nsocl.knn_search(pointset_full_space, n_dim_full,
                                          kraskov_k, theiler_t, nchunkspergpu,
                                          gpuid)

    radii = distances[distances.shape[0]-1,:]

    # get neighbour counts in ranges
    count_var1 = nsocl.range_search(var1, n_dim_var1, radii, theiler_t, nchunkspergpu, gpuid)
    count_var2 = nsocl.range_search(var2, n_dim_var2, radii, theiler_t, nchunkspergpu, gpuid)
    mi = digamma(kraskov_k) + digamma(chunksize) \
        - np.mean(digamma(count_var1 + 1) + digamma(count_var2 +1))
    return mi


# TODO this should take numpy arrays
def jidt_kraskov(self, var1, var2, knn):
    """Calculate mutual information with JIDT's Kraskov implementation."""

    jarLocation = 'infodynamics.jar'
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 MultiInfoCalculatorKraskov1)
    calc = calcClass()
    calc.setProperty('NORMALISE', 'true')
    calc.setProperty('k', str(knn))
    calc.initialise(len(var1[0])+len(var2[0]))
    calc.setObservations(jp.JArray(jp.JDouble, 2)(var1),
                         jp.JArray(jp.JDouble, 2)(var2))
    return calc.computeAverageLocalOfObservations()
