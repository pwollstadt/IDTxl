import pprint
import numpy as np


def swap_chars(s, i_1, i_2):
    """Swap to characters in a string.

    Example:
        >>> print(swap_chars('hellothere', 2, 6))
        'hehlotlere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])


def print_dict(d, indent=4):
    """Use Python's pretty printer to print dictionaries to the console."""
    pp = pprint.PrettyPrinter(indent=indent)
    pp.pprint(d)


def standardise(a, dimension=0, df=1):
    """ Z-standardise a numpy array along a given dimension.

    Standardise array along the axis defined in dimension using the denominator
    (N - df) for the calculation of the standard deviation.

    Args:
        a : numpy array
            data to be standardised
        dimension : int [optional]
            dimension along which array should be standardised
        df : int
            degrees of freedom for the denominator of the standard derivation

    Returns:
        numpy array
            standardised data
    """
    a = (a - a.mean(axis=dimension)) / a.std(axis=dimension, ddof=df)
    return a


def sort_descending(a):
    """Sort array in descending order."""
    # http://stackoverflow.com/questions/26984414/
    #       efficiently-sorting-a-numpy-array-in-descending-order
    return np.sort(a)[::-1]


def argsort_descending(a):
    # http://stackoverflow.com/questions/16486252/
    #       is-it-possible-to-use-argsort-in-descending-order
    return np.array(a).argsort()[::-1]


def remove_row(a, i):
    """Remove a row from a numpy array.

    This is faster than logical indexing ('25 times faster'), because it does
    not make copies, see
    http://scipy.github.io/old-wiki/pages/PerformanceTips

    Args:
        a: 2-dimensional numpy array
        i: row index to be removed
    """
    b = np.empty((a.shape[0] - 1, a.shape[1]))
    b[i:, :] = a[i + 1:, :]
    b[:i, :] = a[:i, :]
    return b.astype(type(a[0][0]))


def remove_column(a, j):
    """Remove a column from a numpy array.

    This is faster than logical indexing ('25 times faster'), because it does
    not make copies, see
    http://scipy.github.io/old-wiki/pages/PerformanceTips

    Args:
        a: 2-dimensional numpy array
        i: column index to be removed
    """
    b = np.empty((a.shape[0], a.shape[1] - 1))
    b[:, j:] = a[:, j+1:]
    b[:, :j] = a[:, :j]
    return b.astype(type(a[0][0]))

def autocorrelation(x):
    # TODO check this, function taken from here:
    # http://stackoverflow.com/questions/14297012/
    #       estimate-autocorrelation-using-python
    # after Wikipedie:
    # https://en.wikipedia.org/wiki/Autocorrelation#Estimation
#==============================================================================
#
#     n = len(x)
#     variance = x.var()
#     x = x - x.mean()
#     r = np.correlate(x, x, mode = 'full')[-n:]
#     # assert n.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
#     result = r / (variance * (n.arange(n, 0, -1)))
#==============================================================================
    return 3


def discretise(a, numBins):
    """Discretise continuous data into discrete values (with 0 as lowest)
    by evenly partitioning the range of the data, one dimension at a time.
    Adapted from infodynamics.utils.MatrixUtils.discretise() from JIDT by J.Lizier
    
    Args:
        a : numpy array
            data to be discretised. Dimensions are
            realisations x variable dimension
        numBins : int
            number of discrete levels or bins to partition the data into

    Returns:
        numpy array
            discretised data
    """
    
    num_samples = a.shape[0]
    if (len(a.shape) == 1):
        # It's a unidimensional array
        discretised_values = np.zeros(num_samples, dtype=np.int_)
        theMin = a.min()
        theMax = a.max()
        binInterval = (theMax - theMin) / numBins
        for t in range(num_samples):
            discretised_values[t] = int((a[t] - theMin) / binInterval)
            if (discretised_values[t] == numBins):
                # This occurs for the maximum value; put it in the largest bin (base - 1)
                discretised_values[t] = discretised_values[t] - 1
        return discretised_values
    
    # Else, multivariate array
    num_dimensions = a.shape[1]
    discretised_values = np.zeros([num_samples, num_dimensions], dtype=np.int_)
    for v in range(a.shape[1]):
        # Bin dimension v:
        theMin = a[:,v].min()
        theMax = a[:,v].max()
        binInterval = (theMax - theMin) / numBins
        for t in range(num_samples):
            discretised_values[t,v] = int((a[t,v] - theMin) / binInterval)
            if (discretised_values[t,v] == numBins):
                # This occurs for the maximum value; put it in the largest bin (base - 1)
                discretised_values[t,v] = discretised_values[t,v] - 1
    return discretised_values


def discretise_max_ent(a, numBins):
    """Discretise continuous data into discrete values (with 0 as lowest)
    by making a maximum entropy partitioning, one dimension at a time.
    Adapted from infodynamics.utils.MatrixUtils.discretiseMaxEntropy() from JIDT by J.Lizier
    
    Args:
        a : numpy array
            data to be discretised. Dimensions are
            realisations x variable dimension
        numBins : int
            number of discrete levels or bins to partition the data into

    Returns:
        numpy array
            discretised data
    """
    
    num_samples = a.shape[0]
    if (len(a.shape) == 1):
        # It's a unidimensional array
        discretised_values = np.zeros(num_samples, dtype=np.int_)
        cuttoff_values = np.zeros(numBins)
        sorted_copy = np.sort(a)
        for bin in range(numBins):
            compartmentSize = int((bin+1)*(num_samples)/numBins)-1;
            cuttoff_values[bin] = sorted_copy[compartmentSize]
        for t in range(num_samples):
            for m in range(numBins):
                if (a[t] <= cuttoff_values[m]):
                    discretised_values[t] = m
                    break
        return discretised_values
    
    # Else, multivariate array
    num_dimensions = a.shape[1]
    discretised_values = np.zeros([num_samples, num_dimensions], dtype=np.int_)
    for v in range(num_dimensions):
        # Bin dimension v:
        cuttoff_values = np.zeros(numBins)
        sorted_copy = np.sort(a[:,v])
        for bin in range(numBins):
            compartmentSize = int((bin+1)*(num_samples)/numBins)-1;
            cuttoff_values[bin] = sorted_copy[compartmentSize]
        for t in range(num_samples):
            for m in range(numBins):
                if (a[t,v] <= cuttoff_values[m]):
                    discretised_values[t,v] = m
                    break
    return discretised_values


def combine_discrete_dimensions(a, numBins):
    """Combine all dimensions for a discrete variable down into a single
    dimensional value for each sample.
    This is done basically by multiplying each dimension
    by a different power of the base (numBins).
    Adapted from infodynamics.utils.MatrixUtils.computeCombinedValues() from JIDT by J.Lizier
    
    Args:
        a : numpy array
            data to be combined across all variable dimensions. Dimensions are
            realisations (samples) x variable dimension
        numBins : int
            number of discrete levels or bins for each variable dimension

    Returns:
        numpy array
            a univariate array -- one entry now for each sample,
            with all dimensions of the data now combined for that sample
    """
    if (len(a.shape) == 1):
        # It's already a unidimensional array
        return a
    
    # Else, 2D array assumed
    num_samples = a.shape[0]
    dimensions = a.shape[1]
    combined_values = np.zeros(num_samples, dtype=np.int_)
    for t in range(num_samples):
        combined_value = 0
        multiplier = 1
        for c in range(dimensions - 1, -1, -1):
            combined_value = combined_value + a[t][c] * multiplier
            multiplier = multiplier * numBins
            if multiplier <= 0:
                # Multiplier has overflown
                raise ArithmaticError('Combination of numBins and number of dimensions of a leads to overflow in making unidimensional array')
        combined_values[t] = int(combined_value)
    return combined_values

