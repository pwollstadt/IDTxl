"""Provide IDTxl utility functions."""
import pprint
import copy as cp
import numpy as np


def swap_chars(s, i_1, i_2):
    """Swap to characters in a string.

    Example:
        >>> print(swap_chars('heLlotHere', 2, 6))
        'heHlotLere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])


def print_dict(d, indent=4):
    """Use Python's pretty printer to print dictionaries to the console."""
    pp = pprint.PrettyPrinter(indent=indent)
    pp.pprint(d)


def standardise(a, dimension=0, df=1):
    """Z-standardise a numpy array along a given dimension.

    Standardise array along the axis defined in dimension using the denominator
    (N - df) for the calculation of the standard deviation.

    Args:
        a : numpy array
            data to be standardised
        dimension : int [optional]
            dimension along which array should be standardised
        df : int [optional]
            degrees of freedom for the denominator of the standard derivation

    Returns:
        numpy array
            standardised data
    """
    # Don't divide by standard devitation if process is constant.
    a_sd = a.std(axis=dimension, ddof=df)
    if np.isclose(a_sd, 0):
        return a - a.mean(axis=dimension)
    else:
        return (a - a.mean(axis=dimension)) / a_sd


def sort_descending(a):
    """Sort array in descending order."""
    # http://stackoverflow.com/questions/26984414/
    #       efficiently-sorting-a-numpy-array-in-descending-order
    return np.sort(a)[::-1]


def argsort_descending(a):
    """Sort array in descending order and return sortind indices."""
    # http://stackoverflow.com/questions/16486252/
    #       is-it-possible-to-use-argsort-in-descending-order
    return np.array(a).argsort()[::-1]


def remove_row(a, i):
    """Remove a row from a numpy array.

    This is faster than logical indexing ('25 times faster'), because it does
    not make copies, see
    http://scipy.github.io/old-wiki/pages/PerformanceTips

    Args:
        a : numpy array
            2-dimensional numpy array
        i : int
            row index to be removed
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
        a : numpy array
            2-dimensional numpy array
        i : int
            column index to be removed
    """
    b = np.empty((a.shape[0], a.shape[1] - 1))
    b[:, j:] = a[:, j+1:]
    b[:, :j] = a[:, :j]
    return b.astype(type(a[0][0]))


def autocorrelation(x):
    """Calculate autocorrelation of a vector."""
    pass


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


def separate_arrays(idx_all, idx_single, a):
    """Separate a single column from all other columns in a 2D-array.

    Return the separated single column and the remaining columns of a 2D-
    array.

    Args:
        idx_all : list<Object>
            list of variables indicating the full set
        idx_single : <Object>
            single variable indicating the column to be separated, variable
            must be contained in idx_all
        a : numpy array
            2D-array with the same length along axis 1 as idx_all
            (.shape[1] == len(idx_all))

    Returns:
        numpy array
            remaining columns in full array
        numpy array
            column at single index
    """
    assert(len(idx_all) == a.shape[1]), ('Length of full index list does '
                                         'not correspond to array size '
                                         'along 1st axis.')
    array_idx_single = idx_all.index(idx_single)
    real_single = np.expand_dims(a[:, array_idx_single], axis=1)
    real_remaining = remove_column(a, array_idx_single)
    return real_remaining, real_single


def combine_discrete_dimensions(a, numBins):
    """Combine multi-dimensional discrete variable into a single dimension.

    Combine all dimensions for a discrete variable down into a single
    dimensional value for each sample. This is done basically by multiplying
    each dimension by a different power of the base (numBins).

    Adapted from infodynamics.utils.MatrixUtils.computeCombinedValues() from
    JIDT by J.Lizier.

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
                raise ArithmeticError('Combination of numBins and number of dimensions of a leads to overflow in making unidimensional array')
        combined_values[t] = int(combined_value)
    return combined_values


def equal_dicts(dict_1, dict_2):
    """Test two dictionaries for equality."""
    if dict_1.keys() != dict_2.keys():
        return False
    for k in dict_1.keys():
        if dict_1[k] != dict_2[k]:
            return False
    return True


def conflicting_entries(dict_1, dict_2):
    """Test two dictionaries for unequal entries.

    Note that only keys that are present in both dicts are compared. If one
    dictionary contains an entry not present in the other dictionary, the
    test passes.
    """
    d1_keys = dict_1.keys()
    d2_keys = dict_2.keys()
    intersect_keys = set(d1_keys).intersection(set(d2_keys))
    for k in intersect_keys:
        if np.array(dict_1[k] != dict_2[k]).any():
            return True
    return False
