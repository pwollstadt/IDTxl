import pprint
import numpy as np

VERBOSE = True


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
    """Calculate autocorrelation of a vector."""
    # TODO check this, function taken from here:
    # http://stackoverflow.com/questions/14297012/
    #       estimate-autocorrelation-using-python
    # after Wikipedie:
    # https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    '''
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert n.allclose(r,
                      N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r / (variance * (n.arange(n, 0, -1)))
    '''
    return 3
