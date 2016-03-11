import numpy as np


def standardize(A, dimension=0, dof=1):
    """ Z-standardization of an numpy array A
    along the axis defined in dimension using the
    denominator (N-dof) for the calculation of
    the standard deviation.
    """
    A = (A - A.mean(axis=dimension)) / A.std(axis=dimension, ddof=dof)
    return A


def sort_descending(a):
    """Sort array in descending order."""
    # http://stackoverflow.com/questions/26984414/
    # efficiently-sorting-a-numpy-array-in-descending-order
    return np.sort(a)[::-1]


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
