def standardize(A, dimension=0, dof=1):
    """ Z-standardization of an numpy array A
    along the axis defined in dimension using the
    denominator (N-dof) for the calculation of
    the standard deviation.
    """
    A = (A - A.mean(axis=dimension)) / A.std(axis=dimension, ddof=dof) 
    return A
