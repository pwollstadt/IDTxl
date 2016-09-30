import copy as cp
import pprint
import json
import numpy as np
from . import Data

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


def save(dat, file_path):
    """Save IDTxl data to disk.

    Save different data types to disk. Supported types are:

    - dictionaries with results, e.g., from Multivariate_te
    - numpy array
    - instance of IDTXL Data object

    Note that while numpy arrays and Data instances are saved in binary for
    performance, dictionaries are saved in the json format, which is human-
    readable and also easily read into other programs (e.g., MATLAB:
    http://undocumentedmatlab.com/blog/json-matlab-integration).

    File extensions are

    - .txt for dictionaries (JSON file)
    - .npy for numpy array
    - .npz for Data instances

    If the extension is not provided in the file_path, the function will add it
    depending on the type of the data to be written.

    Args:
        dat : dict | numpy array | Data object
            data to be saved to disk
        file_path : string
            string with file name (including the path)
    """
    # Check if a file extension is provided in the file_path. Note that the
    # numpy save functions don't need an extension, they are added if missing.
    if file_path.find('.', -4) == -1:
        add_extension = True
    else:
        add_extension = False

    if type(dat) is dict:
        if add_extension:
            file_path = ''.join([file_path, '.txt'])
        # JSON does not recognize numpy arrays and data types, they have to be
        # converted before dumping them.
        dat_json = _remove_numpy(dat)
        if VERBOSE:
            print('writing file {0}'.format(file_path))
        with open(file_path, 'w') as outfile:
            json.dump(obj=dat_json, fp=outfile, sort_keys=True)
    elif type(dat) is np.ndarray:
        # TODO this can't handle scalars, handle this as an exception
        np.save(file_path, dat)
    elif type(dat) is __name__.data.Data:
        np.savez(file_path, data=dat.data, normalised=dat.normalise)


def _remove_numpy(dat):
    """Remove all numpy data structures and types from dictionary.

    JSON can not handle numpy types and data structures, they have to be
    convertedto native python types first.
    """
    dat_json = cp.copy(dat)
    for k in dat_json.keys():
        if VERBOSE:
            print('{0}, type: {1}'.format(dat_json[k], type(dat_json[k])))
        if type(dat_json[k]) is np.ndarray:
            dat_json[k] = dat_json[k].tolist()
    return dat_json


def load(file_path):
    """Load IDTxl data from disk.

    Load different data types to disk. Supported types are:

    - dictionaries with results, e.g., from Multivariate_te
    - numpy array
    - instance of IDTXL Data object

    File extensions are

    - .txt for dictionaries (JSON file)
    - .npy for numpy array
    - .npz for Data instances

    Note that while numpy arrays and Data instances are saved in binary for
    performance, dictionaries are saved in the json format, which is human-
    readable and also easily read into other programs (e.g., MATLAB:
    http://undocumentedmatlab.com/blog/json-matlab-integration).

    Args:
        file_path : string
            string with file name (including the path)

    Returns:

    """
    # Check extension of provided file path, this is needed to determine the
    # file type to be loaded.
    ext = file_path[file_path.find('.', -4) + 1:]
    assert len(ext) == 3, ('Could not determine file format of "file_path", '
                           'please provide one of the following extensions: '
                           '".txt", ".npy", ".npz".')

    # Load data depending on the file type.
    if ext == 'txt':  # JSON file
        print('loading dictionary from disc')
        # with file_path as infile:
        with open(file_path) as json_data:
            d = json.load(json_data)
            # TODO convert lists to np.arrays?
    elif ext == 'npy':  # single numpy array
        print('loading numpy array from disc')
        return np.load(file_path)
    elif ext == 'npz':  # instance of IDTxl Data object
        print('loading data object from disc')
        f = np.load(file_path)
        d = Data(f['data'], dim_order='psr', normalise=False)
        d.normalise = f['normalised']
        return d
