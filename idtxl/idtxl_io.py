"""Provide load and save functionality for IDTxl results."""
import json
import copy as cp
import numpy as np
from .data import Data

VERBOSE = True


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
