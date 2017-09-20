"""Provide I/O functionality.

Provide functions to load and save IDTxl data, provide import functions (e.g.,
mat-files, FieldTrip) and export functions (e.g., networkx, BrainNet Viewer).
"""
import json
import pickle
import h5py
import numpy as np
import copy as cp
from scipy.io import loadmat
from .data import Data

VERBOSE = False


def save(data, file_path):
    """Save IDTxl data to disk.

    Save different data types to disk. Supported types are:

    - dictionaries with results, e.g., from MultivariateTE
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
        data : dict | numpy array | Data object
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

    if type(data) is dict:
        if add_extension:
            file_path = ''.join([file_path, '.txt'])
        # JSON does not recognize numpy arrays and data types, they have to be
        # converted before dumping them.
        data_json = _remove_numpy(data)
        if VERBOSE:
            print('writing file {0}'.format(file_path))
        with open(file_path, 'w') as outfile:
            json.dump(obj=data_json, fp=outfile, sort_keys=True)
    elif type(data) is np.ndarray:
        # TODO this can't handle scalars, handle this as an exception
        np.save(file_path, data)
    elif type(data) is __name__.data.Data:
        np.savez(file_path, data=data.data, normalised=data.normalise)


def _remove_numpy(data):
    """Remove all numpy data structures and types from dictionary.

    JSON can not handle numpy types and data structures, they have to be
    convertedto native python types first.
    """
    data_json = cp.copy(data)
    for k in data_json.keys():
        if VERBOSE:
            print('{0}, type: {1}'.format(data_json[k], type(data_json[k])))
        if type(data_json[k]) is np.ndarray:
            data_json[k] = data_json[k].tolist()
    return data_json


def load(file_path):
    """Load IDTxl data from disk.

    Load different data types to disk. Supported types are:

    - dictionaries with results, e.g., from MultivariateTE
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


def save_pickle(obj, name):
    """Save objects using Python's pickle module.

    Note:
        pickle.HIGHEST_PROTOCOL is a binary format, which may be inconvenient,
        but is good for performance. Protocol 0 is a text format.
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    """Load objects that have been saved using Python's pickle module."""
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def import_fieldtrip(file_name, ft_struct_name, file_version, normalise=True):
    """Convert FieldTrip-style MATLAB-file into an IDTxl Data object.

    Import a MATLAB structure with fields  "trial" (data), "label" (channel
    labels), "time" (time stamps for data samples), and "fsample" (sampling
    rate). This structure is the standard file format in the MATLAB toolbox
    FieldTrip and commonly use to represent neurophysiological data (see also
    http://www.fieldtriptoolbox.org/reference/ft_datatype_raw). The data is
    returned as a IDTxl Data() object.

    The structure is assumed to be saved as a matlab hdf5 file ("-v7.3' or
    higher, .mat) with a SINGLE FieldTrip data structure inside.

    Args:
        file_name : string
            full (matlab) file_name on disk
        ft_struct_name : string
            variable name of the MATLAB structure that is in FieldTrip format
            (autodetect will hopefully be possible later ...)
        file_version : string
            version of the file, e.g. 'v7.3' for MATLAB's 7.3 format
        normalise : bool [optional]
            normalise data after import (default=True)

    Returns:
        Data() instance
            instance of IDTxl Data object, containing data from the 'trial'
            field
        list of strings
            list of channel labels, corresponding to the 'label' field
        numpy array
            time stamps for samples, corresponding to one entry in the 'time'
            field
        int
            sampling rate, corresponding to the 'fsample' field

    @author: Michael Wibral
    """
    if file_version != "v7.3":
        raise RuntimeError('At present only m-files in format 7.3 are '
                           'supported, please consider reopening and resaving '
                           'your m-file in that version.')
        # TODO we could write a fallback option using numpy's loadmat?

    print('Creating Python dictionary from FT data structure: {0}'
          .format(ft_struct_name))
    trial_data = _ft_import_trial(file_name, ft_struct_name)
    label = _ft_import_label(file_name, ft_struct_name)
    fsample = _ft_fsample_2_float(file_name, ft_struct_name)
    timestamps = _ft_import_time(file_name, ft_struct_name)

    data = Data(data=trial_data, dim_order='spr', normalise=normalise)
    return data, label, timestamps, fsample


def _ft_import_trial(file_name, ft_struct_name):
    """Import FieldTrip trial data into Python."""
    ft_file = h5py.File(file_name)
    ft_struct = ft_file[ft_struct_name]  # TODO: ft_struct_name = automagic...

    # Get the trial cells that contain the references (pointers) to the data
    # we need. Then get the data from matrices in cells of a 1 x numtrials cell
    # array in the original FieldTrip structure.
    trial = ft_struct['trial']

    # Get the trial cells that contain the references (pointers) to the data
    # we need. Then get the data from matrices in cells of a 1 x numtrials cell
    # array in the original FieldTrip structure.
    trial = ft_struct['trial']

    # Allocate memory to hold actual data, read shape of first trial to know
    # the data size.
    trial_data_tmp = np.array(ft_file[trial[0][0]])  # get data from 1st trial
    print('Found data with first dimension: {0}, and second: {1}'
          .format(trial_data_tmp.shape[0], trial_data_tmp.shape[1]))
    geometry = trial_data_tmp.shape + (trial.shape[0],)
    trial_data = np.empty(geometry)

    # Get actual data from h5py structure.
    for tt in range(0, trial.shape[0]):
        trialref = trial[tt][0]  # get trial reference
        trial_data[:, :, tt] = np.array(ft_file[trialref])  # get data

    ft_file.close()
    return trial_data


def _ft_import_label(file_name, ft_struct_name):
    """Import FieldTrip labels into Python."""
    # for details of the data handling see comments in _ft_import_trial
    ft_file = h5py.File(file_name)
    ft_struct = ft_file[ft_struct_name]
    ft_label = ft_struct['label']

    if VERBOSE:
        print('Converting FT labels to python list of strings')

    label = []
    for ll in range(0, ft_label.shape[0]):
        # There is only one item in labelref, but we have to index it.
        # Matlab has character arrays that are read as bytes in Python 3.
        # Here, map maps the stuff in labeltmp to characters and "".
        # makes it into a real Python string.
        labelref = ft_label[ll]
        labeltmp = ft_file[labelref[0]]
        strlabeltmp = "".join(map(chr, labeltmp[0:]))
        label.append(strlabeltmp)

    ft_file.close()
    return label


def _ft_import_time(file_name, ft_struct_name):
    """Import FieldTrip time stamps into Python."""
    # for details of the data handling see comments in ft_trial_2_numpyarray
    ft_file = h5py.File(file_name)
    ft_struct = ft_file[ft_struct_name]
    ft_time = ft_struct['time']
    if VERBOSE:
        print('Converting FT time cell array to numpy array')

    np_timeaxis_tmp = np.array(ft_file[ft_time[0][0]])
    geometry = np_timeaxis_tmp.shape + (ft_time.shape[0],)
    timestamps = np.empty(geometry)
    for tt in range(0, ft_time.shape[0]):
        timeref = ft_time[tt][0]
        timestamps[:, :, tt] = np.array(ft_file[timeref])
    ft_file.close()
    return timestamps


def _ft_fsample_2_float(file_name, ft_struct_name):
    ft_file = h5py.File(file_name)
    ft_struct = ft_file[ft_struct_name]
    FTfsample = ft_struct['fsample']
    fsample = int(FTfsample[0])
    if VERBOSE:
        print('Converting FT fsample array (1x1) to numpy array (1x1)')
    return fsample


def import_matarray(file_name, array_name, file_version, dim_order,
                    normalise=True):
    """Read Matlab hdf5 file into IDTxl.

    reads a matlab hdf5 file ("-v7.3' or higher, .mat) with a SINGLE
    array inside and returns a numpy array with dimensions that
    are channel x time x trials, using np.swapaxes where necessary

    Note:
        The import function squeezes the loaded mat-file, i.e., any singleton
        dimension will be removed. Hence do not enter singleton dimension into
        the 'dim_order', e.g., don't pass dim_order='ps' but dim_order='s' if
        you want to load a 1D-array where entries represent samples recorded
        from a single channel.

    Args:
        file_name : string
            full (matlab) file_name on disk
        array_name : string
            variable name of the MATLAB structure to be read
        file_version : string
            version of the file, e.g. 'v7.3' for MATLAB's 7.3 format, currently
            versions 'v4', 'v6', 'v7', and 'v7' are supported
        dim_order : string
            order of dimensions, accepts any combination of the characters
            'p', 's', and 'r' for processes, samples, and replications; must
            have the same length as the data dimensionality, e.g., 'ps' for a
            two-dimensional array of data from several processes over time
        normalise : bool [optional]
            normalise data after import (default=True)

    Returns:
        Data() instance
            instance of IDTxl Data object, containing data from the 'trial'
            field
        list of strings
            list of channel labels, corresponding to the 'label' field
        numpy array
            time stamps for samples, corresponding to one entry in the 'time'
            field
        int
            sampling rate, corresponding to the 'fsample' field

    Created on Wed Mar 19 12:34:36 2014

    @author: Michael Wibral
    """
    if file_version == 'v7.3':
        mat_file = h5py.File(file_name)
        # Assert that at least one of the keys found at the top level of the
        # HDF file  matches the name of the array we wanted
        if array_name not in mat_file.keys():
            raise RuntimeError('Array {0} not in mat file or not a variable '
                               'at the file''s top level.'.format(array_name))

        # 2. Create an object for the matlab array (from the hdf5 hierachy),
        # the trailing [()] ensures everything is read
        mat_data = np.squeeze(np.asarray(mat_file[array_name][()]))

    elif file_version in ['v4', 'v6', 'v7']:
        try:
            m = loadmat(file_name, squeeze_me=True, variable_names=array_name)
        except NotImplementedError as err:
            raise RuntimeError('You may have provided an incorrect file '
                               'version. The mat file was probably saved as '
                               'version 7.3 (hdf5).')
        mat_data = m[array_name]  # loadmat returns a dict containing variables
    else:
        raise ValueError('Unkown file version: {0}.'.format(file_version))

    # Create output: IDTxl data object, list of labels, sampling info in unit
    # time steps (sampling rate of 1).
    print('Creating Data object from matlab array: {0}.'.format(array_name))
    data = Data(mat_data, dim_order=dim_order, normalise=normalise)
    label = []
    for n in range(data.n_processes):
        label.append('channel_{0:03d}'.format(n))
    fsample = 1
    timestamps = np.arange(data.n_samples)
    return data, label, timestamps, fsample
