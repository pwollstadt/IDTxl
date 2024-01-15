"""Import external file formats into IDTxl.

Provide functions to import the following into IDTxl:

    - mat-files (version>7.3, hdf5)
    - FieldTrip-style mat-files (version>7.3, hdf5)

Matlab supports hdf5 only for files saved as version 7.3 or higher:
https://au.mathworks.com/help/matlab/ref/save.html#inputarg_version

Creates a numpy array usable as input to IDTxl.

Methods:
    ft_trial_2_numpyarray(file_name, ft_struct_name)
    matarray2idtxlconverter(file_name, array_name, order) =     takes a file_name,
                    the name of the array variable (array_name) inside,
                    and the order of sensor axis,  time axisand (CHECK THIS!!)
                    repetition axis (as a list)

Note:
    Written for Python 3.4+

Created on Wed Mar 19 12:34:36 2014

@author: Michael Wibral
"""
import h5py
import numpy as np
from scipy.io import loadmat
from idtxl.data import Data

VERBOSE = False


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
        raise RuntimeError(
            "At present only m-files in format 7.3 are "
            "supported, please consider reopening and resaving "
            "your m-file in that version."
        )
        # TODO we could write a fallback option using numpy's loadmat?

    print(
        "Creating Python dictionary from FT data structure: {0}".format(ft_struct_name)
    )
    trial_data = _ft_import_trial(file_name, ft_struct_name)
    label = _ft_import_label(file_name, ft_struct_name)
    fsample = _ft_fsample_2_float(file_name, ft_struct_name)
    timestamps = _ft_import_time(file_name, ft_struct_name)

    dat = Data(data=trial_data, dim_order="spr", normalise=normalise)
    return dat, label, timestamps, fsample


def _ft_import_trial(file_name, ft_struct_name):
    """Import FieldTrip trial data into Python."""
    ft_file = h5py.File(file_name, mode="r+")
    ft_struct = ft_file[ft_struct_name]  # TODO: ft_struct_name = automagic...

    # Get the trial cells that contain the references (pointers) to the data
    # we need. Then get the data from matrices in cells of a 1 x numtrials cell
    # array in the original FieldTrip structure.
    trial = ft_struct["trial"]

    # Get the trial cells that contain the references (pointers) to the data
    # we need. Then get the data from matrices in cells of a 1 x numtrials cell
    # array in the original FieldTrip structure.
    trial = ft_struct["trial"]

    # Allocate memory to hold actual data, read shape of first trial to know
    # the data size.
    trial_data_tmp = np.array(ft_file[trial[0][0]])  # get data from 1st trial
    print(
        "Found data with first dimension: {0}, and second: {1}".format(
            trial_data_tmp.shape[0], trial_data_tmp.shape[1]
        )
    )
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
    ft_file = h5py.File(file_name, mode="r+")
    ft_struct = ft_file[ft_struct_name]
    ft_label = ft_struct["label"]

    if VERBOSE:
        print("Converting FT labels to python list of strings")

    label = []
    for labelref in ft_label:
        # There is only one item in labelref, but we have to index it.
        # Matlab has character arrays that are read as bytes in Python 3.
        # Here, map maps the stuff in labeltmp to characters and "".
        # makes it into a real Python string.
        labeltmp = ft_file[labelref[0]]
        strlabeltmp = "".join([chr(i[0]) for i in labeltmp[0:]])
        label.append(strlabeltmp)

    ft_file.close()
    return label


def _ft_import_time(file_name, ft_struct_name):
    """Import FieldTrip time stamps into Python."""
    # for details of the data handling see comments in ft_trial_2_numpyarray
    ft_file = h5py.File(file_name, mode="r+")
    ft_struct = ft_file[ft_struct_name]
    ft_time = ft_struct["time"]
    if VERBOSE:
        print("Converting FT time cell array to numpy array")

    np_timeaxis_tmp = np.array(ft_file[ft_time[0][0]])
    geometry = np_timeaxis_tmp.shape + (ft_time.shape[0],)
    timestamps = np.empty(geometry)
    for tt in range(0, ft_time.shape[0]):
        timeref = ft_time[tt][0]
        timestamps[:, :, tt] = np.array(ft_file[timeref])
    ft_file.close()
    return timestamps


def _ft_fsample_2_float(file_name, ft_struct_name):
    ft_file = h5py.File(file_name, mode="r+")
    ft_struct = ft_file[ft_struct_name]
    FTfsample = ft_struct["fsample"]
    fsample = int(FTfsample[0])
    if VERBOSE:
        print("Converting FT fsample array (1x1) to numpy array (1x1)")
    return fsample


def import_matarray(file_name, array_name, file_version, dim_order, normalise=True):
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
    if file_version == "v7.3":
        mat_file = h5py.File(file_name, mode="r+")
        # Assert that at least one of the keys found at the top level of the
        # HDF file  matches the name of the array we wanted
        if array_name not in mat_file.keys():
            raise RuntimeError(
                "Array {0} not in mat file or not a variable "
                "at the file"
                "s top level.".format(array_name)
            )

        # 2. Create an object for the matlab array (from the hdf5 hierachy),
        # the trailing [()] ensures everything is read
        mat_data = np.squeeze(np.asarray(mat_file[array_name][()]))

    elif file_version in ["v4", "v6", "v7"]:
        try:
            m = loadmat(file_name, squeeze_me=True, variable_names=array_name)
        except NotImplementedError as err:
            raise RuntimeError(
                "You may have provided an incorrect file "
                "version. The mat file was probably saved as "
                "version 7.3 (hdf5)."
            )
        mat_data = m[array_name]  # loadmat returns a dict containing variables
    else:
        raise ValueError("Unkown file version: {0}.".format(file_version))

    # Create output: IDTxl data object, list of labels, sampling info in unit
    # time steps (sampling rate of 1).
    print("Creating Data object from matlab array: {0}.".format(array_name))
    dat = Data(mat_data, dim_order=dim_order, normalise=normalise)
    label = []
    for n in range(dat.n_processes):
        label.append("channel_{0:03d}".format(n))
    fsample = 1
    timestamps = np.arange(dat.n_samples)
    return dat, label, timestamps, fsample
