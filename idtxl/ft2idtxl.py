"""Provide functions to import FieldTrip mat-files (version>7.3) to IDTxl.

Functions in this module read the (neurophysiological) data from a FieldTrip 
file with the basic fields necessary for its analysis with the MATLAB toolbox 
TRENTOOL (fields ). Creates a numpy array usable as input to IDTxl.

Methods:
    ft_trial_2_numpyarray(filename, FTstructname)  

Note:
    Written for Python 3.4+

Created on Wed Mar 19 12:34:36 2014

@author: Michael Wibral
"""
import h5py
import numpy as np
#import scipy.io as sio
from idtxl.data import Data


def _ft_trial_2_numpyarray(filename, FTstructname):
    """
    ft2txl.ftdata2numpyarray
    reads a matlab hdf5 file ("-v7.3' or higher, .mat) with a SINGLE FieldTrip
    data structure inside and returns a numpy array with dimensions
    are channel x time x trials

    Created on Wed Mar 19 12:34:36 2014

    @author: Michael Wibral
    """

    print('Converting FT trial cell array to numpy array')
    # 1. create a python object that represents the hdf5 file on disk
    FTfile = h5py.File(filename)

    # 2.check if its an hdf5 file

    # 3. Identify the name of the FieldTrip structure inside
    # this will be the only name without '#' and without '/'
    # and containing the subfields
    # /trial, /time, /fsample, /label
    # .. and create an object for the FieldTrip structure
    FTstruct = FTfile[FTstructname]  # TODO: FTstructname = automagic...


    # 4. get the trial cells that contain the references (pointers) to the data we need
    # Note: a valid FT matlab structure always contains the field 'trial'
    trial = FTstruct['trial']

     # 5. Get the trial data (matrices in cells of a 1xnumtrials cell array in
    # the original FieldTrip matlab structure ) by there references stored in
    # the trial variable
    for tt in range(0, trial.shape[0]):
        # oh yes,... zero based indexing.. :-)
        if tt == 0:

            # this is one such array, with a single obejct reference inside...
            trialref = trial[tt]
            # this finally pulls out the numeric data for the trial by a
            # (region) reference
            trialdatatmp = FTfile[trialref[0]]
            trialdata_shape = trialdatatmp.shape
            print(" Found data with first dimension: {0}, and second: {1} ".format(trialdata_shape[0], trialdata_shape[1] ))
            # convert to numpy
            nptrialdatatmp = np.array(trialdatatmp)
            # the shape of the numpy array for a single trial
            npsingletrialdatashape = nptrialdatatmp.shape
            # tuples are immutable we have to create a new one to
            # add information on the number of trials
            geometry = npsingletrialdatashape + (trial.shape[0],)
            # create a numpy array to hold the data from all trials,
            # dimensions are channel x time x trials
            NPData = np.empty(geometry)
            # fill with the first trial
            NPData[:, :, tt] = nptrialdatatmp
        else:
            # if it's not the first trial, simply fill the main numpy array
            # with the other trials
            trialref = trial[tt]
            trialdatatmp = FTfile[trialref[0]]
            nptrialdatatmp = np.array(trialdatatmp)
            NPData[:, :, tt] = nptrialdatatmp

    FTfile.close()
    return NPData

def _ft_label_2_list(filename, FTstructname):
    # for details of the data handling see comments in ft_trial_2_numpyarray
    FTfile = h5py.File(filename)
    FTstruct = FTfile[FTstructname]
    FTlabel = FTstruct['label']

    print('Converting FT labels to python list of strings')

    for ll in range(0, FTlabel.shape[0]):
        if ll == 0:
            labelref = FTlabel[ll]
            labeltmp = FTfile[labelref[0]]  # there is only one item in labelref, but we have to index it
            # matlab has character arrays that are read as bytes in python 3
            # here, map maps the stuff in labeltmp to characters
            # and "". makes it into a real python string
            strlabeltmp = "".join(map(chr, labeltmp[0:]))
            label = [strlabeltmp]
        else:
            labelref = FTlabel[ll]
            labeltmp = FTfile[labelref[0]]
            strlabeltmp = "".join(map(chr, labeltmp[0:]))
            label.append(strlabeltmp)

# just for debi=ugging
#     for plabel in label:
#         print(plabel)
    FTfile.close()
    return label

#     for plabel in label:
#         print(plabel)

def _ft_time_2_numpyarray(filename, FTstructname):
    print('Converting FT time cell array to numpy array')
    # for details of the data handling see comments in ft_trial_2_numpyarray
    FTfile = h5py.File(filename)
    FTstruct = FTfile[FTstructname]
    FTtime = FTstruct['time']  # a  dataset (full of object references, one for each trial)

    for tt in range(0, FTtime.shape[0]):
        if tt == 0:
            timeref = FTtime[tt]  # get the reference (it's still inside an array)
            # unpack the reference from the array it's in and ...
            # get the data for the reference
            timeaxistmp = FTfile[timeref[0]]
            nptimeaxistmp = np.array(timeaxistmp)

            # create an empty numpy array with the right shape
            npsingletimeaxisshape = nptimeaxistmp.shape
            geometry = npsingletimeaxisshape + (FTtime.shape[0],)
            NPtime = np.empty(geometry)
            # and fill it with the data
            NPtime[:, :, tt] = nptimeaxistmp
        else:
            timeref = FTtime[tt]  # get the reference (it's still inside an array)
            # unpack the reference from the array it's in and ...
            # get the data for the reference
            timeaxistmp = FTfile[timeref[0]]
            nptimeaxistmp = np.array(timeaxistmp)
            NPtime[:, :, tt] = nptimeaxistmp
    FTfile.close()
    return NPtime

def _ft_fsample_2_float(filename, FTstructname):
    print('Converting FT fsample array (1x1) to numpy array (1x1)')
    FTfile = h5py.File(filename)
    FTstruct = FTfile[FTstructname]
    FTfsample = FTstruct['fsample']
    NPfsample = np.array(FTfsample[0])
    return NPfsample


# Note: Maybe all these functions should be combined into one convft2txl function that creates
# a dictionary with the keys:  'trial', 'label', 'time', 'fsample'

def ft2idtxlconverter(filename, FTstructname, fileversion):
    """Converts a FieldTrip-style MATLAB-file into an IDTxl Data object.

    Import a MATLAB structure with fields  "trial" (data), "label" (channel
    labels), "time" (time stamps for data samples), and "fsample" (sampling
    rate). This structure is the standard file format in the MATLAB toolbox 
    FieldTrip and commonly use to represent neurophysiological data (see also
    http://www.fieldtriptoolbox.org/). The functions reads a mat-file from 
    disc and returns a dictionary containing the information in the mat-file. 
    Data is represented as an IDTxl Data object.
    
    Args:
        filename : string
            full (matlab) filename on disk            
        FTstructname : string
            variable name of the MATLAB structure that is in FieldTrip format
            (autodetect will hopefully be possible later ...)
        fileversion : string
            version of the file, e.g. "v7.3" for MATLAB's 7.3 format

    Returns:
        dict
            "dataset": instance of IDTxl Data object; "label": list of channel
            labels; "time": numpy array of time stamps; "fsample": sampling 
            rate
    """
    
    # TODO: This will need better error handling !
    if fileversion == "v7.3":
#        try:
        print('Creating Python dictionary from FT data structure: ' + FTstructname)
        NPData = _ft_trial_2_numpyarray(filename, FTstructname)
        label = _ft_label_2_list(filename, FTstructname)
        NPfsample = _ft_fsample_2_float(filename, FTstructname)
        NPtime = _ft_time_2_numpyarray(filename, FTstructname)
        # convert data into IDTxl's Data class
        d = Data()
        # fieldtrip had "channel x timesamples" data,
        # but numpy sees the data as stored internally in the hdf5 file as:
        # "timesamples x channel"
        # we collected the replications
        # in the tirhd diemsnion --> dimension are:
        # s(amples) x p(rocesses) x r(eplications) = 'spr'
        d.set_data(NPData, 'spr')
        TXLdata = {"dataset" : d , "label" : label,
                   "time" : NPtime, "fsample" : NPfsample}

#        except(OSError, RuntimeError):
#            print('incorrect file version, the given file was not a MATLAB'
#                  ' m-file version 7.3')
#            return
    else:
        print('At present only m-files in format 7.3 are aupported,'
              'please consider reopening and resaving your m-file in that'
              'version')
    return TXLdata

