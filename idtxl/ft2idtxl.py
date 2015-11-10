# -*- coding: utf-8 -*-
"""
ft2txl
Module with functions to import FieldTrip matfiles (version>7.3, hdf5)
to TRENTOOL XL (aka TXL).

Functions in this module read the (neurophysiological) data from a Fieldtrip file
with the basic fields necessary for the MATLAB version TRENTOOL and create a numpy
array usable as input to TXL.

provides the functions:
ft_trial_2_numpyarray(filename, FTstructname)
    inputs:
    filename a full (matlab) filename on disk
    FTstructname = name of the MATLAB stucture that is in Fieldtrip format 
    (autodetect will hopefully be possible later ...)
    
    outputs:
    NPData = a numpy 3D array with channel by time by trial
    
Created on Wed Mar 19 12:34:36 2014

@author: Michael Wibral
"""

import h5py
import numpy as np

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

def ft2idtxlconverter(filename, FTstructname):
    print('Creating Python dictionary from FT data structure: ' + FTstructname)
    NPData = _ft_trial_2_numpyarray(filename, FTstructname)
    label = _ft_label_2_list(filename, FTstructname)
    NPfsample = _ft_fsample_2_float(filename, FTstructname)
    NPtime = _ft_time_2_numpyarray(filename, FTstructname)

    TXLdata = {"np_timeseries" : NPData , "label" : label, "time" : NPtime, "fsample" : NPfsample}
    # I wonder whther I should turn this into a list of dicts or even objects - 1 per channel ?
    return TXLdata

