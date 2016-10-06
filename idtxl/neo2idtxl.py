"""
Created on 06/10/2016

Import data via NEO into IDTxl.

Note:
    Written for Python 3.4+

@author: Michael Lindner
"""

import numpy as np
import neo
import os
from os.path import join
from idtxl.data import Data
# from IDTxl import Data
# from . import Data


def _load_file_using_neo(neofilename, filepath):
    """ import data using the neo package """
    supported_formats = 'Spike2IO (.smr), NeoMatlabIO (.mat)'
    # ,AsciiSignalIO (), AsciiSpikeTrainIO (), BrainwareDamIO (),
    # BrainwareF32IO (), BrainwareSrcIO (),
    # NeuroscopeIO (), PickleIO (), PyNNIO (), RawBinarySignalIO (),
    # WinEdrIO (), WinWcpIO ()'

    # get file extension
    fn, file_extension = os.path.splitext(neofilename)
    # Open file depending on the format (given bby the file extension)
    if file_extension[1:4] == 'smr':
        reader = neo.io.Spike2IO(filename=join(filepath, neofilename))
        # load data into neo formated block
        neoblock = reader.read(cascade=True, lazy=False)[0]
    elif file_extension[1:4] == 'mat':
        reader = neo.io.NeoMatlabIO(filename=join(filepath, neofilename))
        neoblock = reader.read_block()
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.AsciiSignalIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #   reader = neo.io.AsciiSpikeTrainIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.BrainwareDamIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.BrainwareF32IO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.BrainwareSrcIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.NeuroscopeIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.PickleIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.PyNNIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #   reader = neo.io.RawBinarySignalIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.WinEdrIO(filename=join(filepath, neofilename))
    # elif file_extension[1:4] == '???':
    #    reader = neo.io.WinWcpIO(filename=join(filepath, neofilename))
    else:
        # rasie error if input file format is not supported
        raise Exception('File extension ' + file_extension[1:4] +
                        'not supported. Supported formats: ' +
                        supported_formats)

    return neoblock


def _neo_get_analogsignals(neoblock):
    """Extract the analogsignals from the neo block format.

    The analogsignals in the neoblock need to have the same length in all
    replications and processes.

    Args:
        neoblock: block format of neo package

    Returns:
        fsample: numpy.array
            sampling frequency of the analogsignals
        time_vector:
            vector containing the time indices for each sample in relation
            to the t_start and t_stop of the neoblock
        label: list
            list of channel labels
        dat: numpy.array
            Three dimensional data array containing the analogsignals for
            each replication and processes
    """
    # get sampling frequency
    fsample = neoblock.segments[0].analogsignals[0].sampling_rate.base

    # get time indices
    time_onset = neoblock.segments[0].analogsignals[0].t_start.magnitude
    time_offset = neoblock.segments[0].analogsignals[0].t_stop.magnitude

    # get number of segments, signals and samples
    nr_replications = len(neoblock.segments)
    nr_processes = len(neoblock.segments[0].analogsignals)
    nr_samples = len(neoblock.segments[0].analogsignals[0].times.base)

    # create time vector
    time_vector = np.arange(time_onset,
                            (time_offset + (time_offset - time_onset) /
                                (nr_samples - 1)),
                            (time_offset - time_onset) / (nr_samples - 1))

    # preallocate data matrix and label list
    dat = np.empty(shape=[nr_replications, nr_processes, nr_samples])
    label = list()

    # loop over segments
    for segm in list(range(0, nr_replications)):
        # get number of signals
        nr_processes = len(neoblock.segments[segm].analogsignals)

        # loop over signals
        for proc in list(range(0, nr_processes)):
            # add analogsignals into the 3D data matrix
            dat[segm, proc, :] = (neoblock.segments[segm].analogsignals[proc]
                                  .times.base)

    for proc in list(range(0, nr_processes)):
        # add channel index into the list of channel labels
        label.append((neoblock.segments[0].analogsignals[proc].channel_index))

    return fsample, time_vector, dat, label


def _neo_get_spiketrains(neoblock):
    """Extract the spiketrains from the neo block format.

    The spiketrains in the neoblock need to have the same length in all
    replications and processes.

    Args:
        neoblock: block format of neo package

    Returns:
        fsample: numpy.array
            sampling frequency of the spiketrains
        time_vector:
            vector containing the time indices for each sample in relation
            to the t_start and t_stop of the neoblock
        label: list
            list of channel labels
        dat: numpy.array
            Three dimensional data array containing the spiketrains for
            each replication and processes
    """
    # get sampling frequency
    fsample = neoblock.segments[0].spiketrians[0].sampling_rate.base

    # get time indices[2]
    time_onset = neoblock.segments[0].spiketrains[0].t_start.magnitude
    time_offset = neoblock.segments[0].spiketrains[0].t_stop.magnitude

    # get number of segments, signals and samples
    nr_replications = len(neoblock.segments)
    nr_processes = len(neoblock.segments[0].spiketrains)
    nr_samples = len(neoblock.segments[0].spiketrains[0].times.magnitude)

    # create time vector
    time_vector = np.arange(time_onset,
                            (time_offset + (time_offset - time_onset) /
                                (nr_samples - 1)),
                            (time_offset - time_onset) / (nr_samples - 1))

    # preallocate data matrix and label list
    dat = np.empty(shape=[nr_replications, nr_processes, nr_samples])
    label = list()

    # loop over segments
    for segm in list(range(0, nr_replications)):
        # get number of signals
        nr_processes = len(neoblock.segments[segm].spiketrains)

        # loop over signals
        for proc in list(range(0, nr_processes)):

            # get analog signals
            spiketrain = (neoblock.segments[segm].spiketrains[proc]
                          .times.magnitude)

            # convert spiketrains onto vectors with zeros and ones:
            # create vector of ones
            spike_vector = np.zeros(shape=(1, nr_samples))

            # convert secs onto vector indices
            # TODO

            dat[segm, proc, :] = spike_vector

    for proc in list(range(0, nr_processes)):
        # add channel index into the list of channel labels
        # TODO: check if channel index exists in spiketrians!!
        label.append((neoblock.segments[0].spiketrains[proc].channel_index))

    return fsample, time_vector, dat, label


def _neo_get_data(neoblock, datatype2extract):

    if datatype2extract == "spiketrains":
        fsample, time_vector, dat, label = _neo_get_spiketrains(neoblock)
    elif datatype2extract == "analogsignals":
        fsample, time_vector, dat, label = _neo_get_analogsignals(neoblock)

    return fsample, time_vector, dat, label


def neo2idtxlconverter(neofilename, filepath, datatype2extract):
    """ Converts a neo format from file to an IDTxl Data object.

    Load data of different data input formats via the neo importer into a neo
    block format and extract the supported data types into the IDTxl data
    format.

    Args:
        neofilename: [string]
                filename of the file to be imported to IDTxl
                Supported file formats:
                .smr = 'Spike2' by Cambridge Electronic Design
                .mat = Matlab neo format
                       (see Matlab function convert_matlab_to_neo.m)

        neofilepath: [string]
                path of the filename

        datatype2extract: [string]
                type of data to be extracted from the neo format:
                supported data types to be imported:
                    analog signals: 'analogsignals'

    Returns:
        IDTxldata_object: IDTxl data object
            containing the 3D array containing the dimensions processes,
            samples, replications
        IDTXLdata: dictionary
            Dictionary containig the 3D data array, a vecotr containg the time
            indices for each sample, the channel labels and the sampling
            frequency.
    """

    neoblock = _load_file_using_neo(neofilename, filepath)

    fsample, time_vec, dat, label = _neo_get_data(neoblock, datatype2extract)

    IDTXLdata_object = Data(dat, 'rps')

    IDTXLdata = {"dataset": IDTXLdata_object, "label": label,
                 "time": time_vec, "fsample": fsample}

    return IDTXLdata_object, IDTXLdata
