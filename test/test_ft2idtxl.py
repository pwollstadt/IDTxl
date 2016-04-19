# -*- coding: utf-8 -*-
"""
Tests the import of fiels in Fieldtrip format to IDTxl's Data class.
Note that ft2idtxl at the moment assumes that fieldtrip data are stored as .mat
files in MATLAB's v7.3 format (which specifies an hdf5 file)

Created on Mon Apr 18 10:53:07 2016

@author: wibral
"""

import numpy as np
from idtxl.data import Data
from idtxl.ft2idtxl import ft2idtxlconverter

converted_data = ft2idtxlconverter("/home/wibral/unison/IDTxl/test/data/"
                                   "ABA04_Up_10-140Hz_v7_3.mat", 'data', 'v7.3')
assert converted_data['dataset'].n_processes == 14, 'wrong number of processes, expected 14, found: {0}'.format(converted_data['d'].n_processes)
assert converted_data['dataset'].n_replications == 135, 'wrong number of replications, expected 135, found: {0}'.format(converted_data['d'].n_replications)
assert converted_data['dataset'].n_samples == 1200, 'wrong number of samples, expected 1200, found: {0}'.format(converted_data['d'].n_samples)
