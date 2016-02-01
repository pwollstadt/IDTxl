# -*- coding: utf-8 -*-
"""
matarray2idtxl
Module with functions to import matrices from matfiles (version>7.3, hdf5)
to IDTxL

Functions in this module read the (neurophysiological) data from a Fieldtrip file
with the basic fields necessary for the MATLAB version TRENTOOL and create a numpy
array usable as input to TXL.

provides the functions:
	matarray2idtxlconverter(filename, arrayname, order) = 	takes a filename,
					the name of the array variable (arrayname) inside,
					and the order of sensor axis,  time axisand (CHECK THIS!!)
					repetition axis (as a list)

@author: Michael Wibral
"""

import h5py
import numpy as np

def _matarray_2_numpyarray(file_name, array_name, order_list):
    """
    matarray2txl._matarray_2_numpyarray
    reads a matlab hdf5 file ("-v7.3' or higher, .mat) with a SINGLE
    array inside and returns a numpy array with dimensions that
    are channel x time x trials, using np.swapaxes where necessary
    
    Created on Wed Mar 19 12:34:36 2014

    @author: Michael Wibral
    """

    print('Converting matlab array from file (v7.3) to numpy array')
    # 1. create a python object that represents the hdf5 file on disk
    mat_file = h5py.File(file_name)
    # assert that at least one of the keys found at the top level
    # of the HDF file  matches the name of the array we wanted
    assert array_name in  mat_file.keys() , "array %r not in mat file or not a variable at the top level" % array_name
    
    # 2. Create an object for the matlab array (from the hdf5 hierachy)
    the_array = mat_file[array_name][()] # trailing [()] ensures everything is read 
    print('From HDF5: ')
    print(the_array)
    # 3. Convert to numpyarray
    the_array = np.asarray(the_array)
    print('as numpy: ')
    print(the_array)
    
    # 4. swapaxes according to the information provided by the user
    the_array = reorder_array(the_array, order_list)
    
    return the_array
    

def reorder_array(the_array, order_list):
	# put time first as by agrreement in IDTxL
	time_dimension = order_list.index("time")
	if time_dimension != 1:
		the_array = np.swapaxes(the_array,1,time_dimension)
		# also swap the list to reflect the new arrangement
		order_list[1], order_list[time_dimension] = \
		order_list[time_dimension], order_list[1]
		
	# put channel second
	channel_dimension = order_list.index("channel")
	if channel_dimension !=2:
		the_array = np.swapaxes(the_array,2,channel_dimension)
		# also swap the list to reflect the new arrangement
		order_list[2], order_list[channel_dimension] = \
		order_list[channel_dimension], order_list[2]
		
	# put repetitions third - unnecessary in principle as n-1 permutations
	# are guaranteed to sort our array dimensions for n dimensions
	#assert order_list.index("repetition") == 3, print('something went wrong with reordering')
	
	# uncomment the following code when expanding
	#repetition_dimension = order_list.index("repetition")
	#if repetition_dimension !=2:
	#	the_array = np.swapaxes(the_array,2,repetition_dimension)
	#	# also swap the list to reflect the new arrangement
	#	order_list[3], order_list[repetition_dimension] = \
	#	order_list[repetition_dimension], order_list[3]
	
	# put further dimensions fourth in future versions...
	return the_array
	

def matarray2idtxl(filename, array_name, order_list):
    print('Creating Python dictionary from matlab array: ' + array_name)
    NPData = _matarray_2_numpyarray(filename, array_name, order_list)
    print(NPData)
    label = [None] * NPData.shape[1]
    for n in range(0, NPData.shape[1]):
		label[n] = "channel{0:04d}.txt".format(n)
		print(label[n])
    
    NPfsample = 1
    NPtime = np.asarray(range(0, NPData.shape[0])) # take unit time steps 

    TXLdata = {"np_timeseries" : NPData , "label" : label, "time" : NPtime, "fsample" : NPfsample}
    
    return TXLdata

