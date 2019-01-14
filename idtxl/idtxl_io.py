"""Provide I/O functionality.

Provide functions to load and save IDTxl data, provide import functions (e.g.,
mat-files, FieldTrip) and export functions (e.g., networkx, BrainNet Viewer).
"""
# import json
import pickle
import h5py
import networkx as nx
import numpy as np
import copy as cp
import itertools as it
from scipy.io import loadmat
from .data import Data
from . import idtxl_exceptions as ex
try:
    import networkx as nx
except ImportError as err:
    ex.package_missing(
        err,
        ('networkx is not available on this system. Install it from '
         'https://pypi.python.org/pypi/networkx/2.0 to export and plot IDTxl '
         'results in this format.'))

VERBOSE = False


# def save(data, file_path):
#     """Save IDTxl data to disk.

#     Save different data types to disk. Supported types are:

#     - dictionaries with results, e.g., from MultivariateTE
#     - numpy array
#     - instance of IDTXL Data object

#     Note that while numpy arrays and Data instances are saved in binary for
#     performance, dictionaries are saved in the json format, which is human-
#     readable and also easily read into other programs (e.g., MATLAB:
#     http://undocumentedmatlab.com/blog/json-matlab-integration).

#     File extensions are

#     - .txt for dictionaries (JSON file)
#     - .npy for numpy array
#     - .npz for Data instances

#     If the extension is not provided in the file_path, the function will add
#     it depending on the type of the data to be written.

#     Args:
#         data : dict | numpy array | Data object
#             data to be saved to disk
#         file_path : string
#             string with file name (including the path)
#     """
#     # Check if a file extension is provided in the file_path. Note that the
#     # numpy save functions don't need an extension, they are added if
#     # missing.
#     if file_path.find('.', -4) == -1:
#         add_extension = True
#     else:
#         add_extension = False

#     if type(data) is dict:
#         if add_extension:
#             file_path = ''.join([file_path, '.txt'])
#         # JSON does not recognize numpy arrays and data types, they have to
#         # be converted before dumping them.
#         data_json = _remove_numpy(data)
#         if VERBOSE:
#             print('writing file {0}'.format(file_path))
#         with open(file_path, 'w') as outfile:
#             json.dump(obj=data_json, fp=outfile, sort_keys=True)
#     elif type(data) is np.ndarray:
#         # TODO this can't handle scalars, handle this as an exception
#         np.save(file_path, data)
#     elif type(data) is __name__.data.Data:
#         np.savez(file_path, data=data.data, normalised=data.normalise)


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


# def load(file_path):
#     """Load IDTxl data from disk.

#     Load different data types to disk. Supported types are:

#     - dictionaries with results, e.g., from MultivariateTE
#     - numpy array
#     - instance of IDTXL Data object

#     File extensions are

#     - .txt for dictionaries (JSON file)
#     - .npy for numpy array
#     - .npz for Data instances

#     Note that while numpy arrays and Data instances are saved in binary for
#     performance, dictionaries are saved in the json format, which is human-
#     readable and also easily read into other programs (e.g., MATLAB:
#     http://undocumentedmatlab.com/blog/json-matlab-integration).

#     Args:
#         file_path : string
#             string with file name (including the path)

#     Returns:

#     """
#     # Check extension of provided file path, this is needed to determine the
#     # file type to be loaded.
#     ext = file_path[file_path.find('.', -4) + 1:]
#     assert len(ext) == 3, ('Could not determine file format of "file_path", '
#                            'please provide one of the following extensions: '
#                            '".txt", ".npy", ".npz".')

#     # Load data depending on the file type.
#     if ext == 'txt':  # JSON file
#         print('loading dictionary from disc')
#         # with file_path as infile:
#         with open(file_path) as json_data:
#             d = json.load(json_data)
#             # TODO convert lists to np.arrays?
#     elif ext == 'npy':  # single numpy array
#         print('loading numpy array from disc')
#         return np.load(file_path)
#     elif ext == 'npz':  # instance of IDTxl Data object
#         print('loading data object from disc')
#         f = np.load(file_path)
#         d = Data(f['data'], dim_order='psr', normalise=False)
#         d.normalise = f['normalised']
#         return d


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

    reads a matlab hdf5 file ("-v7.3' or higher, .mat) or non-hdf5 files with a
    SINGLE array inside and returns an IDTxl Data() object.

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
        except NotImplementedError:
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
    return data


def export_networkx_graph(adjacency_matrix, weights):
    """Export networkx graph object for an inferred network.

    Export a weighted, directed graph object from the network of inferred
    (multivariate) interactions (e.g., multivariate TE), using the networkx
    class for directed graphs (DiGraph). Multiple options for the weight are
    available (see documentation of method get_adjacency_matrix for details).

    Args:
        adjacency_matrix : AdjacencyMatrix instances
            adjacency matrix to be exported, returned by get_adjacency_matrix()
            method of Results() class
        weights : str
            weights for the adjacency matrix (see documentation of method
            get_adjacency_matrix for details)
        fdr : bool [optional]
            return FDR-corrected results (default=True)

    Returns: DiGraph instance
        directed graph of networkx package's DiGraph() class
    """
    # use 'weights' parameter (string) as networkx edge property name and use
    # adjacency matrix entries as edge property values
    G = nx.DiGraph()
    G.add_weighted_edges_from(adjacency_matrix.get_edge_list(), weights)
    return G


def export_networkx_source_graph(results, target, sign_sources=True, fdr=True):
    """Export graph object of source variables for a single target.

    Export graph object from the network of (multivariate) interactions (e.g.,
    multivariate TE) between single source variables and a target process using
    the networkx class for directed graphs (DiGraph). The graph shows the
    information transfer between individual source variables and the target.
    Each node is a tuple with the following format:
    (process index, sample index).

    Args:
        results : Results() instance
            network analysis results
        target : int
            target index
        sign_sources : bool [optional]
            add sources with significant information contribution only
            (default=True)
        fdr : bool [optional]
            return FDR-corrected results (default=True)

    Returns:
        DiGraph instance
            directed graph of networkx package's DiGraph() class
    """
    graph = nx.DiGraph()

    # Replace time index of current value to be consistent with lag-notation
    # in exported graph. Remember the current value's index to later define the
    # proper candidate set.
    current_value = (results.get_single_target(
        target=target, fdr=fdr)['current_value'][0], 0)
    idx_current_value = results.get_single_target(
        target=target, fdr=fdr)['current_value'][1]
    # Add the target as a node and add omnibus p-value as an attribute
    # of the target node
    graph.add_node(current_value,
                   omnibus_te=results.get_single_target(
                                target=target, fdr=fdr)['omnibus_te'],
                   omnibus_sign=results.get_single_target(
                                target=target, fdr=fdr)['omnibus_te'])
    # Get selected source variables
    selected_vars_sources = results.get_single_target(
        target=target, fdr=fdr)['selected_vars_sources']
    # Get selected target variables
    selected_vars_target = results.get_single_target(
        target=target, fdr=fdr)['selected_vars_target']

    if sign_sources:  # Add only significant past variables as nodes.
        graph.add_nodes_from(selected_vars_sources)
        graph.add_nodes_from(selected_vars_target)
    else:   # Add all tested past variables as nodes.
        # Get all sample indices using the current value's actual index.
        samples_tested = np.arange(
            idx_current_value - results.settings.min_lag_sources,
            idx_current_value - results.settings.max_lag_sources,
            -results.settings.tau_sources)
        # Get source indices
        sources_tested = results.get_single_target(
            target=target, fdr=fdr)['sources_tested']
        # Create tuples from source and sample indices
        tested_vars_sources = [i for i in it.product(
            sources_tested, samples_tested)]
        graph.add_nodes_from(tested_vars_sources)

    # Add edges from selected target variables to the target.
    for v in selected_vars_target:
        graph.add_edge(v, current_value)

    # Get TE and p-values fro selected source variables
    selected_sources_te = results.get_single_target(
        target=target, fdr=fdr)['selected_sources_te']
    selected_sources_pval = results.get_single_target(
        target=target, fdr=fdr)['selected_sources_pval']
    # Add edges from selected source variables to the target.
    # Also add TE and p-value as edge attributes
    for (ind, v) in enumerate(selected_vars_sources):
        graph.add_edge(v, current_value,
                       te=selected_sources_te[ind],
                       pval=selected_sources_pval[ind])
    return graph


def export_brain_net_viewer(adjacency_matrix, mni_coord, file_name, **kwargs):
    """Export network to BrainNet Viewer.

    Export networks to BrainNet Viewer (project home page:
    http://www.nitrc.org/projects/bnv/). BrainNet Viewer is a MATLAB
    toolbox offering brain network visualisation (e.g., 'glass' brains).
    The function creates text files [file_name].node and [file_name].edge,
    containing information on node location (in MNI coordinates), directed
    edges, node color and size.

    References:

    - Xia, M., Wang, J., & He, Y. (2013). BrainNet Viewer: A Network
      Visualization Tool for Human Brain Connectomics. PLoS ONE 8(7):e68910.
      https://doi.org/10.1371/journal.pone.0068910

    Args:
        adjacency_matrix : AdjacencyMatrix instance
            adjacency matrix to be exported, returned by get_adjacency_matrix()
            method of Results() class
        mni_coord : numpy array
            MNI coordinates (x,y,z) of the sources, array with size [n 3],
            where n is the number of nodes
        file_name : str
            file name for output files including the file path
        labels : array type of str [optional]
            list of node labels of length n, description or label for each
            node. Note that labels can't contain spaces (causes BrainNet to
            crash), the function will remove any spaces from labels
            (default=no labels)
        node_color : array type of colors [optional]
            BrainNet gives you the option to color nodes according to the
            values in this vector (length n), see BrainNet Manual
        node_size : array type of int [optional]
            BrainNet gives you the option to size nodes according to the
            values in this array (length n), see BrainNet Manual
    """
    # Check input and get default settings for plotting. The default for
    # node labels is a list of '-' (no labels).
    n_nodes = adjacency_matrix.n_nodes()
    n_edges = adjacency_matrix.n_edges()
    labels = kwargs.get('labels', ['-' for i in range(n_nodes)])
    node_color = kwargs.get('node_color', np.ones(n_nodes))
    node_size = kwargs.get('node_size', np.ones(n_nodes))
    if n_edges == 0:
        Warning('No edges in results file. Nothing to plot.')
    assert mni_coord.shape[0] == n_nodes and mni_coord.shape[1] == 3, (
        'MNI coordinates must have shape [n_nodes, 3].')
    assert len(labels) == n_nodes, (
        'Labels must have same length as no. nodes.')
    assert len(node_color) == n_nodes, (
        'Node colors must have same length as no. nodes.')
    assert len(node_size) == n_nodes, (
        'Node size must have same length as no. nodes.')

    # Check, if there are blanks in the labels and delete them, otherwise
    # BrainNet viewer chrashes
    labels_stripped = [l.replace(" ", "") for l in labels]

    # Write node file.
    with open('{0}.node'.format(file_name), 'w') as text_file:
        for n in range(n_nodes):
            print('{0}\t{1}\t{2}\t'.format(*mni_coord[n, :]),
                  file=text_file, end='')
            print('{0}\t{1}\t'.format(node_color[n], node_size[n]),
                  file=text_file, end='')
            print('{0}'.format(labels_stripped[n]), file=text_file)

    # Write edge file.
    with open('{0}.edge'.format(file_name), 'w') as text_file:
        for i in range(n_nodes):
            for j in range(n_nodes):
                print('{0}\t'.format(adjacency_matrix._edge_matrix[i, j]),
                      file=text_file, end='')
            print('', file=text_file)
