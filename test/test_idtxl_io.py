"""Unit tests for IDTxl I/O functions."""
import os
import pickle
import pytest
import numpy as np
from pkg_resources import resource_filename
from idtxl import idtxl_io as io
from idtxl.data import Data
from idtxl.network_comparison import NetworkComparison

# Generate data and load network inference results.
n_nodes = 5
data_0 = Data()
data_0.generate_mute_data(500, 5)
data_1 = Data(np.random.rand(n_nodes, 500, 5), 'psr')

path = os.path.join(os.path.dirname(__file__), 'data/')
res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))

# Generate network comparison results.
comp_settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'stats_type': 'independent',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'alpha_comp': 0.26,
        'n_perm_comp': 200,
        'tail': 'two',
        'permute_in_time': True,
        'perm_type': 'random'
        }
comp = NetworkComparison()
res_within = comp.compare_within(
    comp_settings, res_0, res_1, data_0, data_1)


def test_export_networkx():
    """Test export to networkx DiGrap() object."""
    # raise AssertionError('Test not yet implemented.')
    # Test export of networx graph for network inference results.
    weights = 'binary'
    adj_matrix = res_0.get_adjacency_matrix(weights=weights, fdr=False)
    io.export_networkx_graph(adjacency_matrix=adj_matrix, weights=weights)

    # Test export of source graph
    for s in [True, False]:
        io.export_networkx_source_graph(
            results=res_0, target=1, sign_sources=s, fdr=False)

    # Test export of networx graph for network comparison results.
    for weight in ['union', 'comparison', 'pvalue', 'diff_abs']:
        adj_matrix = res_within.get_adjacency_matrix(weights=weight)
        io.export_networkx_graph(adjacency_matrix=adj_matrix,
                                 weights=weight)
        for s in [True, False]:
            io.export_networkx_source_graph(
                    results=res_0, target=1, sign_sources=s, fdr=False)


def test_export_brain_net():
    """Test export to BrainNet Viewer toolbox."""
    # n_nodes = 5
    # data_0 = Data()
    # data_0.generate_mute_data(500, 5)
    # data_1 = Data(np.random.rand(n_nodes, 500, 5), 'psr')

    # path = os.path.join(os.path.dirname(__file__), 'data/')
    # res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    # res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))

    # Test export of network inference results.
    outfile = '{0}brain_net'.format(path)
    mni_coord = np.random.randint(10, size=(n_nodes, 3))
    node_color = np.random.randint(10, size=n_nodes)
    node_size = np.random.randint(10, size=n_nodes)
    labels = ['node_0', 'node_1', 'node_2', 'node_3', 'node_4']
    adj_matrix = res_0.get_adjacency_matrix(
        weights='max_te_lag', fdr=False,)
    io.export_brain_net_viewer(adjacency_matrix=adj_matrix,
                               mni_coord=mni_coord,
                               file_name=outfile,
                               labels=labels,
                               node_color=node_color,
                               node_size=node_size)

    # Test export of network comparison results.
    for weight in ['union', 'comparison', 'pvalue', 'diff_abs']:
        adj_matrix = res_within.get_adjacency_matrix(weights=weight)
        io.export_brain_net_viewer(adjacency_matrix=adj_matrix,
                                   mni_coord=mni_coord,
                                   file_name=outfile,
                                   labels=labels,
                                   node_color=node_color,
                                   node_size=node_size)

    # Test input checks.
    with pytest.raises(AssertionError):  # no. entries in mni matrix
        io.export_brain_net_viewer(adjacency_matrix=adj_matrix,
                                   mni_coord=mni_coord[:3, :],
                                   file_name=outfile)
    with pytest.raises(AssertionError):  # no. coordinates in mni matrix
        io.export_brain_net_viewer(adjacency_matrix=adj_matrix,
                                   mni_coord=mni_coord[:, :2],
                                   file_name=outfile)
    with pytest.raises(AssertionError):  # length label list
        io.export_brain_net_viewer(adjacency_matrix=adj_matrix,
                                   mni_coord=mni_coord,
                                   file_name=outfile,
                                   labels=['node_1', 'node_2'])
    with pytest.raises(AssertionError):  # length node color list
        io.export_brain_net_viewer(adjacency_matrix=adj_matrix,
                                   mni_coord=mni_coord,
                                   file_name=outfile,
                                   node_color=np.arange(n_nodes + 1))


def test_import_fieldtrip():
    """Test FieldTrip importer."""
    file_path = resource_filename(__name__, 'data/ABA04_Up_10-140Hz_v7_3.mat')
    (data, label, timestamps, fsample) = io.import_fieldtrip(
                                            file_name=file_path,
                                            ft_struct_name='data',
                                            file_version='v7.3')
    assert data.n_processes == 14, (
        'Wrong number of processes, expected 14, found: {0}').format(
            data.n_processes)
    assert data.n_replications == 135, (
        'Wrong number of replications, expected 135, found: {0}').format(
            data.n_replications)
    assert data.n_samples == 1200, (
        'Wrong number of samples, expected 1200, found: {0}').format(
            data.n_samples)

    assert label[0] == 'VirtualChannel_3491_pc1', (
        'Wrong channel name for label 0.')
    assert label[10] == 'VirtualChannel_1573_pc2', (
        'Wrong channel name for label 10.')
    assert label[30] == 'VirtualChannel_1804_pc1', (
        'Wrong channel name for label 30.')
    assert fsample == 600, ('Wrong sampling frequency: {0}'.format(fsample))
    print(timestamps)  # TODO add assertion for this


def test_import_matarray():
    """Test MATLAB importer."""
    n_samples = 20  # no. samples in the example data
    n_processes = 2  # no. processes in the example data
    n_replications = 3  # no. replications in the example data

    # Load hdf5, one to three dimensions.
    data = io.import_matarray(
            file_name=resource_filename(__name__, 'data/one_dim_v7_3.mat'),
            array_name='a',
            dim_order='s',
            file_version='v7.3',
            normalise=False)
    assert data.n_samples == n_samples, (
        'Wrong number of samples: {0}.'.format(data.n_samples))
    assert data.n_processes == 1, (
        'Wrong number of processes: {0}.'.format(data.n_processes))
    assert data.n_replications == 1, (
        'Wrong number of replications: {0}.'.format(data.n_replications))

    data = io.import_matarray(
            file_name=resource_filename(__name__, 'data/two_dim_v7_3.mat'),
            array_name='b',
            dim_order='sp',
            file_version='v7.3',
            normalise=False)
    assert data.n_samples == n_samples, (
        'Wrong number of samples: {0}.'.format(data.n_samples))
    assert data.n_processes == n_processes, (
        'Wrong number of processes: {0}.'.format(data.n_processes))
    assert data.n_replications == 1, (
        'Wrong number of replications: {0}.'.format(data.n_replications))

    data = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='c',
            dim_order='rsp',
            file_version='v7.3',
            normalise=False)
    assert data.n_samples == n_samples, (
        'Wrong number of samples: {0}.'.format(data.n_samples))
    assert data.n_processes == n_processes, (
        'Wrong number of processes: {0}.'.format(data.n_processes))
    assert data.n_replications == n_replications, (
            'Wrong number of replications: {0}.'.format(data.n_replications))

    # Load matlab versions 4, 6, 7.
    file_path = [
        resource_filename(__name__, 'data/two_dim_v4.mat'),
        resource_filename(__name__, 'data/two_dim_v6.mat'),
        resource_filename(__name__, 'data/two_dim_v7.mat')]
    file_version = ['v4', 'v6', 'v7']
    for i in range(3):
        data = io.import_matarray(file_name=file_path[i],
                                  array_name='b',
                                  dim_order='ps',
                                  file_version=file_version[i],
                                  normalise=False)
        assert data.n_processes == n_processes, (
            'Wrong number of processes'.format(data.n_processes))
        assert data.n_samples == n_samples, (
            'Wrong number of samples'.format(data.n_samples))
        assert data.n_replications == 1, (
            'Wrong number of replications'.format(data.n_replications))

    # Load wrong file name.
    with pytest.raises(FileNotFoundError):
        data = io.import_matarray(file_name='test',
                                  array_name='b',
                                  dim_order='ps',
                                  file_version='v6',
                                  normalise=False)

    # Test wrong variable name.
    with pytest.raises(RuntimeError):
        data = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='test',
            dim_order='rsp',
            file_version='v7.3',
            normalise=False)

    # Test wrong dim order.
    with pytest.raises(RuntimeError):
        data = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='c',
            dim_order='rp',
            file_version='v7.3',
            normalise=False)

    # Test wrong file version
    with pytest.raises(RuntimeError):
        data = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='c',
            dim_order='rp',
            file_version='v4',
            normalise=False)


if __name__ == '__main__':
    test_export_brain_net()
    test_export_networkx()
    test_import_matarray()
    test_import_fieldtrip()
