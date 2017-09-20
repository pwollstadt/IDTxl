"""Unit tests for IDTxl I/O functions."""
import os
import pytest
import numpy as np
from pkg_resources import resource_filename
from idtxl import idtxl_io as io
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from test_estimators_jidt import jpype_missing


# @jpype_missing
# def test_save_te_results():
#     """Test saving of TE results."""
#     # Generate some example output
#     data = Data()
#     data.generate_mute_data(100, 2)
#     settings = {
#         'cmi_estimator': 'JidtKraskovCMI',
#         'n_perm_mi': 22,
#         'alpha_mi': 0.05,
#         'tail_mi': 'one',
#         }
#     processes = [2, 3]
#     network_analysis = ActiveInformationStorage(max_lag=5,
#                                                 tau=1,
#                                                 options=settings)
#     res_ais = network_analysis.analyse_network(data, processes)

#     cwd = os.getcwd()
#     fp = ''.join([cwd, '/idtxl_unit_test/'])
#     if not os.path.exists(fp):
#         os.makedirs(fp)
#     io.save(res_ais, file_path=''.join([fp, 'res_ais']))
#     f = io.load(file_path=''.join([fp, 'res_single.txt']))
#     print('THIS MODULE IS NOT YET WORKING!')
#     assert (f is not None), 'File read from disk is None.'


def test_import_fieldtrip():
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
    n_samples = 20  # no. samples in the example data
    n_processes = 2  # no. processes in the example data
    n_replications = 3  # no. replications in the example data

    # Load hdf5, one to three dimensions.
    (data, label, timestamps, fsample) = io.import_matarray(
            file_name=resource_filename(__name__, 'data/one_dim_v7_3.mat'),
            array_name='a',
            dim_order='s',
            file_version='v7.3',
            normalise=False)
    assert fsample == 1, ('Wrong sampling frequency: {0}'.format(fsample))
    assert all(timestamps == np.arange(n_samples)), (
        'Wrong time stamps: {0}'.format(timestamps))
    assert label[0] == 'channel_000', ('Wrong channel label: {0}.'.format(
                                                                    label[0]))
    assert data.n_samples == n_samples, (
        'Wrong number of samples: {0}.'.format(data.n_samples))
    assert data.n_processes == 1, (
        'Wrong number of processes: {0}.'.format(data.n_processes))
    assert data.n_replications == 1, (
        'Wrong number of replications: {0}.'.format(data.n_replications))

    (data, label, timestamps, fsample) = io.import_matarray(
            file_name=resource_filename(__name__, 'data/two_dim_v7_3.mat'),
            array_name='b',
            dim_order='sp',
            file_version='v7.3',
            normalise=False)
    assert fsample == 1, ('Wrong sampling frequency: {0}'.format(fsample))
    assert all(timestamps == np.arange(n_samples)), (
        'Wrong time stamps: {0}'.format(timestamps))
    assert label[0] == 'channel_000', ('Wrong channel label: {0}.'.format(
                                                                    label[0]))
    assert label[1] == 'channel_001', ('Wrong channel label: {0}.'.format(
                                                                    label[1]))
    assert data.n_samples == n_samples, (
        'Wrong number of samples: {0}.'.format(data.n_samples))
    assert data.n_processes == n_processes, (
        'Wrong number of processes: {0}.'.format(data.n_processes))
    assert data.n_replications == 1, (
        'Wrong number of replications: {0}.'.format(data.n_replications))

    (data, label, timestamps, fsample) = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='c',
            dim_order='rsp',
            file_version='v7.3',
            normalise=False)
    assert fsample == 1, ('Wrong sampling frequency: {0}'.format(fsample))
    assert all(timestamps == np.arange(n_samples)), (
        'Wrong time stamps: {0}'.format(timestamps))
    assert label[0] == 'channel_000', ('Wrong channel label: {0}.'.format(
                                                                    label[0]))
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
        (data, label, timestamps, fsample) = io.import_matarray(
                                                file_name=file_path[i],
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
        (data, label, timestamps, fsample) = io.import_matarray(
                                                file_name='test',
                                                array_name='b',
                                                dim_order='ps',
                                                file_version='v6',
                                                normalise=False)

    # Test wrong variable name.
    with pytest.raises(RuntimeError):
        (data, label, timestamps, fsample) = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='test',
            dim_order='rsp',
            file_version='v7.3',
            normalise=False)

    # Test wrong dim order.
    with pytest.raises(RuntimeError):
        (data, label, timestamps, fsample) = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='c',
            dim_order='rp',
            file_version='v7.3',
            normalise=False)

    # Test wrong file version
    with pytest.raises(RuntimeError):
        (data, label, timestamps, fsample) = io.import_matarray(
            file_name=resource_filename(__name__, 'data/three_dim_v7_3.mat'),
            array_name='c',
            dim_order='rp',
            file_version='v4',
            normalise=False)


if __name__ == '__main__':
    test_import_matarray()
#     test_import_fieldtrip()
#     test_save_te_results()
