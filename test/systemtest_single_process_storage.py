"""System test for AIS estimation on example data."""
from idtxl.data import Data
from idtxl.single_process_storage import Single_process_storage


def test_single_source_storage_opencl():
    """Test AIS estimation in MuTE example network."""
    data = Data()
    data.generate_mute_data(1000, 5)
    settings = {
        'cmi_estimator': 'OpenCLKraskovCMI',
        'max_lag': 5,
        'tau': 1,
        'n_perm_mi': 22,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        }
    processes = [1, 2, 3]
    network_analysis = Single_process_storage()
    results = network_analysis.analyse_network(settings, data, processes)
    print('AIS for MUTE data proc 1: {0}'.format(results[1]['ais']))
    print('AIS for MUTE data proc 2: {0}'.format(results[2]['ais']))
    print('AIS for MUTE data proc 3: {0}'.format(results[3]['ais']))


def test_single_source_storage_jidt():
    """Test AIS estimation in MuTE example network."""
    data = Data()
    data.generate_mute_data(1000, 5)
    settings = {
        'cmi_estimator':  'JidtKraskovCMI',
        'max_lag': 5,
        'tau': 1,
        'n_perm_mi': 22,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        }
    processes = [1, 2, 3]
    network_analysis = Single_process_storage()
    results = network_analysis.analyse_network(settings, data, processes)
    print('AIS for MUTE data proc 1: {0}'.format(results[1]['ais']))
    print('AIS for MUTE data proc 2: {0}'.format(results[2]['ais']))
    print('AIS for MUTE data proc 3: {0}'.format(results[3]['ais']))


if __name__ == '__main__':
    test_single_source_storage_jidt()
    test_single_source_storage_opencl()
