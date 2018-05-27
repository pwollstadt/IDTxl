"""System test for AIS estimation on example data."""
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage


def test_active_information_storage_opencl():
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
    network_analysis = ActiveInformationStorage()
    results = network_analysis.analyse_network(settings, data, processes)
    print('AIS for MUTE data proc 1: {0}'.format(
        results.get_single_process(1, fdr=False)['ais']))
    print('AIS for MUTE data proc 2: {0}'.format(
        results.get_single_process(2, fdr=False)['ais']))
    print('AIS for MUTE data proc 3: {0}'.format(
        results.get_single_process(3, fdr=False)['ais']))


def test_active_information_storage_jidt():
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
    network_analysis = ActiveInformationStorage()
    results = network_analysis.analyse_network(settings, data, processes)
    print('AIS for MUTE data proc 1: {0}'.format(
        results.get_single_process(1, fdr=False)['ais']))
    print('AIS for MUTE data proc 2: {0}'.format(
        results.get_single_process(2, fdr=False)['ais']))
    print('AIS for MUTE data proc 3: {0}'.format(
        results.get_single_process(3, fdr=False)['ais']))


if __name__ == '__main__':
    test_active_information_storage_jidt()
    test_active_information_storage_opencl()
