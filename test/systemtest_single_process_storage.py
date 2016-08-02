"""System test for AIS estimation on example data."""
import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_ais
from idtxl.data import Data
import idtxl.idtxl_utils
from idtxl.single_process_storage import Single_process_storage

def test_single_source_storage_opencl():
    """Test AIS estimation in MuTE example network."""
    dat = Data()
    dat.generate_mute_data(1000, 5)
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'opencl_kraskov',
        'n_perm': 22,
        'alpha': 0.05,
        'tail': 'one',
        }
    processes = [1, 2, 3]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res = network_analysis.analyse_network(dat, processes)
    print('AIS for MUTE data proc 1: {0}'.format(res[1]['ais']))
    print('AIS for MUTE data proc 2: {0}'.format(res[2]['ais']))
    print('AIS for MUTE data proc 3: {0}'.format(res[3]['ais']))

def test_single_source_storage_jidt():
    """Test AIS estimation in MuTE example network."""
    dat = Data()
    dat.generate_mute_data(1000, 5)
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm': 22,
        'alpha': 0.05,
        'tail': 'one',
        }
    processes = [1, 2, 3]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res = network_analysis.analyse_network(dat, processes)
    print('AIS for MUTE data proc 1: {0}'.format(res[1]['ais']))
    print('AIS for MUTE data proc 2: {0}'.format(res[2]['ais']))
    print('AIS for MUTE data proc 3: {0}'.format(res[3]['ais']))


if __name__ == '__main__':
    test_single_source_storage_jidt()
    test_single_source_storage_opencl()

