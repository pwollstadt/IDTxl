"""Test AIS analysis class.

This module provides unit tests for the AIS analysis class.
"""
import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_ais
from idtxl.data import Data
import idtxl.idtxl_utils
from idtxl.single_process_storage import Single_process_storage

def test_single_source_storage_gaussian():
    n = 1000
    proc_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    proc_2 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # Cast everything to numpy so the idtxl estimator understands it.
    dat = Data(np.array([proc_1, proc_2]), dim_order='ps')
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_mi': 22,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        }
    processes = [1]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res = network_analysis.analyse_network(dat, processes)
    print('AIS for random normal data without memory (expected is NaN): {0}'.format(res[1]['ais']))
    assert res[1]['ais'] is np.nan, 'Estimator did not return nan for memoryless data.'

def test_compare_jidt_open_cl_estimator():
    """Compare results from OpenCl and JIDT estimators for AIS calculation."""
    dat = Data()
    dat.generate_mute_data(100, 2)
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'opencl_kraskov',
        'n_perm_mi': 22,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        }
    processes = [2, 3]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res_opencl = network_analysis.analyse_network(dat, processes)
    analysis_opts['cmi_calc_name'] = 'jidt_kraskov'
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res_jidt = network_analysis.analyse_network(dat, processes)
    # Note that I require equality up to three digits. Results become more exact for bigger
    # data sizes, but this takes too long for a unit test.
    np.testing.assert_approx_equal(res_opencl[2]['ais'], res_jidt[2]['ais'], significant=3, 
                                   err_msg='AIS results differ between OpenCl and JIDT estimator.')
    np.testing.assert_approx_equal(res_opencl[3]['ais'], res_jidt[3]['ais'], significant=3, 
                                   err_msg='AIS results differ between OpenCl and JIDT estimator.')
    print('AIS for MUTE data proc 2 - opencl: {0} and jidt: {1}'.format(res_opencl[2]['ais'], res_jidt[2]['ais']))
    print('AIS for MUTE data proc 3 - opencl: {0} and jidt: {1}'.format(res_opencl[3]['ais'], res_jidt[3]['ais']))

if __name__ == '__main__':
    test_single_source_storage_gaussian()
    test_compare_jidt_open_cl_estimator()
