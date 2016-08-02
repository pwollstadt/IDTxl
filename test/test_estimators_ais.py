import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_ais
from idtxl.data import Data
import idtxl.idtxl_utils
from idtxl.single_process_storage import Single_process_storage


def test_ais_gaussian():
    """Test multivariate TE estimation on correlated Gaussians.

    Run the multivariate TE algorithm on two sets of random Gaussian data with
    a given covariance. The second data set is shifted by one sample creating
    a source-target delay of one sample. This example is modeled after the
    JIDT demo 4 for transfer entropy. The resulting TE can be compared to the
    analytical result (but expect some error in the estimate).

    Note:
        This test runs considerably faster than other system tests.
        This produces strange small values for non-coupled sources.  TODO
    """
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # Cast everything to numpy so the idtxl estimator understands it.
    source = np.array(source)

    analysis_opts = {
        'kraskov_k': 4,
        'normalise': 'false',
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau': 1,
        'history': 1,
        }
    ais_est = Estimator_ais('jidt_kraskov')
    ais = ais_est.estimate(source, analysis_opts)
    print('AIS for random normal data without memory (calling estimator directly, expected is something close to 0): {0}'.format(ais))

def test_single_source_storage_gaussian():
    n = 1000
    cov = 0.4
    proc_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    proc_2 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # Cast everything to numpy so the idtxl estimator understands it.
    dat = Data(np.array([proc_1, proc_2]), dim_order='ps')
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm': 22,
        'alpha': 0.05,
        'tail': 'one',
        }
    processes = [1]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res = network_analysis.analyse_network(dat, processes)
    print('AIS for random normal data without memory (using analysis class, expected is NaN): {0}'.format(res[1]['ais']))

def test_single_source_storage_jidt():
    dat = Data()
    dat.generate_mute_data(100, 5)
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
    print('AIS for MUTE data proc 1 (using analysis class): {0}'.format(res[1]['ais']))
    print('AIS for MUTE data proc 2 (using analysis class): {0}'.format(res[2]['ais']))
    print('AIS for MUTE data proc 3 (using analysis class): {0}'.format(res[3]['ais']))

def test_single_source_storage_opencl():
    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag = 5
    analysis_opts = {
        # 'cmi_calc_name': 'jidt_kraskov',
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

def test_compare_jidt_open_cl_estimator():
    """Compare results from OpenCl and JIDT estimators for AIS calculation."""
    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'opencl_kraskov',
        'n_perm': 22,
        'alpha': 0.05,
        'tail': 'one',
        }
    processes = [2, 3]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res_opencl = network_analysis.analyse_network(dat, processes)
    analysis_opts['cmi_calc_name'] = 'jidt_kraskov'
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res_jidt = network_analysis.analyse_network(dat, processes)
    np.testing.assert_approx_equal(res_opencl[2]['ais'], res_jidt[2]['ais'], significanct=7, 
                                   err_msg='AIS results differ between OpenCl and JIDT estimator.')
    np.testing.assert_approx_equal(res_opencl[3]['ais'], res_jidt[3]['ais'], significanct=7, 
                                   err_msg='AIS results differ between OpenCl and JIDT estimator.')
    print('AIS for MUTE data proc 2 - opencl: {0} and jidt: {1}'.format(res_opencl[2]['ais'], res_jidt[2]['ais']))
    print('AIS for MUTE data proc 3 - opencl: {0} and jidt: {1}'.format(res_opencl[3]['ais'], res_jidt[3]['ais']))

if __name__ == '__main__':
    # test_single_source_storage_jidt()
    # test_single_source_storage_opencl()
    # test_single_source_storage_gaussian()
    # test_ais_gaussian()
    test_compare_jidt_open_cl_estimator()
