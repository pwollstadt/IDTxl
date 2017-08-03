"""Test AIS analysis class.

This module provides unit tests for the AIS analysis class.
"""
import pytest
import random as rn
import numpy as np
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from test_estimators_jidt import jpype_missing

package_missing = False
try:
    import pyopencl
except ImportError as err:
    package_missing = True
opencl_missing = pytest.mark.skipif(
    package_missing,
    reason="Jpype is missing, JIDT estimators are not available")


@jpype_missing
def test_ActiveInformationStorage_init():
    """Test instance creation for ActiveInformationStorage class."""
    # Test error on missing estimator
    analysis_opts = {'max_lag': 5}
    data = Data()
    data.generate_mute_data(10, 3)
    ais = ActiveInformationStorage()
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(analysis_opts, data, process=0)

    # Test tau larger than maximum lag
    analysis_opts['cmi_estimator'] = 'JidtKraskovCMI'
    analysis_opts['tau'] = 10
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(analysis_opts, data, process=0)
    # Test negative tau and maximum lag
    analysis_opts['tau'] = -10
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(analysis_opts, data, process=0)
    analysis_opts['tau'] = 1
    analysis_opts['max_lag'] = -5
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(analysis_opts, data, process=0)

    # Invalid: process is not an int
    analysis_opts['max_lag'] = 5
    with pytest.raises(RuntimeError):  # no int
        ais.analyse_single_process(analysis_opts, data, process=1.5)
    with pytest.raises(RuntimeError):  # negative
        ais.analyse_single_process(analysis_opts, data, process=-1)
    with pytest.raises(RuntimeError):  # not in data
        ais.analyse_single_process(analysis_opts, data, process=10)
    with pytest.raises(RuntimeError):  # wrong type
        ais.analyse_single_process(analysis_opts, data, process={})

    # Force conditionals
    analysis_opts['add_conditionals'] = [(0, 1), (1, 3)]
    analysis_opts['n_perm_max_stat'] = 21
    analysis_opts['n_perm_min_stat'] = 21
    res = ais.analyse_single_process(analysis_opts, data, process=0)


def test_analyse_network():
    """Test AIS estimation for the whole network."""
    opts = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'max_lag': 5,
        'tau': 1}
    data = Data()
    data.generate_mute_data(10, 3)
    ais = ActiveInformationStorage()
    # Test analysis of 'all' processes
    r = ais.analyse_network(opts, data)
    k = list(r.keys())
    assert all(np.array(k) == np.arange(data.n_processes)), (
                'Network analysis did not run on all targets.')
    # Test check for correct definition of processes
    with pytest.raises(ValueError):  # no list
        ais.analyse_network(opts, data=data, processes={})
    with pytest.raises(ValueError):  # no list of ints
        ais.analyse_network(opts, data=data, processes=[1.5, 0.7])


@jpype_missing
def test_single_source_storage_gaussian():
    n = 1000
    proc_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    proc_2 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # Cast everything to numpy so the idtxl estimator understands it.
    dat = Data(np.array([proc_1, proc_2]), dim_order='ps')
    analysis_opts = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_mi': 50,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        'n_perm_max_stat': 21,
        'max_lag': 5,
        'tau': 1
        }
    processes = [1]
    network_analysis = ActiveInformationStorage()
    res = network_analysis.analyse_network(analysis_opts, dat, processes)
    print('AIS for random normal data without memory (expected is NaN): '
          '{0}'.format(res[1]['ais']))
    assert res[1]['ais'] is np.nan, ('Estimator did not return nan for '
                                     'memoryless data.')


@jpype_missing
@opencl_missing
def test_compare_jidt_open_cl_estimator():
    """Compare results from OpenCl and JIDT estimators for AIS calculation."""
    dat = Data()
    dat.generate_mute_data(100, 2)
    analysis_opts = {
        'cmi_estimator': 'OpenCLKraskovCMI',
        'n_perm_mi': 22,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        'n_perm_max_stat': 21,
        'max_lag': 5,
        'tau': 1
        }
    processes = [2, 3]
    network_analysis = ActiveInformationStorage()
    res_opencl = network_analysis.analyse_network(analysis_opts, dat,
                                                  processes)
    analysis_opts['cmi_estimator'] = 'JidtKraskovCMI'
    res_jidt = network_analysis.analyse_network(analysis_opts, dat, processes)
    # Note that I require equality up to three digits. Results become more
    # exact for bigger data sizes, but this takes too long for a unit test.
    print('AIS for MUTE data proc 2 - opencl: {0} and jidt: {1}'.format(
                                    res_opencl[2]['ais'], res_jidt[2]['ais']))
    print('AIS for MUTE data proc 3 - opencl: {0} and jidt: {1}'.format(
                                    res_opencl[3]['ais'], res_jidt[3]['ais']))
    if not (res_opencl[2]['ais'] is np.nan or res_jidt[2]['ais'] is np.nan):
        assert (res_opencl[2]['ais'] - res_jidt[2]['ais']) < 0.05, (
                       'AIS results differ between OpenCl and JIDT estimator.')
    else:
        assert res_opencl[2]['ais'] == res_jidt[2]['ais'], (
                       'AIS results differ between OpenCl and JIDT estimator.')
    if not (res_opencl[3]['ais'] is np.nan or res_jidt[3]['ais'] is np.nan):
        assert (res_opencl[3]['ais'] - res_jidt[3]['ais']) < 0.05, (
                       'AIS results differ between OpenCl and JIDT estimator.')
    else:
        assert res_opencl[3]['ais'] == res_jidt[3]['ais'], (
                       'AIS results differ between OpenCl and JIDT estimator.')
#    np.testing.assert_approx_equal(res_opencl[2]['ais'], res_jidt[2]['ais'],
#                                   significant=3,
#                                   err_msg=('AIS results differ between '
#                                            'OpenCl and JIDT estimator.'))
#    np.testing.assert_approx_equal(res_opencl[3]['ais'], res_jidt[3]['ais'],
#                                   significant=3,
#                                   err_msg=('AIS results differ between '
#                                            'OpenCl and JIDT estimator.'))


if __name__ == '__main__':
    test_analyse_network()
    test_ActiveInformationStorage_init()
    test_single_source_storage_gaussian()
    test_compare_jidt_open_cl_estimator()
