"""Test network comparison.

This module provides unit/system tests for network comparison between and
within subjects.

@author: patricia
"""
import pytest
import numpy as np
import os
from idtxl.network_comparison import Network_comparison
from idtxl.data import Data
from idtxl import idtxl_io

# The following was ran once to generate example data, which is now in the
# data sub-folder of the test-folder.

## Generate example data
#dat = Data()
#dat.generate_mute_data(100, 5)
#
## analysis settings
#analysis_opts = {
#        'cmi_calc_name': 'jidt_kraskov',
#        'n_perm_max_stat': 50,
#        'n_perm_min_stat': 50,
#        'n_perm_omnibus': 200,
#        'n_perm_max_seq': 50,
#        }
#max_lag_target = 5
#max_lag_sources = 5
#min_lag_sources = 1
#
## network inference for individual data sets
#nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                       analysis_opts)
#res_0 = nw_0.analyse_network(dat, targets=[0, 1], sources='all')
#
#nw_1 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                       analysis_opts)
#res_1 = nw_1.analyse_network(dat,  targets=[1, 2], sources='all')
#
#nw_2 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                       analysis_opts)
#res_2 = nw_2.analyse_network(dat,  targets=[0, 2], sources='all')
#
#nw_3 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                       analysis_opts)
#res_3 = nw_3.analyse_network(dat,  targets=[0, 1, 2], sources='all')
#
#nw_4 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                       analysis_opts)
#res_4 = nw_4.analyse_network(dat,  targets=[1, 2], sources='all')
#
#path = '/home/patriciaw/repos/IDTxl/test/data/'
#idtxl_io.save_pickle(res_0, path + 'mute_res_0')
#idtxl_io.save_pickle(res_1, path + 'mute_res_1')
#idtxl_io.save_pickle(res_2, path + 'mute_res_2')
#idtxl_io.save_pickle(res_3, path + 'mute_res_3')
#idtxl_io.save_pickle(res_4, path + 'mute_res_4')



def test_network_comparison_use_cases():
    dat = Data()
    dat.generate_mute_data(100, 5)

    # Load previously generated example data
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))
    res_2 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_2.pkl'))
    res_3 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_3.pkl'))
    res_4 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_4.pkl'))

#    path = '/home/patriciaw/repos/IDTxl/test/data/'
#    res_0 = idtxl_io.load_pickle(path + 'mute_res_0')
#    res_1 = idtxl_io.load_pickle(path + 'mute_res_1')
#    res_2 = idtxl_io.load_pickle(path + 'mute_res_2')
#    res_3 = idtxl_io.load_pickle(path + 'mute_res_3')
#    res_4 = idtxl_io.load_pickle(path + 'mute_res_4')

    # comparison options
    comparison_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            }
    comparison_opts['n_perm_comp'] = 6
    comparison_opts['alpha_comp'] = 0.2
    comparison_opts['tail'] = 'two'

    print('\n\nTEST 0 - independent within')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    c_0 = comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 1 - dependent within')
    comparison_opts['stats_type'] = 'dependent'
    comp = Network_comparison(comparison_opts)
    c_1 = comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 2 - independent between')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    c_2 = comp.compare_between(network_set_a=np.array((res_0, res_1)),
                               network_set_b=np.array((res_2, res_3)),
                               data_set_a=np.array((dat, dat)),
                               data_set_b=np.array((dat, dat)))

    print('\n\nTEST 3 - dependent between')
    comp = Network_comparison(comparison_opts)
    comparison_opts ['stats_type'] = 'dependent'
    c_3 = comp.compare_between(network_set_a=np.array((res_0, res_1)),
                               network_set_b=np.array((res_2, res_3)),
                               data_set_a=np.array((dat, dat)),
                               data_set_b=np.array((dat, dat)))

    print('\n\nTEST 4 - independent within unbalanced')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    c_4 = comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 5 - independent between unbalanced')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    c_5 = comp.compare_between(network_set_a=np.array((res_0, res_1)),
                               network_set_b=np.array((res_2, res_3, res_4)),
                               data_set_a=np.array((dat, dat)),
                               data_set_b=np.array((dat, dat, dat)))

def test_assertions():
    dat = Data()
    dat.generate_mute_data(100, 5)

    # Load previously generated example data
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))
    res_2 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_2.pkl'))
    res_3 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_3.pkl'))
    res_4 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_4.pkl'))

    # comparison options
    comparison_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'tail': 'two'
            }

    # no. permutations insufficient for requested alpha
    with pytest.raises(RuntimeError):
        comparison_opts['n_perm_comp'] = 6
        comparison_opts['alpha_comp'] = 0.001
        comparison_opts['stats_type'] = 'independent'
        comp = Network_comparison(comparison_opts)

    # data sets have unequal no. realisations
    with pytest.raises(RuntimeError):
        dat2 = Data()
        dat2.generate_mute_data(100, 3)
        print('\n\nTEST 1 - dependent within')
        comparison_opts['stats_type'] = 'dependent'
        comp = Network_comparison(comparison_opts)
        comp.compare_within(res_0, res_1, dat, dat2)

    # no. replications too small for dependent-samples test
    comparison_opts['n_perm_comp'] = 100
    with pytest.raises(RuntimeError):
        comparison_opts['stats_type'] = 'dependent'
        comp = Network_comparison(comparison_opts)
        comp.compare_between(network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((dat, dat)),
                             data_set_b=np.array((dat, dat)))
    with pytest.raises(RuntimeError):
        comparison_opts['stats_type'] = 'dependent'
        comp = Network_comparison(comparison_opts)
        comp.compare_within(res_0, res_1, dat2, dat2)


    print('\n\nTEST 3 - dependent between')
    comp = Network_comparison(comparison_opts)
    comparison_opts ['stats_type'] = 'dependent'
    c_3 = comp.compare_between(network_set_a=np.array((res_0, res_1)),
                               network_set_b=np.array((res_2, res_3)),
                               data_set_a=np.array((dat, dat)),
                               data_set_b=np.array((dat, dat)))

    print('\n\nTEST 4 - independent within unbalanced')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    c_4 = comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 5 - independent between unbalanced')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    c_5 = comp.compare_between(network_set_a=np.array((res_0, res_1)),
                               network_set_b=np.array((res_2, res_3, res_4)),
                               data_set_a=np.array((dat, dat)),
                               data_set_b=np.array((dat, dat, dat)))


if __name__ == '__main__':
    test_assertions()
    test_network_comparison_use_cases()
