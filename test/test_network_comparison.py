"""Test network comparison.

This module provides unit/system tests for network comparison between and
within subjects.

@author: patricia
"""
import os
import random as rn
import pytest
import numpy as np
# from idtxl.multivariate_te import Multivariate_te
from idtxl.network_comparison import Network_comparison
from idtxl.data import Data

# # Generate example data: the following was ran once to generate example data,
# # which is now in the data sub-folder of the test-folder.
# dat = Data()
# dat.generate_mute_data(100, 5)
# # analysis settings
# analysis_opts = {
#     'cmi_calc_name': 'jidt_kraskov',
#     'n_perm_max_stat': 50,
#     'n_perm_min_stat': 50,
#     'n_perm_omnibus': 200,
#     'n_perm_max_seq': 50,
#     }
# max_lag_target = 5
# max_lag_sources = 5
# min_lag_sources = 1
# # network inference for individual data sets
# nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                     analysis_opts)
# res_0 = nw_0.analyse_network(dat, targets=[0, 1], sources='all')
#
# nw_1 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                        analysis_opts)
# res_1 = nw_1.analyse_network(dat,  targets=[1, 2], sources='all')
#
# nw_2 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                        analysis_opts)
# res_2 = nw_2.analyse_network(dat,  targets=[0, 2], sources='all')
#
# nw_3 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                        analysis_opts)
# res_3 = nw_3.analyse_network(dat,  targets=[0, 1, 2], sources='all')
#
# nw_4 = Multivariate_te(max_lag_sources, min_lag_sources, max_lag_target,
#                        analysis_opts)
# res_4 = nw_4.analyse_network(dat,  targets=[1, 2], sources='all')
#
# path = '/home/patriciaw/repos/IDTxl/test/data/'
# idtxl_io.save_pickle(res_0, path + 'mute_res_0')
# idtxl_io.save_pickle(res_1, path + 'mute_res_1')
# idtxl_io.save_pickle(res_2, path + 'mute_res_2')
# idtxl_io.save_pickle(res_3, path + 'mute_res_3')
# idtxl_io.save_pickle(res_4, path + 'mute_res_4')


def test_network_comparison_use_cases():
    """Run all intended use cases, within/between, dependent/independent."""
    dat = Data()
    dat.generate_mute_data(100, 5)

    # Load previously generated example data (pickled)
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
    comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 1 - dependent within')
    comparison_opts['stats_type'] = 'dependent'
    comp = Network_comparison(comparison_opts)
    comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 2 - independent between')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    comp.compare_between(network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3)),
                         data_set_a=np.array((dat, dat)),
                         data_set_b=np.array((dat, dat)))

    print('\n\nTEST 3 - dependent between')
    comp = Network_comparison(comparison_opts)
    comparison_opts['stats_type'] = 'dependent'
    comp.compare_between(network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3)),
                         data_set_a=np.array((dat, dat)),
                         data_set_b=np.array((dat, dat)))

    print('\n\nTEST 4 - independent within unbalanced')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    comp.compare_within(res_0, res_1, dat, dat)

    print('\n\nTEST 5 - independent between unbalanced')
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    comp.compare_between(network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3, res_4)),
                         data_set_a=np.array((dat, dat)),
                         data_set_b=np.array((dat, dat, dat)))


def test_assertions():
    """Test if input checks raise errors."""
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
    comparison_opts['n_perm_comp'] = 6
    comparison_opts['alpha_comp'] = 0.001
    comparison_opts['stats_type'] = 'independent'
    with pytest.raises(RuntimeError):
        comp = Network_comparison(comparison_opts)

    # data sets have unequal no. replications
    dat2 = Data()
    dat2.generate_mute_data(100, 3)
    comparison_opts['stats_type'] = 'dependent'
    comparison_opts['alpha_comp'] = 0.05
    comparison_opts['n_perm_comp'] = 1000
    comp = Network_comparison(comparison_opts)
    with pytest.raises(AssertionError):
        comp.compare_within(res_0, res_1, dat, dat2)

    # data sets have unequal no. realisations
    dat2 = Data()
    dat2.generate_mute_data(80, 5)
    comparison_opts['stats_type'] = 'dependent'
    comparison_opts['alpha_comp'] = 0.05
    comparison_opts['n_perm_comp'] = 21
    comp = Network_comparison(comparison_opts)
    with pytest.raises(RuntimeError):
        comp.compare_within(res_0, res_1, dat, dat2)

    # no. replications/subjects too small for dependent-samples test
    comparison_opts['stats_type'] = 'dependent'
    comparison_opts['n_perm_comp'] = 1000
    comp = Network_comparison(comparison_opts)
    with pytest.raises(RuntimeError):   # between
        comp.compare_between(network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((dat, dat)),
                             data_set_b=np.array((dat, dat)))
    with pytest.raises(RuntimeError):   # within
        comp.compare_within(res_0, res_1, dat2, dat2)

    # no. replications/subjects too small for independent-samples test
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    with pytest.raises(RuntimeError):   # between
        comp.compare_between(network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((dat, dat)),
                             data_set_b=np.array((dat, dat)))
    with pytest.raises(RuntimeError):   # within
        comp.compare_within(res_0, res_1, dat2, dat2)

    # add target to network that is not in the data object
    res_99 = res_0
    res_99[99] = res_99[1]
    comparison_opts['alpha_comp'] = 0.05
    comparison_opts['n_perm_comp'] = 21
    comp = Network_comparison(comparison_opts)
    with pytest.raises(IndexError):
        comp.compare_within(res_0, res_99, dat2, dat2)


def test_create_union_network():
    """Test creation of union of multiple networks."""
    dat1 = Data()
    dat1.generate_mute_data(100, 5)
    dat2 = Data()
    dat2.generate_mute_data(100, 5)

    # Load previously generated example data
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))

    # comparison options
    comparison_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'tail': 'two',
            'n_perm_comp': 6,
            'alpha_comp': 0.2,
            'stats_type': 'independent'
            }
    comp = Network_comparison(comparison_opts)

    src_1 = [(0, 2), (0, 1)]
    src_2 = [(0, 4), (0, 5)]
    res_0[1]['selected_vars_sources'] = src_1
    res_1[1]['selected_vars_sources'] = src_2

    comp._create_union(res_0, res_1)
    ref_targets = [0, 1, 2]
    assert (comp.union['targets'] == ref_targets).all(), (
                                        'Union does not include all targets.')
    assert np.array([True for i in ref_targets if
                     i in comp.union.keys()]).all(), (
                                'Not all targets contained in union network.')
    assert comp.union['max_lag'] == res_0[0]['current_value'][1], (
                                    'The max. lag was not defined correctly.')

    src_union = comp._idx_to_lag(comp.union[1]['selected_vars_sources'],
                                 comp.union['max_lag'])
    assert src_union == (src_1 + src_2), ('Sources for target 1 were not '
                                          'combined correctly.')

    # unequal current values in single networks
    res_0[1]['current_value'] = (1, 7)  # the original is (1, 5)
    with pytest.raises(ValueError):
        comp._create_union(res_0, res_1)


def test_get_permuted_replications():
    """Test if permutation of replications works."""
    # Load previously generated example data
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))

    comparison_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'tail': 'two',
            'n_perm_comp': 6,
            'alpha_comp': 0.2,
            'stats_type': 'dependent'
            }
    comp = Network_comparison(comparison_opts)
    comp._create_union(res_0, res_1)

    # Check permutation for dependent samples test: Replace realisations by
    # zeros and ones, check if realisations get swapped correctly.
    dat1 = Data()
    dat1.normalise = False
    dat1.set_data(np.zeros((5, 100, 5)), 'psr')
    dat2 = Data()
    dat2.normalise = False
    dat2.set_data(np.ones((5, 100, 5)), 'psr')
    [cond_a_perm,
     cv_a_perm,
     cond_b_perm,
     cv_b_perm] = comp._get_permuted_replications(data_a=dat1,
                                                  data_b=dat2,
                                                  target=1)
    n_vars = cond_a_perm.shape[1]
    assert (np.sum(cond_a_perm + cond_b_perm, axis=1) == n_vars).all(), (
                'Dependent samples permutation did not work correctly.')
    assert np.logical_xor(cond_a_perm, cond_b_perm).all(), (
                'Dependent samples permutation did not work correctly.')

    # Check permutations for independent samples test: Check the sum over
    # realisations.
    comparison_opts['stats_type'] = 'independent'
    comp = Network_comparison(comparison_opts)
    comp._create_union(res_0, res_1)
    [cond_a_perm,
     cv_a_perm,
     cond_b_perm,
     cv_b_perm] = comp._get_permuted_replications(data_a=dat1,
                                                  data_b=dat2,
                                                  target=1)
    n_samples = n_vars * dat1.n_realisations((0, comp.union['max_lag']))
    assert np.sum(cond_a_perm + cond_b_perm, axis=None) == n_samples, (
                'Independent samples permutation did not work correctly.')

    # test unequal number of replications
    dat2.generate_mute_data(100, 7)
    with pytest.raises(AssertionError):
        comp._get_permuted_replications(data_a=dat1, data_b=dat2, target=1)


def test_calculate_cmi():
    """Test if the CMI is estimated correctly."""
    dat = Data()
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [0] + [sum(pair) for pair in zip(
        [cov * y for y in source[0:n - 1]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    dat.set_data(np.vstack((source, target)), 'ps')
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    comparison_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'tail': 'two',
        'n_perm_comp': 6,
        'alpha_comp': 0.2,
        'stats_type': 'dependent'
        }
    comp = Network_comparison(comparison_opts)
    comp._create_union(res_0)
    comp.union[1]['selected_vars_sources'] = [(0, 4)]
    cmi = comp._calculate_cmi(dat)
    cmi_expected = np.log(1 / (1 - cov ** 2))
    print('correlated Gaussians: TE result {0:.4f} bits; expected to be '
          '{1:0.4f} bit for the copy'.format(cmi[1][0], cmi_expected))
    np.testing.assert_almost_equal(cmi[1][0], cmi_expected, decimal=1,
                   err_msg='when calculating cmi for correlated Gaussians.')


def test_calculate_mean():
    """Test if mean over CMI estimates is calculated correctly."""
    dat = Data()
    dat.generate_mute_data(100, 5)
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    comparison_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'tail': 'two',
        'n_perm_comp': 6,
        'alpha_comp': 0.2,
        'stats_type': 'dependent'
        }
    comp = Network_comparison(comparison_opts)
    comp._create_union(res_0)
    cmi = comp._calculate_cmi(dat)
    cmi_mean = comp._calculate_mean([cmi, cmi])
    for t in comp.union['targets']:
        assert (cmi_mean[t] == cmi[t]).all(), ('Error in mean of CMI for '
                                               'target {0}'.format(t))


def test_p_value_union():
    """Test if the p-value is calculated correctly."""
    dat = Data()
    dat.generate_mute_data(100, 5)
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))
    comparison_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'tail': 'one',
        'n_perm_comp': 6,
        'alpha_comp': 0.2,
        'stats_type': 'independent'
        }
    comp = Network_comparison(comparison_opts)
    comp.compare_within(res_0, res_1, dat, dat)

    # Replace the surrogate CMI by all zeros for source 0 and all ones for
    # source 1. Set the CMI difference to 0.5 for both sources. Check if this
    # results in one significant and one non-significant result with the
    # correct p-values.
    target = 1
    for p in range(comp.n_permutations):
        comp.cmi_surr[p][target] = np.array([0, 1])
    comp.cmi_diff[target] = np.array([0.5, 0.5])
    [p, s] = comp._p_value_union()
    assert (s[target] == np.array([True, False])).all(), (
                                    'The significance was not determined '
                                    'correctly: {0}'.fomat(s[target]))
    p_1 = 1 / comparison_opts['n_perm_comp']
    p_2 = 1.0
    print(p[target])
    assert (p[target] == np.array([p_1, p_2])).all(), (
                                'The p-value was not calculated correctly: {0}'
                                .fomat(p[target]))

if __name__ == '__main__':
    test_network_comparison_use_cases()
    test_p_value_union()
    test_calculate_mean()
    test_calculate_cmi()
    test_get_permuted_replications()
    test_assertions()
    test_create_union_network()
