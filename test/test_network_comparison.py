"""Test network comparison.

This module provides unit/system tests for network comparison between and
within subjects.

@author: patricia
"""
import os
import pickle
import random as rn
import pytest
import numpy as np
# from idtxl.multivariate_te import MultivariateTE
from idtxl.network_comparison import NetworkComparison
from idtxl.data import Data
from test_estimators_jidt import jpype_missing

# # Generate example data: the following was ran once to generate example data,
# # which is now in the data sub-folder of the test-folder.
# dat = Data()
# dat.generate_mute_data(100, 5)
# # analysis settings
# settings = {
#     'cmi_estimator': 'JidtKraskovCMI',
#     'n_perm_max_stat': 50,
#     'n_perm_min_stat': 50,
#     'n_perm_omnibus': 200,
#     'n_perm_max_seq': 50,
#     'max_lag_target': 5,
#     'max_lag_sources': 5,
#     'min_lag_sources': 1,
#     }
# # network inference for individual data sets
# path = os.path.join(os.path.dirname(__file__) + '/data/'
# nw_0 = MultivariateTE()
# res_0 = nw_0.analyse_network(settings, dat, targets=[0, 1], sources='all')
# pickle.dump(res_0, open(path + 'mute_res_0.pkl', 'wb'))
# res_1 = nw_0.analyse_network(settings, dat,  targets=[1, 2], sources='all')
# pickle.dump(res_1, open(path + 'mute_res_1.pkl', 'wb'))
# res_2 = nw_0.analyse_network(settings, dat,  targets=[0, 2], sources='all')
# pickle.dump(res_2, open(path + 'mute_res_2.pkl', 'wb'))
# res_3 = nw_0.analyse_network(settings, dat,  targets=[0, 1, 2], sources='all')
# pickle.dump(res_3, open(path + 'mute_res_3.pkl', 'wb'))
# res_4 = nw_0.analyse_network(settings, dat,  targets=[1, 2], sources='all')
# pickle.dump(res_4, open(path + 'mute_res_4.pkl', 'wb'))


@jpype_missing
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

#    path = os.path.dirname(__file__) + 'data/'
#    res_0 = idtxl_io.load_pickle(path + 'mute_res_0')
#    res_1 = idtxl_io.load_pickle(path + 'mute_res_1')
#    res_2 = idtxl_io.load_pickle(path + 'mute_res_2')
#    res_3 = idtxl_io.load_pickle(path + 'mute_res_3')
#    res_4 = idtxl_io.load_pickle(path + 'mute_res_4')

    # comparison settings
    comp_settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'alpha_comp': 0.26,
            'n_perm_comp': 4,
            'tail': 'two'
            }

    comp = NetworkComparison()

    print('\n\nTEST 0 - independent within')
    comp_settings['stats_type'] = 'independent'
    comp.compare_within(comp_settings, res_0, res_1, dat, dat)

    print('\n\nTEST 1 - dependent within')
    comp_settings['stats_type'] = 'dependent'
    comp.compare_within(comp_settings, res_0, res_1, dat, dat)

    print('\n\nTEST 2 - independent between')
    comp_settings['stats_type'] = 'independent'
    comp.compare_between(comp_settings,
                         network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3)),
                         data_set_a=np.array((dat, dat)),
                         data_set_b=np.array((dat, dat)))

    print('\n\nTEST 3 - dependent between')
    comp_settings['stats_type'] = 'dependent'
    comp.compare_between(comp_settings,
                         network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3)),
                         data_set_a=np.array((dat, dat)),
                         data_set_b=np.array((dat, dat)))

    print('\n\nTEST 4 - independent within unbalanced')
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    comp.compare_within(comp_settings, res_0, res_1, dat, dat)

    print('\n\nTEST 5 - independent between unbalanced')
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    comp.compare_between(comp_settings,
                         network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3, res_4)),
                         data_set_a=np.array((dat, dat)),
                         data_set_b=np.array((dat, dat, dat)))


@jpype_missing
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

    # comparison settings
    comp_settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'tail': 'two'
            }

    # no. permutations insufficient for requested alpha
    comp_settings['n_perm_comp'] = 6
    comp_settings['alpha_comp'] = 0.001
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):
        comp._initialise(comp_settings)

    # data sets have unequal no. replications
    dat2 = Data()
    dat2.generate_mute_data(100, 3)
    comp_settings['stats_type'] = 'dependent'
    comp_settings['alpha_comp'] = 0.05
    comp_settings['n_perm_comp'] = 1000
    comp = NetworkComparison()
    with pytest.raises(AssertionError):
        comp.compare_within(comp_settings, res_0, res_1, dat, dat2)

    # data sets have unequal no. realisations
    dat2 = Data()
    dat2.generate_mute_data(80, 5)
    comp_settings['stats_type'] = 'dependent'
    comp_settings['alpha_comp'] = 0.05
    comp_settings['n_perm_comp'] = 21
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):
        comp.compare_within(comp_settings, res_0, res_1, dat, dat2)

    # no. replications/subjects too small for dependent-samples test
    comp_settings['stats_type'] = 'dependent'
    comp_settings['n_perm_comp'] = 1000
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):   # between
        comp.compare_between(comp_settings,
                             network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((dat, dat)),
                             data_set_b=np.array((dat, dat)))
    with pytest.raises(RuntimeError):   # within
        comp.compare_within(comp_settings, res_0, res_1, dat2, dat2)

    # no. replications/subjects too small for independent-samples test
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):   # between
        comp.compare_between(comp_settings,
                             network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((dat, dat)),
                             data_set_b=np.array((dat, dat)))
    with pytest.raises(RuntimeError):   # within
        comp.compare_within(comp_settings, res_0, res_1, dat2, dat2)

    # add target to network that is not in the data object
    res_99 = res_0
    res_99[99] = res_99[1]
    comp_settings['alpha_comp'] = 0.05
    comp_settings['n_perm_comp'] = 21
    comp = NetworkComparison()
    with pytest.raises(IndexError):
        comp.compare_within(comp_settings, res_0, res_99, dat2, dat2)


@jpype_missing
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

    # comparison settings
    comp_settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'tail': 'two',
            'n_perm_comp': 6,
            'alpha_comp': 0.2,
            'stats_type': 'independent'
            }
    comp = NetworkComparison()
    comp._initialise(comp_settings)

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


@jpype_missing
def test_get_permuted_replications():
    """Test if permutation of replications works."""
    # Load previously generated example data
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))

    comp_settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'tail': 'two',
            'n_perm_comp': 6,
            'alpha_comp': 0.2,
            'stats_type': 'dependent'
            }
    comp = NetworkComparison()
    comp._initialise(comp_settings)
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
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    comp._initialise(comp_settings)
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


@jpype_missing
def test_calculate_cmi_all_links():
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
    comp_settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'tail': 'two',
        'n_perm_comp': 6,
        'alpha_comp': 0.2,
        'stats_type': 'dependent'
        }
    comp = NetworkComparison()
    comp._initialise(comp_settings)
    comp._create_union(res_0)
    comp.union[1]['selected_vars_sources'] = [(0, 4)]
    cmi = comp._calculate_cmi_all_links(dat)
    cmi_expected = np.log(1 / (1 - cov ** 2))
    print('correlated Gaussians: TE result {0:.4f} bits; expected to be '
          '{1:0.4f} bit for the copy'.format(cmi[1][0], cmi_expected))
    np.testing.assert_almost_equal(
                   cmi[1][0], cmi_expected, decimal=1,
                   err_msg='when calculating cmi for correlated Gaussians.')


@jpype_missing
def test_calculate_mean():
    """Test if mean over CMI estimates is calculated correctly."""
    dat = Data()
    dat.generate_mute_data(100, 5)
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    comp_settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'tail': 'two',
        'n_perm_comp': 6,
        'alpha_comp': 0.2,
        'stats_type': 'dependent'
        }
    comp = NetworkComparison()
    comp._initialise(comp_settings)
    comp._create_union(res_0)
    cmi = comp._calculate_cmi_all_links(dat)
    cmi_mean = comp._calculate_mean([cmi, cmi])
    for t in comp.union['targets']:
        assert (cmi_mean[t] == cmi[t]).all(), ('Error in mean of CMI for '
                                               'target {0}'.format(t))


@jpype_missing
def test_p_value_union():
    """Test if the p-value is calculated correctly."""
    dat = Data()
    dat.generate_mute_data(100, 5)
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_0.pkl'))
    res_1 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_res_1.pkl'))
    comp_settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'n_perm_comp': 6,
        'alpha_comp': 0.2,
        'tail_comp': 'one_bigger',
        'stats_type': 'independent'
        }
    comp = NetworkComparison()
    res_comp = comp.compare_within(comp_settings, res_0, res_1, dat, dat)

    # Replace the surrogate CMI by all zeros for source 0 and all ones for
    # source 1. Set the CMI difference to 0.5 for both sources. Check if this
    # results in one significant and one non-significant result with the
    # correct p-values.
    comp._initialise(comp_settings)
    comp._create_union(res_0, res_1)
    comp._calculate_cmi_diff_within(dat, dat)
    comp._create_surrogate_distribution_within(dat, dat)
    target = 1
    for p in range(comp_settings['n_perm_comp']):
        comp.cmi_surr[p][target] = np.array([0, 1])
    comp.cmi_diff[target] = np.array([0.5, 0.5])
    [p, s] = comp._p_value_union()
    assert (s[target] == np.array([True, False])).all(), (
                                    'The significance was not determined '
                                    'correctly: {0}'.format(s[target]))
    p_1 = 1 / comp_settings['n_perm_comp']
    p_2 = 1.0
    print(p[target])
    assert (p[target] == np.array([p_1, p_2])).all(), (
                                'The p-value was not calculated correctly: {0}'
                                .format(p[target]))


if __name__ == '__main__':
    test_network_comparison_use_cases()
    test_p_value_union()
    test_calculate_mean()
    test_calculate_cmi_all_links()
    test_get_permuted_replications()
    test_assertions()
    test_create_union_network()
