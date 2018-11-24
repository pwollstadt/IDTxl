"""Test network comparison.

This module provides unit/system tests for network comparison between and
within subjects.
"""
import os
import pickle
import random as rn
import pytest
import numpy as np
from idtxl.network_comparison import NetworkComparison
from idtxl.data import Data
from test_estimators_jidt import jpype_missing
from idtxl.idtxl_utils import calculate_mi
from test_estimators_jidt import _get_gauss_data


@jpype_missing
def test_network_comparison_use_cases():
    """Run all intended use cases, within/between, dependent/independent."""
    data = Data()
    data.generate_mute_data(100, 5)

    path = os.path.join(os.path.dirname(__file__), 'data/')
    res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))
    res_2 = pickle.load(open(path + 'mute_results_2.p', 'rb'))
    res_3 = pickle.load(open(path + 'mute_results_3.p', 'rb'))
    res_4 = pickle.load(open(path + 'mute_results_4.p', 'rb'))

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
    comp.compare_within(comp_settings, res_0, res_1, data, data)

    print('\n\nTEST 1 - dependent within')
    comp_settings['stats_type'] = 'dependent'
    comp.compare_within(comp_settings, res_0, res_1, data, data)

    print('\n\nTEST 2 - independent between')
    comp_settings['stats_type'] = 'independent'
    comp.compare_between(comp_settings,
                         network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3)),
                         data_set_a=np.array((data, data)),
                         data_set_b=np.array((data, data)))

    print('\n\nTEST 3 - dependent between')
    comp_settings['stats_type'] = 'dependent'
    comp.compare_between(comp_settings,
                         network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3)),
                         data_set_a=np.array((data, data)),
                         data_set_b=np.array((data, data)))

    print('\n\nTEST 4 - independent within unbalanced')
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    comp.compare_within(comp_settings, res_0, res_1, data, data)

    print('\n\nTEST 5 - independent between unbalanced')
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    comp.compare_between(comp_settings,
                         network_set_a=np.array((res_0, res_1)),
                         network_set_b=np.array((res_2, res_3, res_4)),
                         data_set_a=np.array((data, data)),
                         data_set_b=np.array((data, data, data)))


@jpype_missing
def test_assertions():
    """Test if input checks raise errors."""
    data = Data()
    data.generate_mute_data(100, 5)

    # Load previously generated example data
    path = os.path.join(os.path.dirname(__file__), 'data/')
    res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))
    res_2 = pickle.load(open(path + 'mute_results_2.p', 'rb'))
    res_3 = pickle.load(open(path + 'mute_results_3.p', 'rb'))

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
        comp.compare_within(comp_settings, res_0, res_1, data, dat2)

    # data sets have unequal no. realisations
    dat2 = Data()
    dat2.generate_mute_data(80, 5)
    comp_settings['stats_type'] = 'dependent'
    comp_settings['alpha_comp'] = 0.05
    comp_settings['n_perm_comp'] = 21
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):
        comp.compare_within(comp_settings, res_0, res_1, data, dat2)

    # no. replications/subjects too small for dependent-samples test
    comp_settings['stats_type'] = 'dependent'
    comp_settings['n_perm_comp'] = 1000
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):   # between
        comp.compare_between(comp_settings,
                             network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((data, data)),
                             data_set_b=np.array((data, data)))
    with pytest.raises(RuntimeError):   # within
        comp.compare_within(comp_settings, res_0, res_1, dat2, dat2)

    # no. replications/subjects too small for independent-samples test
    comp_settings['stats_type'] = 'independent'
    comp = NetworkComparison()
    with pytest.raises(RuntimeError):   # between
        comp.compare_between(comp_settings,
                             network_set_a=np.array((res_0, res_1)),
                             network_set_b=np.array((res_2, res_3)),
                             data_set_a=np.array((data, data)),
                             data_set_b=np.array((data, data)))
    with pytest.raises(RuntimeError):   # within
        comp.compare_within(comp_settings, res_0, res_1, dat2, dat2)

    # add target to network that is not in the data object
    dat2 = Data(np.random.rand(2, 1000, 50), dim_order='psr')
    comp_settings['alpha_comp'] = 0.05
    comp_settings['n_perm_comp'] = 21
    comp = NetworkComparison()
    with pytest.raises(IndexError):
        comp.compare_within(comp_settings, res_0, res_2, dat2, dat2)


@jpype_missing
def test_create_union_network():
    """Test creation of union of multiple networks."""
    dat1 = Data()
    dat1.generate_mute_data(100, 5)
    dat2 = Data()
    dat2.generate_mute_data(100, 5)

    # Load previously generated example data
    path = os.path.join(os.path.dirname(__file__), 'data/')
    res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))

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
    res_0._single_target[1].selected_vars_sources = src_1
    res_1._single_target[1].selected_vars_sources = src_2

    comp._create_union(res_0, res_1)
    ref_targets = np.array([0, 1, 2])
    assert (comp.union.targets_analysed == ref_targets).all(), (
        'Union does not include all targets.')
    assert np.array([
        True for i in ref_targets if i in comp.union.keys()]).all(), (
            'Not all targets contained in union network.')
    assert comp.union['max_lag'] == res_0._single_target[1].current_value[1], (
        'The max. lag was not defined correctly.')

    src_union = comp._idx_to_lag(
        comp.union._single_target[1]['selected_vars_sources'],
        comp.union['max_lag'])
    assert src_union == (src_1 + src_2), (
        'Sources for target 1 were not combined correctly.')

    # unequal current values in single networks
    res_0._single_target[1].current_value = (1, 7)  # the original is (1, 5)
    with pytest.raises(ValueError):
        comp._create_union(res_0, res_1)


@jpype_missing
def test_get_permuted_replications():
    """Test if permutation of replications works."""
    # Load previously generated example data
    path = os.path.join(os.path.dirname(__file__), 'data/')
    res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))
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
    expected_mi, source, source_uncorr, target = _get_gauss_data()
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]
    data = Data(np.hstack((source, target)),
                dim_order='sp', normalise=False)
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_results_0.p'))
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
    # Set selected variable to the source, one sample in the past of the
    # current_value (1, 5).
    comp.union._single_target[1]['selected_vars_sources'] = [(0, 4)]
    cmi = comp._calculate_cmi_all_links(data)
    print('correlated Gaussians: TE result {0:.4f} bits; expected to be '
          '{1:0.4f} bit for the copy'.format(cmi[1][0], expected_mi))
    assert np.isclose(cmi[1][0], expected_mi, atol=0.05), (
        'Estimated TE {0:0.6f} differs from expected TE {1:0.6f}.'.format(
            cmi[1][0], expected_mi))


@jpype_missing
def test_calculate_mean():
    """Test if mean over CMI estimates is calculated correctly."""
    data = Data()
    data.generate_mute_data(100, 5)
    res_0 = np.load(os.path.join(os.path.dirname(__file__),
                    'data/mute_results_0.p'))
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
    cmi = comp._calculate_cmi_all_links(data)
    cmi_mean = comp._calculate_mean([cmi, cmi])
    for t in comp.union.targets_analysed:
        assert (cmi_mean[t] == cmi[t]).all(), ('Error in mean of CMI for '
                                               'target {0}'.format(t))
        if len(cmi[t]) == 0:  # skip if no links in results
            continue
        assert (cmi_mean[t] == cmi[t][0]).all(), (
            'Error in mean of CMI for target {0} - actual: ({1}), expected: '
            '({2})'.format(t, cmi_mean[t], cmi[t][0]))

@jpype_missing
def test_p_value_union():
    """Test if the p-value is calculated correctly."""
    data = Data()
    data.generate_mute_data(100, 5)
    path = os.path.join(os.path.dirname(__file__), 'data/')
    res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))
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
    comp.compare_within(comp_settings, res_0, res_1, data, data)

    # Replace the surrogate CMI by all zeros for source 0 and all ones for
    # source 1. Set the CMI difference to 0.5 for both sources. Check if this
    # results in one significant and one non-significant result with the
    # correct p-values.
    comp._initialise(comp_settings)
    comp._create_union(res_0, res_1)
    comp._calculate_cmi_diff_within(data, data)
    comp._create_surrogate_distribution_within(data, data)
    target = 1
    source = 0

    comp.cmi_surr[target] = np.zeros((1, comp_settings['n_perm_comp']))
    comp.cmi_diff[target] = np.array([0.5])
    comp._p_value_union()
    p = comp.pvalue
    s = comp.significance
    assert s[target][source], (
        'The significance was not determined correctly: {0}'.format(s[target]))
    assert p[target][source] == 1 / comp_settings['n_perm_comp'], (
        'The p-value was not calculated correctly: {0}'.format(p[target]))

    comp.cmi_surr[target] = np.ones((1, comp_settings['n_perm_comp']))
    comp.cmi_diff[target] = np.array([0.5])
    comp._p_value_union()
    p = comp.pvalue
    s = comp.significance
    assert not s[target][source], (
        'The significance was not determined correctly: {0}'.format(s[target]))
    assert p[target][source] == 1.0, (
        'The p-value was not calculated correctly: {0}'.format(p[target]))


def test_compare_links_within():
    """Test comparison of two links within a single network."""
    data = Data()
    data.generate_mute_data(100, 5)

    path = os.path.join(os.path.dirname(__file__), 'data/')
    res = pickle.load(open(path + 'mute_results_1.p', 'rb'))

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

    link_a = [0, 1]
    link_b = [0, 2]

    comp = NetworkComparison()
    comp_settings['stats_type'] = 'independent'
    res_indep = comp.compare_links_within(settings=comp_settings,
                                          link_a=link_a,
                                          link_b=link_b,
                                          network=res,
                                          data=data)
    comp_settings['stats_type'] = 'dependent'
    res_dep = comp.compare_links_within(settings=comp_settings,
                                        link_a=[0, 1],
                                        link_b=[0, 2],
                                        network=res,
                                        data=data)
    for r in [res_indep, res_dep]:
        adj_mat_diff = r.get_adjacency_matrix('diff_abs')
        adj_mat_comp = r.get_adjacency_matrix('comparison')
        adj_mat_pval = r.get_adjacency_matrix('pvalue')
        assert (adj_mat_diff._weight_matrix[link_a[0], link_a[1]] ==
                adj_mat_diff._weight_matrix[link_b[0], link_b[1]]), (
                    'Absolute differences for link comparison not equal.')
        assert (adj_mat_comp._weight_matrix[link_a[0], link_a[1]] ==
                adj_mat_comp._weight_matrix[link_b[0], link_b[1]]), (
                    'Comparison results for link comparison not equal.')
        assert (adj_mat_pval._weight_matrix[link_a[0], link_a[1]] ==
                adj_mat_pval._weight_matrix[link_b[0], link_b[1]]), (
                    'P-value for link comparison not equal.')
        assert (r.targets_analysed == [link_a[1], link_b[1]]).all(), (
                'Analysed targets are not correct.')

    with pytest.raises(RuntimeError):
        comp.compare_links_within(settings=comp_settings,
                                  link_a=link_a,
                                  link_b=[3, 4],
                                  network=res,
                                  data=data)


def test_tails():
    """Test one- and two-tailed testing for all stats types."""
    data = Data()
    data.generate_mute_data(100, 5)

    path = os.path.join(os.path.dirname(__file__), 'data/')
    res_0 = pickle.load(open(path + 'mute_results_0.p', 'rb'))
    res_1 = pickle.load(open(path + 'mute_results_1.p', 'rb'))
    res_2 = pickle.load(open(path + 'mute_results_2.p', 'rb'))
    res_3 = pickle.load(open(path + 'mute_results_3.p', 'rb'))

    # comparison settings
    comp_settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'alpha_comp': 0.26,
            'n_perm_comp': 4}

    comp = NetworkComparison()
    for tail in ['two', 'one']:
        for stats_type in ['independent', 'dependent']:
            comp_settings['stats_type'] = stats_type
            comp_settings['tail_comp'] = tail
            c_within = comp.compare_within(
                comp_settings, res_0, res_1, data, data)
            c_between = comp.compare_between(
                comp_settings,
                network_set_a=np.array((res_0, res_1)),
                network_set_b=np.array((res_2, res_3)),
                data_set_a=np.array((data, data)),
                data_set_b=np.array((data, data)))
            adj_mat_within = c_within.get_adjacency_matrix('pvalue')
            adj_mat_within.print_matrix()
            adj_mat_between = c_between.get_adjacency_matrix('pvalue')
            adj_mat_between.print_matrix()


if __name__ == '__main__':
    test_calculate_cmi_all_links()
    test_tails()
    test_compare_links_within()
    test_network_comparison_use_cases()
    test_p_value_union()
    test_create_union_network()
    test_assertions()
    test_calculate_mean()
    test_get_permuted_replications()
