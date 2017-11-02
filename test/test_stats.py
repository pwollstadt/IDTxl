# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:46:09 2016

@author: patricia
"""
import pytest
import numpy as np
from idtxl import stats
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.results import ResultsNetworkInference


def test_omnibus_test():
    print('Write test for omnibus test.')


def test_max_statistic():
    print('Write test for max_statistic.')


def test_min_statistic():
    print('Write test for min_statistic.')


def test_max_statistic_sequential():
    data = Data()
    data.generate_mute_data(104, 10)
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 21,
        'n_perm_max_seq': 21,
        'max_lag_sources': 5,
        'min_lag_sources': 1,
        'max_lag_target': 5
        }
    setup = MultivariateTE()
    setup._initialise(settings, data, sources=[0, 1], target=2)
    setup.current_value = (0, 4)
    setup.selected_vars_sources = [(1, 1), (1, 2)]
    setup.selected_vars_full = [(0, 1), (1, 1), (1, 2)]
    setup._selected_vars_realisations = np.random.rand(
                                    data.n_realisations(setup.current_value),
                                    len(setup.selected_vars_full))
    setup._current_value_realisations = np.random.rand(
                                    data.n_realisations(setup.current_value),
                                    1)
    [sign, p, te] = stats.max_statistic_sequential(analysis_setup=setup,
                                                   data=data)


def test_network_fdr():
    settings = {'n_perm_max_seq': 1000, 'n_perm_omnibus': 1000}
    target_0 = {
        'selected_vars_sources': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 0)],
        'selected_vars_full': [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                               (2, 1), (2, 0)],
        'omnibus_pval': 0.0001,
        'omnibus_sign': True,
        'selected_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047]),
        'selected_sources_te': np.array([1.1, 1.0, 0.8, 0.7, 0.63]),
        }
    target_1 = {
        'selected_vars_sources': [(1, 2), (2, 1), (2, 2)],
        'selected_vars_full': [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2)],
        'omnibus_pval': 0.031,
        'omnibus_sign': True,
        'selected_sources_pval': np.array([0.00001, 0.00014, 0.01]),
        'selected_sources_te': np.array([1.8, 1.75, 0.75]),
        }
    target_2 = {
        'selected_vars_sources': [],
        'selected_vars_full': [(2, 0), (2, 1)],
        'omnibus_pval': 0.41,
        'omnibus_sign': False,
        'selected_sources_pval': None,
        'selected_sources_te': np.array([]),
        }
    res_1 = ResultsNetworkInference(
        n_nodes=3, n_realisations=1000, normalised=True)
    res_1._add_single_target(target=0, settings=settings, results=target_0)
    res_1._add_single_target(target=1, settings=settings, results=target_1)
    res_2 = ResultsNetworkInference(
        n_nodes=3, n_realisations=1000, normalised=True)
    res_2._add_single_target(target=2, settings=settings, results=target_2)

    for correct_by_target in [True, False]:
        settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'alpha_fdr': 0.05,
            'max_lag_sources': 3,
            'min_lag_sources': 1,
            'max_lag_target': 3,
            'correct_by_target': correct_by_target}
        data = Data()
        data.generate_mute_data(n_samples=100, n_replications=3)
        analysis_setup = MultivariateTE()
        analysis_setup._initialise(settings=settings, data=data,
                                   sources=[1, 2], target=0)
        res_pruned = stats.network_fdr(settings, res_1, res_2)
        assert (not res_pruned.single_target[2].selected_vars_sources), (
            'Target ')

        for k in res_pruned.targets_analysed:
            if res_pruned.single_target[k]['selected_sources_pval'] is None:
                assert (
                    not res_pruned.single_target[k]['selected_vars_sources'])
            else:
                assert (
                    len(res_pruned.single_target[k]['selected_vars_sources']) ==
                    len(res_pruned.single_target[k]['selected_sources_pval'])), (
                        'Source list and list of p-values should have '
                        'the same length.')

    # Test function call for single result
    res_pruned = stats.network_fdr(settings, res_1)
    print('successful call on single result dict.')

    # Test None result for insufficient no. permutations
    res_1.settings['n_perm_max_seq'] = 2
    res_2.settings['n_perm_max_seq'] = 2
    res_pruned = stats.network_fdr(settings, res_1, res_2)
    assert not res_pruned.fdr_correction, (
        'results should be None if no. permutations too low.')


def test_find_pvalue():
    test_val = 1
    distribution = np.random.rand(500)  # normally distributed floats in [0,1)
    alpha = 0.05
    tail = 'two'  # 'one' or 'two'
    [s, p] = stats._find_pvalue(test_val, distribution, alpha, tail)
    assert(s is True)

    # If the statistic is bigger than the whole distribution, the p-value
    # should be set to the smallest possible value that could have been
    # expected from a test given the number of permutations.
    test_val = np.inf
    [s, p] = stats._find_pvalue(test_val, distribution, alpha, tail)
    n_perm = distribution.shape[0]
    assert(p == (1 / n_perm))

    # Test assertion that the test distributions is a one-dimensional array.
    with pytest.raises(AssertionError):
        stats._find_pvalue(test_val, np.expand_dims(distribution, axis=1),
                           alpha, tail)
    # Test assertion that no. permutations is high enough to theoretically
    # calculate the requested alpha level.
    with pytest.raises(RuntimeError):
        stats._find_pvalue(test_val, distribution[:5], alpha, tail)
    # Check if wrong parameter for tail raises a value error.
    with pytest.raises(ValueError):
        stats._find_pvalue(test_val, distribution, alpha, tail='foo')


def test_find_table_max():
    tab = np.array([[0, 2, 1], [3, 4, 5], [10, 8, 1]])
    results = stats._find_table_max(tab)
    assert (results == np.array([10,  8,  5])).all(), (
        'Function did not return maximum for each column.')


def test_find_table_min():
    tab = np.array([[0, 2, 1], [3, 4, 5], [10, 8, 1]])
    results = stats._find_table_min(tab)
    assert (results == np.array([0, 2, 1])).all(), (
        'Function did not return minimum for each column.')


def test_sort_table_max():
    tab = np.array([[0, 2, 1], [3, 4, 5], [10, 8, 1]])
    results = stats._sort_table_max(tab)
    assert (results[0, :] == np.array([10,  8,  5])).all(), (
        'Function did not return maximum for first row.')
    assert (results[2, :] == np.array([0, 2, 1])).all(), (
        'Function did not return minimum for last row.')


def test_sort_table_min():
    tab = np.array([[0, 2, 1], [3, 4, 5], [10, 8, 1]])
    results = stats._sort_table_min(tab)
    assert (results[0, :] == np.array([0, 2, 1])).all(), (
        'Function did not return minimum for first row.')
    assert (results[2, :] == np.array([10,  8,  5])).all(), (
        'Function did not return maximum for last row.')


def test_data_type():
    """Test if stats always returns surrogates with the correct data type."""
    # Change data type for the same object instance.
    d_int = np.random.randint(0, 10, size=(3, 50))
    orig_type = type(d_int[0][0])
    data = Data(d_int, dim_order='ps', normalise=False)
    # The concrete type depends on the platform:
    # https://mail.scipy.org/pipermail/numpy-discussion/2011-November/059261.html
    assert data.data_type is orig_type, 'Data type did not change.'
    assert issubclass(type(data.data[0, 0, 0]), np.integer), (
        'Data type is not an int.')
    settings = {'permute_in_time': True, 'perm_type': 'random'}
    surr = stats._get_surrogates(data=data,
                                 current_value=(0, 5),
                                 idx_list=[(1, 3), (2, 4)],
                                 n_perm=20,
                                 perm_settings=settings)
    assert issubclass(type(surr[0, 0]), np.integer), (
        'Realisations type is not an int.')
    surr = stats._generate_spectral_surrogates(data=data,
                                               scale=1,
                                               n_perm=20,
                                               perm_settings=settings)
    assert issubclass(type(surr[0, 0, 0]), np.integer), (
        'Realisations type is not an int.')

    d_float = np.random.randn(3, 50)
    data.set_data(d_float, dim_order='ps')
    assert data.data_type is np.float64, 'Data type did not change.'
    assert issubclass(type(data.data[0, 0, 0]), np.float), (
        'Data type is not a float.')
    surr = stats._get_surrogates(data=data,
                                 current_value=(0, 5),
                                 idx_list=[(1, 3), (2, 4)],
                                 n_perm=20,
                                 perm_settings=settings)
    assert issubclass(type(surr[0, 0]), np.float), (
        'Realisations type is not a float.')
    surr = stats._generate_spectral_surrogates(data=data,
                                               scale=1,
                                               n_perm=20,
                                               perm_settings=settings)
    assert issubclass(type(surr[0, 0, 0]), np.float), ('Realisations type is '
                                                       'not a float.')


if __name__ == '__main__':
    # test_data_type()
    test_network_fdr()
    # test_find_pvalue()
    # test_find_table_max()
    # test_find_table_min()
    # test_sort_table_max()
    # test_sort_table_min()
    # test_omnibus_test()
    # test_max_statistic()
    # test_min_statistic()
    # test_max_statistic_sequential()
