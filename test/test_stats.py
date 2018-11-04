"""Unit tests for stats module."""
import pytest
import numpy as np
from idtxl import stats
from idtxl.multivariate_te import MultivariateTE
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.estimators_jidt import JidtDiscreteCMI
from idtxl.data import Data
from idtxl.results import ResultsNetworkInference, ResultsSingleProcessAnalysis
from test_estimators_jidt import _get_gauss_data


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
    res_1._add_single_result(target=0, settings=settings, results=target_0)
    res_1._add_single_result(target=1, settings=settings, results=target_1)
    res_2 = ResultsNetworkInference(
        n_nodes=3, n_realisations=1000, normalised=True)
    res_2._add_single_result(target=2, settings=settings, results=target_2)

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
        assert (not res_pruned._single_target[2].selected_vars_sources), (
            'Target 2 has not been pruned from results.')

        for k in res_pruned.targets_analysed:
            if res_pruned._single_target[k]['selected_sources_pval'] is None:
                assert (
                    not res_pruned._single_target[k]['selected_vars_sources'])
            else:
                assert (
                    len(res_pruned._single_target[k]['selected_vars_sources']) ==
                    len(res_pruned._single_target[k]['selected_sources_pval'])), (
                        'Source list and list of p-values should have '
                        'the same length.')

    # Test function call for single result
    res_pruned = stats.network_fdr(settings, res_1)
    print('successful call on single result dict.')

    # Test None result for insufficient no. permutations, no FDR-corrected
    # results (the results class throws an error if no FDR-corrected results
    # exist).
    res_1.settings['n_perm_max_seq'] = 2
    res_2.settings['n_perm_max_seq'] = 2
    res_pruned = stats.network_fdr(settings, res_1, res_2)
    with pytest.raises(RuntimeError):
        res_pruned.get_adjacency_matrix('binary', fdr=True)


def test_ais_fdr():
    settings = {'n_perm_max_seq': 1000, 'n_perm_mi': 1000}
    process_0 = {
        'selected_vars': [(0, 1), (0, 2), (0, 3)],
        'ais_pval': 0.0001,
        'ais_sign': True}
    process_1 = {
        'selected_vars': [(1, 0), (1, 1), (1, 2)],
        'ais_pval': 0.031,
        'ais_sign': True}
    process_2 = {
        'selected_vars': [],
        'ais_pval': 0.41,
        'ais_sign': False}
    res_1 = ResultsSingleProcessAnalysis(
        n_nodes=3, n_realisations=1000, normalised=True)
    res_1._add_single_result(process=0, settings=settings, results=process_0)
    res_1._add_single_result(process=1, settings=settings, results=process_1)
    res_2 = ResultsSingleProcessAnalysis(
        n_nodes=3, n_realisations=1000, normalised=True)
    res_2._add_single_result(process=2, settings=settings, results=process_2)

    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'alpha_fdr': 0.05,
        'max_lag': 3}
    data = Data()
    data.generate_mute_data(n_samples=100, n_replications=3)
    analysis_setup = ActiveInformationStorage()
    analysis_setup._initialise(settings=settings, data=data, process=1)
    res_pruned = stats.ais_fdr(settings, res_1, res_2)
    assert (not res_pruned._single_process[2].selected_vars_sources), (
        'Process 2 has not been pruned from results.')

    alpha_fdr = res_pruned.settings.alpha_fdr
    for k in res_pruned.processes_analysed:
        if not res_pruned._single_process[k]['ais_sign']:
            assert (res_pruned._single_process[k]['ais_pval'] > alpha_fdr), (
                'P-value of non-sign. AIS is not 1.')
            assert (not res_pruned._single_process[k]['selected_vars']), (
                'List of significant past variables is not empty')
        else:
            assert (res_pruned._single_process[k]['ais_pval'] < 1), (
                'P-value of sign. AIS is not smaller 1.')
            assert (res_pruned._single_process[k]['selected_vars']), (
                'List of significant past variables is empty')

    # Test function call for single result
    res_pruned = stats.ais_fdr(settings, res_1)
    print('successful call on single result dict.')

    # Test None result for insufficient no. permutations, no FDR-corrected
    # results (the results class throws an error if no FDR-corrected results
    # exist).
    res_1.settings['n_perm_mi'] = 2
    res_2.settings['n_perm_mi'] = 2
    res_pruned = stats.ais_fdr(settings, res_1, res_2)
    with pytest.raises(RuntimeError):
        res_pruned.get_significant_processes(fdr=True)


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


def test_analytical_surrogates():
    # Test generation of analytical surrogates.
    # Generate data and discretise it such that we can use analytical
    # surrogates.
    expected_mi, source1, source2, target = _get_gauss_data(covariance=0.4)
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source1, var2=target)
    data = Data(np.hstack((source_dis, target_dis)),
                dim_order='sp', normalise=False)
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'n_discrete_bins': 5,  # alphabet size of the variables analysed
        'n_perm_max_stat': 100,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 21,
        'n_perm_max_seq': 21,
        'max_lag_sources': 5,
        'min_lag_sources': 1,
        'max_lag_target': 5
        }
    nw = MultivariateTE()
    res = nw.analyse_single_target(settings, data, target=1)
    # Check if generation of analytical surrogates is documented in the
    # settings.
    assert res.settings.analytical_surrogates, (
        'Surrogates were not created analytically.')


if __name__ == '__main__':
    test_ais_fdr()
    test_analytical_surrogates()
    test_data_type()
    test_network_fdr()
    test_find_pvalue()
    test_find_table_max()
    test_find_table_min()
    test_sort_table_max()
    test_sort_table_min()
    test_omnibus_test()
    test_max_statistic()
    test_min_statistic()
    test_max_statistic_sequential()
