"""Test IDTxl results class."""
import pickle
import pytest
from tempfile import TemporaryFile
import copy as cp
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.estimators_jidt import JidtDiscreteCMI
from test_estimators_jidt import jpype_missing

# Use common settings dict that can be used for each test
settings = {
    'cmi_estimator': 'JidtDiscreteCMI',
    'discretise_method': 'none',
    'alph1': 5,
    'alph2': 5,
    'alphc': 5,
    'n_perm_max_stat': 21,
    'n_perm_omnibus': 30,
    'n_perm_max_seq': 30,
    'min_lag_sources': 1,
    'max_lag_sources': 2,
    'max_lag_target': 1}


@jpype_missing
def test_results_multivariate_te():
    """Test results class for multivariate TE network inference."""
    covariance = 0.4
    n = 10000
    delay = 1
    normalisation = False
    source = np.random.normal(0, 1, size=n)
    target_1 = (covariance * source + (1 - covariance) *
                np.random.normal(0, 1, size=n))
    target_2 = (covariance * source + (1 - covariance) *
                np.random.normal(0, 1, size=n))
    expected_mi = np.log(1 / (1 - np.power(covariance, 2)))
    source = source[delay:]
    target_1 = target_1[:-delay]
    target_2 = target_2[:-delay]

    # Discretise data for speed
    settings_dis = {'discretise_method': 'equal',
                    'alph1': 5,
                    'alph2': 5}
    est = JidtDiscreteCMI(settings_dis)
    source_dis, target_1_dis = est._discretise_vars(var1=source, var2=target_1)
    source_dis, target_2_dis = est._discretise_vars(var1=source, var2=target_2)

    data = Data(np.vstack((source_dis, target_1_dis, target_2_dis)),
                dim_order='ps', normalise=normalisation)
    nw = MultivariateTE()

    # Analyse a single target
    res_single = nw.analyse_single_target(
        settings=settings, data=data, target=1)
    est_mi = res_single.single_target[1].omnibus_te
    assert np.isclose(est_mi, expected_mi, atol=0.05), (
        'Estimated TE for discrete variables is not correct. Expected: {0}, '
        'Actual results: {1}.'.format(expected_mi, est_mi))

    # Analyse whole network
    res_network = nw.analyse_network(settings=settings, data=data)
    est_mi = res_network.single_target[1].omnibus_te
    assert np.isclose(est_mi, expected_mi, atol=0.05), (
        'Estimated TE for discrete variables is not correct. Expected: {0}, '
        'Actual results: {1}.'.format(expected_mi, est_mi))
    est_mi = res_network.single_target[2].omnibus_te
    assert np.isclose(est_mi, expected_mi, atol=0.05), (
        'Estimated TE for discrete variables is not correct. Expected: {0}, '
        'Actual results: {1}.'.format(expected_mi, est_mi))

    # Check data parameters in results objects
    n_nodes = 3
    n_realisations = n - delay - max(
        settings['max_lag_sources'], settings['max_lag_target'])
    assert res_network.data.n_nodes == n_nodes, 'Incorrect no. nodes.'
    assert res_single.data.n_nodes == n_nodes, 'Incorrect no. nodes.'
    assert res_network.data.n_realisations == n_realisations, (
        'Incorrect no. realisations.')
    assert res_single.data.n_realisations == n_realisations, (
        'Incorrect no. realisations.')
    assert res_network.data.normalised == normalisation, (
        'Incorrect value for data normalisation.')
    assert res_single.data.normalised == normalisation, (
        'Incorrect value for data normalisation.')
    assert res_network.adjacency_matrix.shape[0] == n_nodes, (
        'Incorrect number of rows in adjacency matrix.')
    assert res_network.adjacency_matrix.shape[1] == n_nodes, (
        'Incorrect number of columns in adjacency matrix.')
    assert res_single.adjacency_matrix.shape[0] == n_nodes, (
        'Incorrect number of rows in adjacency matrix.')
    assert res_single.adjacency_matrix.shape[1] == n_nodes, (
        'Incorrect number of columns in adjacency matrix.')


def test_pickle_results():
    """Test pickling results objects."""
    data = _generate_gauss_data()
    nw = MultivariateTE()
    res_single = nw.analyse_single_target(
        settings=settings, data=data, target=1)
    res_network = nw.analyse_network(settings=settings, data=data)

    outfile = TemporaryFile()
    pickle.dump(res_single, outfile)
    pickle.dump(res_network, outfile)


def test_combine_results():
    """Test combination of results objects."""
    data = _generate_gauss_data()
    nw = MultivariateTE()
    res_network_1 = nw.analyse_network(settings=settings, data=data)

    # Test error for unequal settings
    res_network_2 = cp.deepcopy(res_network_1)
    res_network_2.settings.add_conditionals = 'Test'
    with pytest.raises(RuntimeError):
        res_network_1.combine_results(res_network_2)


def test_add_single_target():
    """Test adding results for a single target."""
    data = _generate_gauss_data()
    nw = MultivariateTE()
    res_network = nw.analyse_single_target(
        settings=settings, data=data, target=1)

    # Test adding target results that already exists
    with pytest.raises(RuntimeError):
        res_network._add_single_target(target=1,
                                       settings=res_network.settings,
                                       results={})
    # Test adding target results with unequal settings
    settings_test = cp.deepcopy(res_network.settings)
    settings_test.add_conditionals = 'Test'
    with pytest.raises(RuntimeError):
        res_network._add_single_target(target=0,
                                       settings=settings_test,
                                       results=res_network.single_target[1])
    # Test adding a target with additional settings, results.settings should be
    # updated
    settings_test = cp.deepcopy(res_network.settings)
    settings_test.new_setting = 'Test'
    res_network._add_single_target(target=0,
                                   settings=settings_test,
                                   results=res_network.single_target[1])
    assert 'new_setting' in res_network.settings.keys(), (
        'Settings dict was not updated.')
    assert res_network.settings.new_setting == 'Test', (
        'Settings dict was not updated correctly.')


def test_delay_reconstruction():
    """Test the reconstruction of information transfer delays from results."""
    covariance = 0.4
    expected_mi = np.log(1 / (1 - np.power(covariance, 2)))
    n = 10000
    delay_1 = 1
    delay_2 = 3
    delay_3 = 5
    normalisation = False
    source = np.random.normal(0, 1, size=n)
    target_1 = (covariance * source + (1 - covariance) *
                np.random.normal(0, 1, size=n))
    target_2 = (covariance * source + (1 - covariance) *
                np.random.normal(0, 1, size=n))
    target_3 = (covariance * source + (1 - covariance) *
                np.random.normal(0, 1, size=n))
    source = source[delay_3:]
    target_1 = target_1[(delay_3-delay_1):-delay_1]
    target_2 = target_2[(delay_3-delay_2):-delay_2]
    target_3 = target_3[:-delay_3]

    # Discretise data for speed
    settings_dis = {'discretise_method': 'equal',
                    'alph1': 5,
                    'alph2': 5}
    est = JidtDiscreteCMI(settings_dis)
    source_dis, target_1_dis = est._discretise_vars(var1=source, var2=target_1)
    source_dis, target_2_dis = est._discretise_vars(var1=source, var2=target_2)
    source_dis, target_3_dis = est._discretise_vars(var1=source, var2=target_3)
    data = Data(
        np.vstack((source_dis, target_1_dis, target_2_dis, target_3_dis)),
        dim_order='ps', normalise=normalisation)

    nw = MultivariateTE()
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'alph1': 5,
        'alph2': 5,
        'alphc': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 1,
        'max_lag_sources': delay_3 + 1,
        'max_lag_target': 1}

    res_network = nw.analyse_single_target(
        settings=settings, data=data, target=1)
    res_network.combine_results(nw.analyse_single_target(
        settings=settings, data=data, target=2))
    res_network.combine_results(nw.analyse_single_target(
        settings=settings, data=data, target=3))
    print(res_network.adjacency_matrix)
    assert res_network.adjacency_matrix[0, 1] == delay_1, (
        'Delay 1 was not reconstructed correctly.')
    assert res_network.adjacency_matrix[0, 2] == delay_2, (
        'Delay 2 was not reconstructed correctly.')
    assert res_network.adjacency_matrix[0, 3] == delay_3, (
        'Delay 3 was not reconstructed correctly.')

    for target in range(1, 4):
        est_mi = res_network.single_target[target].omnibus_te
        assert np.isclose(est_mi, expected_mi, atol=0.05), (
            'Estimated TE for target {0} is not correct. Expected: {1}, '
            'Actual results: {2}.'.format(target, expected_mi, est_mi))


def _generate_gauss_data(covariance=0.4, n=10000, delay=1, normalise=False):
    # Generate two coupled Gaussian time series
    source = np.random.normal(0, 1, size=n)
    target = (covariance * source + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    source = source[delay:]
    target = target[:-delay]

    # Discretise data for speed
    settings = {'discretise_method': 'equal',
                'alph1': 5,
                'alph2': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    return Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=normalise)

if __name__ == '__main__':
    test_pickle_results()
    test_delay_reconstruction()
    test_combine_results()
    test_add_single_target()
    test_results_multivariate_te()
