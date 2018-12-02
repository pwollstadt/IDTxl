# -*- coding: utf-8 -*-
"""Run system tests for network comparison.

Run system tests for network comparison consisting of larger problems and
results from diffenrent network inference algorithms (bivariate/multivariate
mutual information (MI) or transfer entropy (TE)). Test Gaussian, discrete, and
the Kraskov estimator.
"""
import os
import pickle
import numpy as np
from idtxl.network_comparison import NetworkComparison
from idtxl.data import Data
from generate_test_data import generate_continuous_data, generate_discrete_data


def test_network_comparison():
    """Run within/between, dependent/independent test on bivariate MI."""
    # Generate continuous data.
    data_cont = generate_continuous_data(n_replications=10)
    data_disc = generate_discrete_data(n_replications=10)
    data_dummy_cont = _generate_dummy_data(data_cont)
    data_dummy_disc = _generate_dummy_data(data_disc)

    # Set path to load results from network inference on continuous data.
    path = os.path.join(os.path.dirname(__file__), 'data/')

    # Comparison settings.
    res = pickle.load(open('{0}discrete_results_mte_JidtDiscreteCMI.p'.format(
            path), 'rb'))  # load discrete results to get alphabet sizes
    comp_settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'n_perm_max_stat': 50,
            'n_perm_min_stat': 50,
            'n_perm_omnibus': 200,
            'n_perm_max_seq': 50,
            'alpha_comp': 0.26,
            'n_perm_comp': 4,
            'tail': 'two',
            'n_discrete_bins': res.settings['alph1']}
    comp = NetworkComparison()

    # Perform comparison.
    for inference in ['bmi', 'mmi', 'bmi', 'bte', 'mte']:

        # Discrete data
        estimator = 'JidtDiscreteCMI'
        res = pickle.load(open('{0}discrete_results_{1}_{2}.p'.format(
            path, inference, estimator), 'rb'))
        comp_settings['cmi_estimator'] = estimator
        for stats_type in ['dependent', 'independent']:
            print(('\n\n\n######### Running network comparison on {0} '
                   'results ({1} estimator) on discrete data, {2} test.'.format(inference, estimator, stats_type)))
            comp_settings['stats_type'] = stats_type

            c_within = comp.compare_within(
                    comp_settings, res, res, data_disc, data_dummy_disc)
            c_between = comp.compare_between(
                    comp_settings,
                    network_set_a=np.array((res, res)),
                    network_set_b=np.array((res, res)),
                    data_set_a=np.array((data_disc, data_disc)),
                    data_set_b=np.array((data_dummy_disc, data_dummy_disc)))
            _verify_test(c_within, c_between, res)

        # Continous data
        for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
            res = pickle.load(open('{0}continuous_results_{1}_{2}.p'.format(
                path, inference, estimator), 'rb'))
            comp_settings['cmi_estimator'] = estimator
            for stats_type in ['dependent', 'independent']:
                print(('\n\n\n######### Running network comparison on {0} '
                       'results ({1} estimator) on continuous data, {2} test.'.format(inference, estimator, stats_type)))
                comp_settings['stats_type'] = stats_type
                c_within = comp.compare_within(
                    comp_settings, res, res, data_cont, data_dummy_cont)
                c_between = comp.compare_between(
                    comp_settings,
                    network_set_a=np.array((res, res)),
                    network_set_b=np.array((res, res)),
                    data_set_a=np.array((data_cont, data_cont)),
                    data_set_b=np.array((data_dummy_cont, data_dummy_cont)))
                _verify_test(c_within, c_between, res)


def _verify_test(c_within, c_between, res):
    # Test values for verification
    # Get true positives.
    adj_mat_binary = res.get_adjacency_matrix('binary')
    tp = adj_mat_binary._edge_matrix

    for comp, comp_type in zip([c_between, c_within], ['between', 'within']):
        adj_mat_union = comp.get_adjacency_matrix('union')
        adj_mat_pval = comp.get_adjacency_matrix('pvlaue')
        adj_mat_diff = comp.get_adjacency_matrix('diff_abs')
        print(adj_mat_union._weight_matrix)
        assert (adj_mat_union._edge_matrix[tp]).all(), (
            'Missing union link in {} network comparison.'.format(comp_type))
        assert (adj_mat_pval._weight_matrix[tp] < 1).all(), (
            'Wrong p-value in {} network comparison.'.format(comp_type))
        assert (adj_mat_diff._edge_matrix[tp] > 0).all(), (
            'Missed difference in {} network comparison.'.format(comp_type))


def _generate_dummy_data(data):
    """Generate noise with the same dimensions and type as a given data set."""

    if issubclass(data.data_type, np.integer):
        max_value = np.max(data.data)
        d = np.random.randint(max_value, size=data.data.shape)
        data = Data(d, normalise=False)
    elif issubclass(data.data_type, np.float):
        d = np.random.rand(
            data.data.shape[0], data.data.shape[1], data.data.shape[2])
        data = Data(d, normalise=True)
    else:
        raise RuntimeError('Unknown data type.')
    return data


if __name__ == '__main__':
    test_network_comparison()
