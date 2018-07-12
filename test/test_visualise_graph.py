"""Test IDTxl results class."""
import os
import pickle
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from idtxl.visualise_graph import (
    plot_network, plot_selected_vars, plot_network_comparison)
from idtxl.multivariate_te import MultivariateTE
from idtxl.network_comparison import NetworkComparison
from idtxl.data import Data
from idtxl.estimators_jidt import JidtDiscreteCMI
from test_estimators_jidt import jpype_missing


@jpype_missing
def test_plot_network():
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
    source = source[delay:]
    target_1 = target_1[:-delay]
    target_2 = target_2[:-delay]

    # Discretise data for speed
    settings_dis = {'discretise_method': 'equal',
                    'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings_dis)
    source_dis, target_1_dis = est._discretise_vars(var1=source, var2=target_1)
    source_dis, target_2_dis = est._discretise_vars(var1=source, var2=target_2)

    data = Data(np.vstack((source_dis, target_1_dis, target_2_dis)),
                dim_order='ps', normalise=normalisation)

    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,  # alphabet size of the variables analysed
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 1,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'alpha_fdr': 0.5}
    nw = MultivariateTE()

    # Analyse a single target and the whole network
    res_single = nw.analyse_single_target(
        settings=settings, data=data, target=1)
    res_network = nw.analyse_network(settings=settings, data=data)
    graph, fig = plot_network(res_single, 'max_te_lag', fdr=False)
    plt.close(fig)
    graph, fig = plot_network(res_network, 'max_te_lag', fdr=False)
    plt.close(fig)
    for sign_sources in [True, False]:
        graph, fig = plot_selected_vars(
            res_network, target=1, sign_sources=True, fdr=False)
        plt.close(fig)


def test_plot_network_comparison():
    """Test results class for network comparison."""
    data_0 = Data()
    data_0.generate_mute_data(500, 5)
    data_1 = Data(np.random.rand(5, 500, 5), 'psr')

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
            'alpha_comp': 0.26,
            'n_perm_comp': 200,
            'tail': 'two',
            'permute_in_time': True,
            'perm_type': 'random'
            }
    comp = NetworkComparison()

    comp_settings['stats_type'] = 'independent'
    res_within = comp.compare_within(
        comp_settings, res_0, res_1, data_0, data_1)
    comp_settings['stats_type'] = 'independent'
    res_between = comp.compare_between(
        comp_settings,
        network_set_a=np.array(list(it.repeat(res_0, 10))),
        network_set_b=np.array(list(it.repeat(res_1, 10))),
        data_set_a=np.array(list(it.repeat(data_0, 10))),
        data_set_b=np.array(list(it.repeat(data_1, 10))))

    graph, fig = plot_network_comparison(res_between)
    plt.close(fig)
    graph, fig = plot_network_comparison(res_within)
    plt.close(fig)


if __name__ == '__main__':
    test_plot_network_comparison()
    test_plot_network()
