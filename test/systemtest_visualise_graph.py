"""Unit tests for IDTxl graph visualisation."""
import idtxl.visualise_graph as vis
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data


def test_plot_mute_graph():
    """Plot MuTE example network."""
    vis.plot_mute_graph()


def test_visualise_multivariate_te():
    """Visualise output of multivariate TE estimation."""
    data = Data()
    data.generate_mute_data(100, 5)
    settings = {
        'cmi_estimator':  'JidtKraskovCMI',
        'max_lag_sources': 5,
        'min_lag_sources': 4,
        'n_perm_max_stat': 25,
        'n_perm_min_stat': 25,
        'n_perm_omnibus': 50,
        'n_perm_max_seq': 50,
        }
    network_analysis = MultivariateTE()
    results = network_analysis.analyse_network(settings, data,
                                               targets=[0, 1, 2])
    vis.plot_network(results)


def test_plot_selected_vars():
    data = Data()
    data.generate_mute_data(100, 5)
    settings = {
        'cmi_estimator':  'JidtKraskovCMI',
        'max_lag_sources': 5,
        'min_lag_sources': 4,
        'n_perm_max_stat': 25,
        'n_perm_min_stat': 25,
        'n_perm_omnibus': 50,
        'n_perm_max_seq': 50,
        }
    network_analysis = MultivariateTE()
    results = network_analysis.analyse_single_target(settings, data, target=2)
    vis.plot_selected_vars(results)


if __name__ == '__main__':
    test_plot_selected_vars()
    test_visualise_multivariate_te()
    test_plot_mute_graph()
