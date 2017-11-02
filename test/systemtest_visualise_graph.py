"""Plot graph output from multivariate TE estimation.

author: patricia
"""
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl import visualise_graph


def test_plot_mute_graph():
    """Plot MuTE example network."""
    visualise_graph.plot_mute_graph()


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
    # generate graph plots
    visualise_graph.plot_selected_vars(results, target=1, sign_sources=False)
    visualise_graph.plot_network(results, fdr=False)
    visualise_graph.plot_network(results, fdr=True)
    visualise_graph.plot_selected_vars(results, target=1, sign_sources=True)



if __name__ == '__main__':
    test_visualise_multivariate_te()
    test_plot_mute_graph()
