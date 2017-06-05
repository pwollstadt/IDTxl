"""Unit tests for IDTxl graph visualisation."""
import idtxl.visualise_graph as vis
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data


def test_plot_mute_graph():
    """Plot MuTE example network."""
    vis.plot_mute_graph()


def test_visualise_multivariate_te():
    """Visualise output of multivariate TE estimation."""
    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag = 5
    min_lag = 4
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 25,
        'n_perm_min_stat': 25,
        'n_perm_omnibus': 50,
        'n_perm_max_seq': 50,
        }
    network_analysis = MultivariateTE(max_lag, min_lag,
                                      analysis_opts)
    res = network_analysis.analyse_network(dat, targets=[0, 1, 2])
    vis.plot_network(res)


def test_plot_selected_vars():
    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag = 5
    min_lag = 4
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 25,
        'n_perm_min_stat': 25,
        'n_perm_omnibus': 50,
        'n_perm_max_seq': 50,
        }
    network_analysis = MultivariateTE(max_lag, min_lag,
                                      analysis_opts)
    res = network_analysis.analyse_single_target(dat, target=2)
    vis.plot_selected_vars(res)

if __name__ == '__main__':
    test_plot_selected_vars()
    test_visualise_multivariate_te()
    test_plot_mute_graph()