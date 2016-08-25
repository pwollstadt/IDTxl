"""Plot graph output from multivariate TE estimation.

author: patricia
"""
from idtxl.data import Data
from idtxl.multivariate_te import Multivariate_te
from idtxl import visualise_graph

# Generate some example output
data = Data()
data.generate_mute_data(n_replications=2, n_samples=500)
print('Demo data with {0} procs, {1} samples, {2} reps.'.format(
                data.n_processes, data.n_samples, data.n_replications))
opts = {'cmi_calc_name': 'jidt_kraskov'}
mte = Multivariate_te(max_lag_sources=3, max_lag_target=3,
                      min_lag_sources=1, options=opts)
res_single = mte.analyse_single_target(data=data, target=3)
res_full = mte.analyse_network(data=data)

# generate graph plots
g_single = visualise_graph.plot_selected_vars(res_single, mte)
g_full = visualise_graph.plot_network(res_full)
