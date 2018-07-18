# Import classes
from idtxl.multivariate_mi import MultivariateMI
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import matplotlib.pyplot as plt

# a) Generate test data
data = Data()
data.generate_mute_data(n_samples=1000, n_replications=5)

# b) Initialise analysis object and define settings
network_analysis = MultivariateMI()
settings = {'cmi_estimator': 'JidtGaussianCMI',
            'max_lag_sources': 5,
            'min_lag_sources': 1}

# c) Run analysis
results = network_analysis.analyse_network(settings=settings, data=data)

# d) Plot inferred network to console and via matplotlib
results.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=results, weights='max_te_lag', fdr=False)
plt.show()
