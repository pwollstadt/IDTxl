import numpy as np

from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.postprocessing import SignificantSubgraphMining

# Generate example data and perform network inference.
np.random.seed(seed=0)
data_a = Data()
data_a.generate_mute_data(100, 5)
data_b = Data()
data_b.generate_mute_data(100, 5)
settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 50,
    'n_perm_min_stat': 50,
    'n_perm_omnibus': 200,
    'n_perm_max_seq': 50,
    'max_lag_target': 5,
    'max_lag_sources': 5,
    'min_lag_sources': 1,
    'permute_in_time': True
    }
# Run analysis with different targets to simulate different results for both
# analyses.
network_analysis = MultivariateTE()
res_a = network_analysis.analyse_network(settings, data_a, sources='all')
res_b = network_analysis.analyse_network(settings, data_b,  sources='all')

# Perform significant subgraph mining on inferred networks using different
# correction methods. Use the IDTxl format as input, which looks also at the
# time dimension.
sample_size = 10
SSM_between = SignificantSubgraphMining(
    [res_a.get_source_variables(fdr=False) for n in range(sample_size)],
    [res_b.get_source_variables(fdr=False) for n in range(sample_size)],
    design="between",
    alpha=0.05,
    data_format="idtxl"
    )
res_ssm_between = SSM_between.enumerate_significant_subgraphs(
    method="Hommel",  # correction method: "Tarone", "Hommel", or "Westfall-Young"
    verbose=True,
    num_perm=10000,
    max_depth=np.infty
)
# List of subgraphs, each entry is a tuple comprising a list of edges and the
# p-value for this subgraph.
print(res_ssm_between)

SSM_within = SignificantSubgraphMining(
    [res_a.get_source_variables(fdr=False) for n in range(sample_size)],
    [res_b.get_source_variables(fdr=False) for n in range(sample_size)],
    design="within",
    alpha=0.05,
    data_format="idtxl"
    )
res_ssm_within = SSM_within.enumerate_significant_subgraphs(
    method="Westfall-Young",  # correction method: "Tarone", "Hommel", or "Westfall-Young"
    wy_algorithm="simple_depth_first",
    verbose=True,
    num_perm=10000,
    max_depth=np.infty
)
print(res_ssm_within)

SSM_within = SignificantSubgraphMining(
    [res_a.get_source_variables(fdr=False) for n in range(sample_size)],
    [res_b.get_source_variables(fdr=False) for n in range(sample_size)],
    design="within",
    alpha=0.05,
    data_format="idtxl"
    )
res_ssm_within = SSM_within.enumerate_significant_subgraphs(
    method="Tarone",  # correction method: "Tarone", "Hommel", or "Westfall-Young"
    verbose=True,
    num_perm=10000,
    max_depth=np.infty
)
print(res_ssm_within)

# Use adjacency matrices as input. This ignores the time dimension of the
# inferred network.
SSM_within = SignificantSubgraphMining(
    [res_a.get_adjacency_matrix(weights='binary', fdr=False).edge_matrix for n in range(sample_size)],
    [res_b.get_adjacency_matrix(weights='binary', fdr=False).edge_matrix for n in range(sample_size)],
    design="within",
    alpha=0.05,
    data_format="adjacency"
    )
res_ssm_within = SSM_within.enumerate_significant_subgraphs(
    method="Tarone",  # correction method: "Tarone", "Hommel", or "Westfall-Young"
    verbose=True,
    num_perm=10000,
    max_depth=np.infty
)
print(res_ssm_within)
