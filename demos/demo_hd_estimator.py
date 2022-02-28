"""Demo of history dependence estimator by Lukas Rudelt.

Before running this demo, make sure to run the setup file

>>> python3 setup_hde_fast_embedding.py build_ext --inplace

to build the fast embedding cython module. You may have to copy the module from
the generated build folder into the IDTXL/idtxl folder.

To check if the setup was successful, run the fast embedding test in the test
folder:

>>> test_fast_emb.py
"""
# Import classes
import numpy as np
from idtxl.data_spiketime import Data_spiketime
from idtxl.embedding_optimization_ais_Rudelt import OptimizationRudelt

# a) Generate test data
print("\nTest optimization_Rudelt using BBC estimator on Rudelt data")
data = Data_spiketime()             # initialise empty data object
data.load_Rudelt_data()             # load Rudelt spike time data

# b) Initialise analysis object and define settings
# Run optimization with the Bayesian bias criterion (BBC) (alternatively, set
# 'estimation_method' to 'shuffling'). Note that we are here using a subset of
# the default embedding parameters, 'embedding_past_range_set',
# 'embedding_number_of_bins_set', and 'embedding_scaling_exponent_set' to
# reduce the run time of this demo. Use defaults if unsure about the settings.
settings = {
    'embedding_past_range_set': [0.005, 0.31548, 5.0],
    'embedding_number_of_bins_set': [1, 3, 5],
    'embedding_scaling_exponent_set':
        {'number_of_scalings': 3,
         'min_first_bin_size': 0.005,
         'min_step_for_scaling': 0.01},
    'number_of_bootstraps_R_max': 10,
    'number_of_bootstraps_R_tot': 10,
    'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
    'estimation_method': 'bbc',
    'debug': True,
    'visualization': True,
    'output_path': '.',
    'output_prefix': 'systemtest_optimizationRudelt_image1'}
hd_estimator = OptimizationRudelt(settings)

# c) Run analysis and save a plot of the results to current directory.
results_bbc = hd_estimator.optimize(data, processes=[0])
