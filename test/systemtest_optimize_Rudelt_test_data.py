"""System test for Rudelt optimization on example data (creating result image)."""

from idtxl.data_spiketime import Data_spiketime
from idtxl.embedding_optimization_ais_Rudelt import OptimizationRudelt

print("\nTest optimization_Rudelt using bbc estimator on Rudelt data")
data = Data_spiketime()  # initialise empty data object
data.load_Rudelt_data()  # load Rudelt spike time data

settings = {
    "embedding_past_range_set": [
        0.005,
        0.00998,
        0.01991,
        0.03972,
        0.07924,
        0.15811,
        0.31548,
        0.62946,
        1.25594,
        2.50594,
        5.0,
    ],
    "number_of_bootstraps_R_max": 10,
    "number_of_bootstraps_R_tot": 10,
    "auto_MI_bin_size_set": [0.01, 0.025, 0.05, 0.25],
    "estimation_method": "bbc",
    "debug": True,
    "visualization": True,
    "output_path": "./test_vis",
    "output_prefix": "systemtest_optimizationRudelt_image1",
}

processes = [0]
optimization_rudelt = OptimizationRudelt(settings)
results_bbc = optimization_rudelt.optimize(data, processes)
