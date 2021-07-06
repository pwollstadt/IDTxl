"""System test for Rudelt optimization on example data."""
from idtxl.data_spiketime import Data_spiketime
from idtxl.optimization_Rudelt import OptimizationRudelt

def test_optimization_Rudelt():
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811, 0.31548, 0.62946, 1.25594,
                                       2.50594, 5.0],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
                'estimation_method': 'bbc',
                'debug': True}

    processes = [0]
    replications = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results = optimization_rudelt.optimize(data, processes, replications)




if __name__ == '__main__':
    test_optimization_Rudelt()
