"""System test for Rudelt optimization on example data."""
from idtxl.data_spiketime import Data_spiketime
from idtxl.optimization_Rudelt import OptimizationRudelt
import numpy as np
import os


def test_optimization_Rudelt_bbc():
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811,
                                             0.31548, 0.62946, 1.25594, 2.50594, 5.0],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
                'estimation_method': 'bbc',
                'debug': True}
    #settings = {
    #    'embedding_past_range_set': [0.005, 1.25594,
    #                                 2.50594, 5.0],
    #    'embedding_number_of_bins_set': [1, 3, 5],
    #    'number_of_bootstraps_R_max': 10,
    #    'number_of_bootstraps_R_tot': 10,
    #    'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
    #    'estimation_method': 'bbc',
    #    'debug': True}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)


def test_optimization_Rudelt_shuffling():
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811,
                                             0.31548, 0.62946, 1.25594, 2.50594, 5.0],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
                'estimation_method': 'shuffling',
                'debug': True}
    #settings = {
    #    'embedding_past_range_set': [0.005, 1.25594,
    #                                 2.50594, 5.0],
    #    'embedding_number_of_bins_set': [1, 3, 5],
    #    'number_of_bootstraps_R_max': 10,
    #    'number_of_bootstraps_R_tot': 10,
    #    'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
    #    'estimation_method': 'shuffling',
    #    'debug': True}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_shuffling = optimization_rudelt.optimize(data, processes)


def test_optimization_Rudelt_bbc_multiple_processes():
    # add multiple processes to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    nr_processes = 10
    spiketimedata = np.empty(shape=(nr_processes), dtype=np.ndarray)
    nr_spikes = np.empty(shape=(nr_processes), dtype=int)

    for i in range(nr_processes):
        if i == 5:
            spiketimedata[i] = spiketimes
            nr_spikes[i] = len(spiketimes)
        elif i == 2:
            spiketimedata[i] = spiketimes
            nr_spikes[i] = len(spiketimes)
        else:
            ran = np.random.rand(len(spiketimes)) * 1000
            new = spiketimes + ran
            sampl = int(np.random.uniform(low=0.6 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            nr_spikes[i] = sampl
            spiketimedata[i] = new[0:sampl]

    data = Data_spiketime()
    data.set_data(spiketimedata)

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811,
                                             0.31548, 0.62946, 1.25594, 2.50594, 5.0],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
                'estimation_method': 'bbc',
                'debug': True}
    #settings = {
    #    'embedding_past_range_set': [0.005, 1.25594,
    #                                 2.50594],
    #    'embedding_number_of_bins_set': [1, 3],
    #    'number_of_bootstraps_R_max': 5,
    #    'number_of_bootstraps_R_tot': 5,
    #    'analyse_auto_MI': False,
    #    'estimation_method': 'bbc',
    #    'debug': True}

    processes = [5, 1, 2]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res5 = results_bbc.get_single_process(5)
    res2 = results_bbc.get_single_process(2)
    res1 = results_bbc.get_single_process(1)

    assert (res2.R_tot == res5.R_tot), ("R_tot of the same process differ. Needs to be identical")
    assert (res2.R_tot != res1.R_tot), ("R_tot of the different processes are Identical. Needs to be different")


if __name__ == '__main__':
    test_optimization_Rudelt_bbc_multiple_processes()
    test_optimization_Rudelt_bbc()
    test_optimization_Rudelt_shuffling()
