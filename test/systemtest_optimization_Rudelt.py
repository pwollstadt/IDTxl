"""Provides unit test for Rudelt optimization on example data."""

from idtxl.data_spiketime import Data_spiketime
from idtxl.embedding_optimization_ais_Rudelt import OptimizationRudelt
import numpy as np
import os
import sys
from idtxl.hde_utils import hde_visualize_results as vis
import math


def test_optimization_Rudelt_bbc():
    print("\nTest optimization_Rudelt using bbc estimator on Rudelt data")
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811,
                                             0.31548, 0.62946, 1.25594, 2.50594, 5.0],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
                'estimation_method': 'bbc',
                'visualization': True,
                'output_path': os.path.abspath(os.path.dirname(sys.argv[0])),
                'output_prefix': 'systemtest_optimizationRudelt_image1'}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res = results_bbc.get_single_process(0)
    assert (round(res.T_D, 3) == 0.315), "Estimated T_D incorrect"
    assert (round(res.tau_R, 3) == 0.081), "Estimated tau_R incorrect"
    assert (round(res.R_tot, 3) == 0.122), "Estimated R_tot incorrect"
    assert (res.R_tot_CI[0] is None), "R_tot_CO_lo needs to be None"
    assert (res.R_tot_CI[1] is None), "R_tot_CO_hi needs to be None"
    assert (res.opt_number_of_bins_d == 5), "Estimated opt number of bins incorrect"
    assert (round(res.opt_scaling_k, 1) == 0.0), "Estimated opt scaling k size incorrect"
    assert (round(res.opt_first_bin_size, 3) == 0.063), "Estimated opt first bin size incorrect"
    assert (round(res.firing_rate, 1) == 4.0), "Calculated firing rate incorrect"
    assert (round(res.recording_length, 1) == 952.2), "Calculated recording length incorrect"
    assert ('auto_MI' in res), "auto_MI results do not exist"

    f1 = 'systemtest_optimizationRudelt_image1_process' + str(processes[0]) + '.svg'
    filename1 = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), f1)
    try:
        os.path.isfile(filename1)
        os.remove(filename1)
    except IOError:
        print('File: \n{0}\nwas not created!'.format(filename1))
        return

    f2 = 'systemtest_optimizationRudelt_image2_process' + str(processes[0]) + '.svg'
    filename2 = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), f2)

    vis(results_bbc, 0, filename=filename2)
    try:
        os.path.isfile(filename2)
        os.remove(filename2)
    except IOError:
        print('File: \n{0}\nwas not created!'.format(filename2))
        return

    print("\tpassed")


def test_optimization_Rudelt_bbc_noAveragedR_noAutoMI_noOutputImage():
    print("\nTest optimization_Rudelt using bbc estimator on Rudelt data with no averagedR, no MI and no visualization")
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.15811, 1.25594, 5.0],
                'embedding_number_of_bins_set': [1, 3, 5],
                'embedding_scaling_exponent_set': {'number_of_scalings': 10,
                                                   'min_first_bin_size': 0.005,
                                                   'min_step_for_scaling': 0.01},
                'return_averaged_R': False,
                'analyse_auto_MI': False,
                'estimation_method': 'bbc',
                'visualization': False}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res = results_bbc.get_single_process(0)
    assert (round(res.T_D, 3) == 1.256), "Estimated T_D incorrect"
    assert (round(res.tau_R, 3) == 0.549), "Estimated tau_R incorrect"
    assert (round(res.R_tot, 3) == 0.112), "Estimated R_tot incorrect"
    assert (round(res.AIS_tot, 3) == 0.011), "Estimated AIS_tot incorrect"
    assert (math.isclose(res.R_tot_CI[0], 0.107, rel_tol=1e-02)), "Estimated R_tot_CO_lo incorrect"
    assert (math.isclose(res.R_tot_CI[1], 0.116, rel_tol=1e-02)), "Estimated R_tot_CO_hi incorrect"
    assert (res.opt_number_of_bins_d == 5), "Estimated opt number of bins incorrect"
    assert (round(res.opt_scaling_k, 2) == 0.44), "Estimated opt scaling k size incorrect"
    assert (round(res.opt_first_bin_size, 3) == 0.014), "Estimated opt first bin size incorrect"
    assert (round(res.firing_rate, 1) == 4.0), "Calculated firing rate incorrect"
    assert (round(res.recording_length, 1) == 952.2), "Calculated recording length incorrect"
    assert (round(res.H_spiking, 3) == 0.098), "Calculated H_spiking incorrect"
    assert ('auto_MI' not in res), "auto_MI results does exist but should not"

    print("\tpassed")


def test_optimization_Rudelt_shuffling():
    print("\nTest optimization_Rudelt using shuffling estimator on Rudelt data")
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811,
                                             0.31548, 0.62946, 1.25594, 2.50594, 5.0],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
                'estimation_method': 'shuffling'}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_shuffling = optimization_rudelt.optimize(data, processes)

    res = results_shuffling.get_single_process(0)
    assert (math.isclose(res.T_D, 0.315, rel_tol=1e-02)), "Estimated T_D {0} incorrect".format(str(res.T_D))
    assert (math.isclose(res.tau_R, 0.08, rel_tol=1e-01)), "Estimated tau_R {0} incorrect".format(str(res.tau_R))
    assert (math.isclose(res.R_tot, 0.12, rel_tol=1e-02)), "Estimated R_tot {0} incorrect".format(str(res.R_tot))
    assert (res.opt_number_of_bins_d == 5), "Estimated opt number of bins incorrect"
    assert (round(res.opt_scaling_k, 1) == 0.0), "Estimated opt scaling k size incorrect"
    assert (round(res.firing_rate, 1) == 4.0), "Calculated firing rate incorrect"
    assert (round(res.recording_length, 1) == 952.2), "Calculated recording length incorrect"

    print("\tpassed")


def test_optimization_Rudelt_shuffling_noAveragedR_noAutoMI_noOutputImage():
    print("\nTest optimization_Rudelt using shuffling estimator on Rudelt data with no averagedR, no MI and no visualization")
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    settings = {'embedding_past_range_set': [0.005, 0.00998, 0.15811, 1.25594, 5.0],
                'embedding_number_of_bins_set': [1, 3, 5],
                'embedding_scaling_exponent_set': {'number_of_scalings': 10,
                                                   'min_first_bin_size': 0.005,
                                                   'min_step_for_scaling': 0.01},
                'return_averaged_R': False,
                'analyse_auto_MI': False,
                'estimation_method': 'shuffling',
                'visualization': False}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res = results_bbc.get_single_process(0)
    assert (round(res.T_D, 3) == 1.256), "Estimated T_D incorrect"
    assert (round(res.tau_R, 3) == 0.549), "Estimated tau_R incorrect"
    assert (round(res.R_tot, 3) == 0.111), "Estimated R_tot incorrect"
    assert (round(res.AIS_tot, 3) == 0.011), "Estimated AIS_tot incorrect"
    assert (math.isclose(res.R_tot_CI[0], 0.107, rel_tol=1e-02)), "Estimated R_tot_CO_lo incorrect"
    assert (math.isclose(res.R_tot_CI[1], 0.116, rel_tol=1e-02)), "Estimated R_tot_CO_hi incorrect"
    assert (res.opt_number_of_bins_d == 5), "Estimated opt number of bins incorrect"
    assert (round(res.opt_scaling_k, 2) == 0.44), "Estimated opt scaling k size incorrect"
    assert (round(res.opt_first_bin_size, 3) == 0.014), "Estimated opt first bin size incorrect"
    assert (round(res.firing_rate, 1) == 4.0), "Calculated firing rate incorrect"
    assert (round(res.recording_length, 1) == 952.2), "Calculated recording length incorrect"
    assert (round(res.H_spiking, 3) == 0.098), "Calculated H_spiking incorrect"
    assert ('auto_MI' not in res), "auto_MI results does exist but should not"

    print("\tpassed")


def test_optimization_Rudelt_bbc_multiple_processes():
    print("\nTest optimization_Rudelt using bbc estimator on multiple processes")

    # add multiple processes to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    nr_processes = 6
    spiketimedata = np.empty(shape=nr_processes, dtype=np.ndarray)
    nr_spikes = np.empty(shape=nr_processes, dtype=int)

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
            sampl = int(np.random.uniform(low=0.5 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            nr_spikes[i] = sampl
            spiketimedata[i] = new[0:sampl]

    data = Data_spiketime()
    data.set_data(spiketimedata)

    # Only limited value set for speeding up the test. Correct results are not important,
    # because only the relative results between equal and nonequal data are checked
    settings = {'embedding_past_range_set': [0.005, 0.31548, 0.62946, 2.50594, 5.0],
                'embedding_number_of_bins_set': [1, 3, 5],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'analyse_auto_MI': False,
                'estimation_method': 'bbc'}

    processes = [5, 1, 2]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res5 = results_bbc.get_single_process(5)
    res2 = results_bbc.get_single_process(2)
    res1 = results_bbc.get_single_process(1)

    assert (res2.R_tot == res5.R_tot), "R_tot of the same process differ. Needs to be identical"
    assert (res2.R_tot != res1.R_tot), "R_tot of the different processes are Identical. Needs to be different"
    assert (res2.T_D == res5.T_D), "T_D of the same process differ. Needs to be identical"
    assert (res2.T_D != res1.T_D), "T_D of the different processes are Identical. Needs to be different"
    assert (res2.tau_R == res5.tau_R), "tau_R of the same process differ. Needs to be identical"

    print("\tpassed")


def test_optimization_Rudelt_shuffling_multiple_processes():
    print("\nTest optimization_Rudelt using bbc estimator on multiple processes")

    # add multiple processes to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    nr_processes = 6
    spiketimedata = np.empty(shape=nr_processes, dtype=np.ndarray)
    nr_spikes = np.empty(shape=nr_processes, dtype=int)

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
            sampl = int(np.random.uniform(low=0.5 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            nr_spikes[i] = sampl
            spiketimedata[i] = new[0:sampl]

    data = Data_spiketime()
    data.set_data(spiketimedata)

    # Only limited value set for speeding up the test. Correct results are not important,
    # because only the relative results between equal and nonequal data are checked
    settings = {'embedding_past_range_set': [0.005, 0.31548, 0.62946, 2.50594, 5.0],
                'embedding_number_of_bins_set': [1, 3, 5],
                'number_of_bootstraps_R_max': 10,
                'number_of_bootstraps_R_tot': 10,
                'analyse_auto_MI': False,
                'estimation_method': 'shuffling'}

    processes = [5, 1, 2]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res5 = results_bbc.get_single_process(5)
    res2 = results_bbc.get_single_process(2)
    res1 = results_bbc.get_single_process(1)

    assert (math.isclose(res2.R_tot, res5.R_tot, rel_tol=1e-02)), \
        "R_tot of the same process differ. Needs to be identical"
    assert (res2.R_tot != res1.R_tot), \
        "R_tot of the different processes are Identical. Needs to be different"
    assert (math.isclose(res2.T_D, res5.T_D, rel_tol=1e-02)), \
        "T_D of the same process differ. Needs to be identical"
    assert (res2.T_D != res1.T_D), \
        "T_D of the different processes are Identical. Needs to be different"
    assert (math.isclose(res2.tau_R, res5.tau_R, rel_tol=1e-02)), \
        "tau_R of the same process differ. Needs to be identical"

    print("\tpassed")


def test_optimization_Rudelt_bbc_noAutoMI_OutputImage():
    print("\nTest optimization_Rudelt output image creation without auto MI")
    data = Data_spiketime()             # initialise empty data object
    data.load_Rudelt_data()             # load Rudelt spike time data

    # Only limited value set for speeding up the test. Results are not important,
    # because only the output image creation is tested
    settings = {
        'embedding_past_range_set': [0.005, 0.62946],
        'embedding_number_of_bins_set': [1, 3],
        'number_of_bootstraps_R_max': 10,
        'number_of_bootstraps_R_tot': 10,
        'analyse_auto_MI': False,
        'estimation_method': 'bbc',
        'visualization': True,
        'output_path': os.path.abspath(os.path.dirname(sys.argv[0])),
        'output_prefix': 'systemtest_optimizationRudelt_image1'}

    processes = [0]
    optimization_rudelt = OptimizationRudelt(settings)
    results_bbc = optimization_rudelt.optimize(data, processes)

    res = results_bbc.get_single_process(0)
    assert ('auto_MI' not in res), "auto_MI results does exist but was not calcuated"

    f1 = 'systemtest_optimizationRudelt_image1_process'+ str(processes[0]) + '.svg'
    filename1 = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), f1)
    try:
        os.path.isfile(filename1)
        os.remove(filename1)
    except IOError:
        print('File: \n{0}\nwas not created!'.format(filename1))
        return

    f2 = 'systemtest_optimizationRudelt_image2_process'+ str(processes[0]) + '.svg'
    filename2 = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), f2)

    vis(results_bbc, 0, filename=filename2)
    try:
        os.path.isfile(filename2)
        os.remove(filename2)
    except IOError:
        print('File: \n{0}\nwas not created!'.format(filename2))
        return

    print("\tpassed")


if __name__ == '__main__':
    test_optimization_Rudelt_bbc()
    test_optimization_Rudelt_bbc_noAveragedR_noAutoMI_noOutputImage()
    test_optimization_Rudelt_shuffling()
    test_optimization_Rudelt_shuffling_noAveragedR_noAutoMI_noOutputImage()
    test_optimization_Rudelt_bbc_multiple_processes()
    test_optimization_Rudelt_shuffling_multiple_processes()
    test_optimization_Rudelt_bbc_noAutoMI_OutputImage()
