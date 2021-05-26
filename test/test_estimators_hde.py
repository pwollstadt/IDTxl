"""Test HDE estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against JIDT estimators.
"""
import math
import pytest
import numpy as np
from idtxl.estimators_hd import hdEstimatorShuffling
from test_estimators_jidt import _get_gauss_data
import pprint

settings = {'debug': False,
            'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811, 0.31548, 0.62946, 1.25594,
                                       2.50594, 5.0],
            'number_of_bootstraps_R_max': 10,
            'number_of_bootstraps_R_tot': 10,
            'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
            'parallel': True}
            #'numberCPUs': 8
            #}

data = np.loadtxt('/home/mlindner/Dropbox/hdestimator-master/sample_data/spike_times.dat', dtype=float)

est = hdEstimatorShuffling(settings)

results_shuffling = est.estimate(data)
print("tau_R: ", str(results_shuffling['tau_R']))
print("T_D: ", str(results_shuffling['T_D']))
print("R_tot: ", str(results_shuffling['R_tot']))
print("AIS_tot: ", str(results_shuffling['AIS_tot']))
print("opt_number_of_bins_d: ", str(results_shuffling['opt_number_of_bins_d']))
print("opt_scaling_k: ", str(results_shuffling['opt_scaling_k']))


from matplotlib import pyplot as pl


a=1

for key in results_shuffling['auto_MI']['auto_MI'].keys():
    X = np.linspace(min(results_shuffling['settings']['embedding_number_of_bins_set']),
                    max(results_shuffling['settings']['embedding_number_of_bins_set']),
                    len(results_shuffling['settings']['auto_MI']['auto_MI'][key]))
    Y = results_shuffling['settings']['auto_MI']['auto_MI'][key]
pl.plot(X,Y)
pl.show()

