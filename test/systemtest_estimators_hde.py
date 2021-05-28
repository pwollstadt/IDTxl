"""example for HDE estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against JIDT estimators.
"""
import numpy as np
from idtxl.estimators_hd import hdEstimatorShuffling
import matplotlib.pyplot as plt

settings = {'debug': False,
            'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811, 0.31548, 0.62946, 1.25594,
                                       2.50594, 5.0],
            'number_of_bootstraps_R_max': 10,
            'number_of_bootstraps_R_tot': 10,
            'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25]}
            #'parallel': True,
            #'numberCPUs': 8}

data = np.loadtxt('/home/mlindner/Dropbox/hdestimator-master/sample_data/spike_times.dat', dtype=float)

est = hdEstimatorShuffling(settings)
results_shuffling = est.estimate(data)
print("Shuffling estimator")
print("tau_R: ", str(results_shuffling['tau_R']))
print("T_D: ", str(results_shuffling['T_D']))
print("R_tot: ", str(results_shuffling['R_tot']))
print("AIS_tot: ", str(results_shuffling['AIS_tot']))
print("opt_number_of_bins_d: ", str(results_shuffling['opt_number_of_bins_d']))
print("opt_scaling_k: ", str(results_shuffling['opt_scaling_k']))
print("---------------------------------")

"""
est = hdEstimatorBBC(settings)
results_bbc = est.estimate(data)
print("bbc estimator")
print("tau_R: ", str(results_bbc['tau_R']))
print("T_D: ", str(results_bbc['T_D']))
print("R_tot: ", str(results_bbc['R_tot']))
print("AIS_tot: ", str(results_bbc['AIS_tot']))
print("opt_number_of_bins_d: ", str(results_bbc['opt_number_of_bins_d']))
print("opt_scaling_k: ", str(results_bbc['opt_scaling_k']))
print("---------------------------------")
"""

ax = plt.subplot(221)
y = results_shuffling['HD_max_R']
x = np.array(results_shuffling['settings']['embedding_past_range_set'])
ax.plot(x, y)
ax.axvline(x=results_shuffling['tau_R'], color='k', ls='--', label=r"$\tau_R$")
ax.set_xscale('log')
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=6)
ax.title.set_text('History Dependence Shuffle')

ax2 = plt.subplot(223)
leg = []
for key in results_shuffling['auto_MI'].keys():
    x = results_shuffling['auto_MI'][key][0]
    y = results_shuffling['auto_MI'][key][1]
    leg.append(key)
    ax2.plot(x, y, label=str(float(key)*1000))
ax2.set_xscale('log')
ax2.title.set_text('auto MI Shuffle')
ax2.legend(loc="upper right")
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.show()

"""
ax3 = plt.subplot(222)
y = results_shuffling['HD_max_R']
x = np.array(results_bbc['settings']['embedding_past_range_set'])
ax3.plot(x, y)
ax3.axvline(x=results_bbc['tau_R'], color='k', ls='--', label=r"$\tau_R$")
ax3.set_xscale('log')
ax3.set_xticks(x)
ax3.set_xticklabels(x, fontsize=6)
ax3.title.set_text('History Dependence bbc')


ax4 = plt.subplot(224)
leg = []
for key in results_bbc['auto_MI'].keys():
    x = results_bbc['auto_MI'][key][0]
    y = results_bbc['auto_MI'][key][1]
    leg.append(key)
    ax2.plot(x, y, label=str(float(key)*1000))
ax4.set_xscale('log')
ax4.title.set_text('auto MI bbc')
ax4.legend(loc="upper right")
plt.show()

"""