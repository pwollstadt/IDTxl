# Import classes
import numpy as np
from idtxl.partial_information_decomposition import (
                                        PartialInformationDecomposition)
from idtxl.data import Data

# a) Generate test data
n = 100
alph = 2
x = np.random.randint(0, alph, n)
y = np.random.randint(0, alph, n)
z = np.logical_xor(x, y).astype(int)
data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

# b) Initialise analysis object and define settings for both PID estimators
pid = PartialInformationDecomposition()
settings_tartu = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}
settings_sydney = {
    'alph_s1': alph,
    'alph_s2': alph,
    'alph_t': alph,
    'max_unsuc_swaps_row_parm': 60,
    'num_reps': 63,
    'max_iters': 1000,
    'pid_estimator': 'SydneyPID',
    'lags_pid': [0, 0]}

# c) Run Tartu estimator
results_tartu = pid.analyse_single_target(
    settings=settings_tartu, data=data, target=2, sources=[0, 1])

# d) Run Sydney estimator
pid = PartialInformationDecomposition()
results_sydney = pid.analyse_single_target(
    settings=settings_sydney, data=data, target=2, sources=[0, 1])

# e) Print results to console
print('\nLogical XOR')
print('Estimator            Sydney\t\tTartu\t\tExpected\n')
print('Uni s1               {0:.4f}\t\t{1:.4f}\t\t{2:.2f}'.format(
    results_sydney.get_single_target(2)['unq_s1'],
    results_tartu.get_single_target(2)['unq_s1'],
    0))
print('Uni s2               {0:.4f}\t\t{1:.4f}\t\t{2:.2f}'.format(
    results_sydney.get_single_target(2)['unq_s2'],
    results_tartu.get_single_target(2)['unq_s2'],
    0))
print('Shared s1_s2         {0:.4f}\t\t{1:.4f}\t\t{2:.2f}'.format(
    results_sydney.get_single_target(2)['shd_s1_s2'],
    results_tartu.get_single_target(2)['shd_s1_s2'],
    0))
print('Synergy s1_s2        {0:.4f}\t\t{1:.4f}\t\t{2:.2f}'.format(
    results_sydney.get_single_target(2)['syn_s1_s2'],
    results_tartu.get_single_target(2)['syn_s1_s2'],
    1))
