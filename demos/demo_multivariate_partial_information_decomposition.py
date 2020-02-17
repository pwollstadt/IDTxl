# Import classes
import numpy as np
from idtxl.multivariate_partial_information_decomposition import (
                                        MultivariatePartialInformationDecomposition)
from idtxl.data import Data

# a) Generate test data
n = 100
alph = 2
x = np.random.randint(0, alph, n)
y = np.random.randint(0, alph, n)
z = np.logical_xor(x, y).astype(int)
data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

# b) Initialise analysis object and define settings for both PID estimators
pid = MultivariatePartialInformationDecomposition()
settings_SxPID = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0]}

# c) Run Tartu estimator
results_SxPID = pid.analyse_single_target(
    settings=settings_SxPID, data=data, target=2, sources=[0, 1])


# e) Print results to console
print('\nLogical XOR')
print('Estimator            SxPID\t\tExpected\n')
print('Uni s1               {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),)][2],
    .58))
print('Uni s2               {0:.4f}\t\t{1:.2f}'.format(
    results_SxPID.get_single_target(2)['avg'][((2,),)][2],
    0.58))
print('Shared s1_s2         {0:.4f}\t\t{1:.2f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),(2,),)][2],
    -0.58))
print('Synergy s1_s2        {0:.4f}\t\t{1:.2f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,2,),)][2],
    0.41))
