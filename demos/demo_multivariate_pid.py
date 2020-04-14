# Import classes
import numpy as np
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data

# a) Generate test data
n = 100
alph = 2
x = np.random.randint(0, alph, n)
y = np.random.randint(0, alph, n)
z = np.logical_xor(x, y).astype(int)
data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

# b) Initialise analysis object and define settings for SxPID estimators
pid = MultivariatePID()
settings_SxPID = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0]}

# c) Run Goettingen estimator
results_SxPID = pid.analyse_single_target(
    settings=settings_SxPID, data=data, target=2, sources=[0, 1])


# e) Print results to console
print('\nLogical XOR')
print('Estimator            SxPID\t\tExpected\n')
print('Uni s1               {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),)][2],
    .5896))
print('Uni s2               {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((2,),)][2],
    0.5896))
print('Shared s1_s2         {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),(2,),)][2],
    -0.5896))
print('Synergy s1_s2        {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,2,),)][2],
    0.415))

# Some special Examples

# Pointwise Unique
x = np.asarray([0, 1, 0, 2])
y = np.asarray([1, 0, 2, 0])
z = np.asarray([1, 1, 2, 2])
data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

pid = MultivariatePID()
settings_SxPID = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0]}

results_SxPID = pid.analyse_single_target(
    settings=settings_SxPID, data=data, target=2, sources=[0, 1])

print('\nLogical PwUnq')
print('Estimator            SxPID\t\tExpected\n')
print('Uni s1               {0:.4f}\t\t{1:.1f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),)][2],
    .5))
print('Uni s2               {0:.4f}\t\t{1:.1f}'.format(
    results_SxPID.get_single_target(2)['avg'][((2,),)][2],
    0.5))
print('Shared s1_s2         {0:.4f}\t\t{1:.1f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),(2,),)][2],
    0.))
print('Synergy s1_s2        {0:.4f}\t\t{1:.1f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,2,),)][2],
    0.))

# Redundancy Error
x = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0])
z = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

pid = MultivariatePID()
settings_SxPID = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0]}

results_SxPID = pid.analyse_single_target(
    settings=settings_SxPID, data=data, target=2, sources=[0, 1])

print('\nLogical RndErr')
print('Estimator            SxPID\t\tExpected\n')
print('Uni s1               {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),)][2],
    .4433))
print('Uni s2               {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((2,),)][2],
    -0.368))
print('Shared s1_s2         {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,),(2,),)][2],
    0.5567))
print('Synergy s1_s2        {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(2)['avg'][((1,2,),)][2],
    0.368))

# Three bits hash
s1 = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
s2 = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
s3 = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
z  = np.asarray([0, 1, 1, 0, 1, 0, 0, 1])
data = Data(np.vstack((s1, s2, s3, z)), 'ps', normalise=False)

pid = MultivariatePID()
settings_SxPID = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0, 0]}

results_SxPID = pid.analyse_single_target(
    settings=settings_SxPID, data=data, target=3, sources=[0, 1, 2])

print('\nLogical PwUnq')
print('Estimator                                                SxPID\t\tExpected\n')
print('Uni s1                                                   {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,),)][2], 0.3219))
print('Uni s2                                                   {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((2,),)][2], 0.3219))
print('Uni s3                                                   {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((3,),)][2], 0.3219))
print('Synergy s1_s2_s3                                         {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,2,3),)][2], 0.2451))
print('Synergy s1_s2                                            {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,2,),)][2], 0.1699))
print('Synergy s1_s3                                            {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,3,),)][2], 0.1699))
print('Synergy s2_s3                                            {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((2,3,),)][2], 0.1699))
print('Shared s1_s2_s3                                          {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,),(2,),(3,),)][2], 0.1926))
print('Shared of (s1, s2)                                       {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,), (2,),)][2], -0.1926))
print('Shared of (s1, s2)                                       {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,), (3,),)][2], -0.1926))
print('Shared of (s2, s3)                                       {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((2,), (3,),)][2], -0.1926))
print('Shared of (Synergy s1_s2, Synergy s1_s3)                 {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,2,), (1,3),)][2], 0.0931))
print('Shared of (Synergy s1_s2, Synergy s2_s3)                 {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,2,), (2,3),)][2], 0.0931))
print('Shared of (Synergy s1_s3, Synergy s2_s3)                 {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,3,), (2,3),)][2], 0.0931))
print('Shared of (Synergy s1_s2, Synergy s1_s3, Synergy s2_s3)  {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,2,), (1,3), (2,3),)][2], -0.2268))
print('Shared of (s1, Synergy s2_s3)                            {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((1,), (2,3),)][2], -0.1292))
print('Shared of (s2, Synergy s1_s3)                            {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((2,), (1,3),)][2], -0.1292))
print('Shared of (s3, Synergy s1_s2)                            {0:.4f}\t\t{1:.4f}'.format(
    results_SxPID.get_single_target(3)['avg'][((3,), (1,2),)][2], -0.1292))
