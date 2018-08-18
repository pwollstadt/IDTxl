"""Demonstrate the usage of IDTxl's core estimators."""
import numpy as np
from idtxl.estimators_jidt import (JidtDiscreteAIS, JidtDiscreteCMI,
                                   JidtDiscreteMI, JidtDiscreteTE,
                                   JidtKraskovAIS, JidtKraskovCMI,
                                   JidtKraskovMI, JidtKraskovTE,
                                   JidtGaussianAIS, JidtGaussianCMI,
                                   JidtGaussianMI, JidtGaussianTE)
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI
from idtxl.estimators_pid import SydneyPID, TartuPID
from idtxl.idtxl_utils import calculate_mi

# Generate Gaussian test data
n = 10000
covariance = 0.4
corr_expected = covariance / (1 * np.sqrt(covariance**2 + (1-covariance)**2))
expected_mi = calculate_mi(corr_expected)
source_cor = np.random.normal(0, 1, size=n)  # correlated src
source_uncor = np.random.normal(0, 1, size=n)  # uncorrelated src
target = (covariance * source_cor +
          (1 - covariance) * np.random.normal(0, 1, size=n))

# JIDT Discrete estimators
settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
est = JidtDiscreteCMI(settings)
cmi = est.estimate(source_cor, target, source_uncor)
print('Estimated CMI: {0:.5f}, expected CMI: {1:.5f}'.format(cmi, expected_mi))
est = JidtDiscreteMI(settings)
mi = est.estimate(source_cor, target)
print('Estimated MI: {0:.5f}, expected MI: {1:.5f}'.format(mi, expected_mi))
settings['history_target'] = 1
est = JidtDiscreteTE(settings)
te = est.estimate(source_cor[1:n], target[0:n - 1])
print('Estimated TE: {0:.5f}, expected TE: {1:.5f}'.format(te, expected_mi))
settings['history'] = 1
est = JidtDiscreteAIS(settings)
ais = est.estimate(target)
print('Estimated AIS: {0:.5f}, expected AIS: ~0'.format(ais))

# JIDT Kraskov estimators
settings = {}
est = JidtKraskovCMI(settings)
cmi = est.estimate(source_cor, target, source_uncor)
print('Estimated CMI: {0:.5f}, expected CMI: {1:.5f}'.format(cmi, expected_mi))
est = JidtKraskovMI(settings)
mi = est.estimate(source_cor, target)
print('Estimated MI: {0:.5f}, expected MI: {1:.5f}'.format(mi, expected_mi))
settings['history_target'] = 1
est = JidtKraskovTE(settings)
te = est.estimate(source_cor[1:n], target[0:n - 1])
print('Estimated TE: {0:.5f}, expected TE: {1:.5f}'.format(te, expected_mi))
settings['history'] = 1
est = JidtKraskovAIS(settings)
ais = est.estimate(target)
print('Estimated AIS: {0:.5f}, expected AIS: ~0'.format(ais))

# JIDT Gaussian estimators
settings = {}
est = JidtGaussianCMI(settings)
cmi = est.estimate(source_cor, target, source_uncor)
print('Estimated CMI: {0:.5f}, expected CMI: {1:.5f}'.format(cmi, expected_mi))
est = JidtGaussianMI(settings)
mi = est.estimate(source_cor, target)
print('Estimated MI: {0:.5f}, expected MI: {1:.5f}'.format(mi, expected_mi))
settings['history_target'] = 1
est = JidtGaussianTE(settings)
te = est.estimate(source_cor[1:n], target[0:n - 1])
print('Estimated TE: {0:.5f}, expected TE: {1:.5f}'.format(te, expected_mi))
settings['history'] = 1
est = JidtGaussianAIS(settings)
ais = est.estimate(target)
print('Estimated AIS: {0:.5f}, expected AIS: ~0'.format(ais))

# OpenCL Kraskov estimators
settings = {}
est = OpenCLKraskovCMI(settings)
cmi = est.estimate(source_cor, target, source_uncor)
print('Estimated CMI: {0:.5f}, expected CMI: {1:.5f}'.format(cmi[0],
                                                             expected_mi))
est = OpenCLKraskovMI(settings)
mi = est.estimate(source_cor, target)
print('Estimated MI: {0:.5f}, expected MI: {1:.5f}'.format(mi[0], expected_mi))

# Generate binary test data
alph_x = 2
alph_y = 2
alph_z = 2
x = np.random.randint(0, alph_x, n)
y = np.random.randint(0, alph_y, n)
z = np.logical_xor(x, y).astype(int)

# PID estimators
settings = {
        'alph_s1': alph_x,
        'alph_s2': alph_y,
        'alph_t': alph_z,
        'max_unsuc_swaps_row_parm': 60,
        'num_reps': 63,
        'max_iters': 100}
est_sydney = SydneyPID(settings)
est_tartu = TartuPID(settings)
pid_sydney = est_sydney.estimate(x, y, z)
print('Estimated synergy (Sydney): {0:.5f}, expected synergy: ~1'.format(
                                                    pid_sydney['syn_s1_s2']))
pid_tartu = est_tartu.estimate(x, y, z)
print('Estimated synergy (Tartu): {0:.5f}, expected synergy: ~1'.format(
                                                    pid_tartu['syn_s1_s2']))