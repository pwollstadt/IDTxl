"""Provide unit tests for high-level PID estimation.

@author: patricia
"""
import time as tm
import numpy as np
from idtxl.partial_information_decomposition import (
                                        PartialInformationDecomposition)
from idtxl.data import Data
import idtxl.idtxl_utils as utils


def test_pid_xor_data():
    """Test basic calls to PID class."""

    n = 20
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Run Tartu estimator
    settings = {'pid_estimator': 'TartuPID'}
    pid = PartialInformationDecomposition()
    tic = tm.time()
    est_tartu = pid.analyse_single_target(settings, data=data, target=2,
                                          sources=[0, 1], lags=[0, 0])
    t_tartu = tm.time() - tic

    # Run Sydney estimator
    settings = {
        'n_perm': 11,
        'alpha': 0.1,
        'alph_s1': alph,
        'alph_s2': alph,
        'alph_t': alph,
        'max_unsuc_swaps_row_parm': 60,
        'num_reps': 63,
        'max_iters': 1000,
        'pid_estimator': 'SydneyPID'}
    pid = PartialInformationDecomposition()
    tic = tm.time()
    est_sydney = pid.analyse_single_target(settings, data=data, target=2,
                                           sources=[0, 1], lags=[0, 0])
    t_sydney = tm.time() - tic

    print('\nResults Tartu estimator:')
    utils.print_dict(est_tartu)
    print('\nResults Sydney estimator:')
    utils.print_dict(est_sydney)

    print('\nLogical XOR')
    print('Estimator            Sydney\t\tTartu\n')
    print('PID evaluation       {:.3f} s\t\t{:.3f} s\n'.format(t_sydney,
                                                               t_tartu))
    print('Uni s1               {0:.8f}\t\t{1:.8f}'.format(
        est_sydney.single_target[2]['unq_s1'],
        est_tartu.single_target[2]['unq_s1']))
    print('Uni s2               {0:.8f}\t\t{1:.8f}'.format(
        est_sydney.single_target[2]['unq_s2'],
        est_tartu.single_target[2]['unq_s2']))
    print('Shared s1_s2         {0:.8f}\t\t{1:.8f}'.format(
        est_sydney.single_target[2]['shd_s1_s2'],
        est_tartu.single_target[2]['shd_s1_s2']))
    print('Synergy s1_s2        {0:.8f}\t\t{1:.8f}'.format(
        est_sydney.single_target[2]['syn_s1_s2'],
        est_tartu.single_target[2]['syn_s1_s2']))
    assert 0.9 < est_sydney.single_target[2]['syn_s1_s2'] <= 1.1, (
            'Sydney estimator incorrect synergy: {0}, should approx. 1'.format(
                est_sydney.single_target[2]['syn_s1_s2']))
    assert 0.9 < est_tartu.single_target[2]['syn_s1_s2'] <= 1.1, (
            'Tartu estimator incorrect synergy: {0}, should approx. 1'.format(
                est_tartu.single_target[2]['syn_s1_s2']))
    # TODO test this: est = pid.analyse_network(data=data, target=0)


def test_multivariate_sources():
    pass  # TODO implement multivariate sources


if __name__ == '__main__':
    test_pid_xor_data()
