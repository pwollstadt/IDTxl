"""Provide unit tests for PID estimators.

Created on Mon Apr 11 21:51:56 2016

@author: wibral
"""
import time as tm
import numpy as np
from idtxl.set_estimator import Estimator_pid


ALPH_X = 2
ALPH_Y = 2
ALPH_Z = 2

X = np.asarray([0, 0, 1, 1])
# X = np.squeeze(nm.repmat(X, 1, 100000))
Y = np.asarray([0, 1, 0, 1])
# Y = np.squeeze(nm.repmat(Y, 1, 100000))

# alternative way of generating the distributions stochastically
# n = 10000000
# X = np.random.randint(0, ALPH_X, n)
# Y = np.random.randint(0, ALPH_Y, n)


analysis_opts = {
    'alph_s1': ALPH_X,
    'alph_s2': ALPH_Y,
    'alph_t': ALPH_Z,
    'max_unsuc_swaps_row_parm': 60,
    'num_reps': 63,
    'max_iters': 1000
}
pid_sydney = Estimator_pid('pid_sydney')
pid_tartu = Estimator_pid('pid_tartu')


def test_pid_and():
    """Test PID estimator on logical AND."""
    Z = np.logical_and(X, Y).astype(int)
    est_sydney, est_tartu = _estimate(Z)

    assert np.isclose(0.5, est_sydney['syn_s1_s2']), ('Synergy is not 0.5. for'
                                                      'Sydney estimator.')
    assert np.isclose(0.5, est_tartu['syn_s1_s2']), ('Synergy is not 0.5. for'
                                                     'Tartu estimator.')


def test_pid_xor():
    """Test PID estimator on logical XOR."""
    Z = np.logical_xor(X, Y).astype(int)
    est_sydney, est_tartu = _estimate(Z)

    assert np.isclose(1, est_sydney['syn_s1_s2']), 'Synergy is not 1.'
    assert np.isclose(1, est_tartu['syn_s1_s2']), 'Synergy is not 1.'


def test_pip_source_copy():
    """Test PID estimator on copied source."""
    Z = X
    est_sydney, est_tartu = _estimate(Z)

    assert np.isclose(1, est_sydney['unq_s1']), ('Unique information 1 is not '
                                                 '1 for Sydney estimator.')
    assert np.isclose(0, est_sydney['unq_s2']), ('Unique information 2 is not '
                                                 '0 for Sydney estimator.')
    assert np.isclose(0, est_sydney['shd_s1_s2']), ('Shared information is not'
                                                    '0 for Sydney estimator.')
    assert np.isclose(0, est_sydney['syn_s1_s2']), ('Synergy is not 0 for '
                                                    'Sydney estimator.')
    assert np.isclose(0, est_tartu['shd_s1_s2']), ('Shared information is not'
                                                   '0 for Tartu estimator.')
    assert np.isclose(0, est_tartu['syn_s1_s2']), ('Synergy is not 0 for '
                                                   'Tartu estimator.')


def test_xor_long():
    """Test PID estimation with Sydney estimator on XOR with higher N."""
    # logical AND
    n = 1000
    alph = 2
    s1 = np.random.randint(0, alph, n)
    s2 = np.random.randint(0, alph, n)
    target = np.logical_xor(s1, s2).astype(int)
    analysis_opts = {
        'alph_s1': 2,
        'alph_s2': 2,
        'alph_t': 2,
        'max_unsuc_swaps_row_parm': 3,
        'num_reps': 63,
        'max_iters': 10000
    }
    print('\n\nTesting PID estimator on binary XOR, pointset size: {0}, '
          'iterations: {1}'.format(n, analysis_opts['max_iters']))

    # Sydney estimator
    tic = tm.time()
    est_sydney = pid_sydney.estimate(s1, s2, target, analysis_opts)
    t_sydney = tm.time() - tic
    # Tartu estimator
    tic = tm.time()
    est_tartu = pid_tartu.estimate(s1, s2, target, analysis_opts)
    t_tartu = tm.time() - tic

    print('\nLogical XOR - N = 1000')
    print('Estimator            Sydney\t\tTartu\n')
    print('PID evaluation       {:.3f} s\t\t{:.3f} s\n'.format(t_sydney,
                                                               t_tartu))
    print('Uni s1               {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['unq_s1'],
                                                    est_tartu['unq_s1']))
    print('Uni s2               {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['unq_s2'],
                                                    est_tartu['unq_s2']))
    print('Shared s1_s2         {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['shd_s1_s2'],
                                                    est_tartu['shd_s1_s2']))
    print('Synergy s1_s2        {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['syn_s1_s2'],
                                                    est_tartu['syn_s1_s2']))
    assert 0.9 < est_sydney['syn_s1_s2'] <= 1.1, (
            'Sydney estimator incorrect synergy: {0}, expected was {1}'.format(
                    est_sydney['syn_s1s2'], 0.98))
    assert 0.9 < est_tartu['syn_s1_s2'] <= 1.1, (
            'Tartu estimator incorrect synergy: {0}, expected was {1}'.format(
                    est_sydney['syn_s1s2'], 0.98))


def _estimate(Z):
    """Estimate PID for a given target."""
    # Sydney estimator
    tic = tm.time()
    est_sydney = pid_sydney.estimate(X, Y, Z, analysis_opts)
    t_sydney = tm.time() - tic
    # Tartu estimator
    tic = tm.time()
    est_tartu = pid_tartu.estimate(X, Y, Z, analysis_opts)
    t_tartu = tm.time() - tic

    print('\nCopied source')
    print('Estimator            Sydney\t\tTartu\n')
    print('PID evaluation       {:.3f} s\t\t{:.3f} s\n'.format(t_sydney,
                                                               t_tartu))
    print('Uni s1               {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['unq_s1'],
                                                    est_tartu['unq_s1']))
    print('Uni s2               {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['unq_s2'],
                                                    est_tartu['unq_s2']))
    print('Shared s1_s2         {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['shd_s1_s2'],
                                                    est_tartu['shd_s1_s2']))
    print('Synergy s1_s2        {0:.8f}\t\t{1:.8f}'.format(
                                                    est_sydney['syn_s1_s2'],
                                                    est_tartu['syn_s1_s2']))
    return est_sydney, est_tartu


if __name__ == '__main__':
    test_xor_long()
    test_pid_and()
    test_pid_xor()
    test_pip_source_copy()
