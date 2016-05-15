"""Unit tests for fast PID estimator.
"""
import time as tm
import numpy as np
import numpy.matlib as nm
import idtxl.estimators_fast_pid_ext_rep as epid

n = 1000
ALPH_X = 2
ALPH_Y = 2
ALPH_Z = 2

X = np.asarray([0, 0, 1, 1]) # np.random.randint(0, ALPH_X, n)
#X = np.squeeze(nm.repmat(X, 1, 100000))
Y = np.asarray([0, 1, 0, 1]) # np.random.randint(0, ALPH_Y, n)
#Y = np.squeeze(nm.repmat(Y, 1, 100000))


CFG = {
    'alph_s1': ALPH_X,
    'alph_s2': ALPH_Y,
    'alph_t': ALPH_Z,
    'max_unsuc_swaps_row_parm': 60,
    'num_reps': 63,
    'max_iters': 1000
}

def test_pid_and():
    """Test PID estimator on logical AND."""
    z = np.logical_and(X, Y).astype(int)
    tic = tm.time()
    est = epid.pid(X, Y, z, CFG)
    toc = tm.time()
    print('\nLogical AND')
    print('PID evaluation       {:.3f} seconds\n'.format(toc - tic))
    print('Uni s1              ', est['unq_s1'])
    print('Uni s2              ', est['unq_s2'])
    print('Shared s1_s2        ', est['shd_s1_s2'])
    print('Synergy s1_s2       ', est['syn_s1_s2'])
    assert np.isclose(0.5, est['syn_s1_s2'][0]), 'Synergy is not 0.5.'


def test_pid_xor():
    """Test PID estimator on logical XOR."""
    z = np.logical_xor(X, Y).astype(int)
    tic = tm.time()
    est = epid.pid(X, Y, z, CFG)
    toc = tm.time()

    print('\nLogical XOR')
    print('PID evaluation       {:.3f} seconds\n'.format(toc - tic))
    print('Uni s1              ', est['unq_s1'])
    print('Uni s2              ', est['unq_s2'])
    print('Shared s1_s2        ', est['shd_s1_s2'])
    print('Synergy s1_s2       ', est['syn_s1_s2'])
    assert np.isclose(1, est['syn_s1_s2'][0]), 'Synergy is not 1.'


def test_pip_source_copy():
    """Test PID estimator on copied source."""
    z = X
    tic = tm.time()
    est = epid.pid(X, Y, z, CFG)
    toc = tm.time()

    print('\nCopied source')
    print('PID evaluation       {:.3f} seconds\n'.format(toc - tic))
    print('Uni s1              ', est['unq_s1'])
    print('Uni s2              ', est['unq_s2'])
    print('Shared s1_s2        ', est['shd_s1_s2'])
    print('Synergy s1_s2       ', est['syn_s1_s2'])
    assert np.isclose(1, est['unq_s1'][0]), 'Unique information 1 is not 0.'
    assert np.isclose(0, est['unq_s2'][0]), 'Unique information 2 is not 0.'
    assert np.isclose(0, est['shd_s1_s2'][0]), 'Shared information is not 0.'
    assert np.isclose(0, est['syn_s1_s2'][0]), 'Synergy is not 0.'


if __name__ == '__main__':
    test_pid_and()
    test_pid_xor()
    test_pip_source_copy()
