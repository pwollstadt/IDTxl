"""Provide unit tests for PID estimators."""
import time as tm
import numpy as np
import pytest
from idtxl.estimators_pid import SydneyPID, TartuPID

package_missing = False
try:
    import ecos
except ImportError:
    package_missing = True
optimiser_missing = pytest.mark.skipif(
    package_missing,
    reason='ECOS is missing.')

no_float128 = False
try:
    np.float128()
except AttributeError as err:
    if "'module' object has no attribute 'float128'" == err.args[0]:
        no_float128 = True
    else:
        raise
float128_not_available = pytest.mark.skipif(
    no_float128,
    reason="type float128 not available on current architecture")


ALPH_X = 2
ALPH_Y = 2
ALPH_Z = 2

SETTINGS = {
        'alph_s1': ALPH_X,
        'alph_s2': ALPH_Y,
        'alph_t': ALPH_Z,
        'max_unsuc_swaps_row_parm': 60,
        'num_reps': 63,
        'max_iters': 1000}

X = np.asarray([0, 0, 1, 1])
# X = np.squeeze(nm.repmat(X, 1, 100000))
Y = np.asarray([0, 1, 0, 1])
# Y = np.squeeze(nm.repmat(Y, 1, 100000))

# alternative way of generating the distributions stochastically
# n = 10000000
# X = np.random.randint(0, ALPH_X, n)
# Y = np.random.randint(0, ALPH_Y, n)


@optimiser_missing
def test_tartu_estimator():
    # Test Tartu estimator on logical and
    pid_tartu = TartuPID(SETTINGS)
    Z = np.logical_and(X, Y).astype(int)
    est_tartu = pid_tartu.estimate(X, Y, Z)
    assert np.isclose(0.5, est_tartu['syn_s1_s2']), (
        'Synergy is not 0.5. for Tartu estimator.')


@float128_not_available
def test_sydney_estimator():
    # Test Sydney estimator on logical and
    pid_sydney = SydneyPID(SETTINGS)
    Z = np.logical_and(X, Y).astype(int)
    est_sydney = pid_sydney.estimate(X, Y, Z)
    assert np.isclose(0.5, est_sydney['syn_s1_s2']), (
        'Synergy is not 0.5. for Sydney estimator.')


@float128_not_available
@optimiser_missing
def test_pid_and():
    """Test PID estimator on logical AND."""
    Z = np.logical_and(X, Y).astype(int)
    est_sydney, est_tartu = _estimate(Z)

    assert np.isclose(0.5, est_sydney['syn_s1_s2']), ('Synergy is not 0.5. for'
                                                      'Sydney estimator.')
    assert np.isclose(0.5, est_tartu['syn_s1_s2']), ('Synergy is not 0.5. for'
                                                     'Tartu estimator.')


@float128_not_available
@optimiser_missing
def test_pid_xor():
    """Test PID estimator on logical XOR."""
    Z = np.logical_xor(X, Y).astype(int)
    est_sydney, est_tartu = _estimate(Z)

    assert np.isclose(1, est_sydney['syn_s1_s2']), 'Synergy is not 1.'
    assert np.isclose(1, est_tartu['syn_s1_s2']), 'Synergy is not 1.'


@float128_not_available
@optimiser_missing
def test_pip_source_copy():
    """Test PID estimator on copied source."""
    Z = X
    est_sydney, est_tartu = _estimate(Z)

    assert np.isclose(1, est_sydney['unq_s1'], atol=1.e-7), (
        'Unique information 1 is not 1 for Sydney estimator ({0}).'.format(
            est_sydney['unq_s1']))
    assert np.isclose(0, est_sydney['unq_s2'], atol=1.e-7), (
        'Unique information 2 is not 0 for Sydney estimator ({0}).'.format(
            est_sydney['unq_s2']))
    assert np.isclose(0, est_sydney['shd_s1_s2'], atol=1.e-7), (
        'Shared information is not 0 for Sydney estimator ({0}).'.format(
            est_sydney['shd_s1_s2']))
    assert np.isclose(0, est_sydney['syn_s1_s2'], atol=1.e-7), (
        'Synergy is not 0 for Sydney estimator ({0}).'.format(
            est_sydney['syn_s1_s2']))
    assert np.isclose(0, est_tartu['shd_s1_s2'], atol=1.e-7), (
        'Shared information is not 0 for Tartu estimator ({0}).'.format(
            est_tartu['shd_s1_s2']))
    assert np.isclose(0, est_tartu['syn_s1_s2'], atol=1.e-7), (
        'Synergy is not 0 for Tartu estimator ({0}).'.format(
            est_tartu['syn_s1_s2']))


@float128_not_available
@optimiser_missing
def test_xor_long():
    """Test PID estimation with Sydney estimator on XOR with higher N."""
    # logical AND
    n = 1000
    alph = 2
    s1 = np.random.randint(0, alph, n)
    s2 = np.random.randint(0, alph, n)
    target = np.logical_xor(s1, s2).astype(int)
    settings = {
        'alph_s1': alph,
        'alph_s2': alph,
        'alph_t': alph,
        'max_unsuc_swaps_row_parm': 3,
        'num_reps': 63,
        'max_iters': 10000
    }
    print('\n\nTesting PID estimator on binary XOR, pointset size: {0}, '
          'iterations: {1}'.format(n, settings['max_iters']))

    # Sydney estimator
    pid_sydney = SydneyPID(settings)
    pid_tartu = TartuPID(settings)
    tic = tm.time()
    est_sydney = pid_sydney.estimate(s1, s2, target)
    t_sydney = tm.time() - tic
    # Tartu estimator
    tic = tm.time()
    est_tartu = pid_tartu.estimate(s1, s2, target)
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
    pid_sydney = SydneyPID(SETTINGS)
    pid_tartu = TartuPID(SETTINGS)
    tic = tm.time()
    est_sydney = pid_sydney.estimate(X, Y, Z)
    t_sydney = tm.time() - tic
    # Tartu estimator
    tic = tm.time()
    est_tartu = pid_tartu.estimate(X, Y, Z)
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


@float128_not_available
@optimiser_missing
def test_int_types():
    """Test PID estimator on different integer types."""
    Z = np.logical_xor(X, Y).astype(np.int32)
    print(type(Z))
    print(type(Z[0]))
    est_sydney, est_tartu = _estimate(Z)
    assert np.isclose(1, est_sydney['syn_s1_s2']), 'Synergy is not 1.'
    assert np.isclose(1, est_tartu['syn_s1_s2']), 'Synergy is not 1.'

    Z = np.logical_xor(X, Y).astype(np.int64)
    print(Z)
    print(type(Z))
    print(type(Z[0]))
    est_sydney, est_tartu = _estimate(Z)
    assert np.isclose(1, est_sydney['syn_s1_s2']), 'Synergy is not 1.'
    assert np.isclose(1, est_tartu['syn_s1_s2']), 'Synergy is not 1.'

    Z = [0, 1, 1, 0]
    print(type(Z))
    print(type(Z[0]))
    with pytest.raises(TypeError):  # Test incorrect input type
        est_sydney, est_tartu = _estimate(Z)


@float128_not_available
@optimiser_missing
def test_non_binary_alphabet():
    """Test PID estimators on larger alphabets."""
    n = 1000
    alph_s1 = 5
    alph_s2 = 3
    s1 = np.random.randint(0, alph_s1, n)
    s2 = np.random.randint(0, alph_s2, n)
    target = s1 + s2
    settings = {
        'alph_s1': alph_s1,
        'alph_s2': alph_s2,
        'alph_t': np.unique(target).shape[0],
        'max_unsuc_swaps_row_parm': 60,
        'num_reps': 63,
        'max_iters': 1000
    }
    pid_sydney = SydneyPID(settings)
    pid_tartu = TartuPID(settings)

    # Sydney estimator
    tic = tm.time()
    est_sydney = pid_sydney.estimate(s1, s2, target)
    t_sydney = tm.time() - tic
    # Tartu estimator
    tic = tm.time()
    est_tartu = pid_tartu.estimate(s1, s2, target)
    t_tartu = tm.time() - tic

    print('\nInt addition - N = 1000')
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
    assert np.isclose(est_tartu['syn_s1_s2'], est_sydney['syn_s1_s2'],
                      atol=1e-03), 'Synergies are not equal.'
    assert np.isclose(est_tartu['shd_s1_s2'], est_sydney['shd_s1_s2'],
                      atol=1e-03), 'Shareds are not equal.'
    assert np.isclose(est_tartu['unq_s1'], est_sydney['unq_s1'],
                      atol=1e-03), 'Unique1 is not equal.'
    assert np.isclose(est_tartu['unq_s2'], est_sydney['unq_s2'],
                      atol=1e-03), 'Unique2 is not equal.'


if __name__ == '__main__':
    test_non_binary_alphabet()
    test_xor_long()
    test_pid_and()
    test_pid_xor()
    test_pip_source_copy()
    test_int_types()
