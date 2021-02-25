"""Provide unit tests for Multivariate PID estimators."""
import time as tm
import numpy as np
import pytest
from idtxl.estimators_multivariate_pid import SxPID


SETTINGS = {}

X = np.asarray([0, 0, 1, 1])
Y = np.asarray([0, 1, 0, 1])


def test_pointwise_pid_and():
    """Test Goettingen estimator on pointwise level"""
    Z = np.logical_and(X, Y).astype(int)
    est_goettingen = _estimate(Z)
    true_values = {
        (0, 0, 0): 0.4150374992788438,
        (0, 1, 0): -0.16992500144231237,
        (1, 0, 0): -0.16992500144231237,
        (1, 1, 1): 0.4150374992788438
    }
    for i in range(4):
        assert np.isclose(true_values[(X[i], Y[i], Z[i])],
                          est_goettingen['ptw'][(X[i], Y[i], Z[i])][((1,), (2,),)][2]), (
                              'pointwise shared at ({0},{1},{2}) is not {3:.8f}'.format(
                                  X[i], Y[i], Z[i],
                                  est_goettingen['ptw'][(X[i], Y[i], Z[i])][((1,), (2,),)][2]))


def test_average_pid_and():
    """Test Goettingen estimator on logical AND."""
    Z = np.logical_and(X, Y).astype(int)
    est_goettingen = _estimate(Z)
    assert np.isclose(0.12255624891826572, est_goettingen['avg'][((1,), (2,),)][2]), (
        'Average Shared is not 0.1225...')


def test_average_pid_xor():
    """Test Goettingen estimator on logical XOR."""
    Z = np.logical_xor(X, Y).astype(int)
    est_goettingen = _estimate(Z)

    assert np.isclose(-0.5849625007211562, est_goettingen['avg'][((1,), (2,),)][2]), (
        'Average Shared is not -0.5849...')


def test_average_pid_source_copy():
    """Test Goettingen estimator on copied source."""
    Z = X
    est_goettingen = _estimate(Z)

    assert np.isclose(0.5849625007211562, est_goettingen['avg'][((1,),)][2], atol=1.e-7), (
        'Unique information 1 is not 1 for SxPID estimator ({0}).'.format(
            est_goettingen['avg'][((1,),)][2]))
    assert np.isclose(-0.4150374992788438, est_goettingen['avg'][((2,),)][2], atol=1.e-7), (
        'Unique information 2 is not 0 for SxPID estimator ({0}).'.format(
            est_goettingen['avg'][((2,),)][2]))
    assert np.isclose(0.4150374992788438, est_goettingen['avg'][((1,), (2,),)][2], atol=1.e-7), (
        'Shared information is not 0 for SxPID estimator ({0}).'.format(
            est_goettingen['avg'][((1,), (2,),)][2]))
    assert np.isclose(0.4150374992788438, est_goettingen['avg'][((1, 2,),)][2], atol=1.e-7), (
        'Synergy is not 0 for SxPID estimator ({0}).'.format(
            est_goettingen['avg'][((1, 2,),)][2]))


def _estimate(T):
    """Estimate PID for a given target."""

    # Goettingen estimator
    pid_goettingen = SxPID(SETTINGS)
    tic = tm.time()
    est_goettingen = pid_goettingen.estimate([X, Y], T)
    t_goettingen = tm.time() - tic

    print('\nCopied source')
    print('Estimator            SxPID\n')
    print('PID evaluation       {:.3f} s\n'.format(t_goettingen))
    print('Uni s1               {0:.8f}'.format(
                                         est_goettingen['avg'][((1,),)][2]))
    print('Uni s2               {0:.8f}'.format(
                                         est_goettingen['avg'][((2,),)][2]))
    print('Shared s1_s2         {0:.8f}'.format(
                                         est_goettingen['avg'][((1,),(2,),)][2]))
    print('Synergy s1_s2        {0:.8f}'.format(
                                         est_goettingen['avg'][((1,2,),)][2]))

    return est_goettingen


def test_average_pid_xor_long():
    """Test PID estimation with Goettingen estimator on XOR with higher N."""
    # logical XOR - N
    n = 1000
    alph = 2
    s1 = np.random.randint(0, alph, n)
    s2 = np.random.randint(0, alph, n)
    target = np.logical_xor(s1, s2).astype(int)
    print('\n\nTesting PID estimator on binary XOR, pointset size: {0}.'.format(n))
    # SxPID estimator
    tic = tm.time()
    pid_goettingen = SxPID(SETTINGS)
    est_goettingen = pid_goettingen.estimate([s1, s2], target)
    t_goettingen = tm.time() - tic

    print('\nLogical XOR - N = 1000')
    print('Estimator            SxPID\n')
    print('PID evaluation       {:.3f} s\n'.format(t_goettingen))
    print('Uni s1               {0:.8f}'.format(
        est_goettingen['avg'][((1,),)][2]))
    print('Uni s2               {0:.8f}'.format(
        est_goettingen['avg'][((2,),)][2]))
    print('Shared s1_s2         {0:.8f}'.format(
        est_goettingen['avg'][((1,), (2,),)][2]))
    print('Synergy s1_s2        {0:.8f}'.format(
        est_goettingen['avg'][((1, 2,),)][2]))

    # atol = 10/n since the divergence from the uniform dist of logic xor is in the order of 10/n
    assert np.isclose(-0.5849625007211562, est_goettingen['avg'][((1,), (2,),)][2], atol=10/n), (
        'Average Shared is not -0.5849...')


def test_int_types():
    """Test Goettingen estimator on different integer types."""
    Z = np.logical_xor(X, Y).astype(np.int32)
    print(type(Z))
    print(type(Z[0]))
    est_goettingen = _estimate(Z)

    assert np.isclose(-0.5849625007211562, est_goettingen['avg'][((1,),(2,),)][2]), (
        'Average Shared is not -0.5849...')

    Z = np.logical_xor(X, Y).astype(np.int64)
    print(Z)
    print(type(Z))
    print(type(Z[0]))
    est_goettingen = _estimate(Z)

    assert np.isclose(-0.5849625007211562, est_goettingen['avg'][((1,),(2,),)][2]), (
        'Average Shared is not -0.5849...')

    Z = [0, 1, 1, 0]
    print(type(Z))
    print(type(Z[0]))
    with pytest.raises(TypeError):  # Test incorrect input type
        est_goettingen = _estimate(Z)


# Not sure if it is useful for SxPID
def test_non_binary_alphabet():
    """Test Goettingen estimator on larger alphabets."""
    n = 1000
    alph_s1 = 5
    alph_s2 = 3
    s1 = np.random.randint(0, alph_s1, n)
    s2 = np.random.randint(0, alph_s2, n)
    target = s1 + s2
    pid_goettingen = SxPID(SETTINGS)

    # Goettingen SxPID estimator
    tic = tm.time()
    est_goettingen = pid_goettingen.estimate([s1, s2], target)
    t_goettingen = tm.time() - tic

    print('\nInt addition - N = 1000')
    print('Estimator            SxPID\n')
    print('PID evaluation       {:.3f} s\n'.format(t_goettingen))
    print('Uni s1               {0:.8f}'.format(
                                         est_goettingen['avg'][((1,),)][2]))
    print('Uni s2               {0:.8f}'.format(
                                         est_goettingen['avg'][((2,),)][2]))
    print('Shared s1_s2         {0:.8f}'.format(
                                         est_goettingen['avg'][((1,),(2,),)][2]))
    print('Synergy s1_s2        {0:.8f}'.format(
                                         est_goettingen['avg'][((1,2,),)][2]))

    # assert np.isclose(est_tartu['syn_s1_s2'], est_sydney['syn_s1_s2'],
    #                   atol=1e-03), 'Synergies are not equal.'
    # assert np.isclose(est_tartu['shd_s1_s2'], est_sydney['shd_s1_s2'],
    #                   atol=1e-03), 'Shareds are not equal.'
    # assert np.isclose(est_tartu['unq_s1'], est_sydney['unq_s1'],
    #                   atol=1e-03), 'Unique1 is not equal.'
    # assert np.isclose(est_tartu['unq_s2'], est_sydney['unq_s2'],
    #                   atol=1e-03), 'Unique2 is not equal.'


def test_average_pid_three_hash():
    """Test Goettingen estimator on binary three hash."""
    s1 = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    s2 = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    s3 = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    target = np.logical_xor(s3, np.logical_xor(s1, s2).astype(int)).astype(int)
    pid_goettingen = SxPID(SETTINGS)

    # Goettingen SxPID estimator
    tic = tm.time()
    est_goettingen = pid_goettingen.estimate([s1, s2, s3], target)
    t_goettingen = tm.time() - tic

    print('\n Binary three hash')
    print('Estimator                                                SxPID\n')
    print('PID evaluation (Average)                                 {:.3f} s\n'.format(
        t_goettingen))
    print('Uni s1                                                   {0:.8f}'.format(
        est_goettingen['avg'][((1,),)][2]))
    print('Uni s2                                                   {0:.8f}'.format(
        est_goettingen['avg'][((2,),)][2]))
    print('Uni s3                                                   {0:.8f}'.format(
        est_goettingen['avg'][((3,),)][2]))
    print('Synergy s1_s2_s3                                         {0:.8f}'.format(
        est_goettingen['avg'][((1, 2, 3),)][2]))
    print('Synergy s1_s2                                            {0:.8f}'.format(
        est_goettingen['avg'][((1, 2,),)][2]))
    print('Synergy s1_s3                                            {0:.8f}'.format(
        est_goettingen['avg'][((1, 3,),)][2]))
    print('Synergy s2_s3                                            {0:.8f}'.format(
        est_goettingen['avg'][((2, 3,),)][2]))
    print('Shared s1_s2_s3                                           {0:.8f}'.format(
        est_goettingen['avg'][((1,), (2,), (3,),)][2]))
    print('Shared of (s1, s2)                                       {0:.8f}'.format(
        est_goettingen['avg'][((1,), (2,),)][2]))
    print('Shared of (s1, s2)                                       {0:.8f}'.format(
        est_goettingen['avg'][((1,), (3,),)][2]))
    print('Shared of (s2, s3)                                       {0:.8f}'.format(
        est_goettingen['avg'][((2,), (3,),)][2]))
    print('Shared of (Synergy s1_s2, Synergy s1_s3)                 {0:.8f}'.format(
        est_goettingen['avg'][((1, 2,), (1, 3),)][2]))
    print('Shared of (Synergy s1_s2, Synergy s2_s3)                 {0:.8f}'.format(
        est_goettingen['avg'][((1, 2,), (2, 3),)][2]))
    print('Shared of (Synergy s1_s3, Synergy s2_s3)                 {0:.8f}'.format(
        est_goettingen['avg'][((1, 3,), (2, 3),)][2]))
    print('Shared of (Synergy s1_s2, Synergy s1_s3, Synergy s2_s3)  {0:.8f}'.format(
        est_goettingen['avg'][((1, 2,), (1, 3), (2, 3),)][2]))
    print('Shared of (s1, Synergy s2_s3)                            {0:.8f}'.format(
        est_goettingen['avg'][((1,), (2, 3),)][2]))
    print('Shared of (s2, Synergy s1_s3)                            {0:.8f}'.format(
        est_goettingen['avg'][((2,), (1, 3),)][2]))
    print('Shared of (s3, Synergy s1_s2)                            {0:.8f}'.format(
        est_goettingen['avg'][((3,), (1, 2),)][2]))

    assert np.isclose(0.1926450779423959, est_goettingen['avg'][((1,), (2,), (3,),)][2]), (
        'Average Shared is not 0.1926...')
    assert np.isclose(-0.22686079328030903, est_goettingen['avg'][((1, 2,), (1, 3), (2, 3),)][2]), (
        'Average Shared of (Synergy s1_s2, Synergy s1_s3, Synergy s2_s3) is not -0.2268...')
    assert np.isclose(0.24511249783653177, est_goettingen['avg'][((1, 2, 3),)][2]), (
        'Average Synergy s1_s2_s3 is not 0.2451...')


if __name__ == '__main__':
    test_pointwise_pid_and()
    test_average_pid_xor_long()
    test_average_pid_and()
    test_average_pid_xor()
    test_average_pid_source_copy()
    test_average_pid_three_hash()
    test_non_binary_alphabet()
    test_int_types()
