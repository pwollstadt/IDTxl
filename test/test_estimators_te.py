"""Test TE estimators.

This module provides unit tests for TE estimators.

Created on Fri Oct 21 11:29:06 2016

@author: jlizier
"""
import pytest
import random as rn
import numpy as np
import math
from idtxl.set_estimator import Estimator_te
from test_estimators_cmi import jpype_missing


@jpype_missing
def test_jidt_kraskov_input():
    """Test handling of wrong inputs to the JIDT Kraskov TE-estimator."""
    te_est = Estimator_te('jidt_kraskov')
    source = np.empty((100))
    target = np.empty((100))

    # Wrong type for options dictinoary
    with pytest.raises(TypeError):
        te_est.estimate(source=source, target=target, opts=None)
    # Missing history for the target
    analysis_opts = {}
    with pytest.raises(RuntimeError):
        te_est.estimate(source=source, target=target, opts=analysis_opts)
    # Run analysis with all default vales
    analysis_opts = {'history_target': 3}
    te_est.estimate(source=source, target=target, opts=analysis_opts)


@jpype_missing
def test_jidt_discrete_input():
    """Test handling of wrong inputs to the JIDT discrete TE-estimator."""
    calculator_name = 'jidt_discrete'
    te_est = Estimator_te(calculator_name)
    n = 100
    source = np.zeros(n, np.int_)
    target = np.zeros(n, np.int_)

    # Wrong type for options dictinoary
    with pytest.raises(TypeError):
        te_est.estimate(source=source, target=target, opts=None)
    # Missing history for the target
    analysis_opts = {}
    with pytest.raises(RuntimeError):
        te_est.estimate(source=source, target=target, opts=analysis_opts)
    # Run analysis with all default vales
    analysis_opts = {'history_target': 3}
    te_est.estimate(source=source, target=target, opts=analysis_opts)

    # Test handling of incorrect alphabet sizes
    analysis_opts = {
        'history_target': 3,
        'alph_source': 2,
        'alph_target': 2
    }
    source = np.random.randint(5, size=(n, 4))
    target = np.random.randint(2, size=(n, 4))
    with pytest.raises(RuntimeError):
        te_est.estimate(source=source, target=target, opts=analysis_opts)
    source = np.random.randint(2, size=(n, 4))
    target = np.random.randint(5, size=(n, 4))
    with pytest.raises(RuntimeError):
        te_est.estimate(source=source, target=target, opts=analysis_opts)

    # Test multidimensional variables
    analysis_opts = {
        'history_target': 3,
        'alph_source': 2,
        'alph_target': 2
    }
    source = np.random.randint(2, size=(n, 2))
    target = np.random.randint(2, size=(n, 2))
    te_est.estimate(source=source, target=target, opts=analysis_opts)

    # Test discretisation methods
    analysis_opts = {
        'history_target': 3,
        'discretise_method': 'equal'
    }
    source = np.random.randn(n, 2)
    target = np.random.randn(n, 2)
    te_est.estimate(source=source, target=target, opts=analysis_opts)
    analysis_opts['discretise_method'] = 'max_ent'
    te_est.estimate(source=source, target=target, opts=analysis_opts)
    analysis_opts['num_discrete_bins'] = 3
    te_est.estimate(source=source, target=target, opts=analysis_opts)
    analysis_opts['discretise_method'] = 'equal'
    te_est.estimate(source=source, target=target, opts=analysis_opts)


@jpype_missing
def test_te_corr_gaussian():
    """Test TE estimation on correlated Gaussians.

    Run TE estimation on two sets of random Gaussian data with a given
    covariance. The second data set is shifted by one sample creating a source-
    target delay of one sample. This example is modeled after the JIDT demo 4
    for transfer entropy. The resulting TE can be compared to the analytical
    result (but expect some error in the estimate).

    Note:
        This test runs considerably faster than other system tests.
    """
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [0] + [sum(pair) for pair in zip(
        [cov * y for y in source[0:n - 1]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau_target': 1,
        'tau_source': 1,
        'source_target_delay': 1,
        'history_target': 1,
        'history_source': 1,
        }
    te_est = Estimator_te('jidt_kraskov')
    te_res = te_est.estimate(np.array(source), np.array(target),
                             analysis_opts)
    te_exp = np.log(1 / (1 - cov ** 2))
    print('Correlated Gaussians: TE result {0:.4f} bits; expected to be '
          '{1:0.4f} bit for the copy'
          .format(te_res, te_exp))
    np.testing.assert_approx_equal(
                            te_res, te_exp, significant=1,
                            err_msg='TE is not close to expected result.')


@jpype_missing
def test_te_estimator_jidt_discrete():
    """Test TE estimation on sets of discrete random data."""
    opts = {'num_discrete_bins': 2,
            'history_target': 1,
            'discretise_method': 'none'}
    calculator_name = 'jidt_discrete'
    est = Estimator_te(calculator_name)
    n = 1001  # Need this to be an odd number for the test to work
    source_1 = np.zeros(n, np.int_)
    target = np.zeros(n, np.int_)

    # Test 1: target is an independent copy of the source, with no k=1
    # dependence
    target[0] = 1
    for t in range(n):
        # Create source array: 0,0,1,1,0,0,1,1,...
        source_1[t] = math.floor(t / 2) % 2
        if (t > 0):
            target[t] = source_1[t-1]
    res_1 = est.estimate(source_1, target, opts)
    print('Example 1: TE result {0:.4f} bits; expected to be 1 bit for the '
          'copy'.format(res_1))
    assert (np.abs(res_1 - 1) < 0.0001), ('TE calculation for copy failed.')

    # Test 2: target is a copy of the source, but dictacted by k=2 history
    # of source
    opts['history_target'] = 2
    res_2 = est.estimate(source_1, target, opts)
    print('Example 2: TE result {0:.4f} bits; expected to be 0 bit for k={1:d}'
          ' dependent copy'.format(res_2, opts['history_target']))
    assert(np.abs(res_2 - 0) < 0.0001), ('TE calculation for k=2 dependent '
                                         'copy failed.')

    # Test 3: target is an XOR of source and target past
    opts['history_target'] = 1
    source_2 = np.array([rn.randint(0, 1) for t in range(n)])
    for t in range(1, n):
        target[t] = target[t-1] ^ source_2[t-1]
    res_3 = est.estimate(source_2, target, opts)
    print('Example 3: TE result {0:.4f} bits; expected to be 1 bit for k={1:d}'
          ' dependent XOR'.format(res_3, opts['history_target']))
    assert (np.abs(res_3 - 1) < 0.01), ('TE calculation for XOR failed.')

    # Test 4: target is an XOR of source and delayed target past
    opts['history_target'] = 1
    target[1] = 0
    for t in range(2, n):
        target[t] = target[t-2] ^ source_2[t-1]
    res_4a = est.estimate(source_2, target, opts)
    print('Example 4a: TE result {0:.4f} bits; expected to be 0 bit for '
          'delayed XOR beyond k=1'.format(res_4a))
    assert (np.abs(res_4a - 0) < 0.01), ('TE calculation for delayed XOR '
                                         'failed.')
    opts['history_target'] = 2
    res_4b = est.estimate(source_2, target, opts)
    print('Example 4b: TE result {0:.4f} bits; expected to be 1 bit for '
          'delayed XOR at k=2'.format(res_4b))
    assert (np.abs(res_4b - 1) < 0.01), ('TE calculation for delayed XOR '
                                         'failed.')

    # Test 5: target is an XOR of source and delayed target past
    opts['history_target'] = 1
    for t in range(2, n):
        target[t] = target[t-1] ^ source_2[t-2]
    res_5a = est.estimate(source_2, target, opts)
    print('Example 5a: TE result {0:.4f} bits; expected to be 0 bit for '
          'delayed XOR beyond source delay 1'.format(res_5a))
    assert (np.abs(res_5a - 0) < 0.01), ('TE calculation for delayed XOR '
                                         'failed.')
    opts['history_source'] = 2
    res_5b = est.estimate(source_2, target, opts)
    print('Example 5b: TE result {0:.4f} bits; expected to be 1 bit for '
          'delayed XOR at source history length 2'.format(res_5b))
    assert (np.abs(res_5b - 1) < 0.01), ('TE calculation for delayed XOR '
                                         'failed.')
    opts['history_source'] = 1
    opts['source_target_delay'] = 2
    res_5c = est.estimate(source_2, target, opts)
    print('Example 5c: TE result {0:.4f} bits; expected to be 1 bit for '
          'delayed XOR at source delay 2'.format(res_5c))
    assert (np.abs(res_5c - 1) < 0.01), ('TE calculation for delayed XOR '
                                         'failed.')


@jpype_missing
def test_te_local_values():
    """Test local TE estimation."""
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [0] + [sum(pair) for pair in zip(
        [cov * y for y in source[0:n - 1]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': True,
        'tau_target': 1,
        'tau_source': 1,
        'source_target_delay': 1,
        'history_target': 1,
        'history_source': 1,
        }
    te_est = Estimator_te('jidt_kraskov')
    te_res = te_est.estimate(np.array(source), np.array(target),
                             analysis_opts)
    analysis_opts['local_values'] = False
    te_avg = te_est.estimate(np.array(source), np.array(target),
                             analysis_opts)
    assert te_res.shape[0] == n, 'Local TE estimator did not return an array.'
    assert np.isclose(np.mean(te_res), te_avg, rtol=0.01), (
        'Average local TE is not equal to estimated average TE.')


@jpype_missing
def test_te_estimator_jidt_gaussian():
    """Test Gaussian TE estimation on correlated Gaussians.

    Generate two sets of random Gaussian data with a given covariance.
    The second data set is shifted by one sample creating a source-target delay
    of one sample. The resulting TE is compared to the analytical result.
    """
    # Set length of time series and correlation coefficient
    n = 1000
    cov = 0.4
    # Allowing loose tolerance, because of effective
    # non-zero correlation due to the finite length of the time series
    assert_tolerance_1 = 0.01
    # Generate random normally-distributed source time series
    source = [rn.normalvariate(0, 1) for r in range(n)]
    # Generate correlated and shifted target time series
    target = [0] + [sum(pair) for pair in zip(
        [cov * y for y in source[0:n - 1]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    # Compute effective correlation from finite time series
    corr_effective = np.corrcoef(source[:-1], target[1:])[1, 0]
    # Compute theoretical value for MI using the effective correlation
    theoretical_res = - 0.5 * np.log(1 - corr_effective ** 2)
    # Cast everything to numpy so the idtxl estimator understands it.
    source = np.array(source)
    target = np.array(target)
    # Call JIDT to perform estimation
    opts = {
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau_target': 1,
        'tau_source': 1,
        'source_target_delay': 1,
        'history_target': 1,
        'history_source': 1,
        }
    est = Estimator_te('jidt_gaussian')
    res_1 = est.estimate(source=source, target=target, opts=opts)
    print('TE result: {0:.4f} nats; expected to be '
          '{1:.4f} nats.'.format(res_1, theoretical_res))
    assert (np.abs(res_1 - theoretical_res) < assert_tolerance_1),\
        ('TE test for Gaussians estimator failed'
         '(error larger than {1:.4f}).'.format(assert_tolerance_1))


if __name__ == '__main__':
    test_jidt_discrete_input()
    test_jidt_kraskov_input()
    test_te_local_values()
    test_te_estimator_jidt_discrete()
    test_te_corr_gaussian()
    test_te_estimator_jidt_gaussian()
