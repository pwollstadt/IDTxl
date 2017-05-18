"""Test MI estimators.

This module provides unit tests for MI estimators.

Created on Fri Oct 7 11:29:06 2016

@author: joseph, patricia
"""
import pytest
import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_mi
from test_estimators_cmi import jpype_missing


@jpype_missing
def test_mi_corr_gaussian():
    """Test MI estimation on correlated Gaussians.

    Run MI estimation on two sets of random Gaussian data with a given
    covariance. The second data set is shifted by one sample creating a source-
    target delay of one sample. This example is modeled after the JIDT demo 4
    for transfer entropy. The resulting MI can be compared to the analytical
    result (but expect some error in the estimate).

    Note:
        This test runs considerably faster than other system tests.
    """
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source[0:n]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n)]])]
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
    mi_est = Estimator_mi('jidt_kraskov')
    mi_res = mi_est.estimate(np.expand_dims(np.array(source), axis=1),
                             np.expand_dims(np.array(target), axis=1),
                             analysis_opts)
    mi_exp = np.log(1 / (1 - cov ** 2))
    print('Correlated Gaussians: MI result {0:.4f} bits; expected to be '
          '{1:0.4f} bit for the copy'
          .format(mi_res, mi_exp))
    np.testing.assert_approx_equal(
                            mi_res, mi_exp, significant=1,
                            err_msg='MI is not close to expected result.')


@jpype_missing
def test_lagged_mi_corr_gaussian():
    """Test MI estimation on correlated Gaussians.

    Run MI estimation on two sets of random Gaussian data with a given
    covariance. The second data set is shifted by one sample creating a source-
    target delay of one sample. This example is modeled after the JIDT demo 4
    for transfer entropy. The resulting MI can be compared to the analytical
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
        'lag': 1
        }
    mi_est = Estimator_mi('jidt_kraskov')
    mi_res = mi_est.estimate(np.expand_dims(np.array(source), axis=1),
                             np.expand_dims(np.array(target), axis=1),
                             analysis_opts)
    mi_exp = np.log(1 / (1 - cov ** 2))
    print('Correlated Gaussians: MI result {0:.4f} bits; expected to be '
          '{1:0.4f} bit for the copy'
          .format(mi_res, mi_exp))
    np.testing.assert_approx_equal(
                            mi_res, mi_exp, significant=1,
                            err_msg='MI is not close to expected result.')


@jpype_missing
def test_jidt_kraskov_input():
    """Test handling of wrong inputs to the JIDT Kraskov MI-estimator."""
    source_1 = np.empty((100, 1))
    source_2 = np.empty((100, 1))
    mi_est = Estimator_mi('jidt_kraskov')

    # Wrong type for options dictinoary
    with pytest.raises(TypeError):
        mi_est.estimate(var1=source_1, var2=source_2, opts=1)
    # Run without a options dict altogether
    mi_est.estimate(var1=source_1, var2=source_2)
    # Run analysis with all default vales
    mi_est.estimate(var1=source_1, var2=source_2, opts={})


@jpype_missing
def test_mi_estimator_jidt_discrete():
    """Test MI estimation on two sets of discrete data."""
    n = 1000
    source_1 = np.zeros(n, np.int_)
    source_2 = np.zeros(n, np.int_)
    target = np.zeros(n, np.int_)
    for t in range(n):
        source_1[t] = (t >= n/2) * 1
        source_2[t] = t % 2
        target[t] = source_1[t]
    opts = {'num_discrete_bins': 2,
            'lag': 0,
            'discretise_method': 'none'}
    calculator_name = 'jidt_discrete'
    est = Estimator_mi(calculator_name)
    res_1 = est.estimate(var1=source_1, var2=target, opts=opts)
    res_2 = est.estimate(var1=source_2, var2=target, opts=opts)
    print('Example 1: MI result {0:.4f} bits; expected to be 1 bit for the '
          'copy'.format(res_1))
    print('Example 2: MI result {0:.4f} bits; expected to be 0 bits for the '
          'uncorrelated variable.'.format(res_2))
    assert (res_1 == 1), ('MI calculation for copy failed.')
    assert (res_2 == 0), ('MI calculation for no correlation failed.')


@jpype_missing
def test_mi_local_values():
    """Test local MI estimation."""
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [0] + [sum(pair) for pair in zip(
        [cov * y for y in source[0:n - 1]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    source = np.expand_dims(np.array(source), axis=1)
    target = np.expand_dims(np.array(target), axis=1)
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': True
        }
    mi_est = Estimator_mi('jidt_kraskov')
    mi_res = mi_est.estimate(source, target, analysis_opts)
    assert mi_res.shape[0] == n, 'Local MI estimator did not return an array.'
    analysis_opts['local_values'] = False
    mi_avg = mi_est.estimate(source, target, analysis_opts)
    assert np.isclose(np.mean(mi_res), mi_avg, rtol=0.01), (
        'Average local MI is not equal to estimated average MI.')


@jpype_missing
def test_mi_estimator_jidt_discrete_discretisation():
    """Test MI estimation discretisation methods and error handling."""
    opts = {'num_discrete_bins': 2,
            'lag': 0,
            'discretise_method': 'none'}
    calculator_name = 'jidt_discrete'
    est = Estimator_mi(calculator_name)
    n = 1000  # Need this to be an even number for the test to work
    cov = 0.4

    # Test 1: input is not an integer array
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    source_2 = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1)
    source_2 = np.expand_dims(np.array(source_2), axis=1)
    target = np.expand_dims(np.array(target), axis=1)

    with pytest.raises(AssertionError):
        est.estimate(var1=source_1, var2=target, opts=opts)

    # Test 2: alphabet sizes not specified correctly
    source_1 = np.zeros(n, np.int_)
    source_2 = np.zeros(n, np.int_)
    source_3 = np.zeros(n, np.int_)
    target = np.zeros(n, np.int_)

    for t in range(n):
        source_1[t] = (t >= n/2) * 1
        source_2[t] = t % 2
        source_3[t] = (t < n/2) * source_2[t] + (t >= n/2) * (1 - source_2[t])
        target[t] = source_1[t]

    opts = {'num_discrete_bins': 0,
            'lag': 0,
            'discretise_method': 'none'}
    with pytest.raises(AssertionError):
        est.estimate(var1=source_2, var2=target, opts=opts)

# TODO: add assertions for the right values


if __name__ == '__main__':
    test_mi_corr_gaussian()
    test_lagged_mi_corr_gaussian()
    test_jidt_kraskov_input()
    test_mi_local_values()
    test_mi_estimator_jidt_discrete()
    test_mi_estimator_jidt_discrete_discretisation()
