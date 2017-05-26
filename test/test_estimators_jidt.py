"""Test JIDT estimators.

This module provides unit tests for JIDT estimators.

Unit tests are adapted from the JIDT demos:
    https://github.com/jlizier/jidt/raw/master/demos/python/

Created on Thu Jun 1 2017

@author: patricia
"""
import math
import pytest
import random as rn
import numpy as np
from idtxl.estimators_jidt import (JidtKraskovCMI, JidtKraskovMI,
                                   JidtKraskovAIS, JidtKraskovTE,
                                   JidtDiscreteCMI, JidtDiscreteMI,
                                   JidtDiscreteAIS, JidtDiscreteTE,
                                   JidtGaussianCMI, JidtGaussianMI,
                                   JidtGaussianAIS, JidtGaussianTE)

package_missing = False
try:
    import jpype
except ImportError as err:
    package_missing = True
jpype_missing = pytest.mark.skipif(
        package_missing,
        reason="Jpype is missing, JIDT estimators are not available")


def _assert_result(res, expected_res, estimator, measure, tol=0.05):
    # Compare estimates with analytic results and print output.
    print('{0} - {1} result: {2:.4f} nats; expected to be close to {3:.4f} '
          'nats.'.format(estimator, measure, res, expected_res))
    assert np.isclose(res, expected_res, atol=tol), (
                        '{0} calculation failed (error larger than '
                        '{1}).'.format(measure, tol))


def _compare_result(res1, res2, estimator1, estimator2, measure, tol=0.05):
    # Compare estimates with each other and print output.
    print('{0} vs. {1} - {2} result: {3:.4f} nats vs. {4:.4f} '
          'nats.'.format(estimator1, estimator2, measure, res1, res2))
    assert np.isclose(res1, res2, atol=tol), (
                        '{0} calculation failed (error larger than '
                        '{1}).'.format(measure, tol))


def _get_gauss_data(n=10000, covariance=0.4, expand=True):
    """Generate correlated and uncorrelated Gaussian variables.

    Generate two sets of random normal data, where one set has a given
    covariance and the second is uncorrelated.
    """
    expected_mi = math.log(1 / (1 - math.pow(covariance, 2)))
    src_corr = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    src_uncorr = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
                    [covariance * y for y in src_corr[0:n]],
                    [(1-covariance) * y for y in [
                        rn.normalvariate(0, 1) for r in range(n)]])]
    # Make everything numpy arrays so jpype understands it. Add an additional
    # axis if requested (MI/CMI estimators accept 2D arrays, TE/AIS only 1D).
    if expand:
        src_corr = np.expand_dims(np.array(src_corr), axis=1)
        src_uncorr = np.expand_dims(np.array(src_uncorr), axis=1)
        target = np.expand_dims(np.array(target), axis=1)
    else:
        src_corr = np.array(src_corr)
        src_uncorr = np.array(src_uncorr)
        target = np.array(target)
    return expected_mi, src_corr, src_uncorr, target


def _get_ar_data(n=10000, expand=False):
    """Simulate a process with memory using an AR process of order 2.

    Return data with memory and random data without memory.
    """
    order = 2
    source1 = np.zeros(n + order)
    source1[0:order] = np.random.normal(size=(order))
    term_1 = 0.95 * np.sqrt(2)
    for n in range(order, n + order):
        source1[n] = (term_1 * source1[n - 1] - 0.9025 * source1[n - 2] +
                      np.random.normal())
    source2 = np.random.randn(n + order)
    if expand:
        return np.expand_dims(source1, axis=1), np.expand_dims(source2, axis=1)
    else:
        return source1, source2


def _get_mem_binary_data(n=10000, expand=False):
    """Simulate simple binary process with memory.

    Return data with memory and random data without memory.
    """
    source1 = np.zeros(n + 2)
    source1[0:2] = np.random.randint(2, size=(2))
    for n in range(2, n + 2):
        source1[n] = np.logical_xor(source1[n - 1], np.random.rand() > 0.15)
    source1 = source1.astype(int)
    source2 = np.random.randint(2, size=(n + 2))
    if expand:
        return np.expand_dims(source1, axis=1), np.expand_dims(source2, axis=1)
    else:
        return source1, source2


@jpype_missing
def test_mi_gauss_data():
    """Test MI estimators on correlated Gauss data.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it; in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data()

    # Test Kraskov
    mi_estimator = JidtKraskovMI(opts={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovMI', 'CMI (uncorr., no cond.)')

    # Test Gaussian
    mi_estimator = JidtGaussianMI(opts={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianMI', 'CMI (uncorr., no cond.)')

    # Test Discrete
    opts = {'discretise_method': 'equal', 'num_discrete_bins': 5}
    mi_estimator = JidtDiscreteMI(opts=opts)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtDiscreteMI', 'CMI (uncorr., no cond.)')


@jpype_missing
def test_cmi_gauss_data_no_cond():
    """Test estimators on correlated Gauss data without a conditional.

    The estimators should return the MI if no conditional variable is
    provided.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it; in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data()

    # Test Kraskov
    mi_estimator = JidtKraskovCMI(opts={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovCMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovCMI', 'CMI (uncorr., no cond.)')

    # Test Gaussian
    mi_estimator = JidtGaussianCMI(opts={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianCMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianCMI', 'CMI (uncorr., no cond.)')

    # Test Discrete
    opts = {'discretise_method': 'equal', 'num_discrete_bins': 5}
    mi_estimator = JidtDiscreteCMI(opts=opts)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteCMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtDiscreteCMI', 'CMI (uncorr., no cond.)')


@jpype_missing
def test_cmi_gauss_data():
    """Test CMI estimation on two sets of Gaussian random data.

    The first test is on correlated variables, the second on uncorrelated
    variables.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it; in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data()

    # Test Kraskov
    mi_estimator = JidtKraskovCMI(opts={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovCMI', 'CMI (corr.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovCMI', 'CMI (uncorr.)')

    # Test Gaussian
    mi_estimator = JidtGaussianCMI(opts={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianCMI', 'CMI (corr.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianCMI', 'CMI (uncorr.)')

    # Test Discrete
    opts = {'discretise_method': 'equal', 'num_discrete_bins': 5}
    mi_estimator = JidtDiscreteCMI(opts=opts)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteCMI', 'CMI (corr.)')
    _assert_result(mi_uncor, 0, 'JidtDiscreteCMI', 'CMI (uncorr.)')


@jpype_missing
def test_te_gauss_data():
    """Test TE estimation on two sets of Gaussian random data.

    The first test is on correlated variables, the second on uncorrelated
    variables.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it; in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(expand=False)
    # add delay of one sample
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    opts = {'discretise_method': 'equal',
            'num_discrete_bins': 4,
            'history_target': 1}
    # Test Kraskov
    mi_estimator = JidtKraskovTE(opts=opts)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovTE', 'TE (corr.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovTE', 'TE (uncorr.)')

    # Test Gaussian
    mi_estimator = JidtGaussianTE(opts=opts)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianTE', 'TE (corr.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianTE', 'TE (uncorr.)')

    # Test Discrete
    mi_estimator = JidtDiscreteTE(opts=opts)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteTE', 'TE (corr.)')
    _assert_result(mi_uncor, 0, 'JidtDiscreteTE', 'TE (uncorr.)')


@jpype_missing
def test_ais_gauss_data():
    """Test AIS estimation on an autoregressive process.

    The first test is on correlated variables, the second on uncorrelated
    variables.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it; in fact, there will
    be a large variance around it.
    """
    source1, source2 = _get_ar_data()

    opts = {'discretise_method': 'equal',
            'num_discrete_bins': 4,
            'history': 2}

    # Test Kraskov
    mi_estimator = JidtKraskovAIS(opts=opts)
    mi_cor_k = mi_estimator.estimate(source1)
    mi_uncor = mi_estimator.estimate(source2)
    _assert_result(mi_uncor, 0, 'JidtKraskovAIS', 'AIS (uncorr.)')

    # Test Gaussian
    mi_estimator = JidtGaussianAIS(opts=opts)
    mi_cor_g = mi_estimator.estimate(source1)
    mi_uncor = mi_estimator.estimate(source2)
    _assert_result(mi_uncor, 0, 'JidtGaussianAIS', 'AIS (uncorr.)')

    # TODO is this a meaningful test?
    # # Test Discrete
    # mi_estimator = JidtDiscreteAIS(opts=opts)
    # mi_cor_d = mi_estimator.estimate(source1)
    # mi_uncor = mi_estimator.estimate(source2)
    # _assert_result(mi_uncor, 0, 'JidtDiscreteAIS', 'AIS (uncorr.)', tol=0.5)

    # Compare results for AR process.
    _compare_result(mi_cor_k, mi_cor_g, 'JidtKraskovAIS', 'JidtGaussianAIS',
                    'AIS (AR process)')
    # _compare_result(mi_cor_k, mi_cor_d, 'JidtKraskovAIS', 'JidtDiscreteAIS',
    #                 'AIS (AR process)')


@jpype_missing
def test_one_two_dim_input_kraskov():
    """Test one- and two-dimensional input for Kraskov estimators."""

    expected_mi, src_one, s, target_one = _get_gauss_data(expand=False)
    src_two = np.expand_dims(src_one, axis=1)
    target_two = np.expand_dims(target_one, axis=1)
    ar_src_one, s = _get_ar_data(expand=False)
    ar_src_two = np.expand_dims(ar_src_one, axis=1)

    # MI
    mi_estimator = JidtKraskovMI(opts={})
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtKraskovMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtKraskovMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovMI one dim', 'JidtKraskovMI two dim', 'MI')
    # CMI
    cmi_estimator = JidtKraskovCMI(opts={})
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtKraskovCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtKraskovCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovMI one dim', 'JidtKraskovMI two dim', 'CMI')
    # TE
    te_estimator = JidtKraskovTE(opts={'history_target': 1})
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'JidtKraskovTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_two, expected_mi, 'JidtKraskovTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovMI one dim', 'JidtKraskovMI two dim', 'TE')
    # AIS
    ais_estimator = JidtKraskovAIS(opts={'history': 2})
    mi_cor_one = ais_estimator.estimate(ar_src_one)
    mi_cor_two = ais_estimator.estimate(ar_src_two)
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovAIS one dim', 'JidtKraskovAIS two dim',
                    'AIS (AR process)')


@jpype_missing
def test_one_two_dim_input_gaussian():
    """Test one- and two-dimensional input for Gaussian estimators."""

    expected_mi, src_one, s, target_one = _get_gauss_data(expand=False)
    src_two = np.expand_dims(src_one, axis=1)
    target_two = np.expand_dims(target_one, axis=1)
    ar_src_one, s = _get_ar_data(expand=False)
    ar_src_two = np.expand_dims(ar_src_one, axis=1)

    # MI
    mi_estimator = JidtGaussianMI(opts={})
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtGaussianMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtGaussianMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianMI one dim', 'JidtGaussianMI two dim', 'MI')
    # CMI
    cmi_estimator = JidtGaussianCMI(opts={})
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtGaussianCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtGaussianCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianMI one dim', 'JidtGaussianMI two dim', 'CMI')
    # TE
    te_estimator = JidtGaussianTE(opts={'history_target': 1})
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'JidtGaussianTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_two, expected_mi, 'JidtGaussianTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianMI one dim', 'JidtGaussianMI two dim', 'TE')
    # AIS
    ais_estimator = JidtGaussianAIS(opts={'history': 2})
    mi_cor_one = ais_estimator.estimate(ar_src_one)
    mi_cor_two = ais_estimator.estimate(ar_src_two)
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianAIS one dim', 'JidtGaussianAIS two dim',
                    'AIS (AR process)')


@jpype_missing
def test_one_two_dim_input_discrete():
    """Test one- and two-dimensional input for discrete estimators."""

    expected_mi, src_one, s, target_one = _get_gauss_data(expand=False)
    src_two = np.expand_dims(src_one, axis=1)
    target_two = np.expand_dims(target_one, axis=1)
    ar_src_one, s = _get_ar_data(expand=False)
    ar_src_two = np.expand_dims(ar_src_one, axis=1)

    opts = {'discretise_method': 'equal',
            'num_discrete_bins': 4,
            'history_target': 1,
            'history': 2}
    # MI
    mi_estimator = JidtDiscreteMI(opts=opts)
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtDiscreteMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtDiscreteMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteMI one dim', 'JidtDiscreteMI two dim', 'MI')
    # CMI
    cmi_estimator = JidtDiscreteCMI(opts=opts)
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtDiscreteCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtDiscreteCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteMI one dim', 'JidtDiscreteMI two dim', 'CMI')
    # TE
    te_estimator = JidtDiscreteTE(opts=opts)
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'JidtDiscreteTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_two, expected_mi, 'JidtDiscreteTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteMI one dim', 'JidtDiscreteMI two dim', 'TE')
    # AIS
    ais_estimator = JidtDiscreteAIS(opts=opts)
    mi_cor_one = ais_estimator.estimate(ar_src_one)
    mi_cor_two = ais_estimator.estimate(ar_src_two)
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteAIS one dim', 'JidtDiscreteAIS two dim',
                    'AIS (AR process)')


@jpype_missing
def test_local_values():
    """Test estimation of local values and their return type."""
    expected_mi, source, s, target = _get_gauss_data(expand=False)
    ar_proc, s = _get_ar_data(expand=False)

    opts = {'discretise_method': 'equal',
            'num_discrete_bins': 4,
            'history_target': 1,
            'history': 2,
            'local_values': True}

    # MI - Discrete
    mi_estimator = JidtDiscreteMI(opts=opts)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtDiscreteMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Gaussian
    mi_estimator = JidtGaussianMI(opts=opts)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtGaussianMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Kraskov
    mi_estimator = JidtKraskovMI(opts=opts)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtKraskovMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # CMI - Discrete
    cmi_estimator = JidtDiscreteCMI(opts=opts)
    mi = cmi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtDiscreteCMI', 'CMI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Gaussian
    mi_estimator = JidtGaussianCMI(opts=opts)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtGaussianCMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Kraskov
    mi_estimator = JidtKraskovCMI(opts=opts)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtKraskovCMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Discrete
    te_estimator = JidtDiscreteTE(opts=opts)
    mi = te_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'JidtDiscreteTE', 'TE')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Gaussian
    mi_estimator = JidtGaussianTE(opts=opts)
    mi = mi_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'JidtGaussianTE', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Kraskov
    mi_estimator = JidtKraskovTE(opts=opts)
    mi = mi_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'JidtKraskovTE', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # AIS - Kraskov
    ais_estimator = JidtKraskovAIS(opts=opts)
    mi_k = ais_estimator.estimate(ar_proc)
    assert type(mi_k) is np.ndarray, 'Local values are not a numpy array.'

    # AIS - Discrete
    ais_estimator = JidtDiscreteAIS(opts=opts)
    mi_d = ais_estimator.estimate(ar_proc)
    assert type(mi_d) is np.ndarray, 'Local values are not a numpy array.'
    # TODO should we compare these?
    # _compare_result(np.mean(mi_k), np.mean(mi_d),
    #                 'JidtKraskovAIS', 'JidtDiscreteAIS', 'AIS (AR process)')

    # AIS - Gaussian
    ais_estimator = JidtGaussianAIS(opts=opts)
    mi_g = ais_estimator.estimate(ar_proc)
    assert type(mi_g) is np.ndarray, 'Local values are not a numpy array.'
    _compare_result(np.mean(mi_k), np.mean(mi_g),
                    'JidtKraskovAIS', 'JidtGaussianAIS', 'AIS (AR process)')


@jpype_missing
def test_discrete_ais():
    """Test results for discrete AIS estimation against other estimators."""

    opts = {'discretise_method': 'none',
            'alph': 2,
            'history': 2,
            'local_values': False}

    proc1, proc2 = _get_mem_binary_data()

    # Compare discrete to Gaussian estimator
    ais_estimator = JidtDiscreteAIS(opts=opts)
    mi_d = ais_estimator.estimate(proc1)

    ais_estimator = JidtGaussianAIS(opts=opts)
    mi_g = ais_estimator.estimate(proc1.astype(np.float))
    _compare_result(np.mean(mi_d), np.mean(mi_g), 'JidtDiscreteAIS',
                    'JidtGaussianAIS', 'AIS (AR process)', tol=0.07)

    # Compare discrete to Gaussian estimator on memoryless data
    ais_estimator = JidtDiscreteAIS(opts=opts)
    mi_d = ais_estimator.estimate(proc2)

    ais_estimator = JidtGaussianAIS(opts=opts)
    mi_g = ais_estimator.estimate(proc2.astype(np.float))
    _compare_result(np.mean(mi_d), np.mean(mi_g), 'JidtDiscreteAIS',
                    'JidtGaussianAIS', 'AIS (AR process, no mem.)', tol=0.05)
    _assert_result(mi_d, 0, 'JidtDiscreteAIS', 'MI (no memory)')
    _assert_result(mi_g, 0, 'JidtGaussianAIS', 'MI (no memory)')


def test_invalid_opts_input():
    """Test handling of wrong inputs for options dictionary."""

    # Wrong input type for opts dict.
    with pytest.raises(TypeError): JidtDiscreteMI(opts=1)
    with pytest.raises(TypeError): JidtDiscreteCMI(opts=1)
    with pytest.raises(TypeError): JidtDiscreteAIS(opts=1)
    with pytest.raises(TypeError): JidtDiscreteTE(opts=1)
    with pytest.raises(TypeError): JidtGaussianMI(opts=1)
    with pytest.raises(TypeError): JidtGaussianCMI(opts=1)
    with pytest.raises(TypeError): JidtGaussianAIS(opts=1)
    with pytest.raises(TypeError): JidtGaussianTE(opts=1)
    with pytest.raises(TypeError): JidtKraskovMI(opts=1)
    with pytest.raises(TypeError): JidtKraskovCMI(opts=1)
    with pytest.raises(TypeError): JidtKraskovAIS(opts=1)
    with pytest.raises(TypeError): JidtKraskovTE(opts=1)

    # Test if opts dict is initialised correctly.
    e = JidtDiscreteMI()
    assert type(e.opts) is dict, 'Did not initialise options as dictionary.'
    e = JidtDiscreteCMI()
    assert type(e.opts) is dict, 'Did not initialise options as dictionary.'
    e = JidtGaussianMI()
    assert type(e.opts) is dict, 'Did not initialise options as dictionary.'
    e = JidtGaussianCMI()
    assert type(e.opts) is dict, 'Did not initialise options as dictionary.'
    e = JidtKraskovMI()
    assert type(e.opts) is dict, 'Did not initialise options as dictionary.'
    e = JidtKraskovCMI()
    assert type(e.opts) is dict, 'Did not initialise options as dictionary.'

    # History parameter missing for AIS and TE estimation.
    with pytest.raises(RuntimeError): JidtDiscreteAIS(opts={})
    with pytest.raises(RuntimeError): JidtDiscreteTE(opts={})
    with pytest.raises(RuntimeError): JidtGaussianAIS(opts={})
    with pytest.raises(RuntimeError): JidtGaussianTE(opts={})
    with pytest.raises(RuntimeError): JidtKraskovAIS(opts={})
    with pytest.raises(RuntimeError): JidtKraskovTE(opts={})


def test_invalid_history_parameters():
    """Ensure invalid history parameters raise a RuntimeError."""

    # TE: Parameters are not integers
    opts = {'history_target': 4, 'history_source': 4,
            'tau_source': 2, 'tau_target': 2.5}
    with pytest.raises(AssertionError): JidtDiscreteTE(opts=opts)
    with pytest.raises(AssertionError): JidtGaussianTE(opts=opts)
    with pytest.raises(AssertionError): JidtKraskovTE(opts=opts)
    opts['tau_source'] = 2.5
    opts['tau_target'] = 2
    with pytest.raises(AssertionError): JidtDiscreteTE(opts=opts)
    with pytest.raises(AssertionError): JidtGaussianTE(opts=opts)
    with pytest.raises(AssertionError): JidtKraskovTE(opts=opts)
    opts['history_source'] = 2.5
    opts['tau_source'] = 2
    with pytest.raises(AssertionError): JidtDiscreteTE(opts=opts)
    with pytest.raises(AssertionError): JidtGaussianTE(opts=opts)
    with pytest.raises(AssertionError): JidtKraskovTE(opts=opts)
    opts['history_target'] = 2.5
    opts['history_source'] = 4
    with pytest.raises(AssertionError): JidtDiscreteTE(opts=opts)
    with pytest.raises(AssertionError): JidtGaussianTE(opts=opts)
    with pytest.raises(AssertionError): JidtKraskovTE(opts=opts)

    # AIS: Parameters are not integers.
    opts = {'history': 4, 'tau': 2.5}
    with pytest.raises(AssertionError): JidtGaussianAIS(opts=opts)
    with pytest.raises(AssertionError): JidtKraskovAIS(opts=opts)
    opts = {'history': 4.5, 'tau': 2}
    with pytest.raises(AssertionError): JidtDiscreteAIS(opts=opts)
    with pytest.raises(AssertionError): JidtGaussianAIS(opts=opts)
    with pytest.raises(AssertionError): JidtKraskovAIS(opts=opts)


# def test_discretisation():
#     """Test discretisation for continuous data."""
#     n = 1000
#     source = np.random.randn(n)
#     target = np.random.randn(n)

#     opts = {'discretise_method': 'equal', 'num_discrete_bins': 4, 'history': 1,
#             'history_target': 1}
#     est = JidtDiscreteAIS(opts)
#     est = JidtDiscreteTE(opts)
#     est = JidtDiscreteCMI(opts)
#     est = JidtDiscreteMI(opts)
#     opts['discretise_method'] = 'max_ent'
#     est = JidtDiscreteAIS(opts)
#     est = JidtDiscreteTE(opts)
#     est = JidtDiscreteCMI(opts)
#     est = JidtDiscreteMI(opts)


def test_lagged_mi():
    """Test estimation of lagged MI."""
    n = 10000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]
    target = [0] + [sum(pair) for pair in zip(
                        [cov * y for y in source[0:n - 1]],
                        [(1 - cov) * y for y in
                            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    source = np.array(source)
    target = np.array(target)
    opts = {'discretise_method': 'equal', 'num_discrete_bins': 4, 'history': 1,
            'history_target': 1, 'lag': 1, 'source_target_delay': 1}

    est_te_k = JidtKraskovTE(opts)
    te_k = est_te_k.estimate(source, target)
    est_te_d = JidtDiscreteTE(opts)
    te_d = est_te_d.estimate(source, target)
    est_d = JidtDiscreteMI(opts)
    mi_d = est_d.estimate(source, target)
    est_k = JidtKraskovMI(opts)
    mi_k = est_k.estimate(source, target)
    est_g = JidtGaussianMI(opts)
    mi_g = est_g.estimate(source, target)
    _compare_result(mi_d, te_d, 'JidtDiscreteMI', 'JidtDiscreteTE',
                    'lagged MI', tol=0.05)
    _compare_result(mi_k, te_k, 'JidtKraskovMI', 'JidtKraskovTE',
                    'lagged MI', tol=0.05)
    _compare_result(mi_g, te_k, 'JidtGaussianMI', 'JidtKraskovTE',
                    'lagged MI', tol=0.05)


if __name__ == '__main__':
    test_lagged_mi()
    # test_discretisation()
    test_invalid_history_parameters()
    test_invalid_opts_input()
    test_discrete_ais()
    test_local_values()
    test_te_gauss_data()
    test_one_two_dim_input_kraskov()
    test_one_two_dim_input_gaussian()
    test_one_two_dim_input_discrete()
    test_ais_gauss_data()
    test_te_gauss_data()
    test_cmi_gauss_data()
    test_cmi_gauss_data_no_cond()
    test_mi_gauss_data()