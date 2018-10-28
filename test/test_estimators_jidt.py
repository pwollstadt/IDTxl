"""Test JIDT estimators.

This module provides unit tests for JIDT estimators.

Unit tests are adapted from the JIDT demos:
    https://github.com/jlizier/jidt/raw/master/demos/python/
"""
import math
import pytest
import random as rn
import numpy as np
from scipy.special import digamma
from idtxl.estimators_jidt import (JidtKraskovCMI, JidtKraskovMI,
                                   JidtKraskovAIS, JidtKraskovTE,
                                   JidtDiscreteCMI, JidtDiscreteMI,
                                   JidtDiscreteAIS, JidtDiscreteTE,
                                   JidtGaussianCMI, JidtGaussianMI,
                                   JidtGaussianAIS, JidtGaussianTE)
from idtxl.idtxl_utils import calculate_mi
import idtxl.idtxl_exceptions as ex

package_missing = False
try:
    import jpype
except ImportError as err:
    package_missing = True
jpype_missing = pytest.mark.skipif(
        package_missing,
        reason="Jpype is missing, JIDT estimators are not available")


def _assert_result(results, expected_res, estimator, measure, tol=0.05):
    # Compare estimates with analytic results and print output.
    print('{0} - {1} result: {2:.4f} nats; expected to be close to {3:.4f} '
          'nats.'.format(estimator, measure, results, expected_res))
    assert np.isclose(results, expected_res, atol=tol), (
        '{0} calculation failed (error larger than {1}).'.format(measure, tol))


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
    corr_expected = covariance / (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
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
    mi_estimator = JidtKraskovMI(settings={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovMI', 'CMI (uncorr., no cond.)')

    # Test Gaussian
    mi_estimator = JidtGaussianMI(settings={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianMI', 'CMI (uncorr., no cond.)')

    # Test Discrete
    settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
    mi_estimator = JidtDiscreteMI(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteMI', 'CMI (no cond.)', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'JidtDiscreteMI', 'CMI (uncorr., no cond.)', 0.08) # More variability here


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
    mi_estimator = JidtKraskovCMI(settings={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovCMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovCMI', 'CMI (uncorr., no cond.)')

    # Test Gaussian
    mi_estimator = JidtGaussianCMI(settings={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianCMI', 'CMI (no cond.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianCMI', 'CMI (uncorr., no cond.)')

    # Test Discrete
    settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
    mi_estimator = JidtDiscreteCMI(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteCMI', 'CMI (no cond.)', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'JidtDiscreteCMI', 'CMI (uncorr., no cond.)', 0.08) # More variability here


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
    mi_estimator = JidtKraskovCMI(settings={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovCMI', 'CMI (corr.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovCMI', 'CMI (uncorr.)')

    # Test Gaussian
    mi_estimator = JidtGaussianCMI(settings={})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianCMI', 'CMI (corr.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianCMI', 'CMI (uncorr.)')

    # Test Discrete
    settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
    mi_estimator = JidtDiscreteCMI(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteCMI', 'CMI (corr.)', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'JidtDiscreteCMI', 'CMI (uncorr.)', 0.08) # More variability here


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
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1}
    # Test Kraskov
    mi_estimator = JidtKraskovTE(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtKraskovTE', 'TE (corr.)')
    _assert_result(mi_uncor, 0, 'JidtKraskovTE', 'TE (uncorr.)')

    # Test Gaussian
    mi_estimator = JidtGaussianTE(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtGaussianTE', 'TE (corr.)')
    _assert_result(mi_uncor, 0, 'JidtGaussianTE', 'TE (uncorr.)')

    # Test Discrete
    mi_estimator = JidtDiscreteTE(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'JidtDiscreteTE', 'TE (corr.)', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'JidtDiscreteTE', 'TE (uncorr.)', 0.08) # More variability here


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

    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history': 2}

    # Test Kraskov
    mi_estimator = JidtKraskovAIS(settings=settings)
    mi_cor_k = mi_estimator.estimate(source1)
    mi_uncor = mi_estimator.estimate(source2)
    _assert_result(mi_uncor, 0, 'JidtKraskovAIS', 'AIS (uncorr.)')

    # Test Gaussian
    mi_estimator = JidtGaussianAIS(settings=settings)
    mi_cor_g = mi_estimator.estimate(source1)
    mi_uncor = mi_estimator.estimate(source2)
    _assert_result(mi_uncor, 0, 'JidtGaussianAIS', 'AIS (uncorr.)')

    # TODO is this a meaningful test?
    # # Test Discrete
    # mi_estimator = JidtDiscreteAIS(settings=settings)
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
    mi_estimator = JidtKraskovMI(settings={})
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtKraskovMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtKraskovMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovMI one dim', 'JidtKraskovMI two dim', 'MI')
    # CMI
    cmi_estimator = JidtKraskovCMI(settings={})
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtKraskovCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtKraskovCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovMI one dim', 'JidtKraskovMI two dim', 'CMI')
    # TE
    te_estimator = JidtKraskovTE(settings={'history_target': 1})
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'JidtKraskovTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_two, expected_mi, 'JidtKraskovTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtKraskovMI one dim', 'JidtKraskovMI two dim', 'TE')
    # AIS
    ais_estimator = JidtKraskovAIS(settings={'history': 2})
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
    mi_estimator = JidtGaussianMI(settings={})
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtGaussianMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtGaussianMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianMI one dim', 'JidtGaussianMI two dim', 'MI')
    # CMI
    cmi_estimator = JidtGaussianCMI(settings={})
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtGaussianCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtGaussianCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianMI one dim', 'JidtGaussianMI two dim', 'CMI')
    # TE
    te_estimator = JidtGaussianTE(settings={'history_target': 1})
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'JidtGaussianTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_two, expected_mi, 'JidtGaussianTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtGaussianMI one dim', 'JidtGaussianMI two dim', 'TE')
    # AIS
    ais_estimator = JidtGaussianAIS(settings={'history': 2})
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

    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1,
                'history': 2}
    # MI
    mi_estimator = JidtDiscreteMI(settings=settings)
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtDiscreteMI', 'MI', 0.08) # More variability here
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtDiscreteMI', 'MI', 0.08) # More variability here
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteMI one dim', 'JidtDiscreteMI two dim', 'MI')
    # CMI
    cmi_estimator = JidtDiscreteCMI(settings=settings)
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'JidtDiscreteCMI', 'CMI', 0.08) # More variability here
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'JidtDiscreteCMI', 'CMI', 0.08) # More variability here
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteMI one dim', 'JidtDiscreteMI two dim', 'CMI')
    # TE
    te_estimator = JidtDiscreteTE(settings=settings)
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'JidtDiscreteTE', 'TE', 0.08) # More variability here
    mi_cor_two = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_two, expected_mi, 'JidtDiscreteTE', 'TE', 0.08) # More variability here
    _compare_result(mi_cor_one, mi_cor_two,
                    'JidtDiscreteMI one dim', 'JidtDiscreteMI two dim', 'TE')
    # AIS
    ais_estimator = JidtDiscreteAIS(settings=settings)
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

    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1,
                'history': 2,
                'local_values': True}

    # MI - Discrete
    mi_estimator = JidtDiscreteMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtDiscreteMI', 'MI', 0.08) # More variability here
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Gaussian
    mi_estimator = JidtGaussianMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtGaussianMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Kraskov
    mi_estimator = JidtKraskovMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtKraskovMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # CMI - Discrete
    cmi_estimator = JidtDiscreteCMI(settings=settings)
    mi = cmi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtDiscreteCMI', 'CMI', 0.08) # More variability here
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Gaussian
    mi_estimator = JidtGaussianCMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtGaussianCMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Kraskov
    mi_estimator = JidtKraskovCMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'JidtKraskovCMI', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Discrete
    te_estimator = JidtDiscreteTE(settings=settings)
    mi = te_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'JidtDiscreteTE', 'TE', 0.08) # More variability here
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Gaussian
    mi_estimator = JidtGaussianTE(settings=settings)
    mi = mi_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'JidtGaussianTE', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Kraskov
    mi_estimator = JidtKraskovTE(settings=settings)
    mi = mi_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'JidtKraskovTE', 'MI')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # AIS - Kraskov
    ais_estimator = JidtKraskovAIS(settings=settings)
    mi_k = ais_estimator.estimate(ar_proc)
    assert type(mi_k) is np.ndarray, 'Local values are not a numpy array.'

    # AIS - Discrete
    ais_estimator = JidtDiscreteAIS(settings=settings)
    mi_d = ais_estimator.estimate(ar_proc)
    assert type(mi_d) is np.ndarray, 'Local values are not a numpy array.'
    # TODO should we compare these?
    # _compare_result(np.mean(mi_k), np.mean(mi_d),
    #                 'JidtKraskovAIS', 'JidtDiscreteAIS', 'AIS (AR process)')

    # AIS - Gaussian
    ais_estimator = JidtGaussianAIS(settings=settings)
    mi_g = ais_estimator.estimate(ar_proc)
    assert type(mi_g) is np.ndarray, 'Local values are not a numpy array.'
    _compare_result(np.mean(mi_k), np.mean(mi_g),
                    'JidtKraskovAIS', 'JidtGaussianAIS', 'AIS (AR process)')


@jpype_missing
def test_discrete_ais():
    """Test results for discrete AIS estimation against other estimators."""

    settings = {'discretise_method': 'none',
                'alph': 2,
                'history': 2,
                'local_values': False}

    proc1, proc2 = _get_mem_binary_data()

    # Compare discrete to Gaussian estimator
    ais_estimator = JidtDiscreteAIS(settings=settings)
    mi_d = ais_estimator.estimate(proc1)

    ais_estimator = JidtGaussianAIS(settings=settings)
    mi_g = ais_estimator.estimate(proc1.astype(np.float))
    _compare_result(np.mean(mi_d), np.mean(mi_g), 'JidtDiscreteAIS',
                    'JidtGaussianAIS', 'AIS (AR process)', tol=0.07)

    # Compare discrete to Gaussian estimator on memoryless data
    ais_estimator = JidtDiscreteAIS(settings=settings)
    mi_d = ais_estimator.estimate(proc2)

    ais_estimator = JidtGaussianAIS(settings=settings)
    mi_g = ais_estimator.estimate(proc2.astype(np.float))
    _compare_result(np.mean(mi_d), np.mean(mi_g), 'JidtDiscreteAIS',
                    'JidtGaussianAIS', 'AIS (AR process, no mem.)', tol=0.05)
    _assert_result(mi_d, 0, 'JidtDiscreteAIS', 'MI (no memory)')
    _assert_result(mi_g, 0, 'JidtGaussianAIS', 'MI (no memory)')


def test_invalid_settings_input():
    """Test handling of wrong inputs for settings dictionary."""

    # Wrong input type for settings dict.
    with pytest.raises(TypeError): JidtDiscreteMI(settings=1)
    with pytest.raises(TypeError): JidtDiscreteCMI(settings=1)
    with pytest.raises(TypeError): JidtDiscreteAIS(settings=1)
    with pytest.raises(TypeError): JidtDiscreteTE(settings=1)
    with pytest.raises(TypeError): JidtGaussianMI(settings=1)
    with pytest.raises(TypeError): JidtGaussianCMI(settings=1)
    with pytest.raises(TypeError): JidtGaussianAIS(settings=1)
    with pytest.raises(TypeError): JidtGaussianTE(settings=1)
    with pytest.raises(TypeError): JidtKraskovMI(settings=1)
    with pytest.raises(TypeError): JidtKraskovCMI(settings=1)
    with pytest.raises(TypeError): JidtKraskovAIS(settings=1)
    with pytest.raises(TypeError): JidtKraskovTE(settings=1)

    # Test if settings dict is initialised correctly.
    e = JidtDiscreteMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = JidtDiscreteCMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = JidtGaussianMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = JidtGaussianCMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = JidtKraskovMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = JidtKraskovCMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'

    # History parameter missing for AIS and TE estimation.
    with pytest.raises(RuntimeError): JidtDiscreteAIS(settings={})
    with pytest.raises(RuntimeError): JidtDiscreteTE(settings={})
    with pytest.raises(RuntimeError): JidtGaussianAIS(settings={})
    with pytest.raises(RuntimeError): JidtGaussianTE(settings={})
    with pytest.raises(RuntimeError): JidtKraskovAIS(settings={})
    with pytest.raises(RuntimeError): JidtKraskovTE(settings={})


def test_invalid_history_parameters():
    """Ensure invalid history parameters raise a RuntimeError."""

    # TE: Parameters are not integers
    settings = {'history_target': 4, 'history_source': 4,
                'tau_source': 2, 'tau_target': 2.5}
    with pytest.raises(AssertionError): JidtDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): JidtGaussianTE(settings=settings)
    with pytest.raises(AssertionError): JidtKraskovTE(settings=settings)
    settings['tau_source'] = 2.5
    settings['tau_target'] = 2
    with pytest.raises(AssertionError): JidtDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): JidtGaussianTE(settings=settings)
    with pytest.raises(AssertionError): JidtKraskovTE(settings=settings)
    settings['history_source'] = 2.5
    settings['tau_source'] = 2
    with pytest.raises(AssertionError): JidtDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): JidtGaussianTE(settings=settings)
    with pytest.raises(AssertionError): JidtKraskovTE(settings=settings)
    settings['history_target'] = 2.5
    settings['history_source'] = 4
    with pytest.raises(AssertionError): JidtDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): JidtGaussianTE(settings=settings)
    with pytest.raises(AssertionError): JidtKraskovTE(settings=settings)

    # AIS: Parameters are not integers.
    settings = {'history': 4, 'tau': 2.5}
    with pytest.raises(AssertionError): JidtGaussianAIS(settings=settings)
    with pytest.raises(AssertionError): JidtKraskovAIS(settings=settings)
    settings = {'history': 4.5, 'tau': 2}
    with pytest.raises(AssertionError): JidtDiscreteAIS(settings=settings)
    with pytest.raises(AssertionError): JidtGaussianAIS(settings=settings)
    with pytest.raises(AssertionError): JidtKraskovAIS(settings=settings)


# def test_discretisation():
#     """Test discretisation for continuous data."""
#     n = 1000
#     source = np.random.randn(n)
#     target = np.random.randn(n)

#     settings = {'discretise_method': 'equal', 'n_discrete_bins': 4, 'history': 1,
#             'history_target': 1}
#     est = JidtDiscreteAIS(settings)
#     est = JidtDiscreteTE(settings)
#     est = JidtDiscreteCMI(settings)
#     est = JidtDiscreteMI(settings)
#     settings['discretise_method'] = 'max_ent'
#     est = JidtDiscreteAIS(settings)
#     est = JidtDiscreteTE(settings)
#     est = JidtDiscreteCMI(settings)
#     est = JidtDiscreteMI(settings)


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
    settings = {
        'discretise_method': 'equal',
        'n_discrete_bins': 4,
        'history': 1,
        'history_target': 1,
        'lag_mi': 1,
        'source_target_delay': 1}

    est_te_k = JidtKraskovTE(settings)
    te_k = est_te_k.estimate(source, target)
    est_te_d = JidtDiscreteTE(settings)
    te_d = est_te_d.estimate(source, target)
    est_d = JidtDiscreteMI(settings)
    mi_d = est_d.estimate(source, target)
    est_k = JidtKraskovMI(settings)
    mi_k = est_k.estimate(source, target)
    est_g = JidtGaussianMI(settings)
    mi_g = est_g.estimate(source, target)
    _compare_result(mi_d, te_d, 'JidtDiscreteMI', 'JidtDiscreteTE',
                    'lagged MI', tol=0.05)
    _compare_result(mi_k, te_k, 'JidtKraskovMI', 'JidtKraskovTE',
                    'lagged MI', tol=0.05)
    _compare_result(mi_g, te_k, 'JidtGaussianMI', 'JidtKraskovTE',
                    'lagged MI', tol=0.05)


def test_insufficient_no_points():
    """Test if estimation aborts for too few data points."""
    expected_mi, source1, source2, target = _get_gauss_data(n=4)

    settings = {
        'kraskov_k': 4,
        'theiler_t': 0,
        'history': 1,
        'history_target': 1,
        'lag_mi': 1,
        'source_target_delay': 1}

    # Test first settings combination with k==N
    est = JidtKraskovTE(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = JidtKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = JidtKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)
    est = JidtKraskovAIS(settings)
    with pytest.raises(RuntimeError): est.estimate(source1)

    # Test a second combination with a Theiler-correction != 0
    settings['theiler_t'] = 1
    settings['kraskov_k'] = 2

    est = JidtKraskovTE(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = JidtKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = JidtKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)
    est = JidtKraskovAIS(settings)
    with pytest.raises(RuntimeError): est.estimate(source1)


@jpype_missing
def test_discrete_mi_memerror():
    """Test exception handling for memory exhausted exceptions."""
    var1, var2 = _get_mem_binary_data()
    
    # Check that we catch instantiation error for an enormous history:
    caughtException = False
    settings = {'n_discrete_bins': 1000000000};
    result = -1;
    try:
        mi_estimator = JidtDiscreteMI(settings=settings)
        result = mi_estimator.estimate(var1, var2)
        print('Result of MI calc (which should not have completed) was ', result);
    except ex.JidtOutOfMemoryError:
        caughtException = True
        print('ex.JidtOutOfMemoryError caught as required')
    assert caughtException, 'No exception instantiating MI calculator with 10^18 bins'
    # Check that we instantiate correctly for a small history, even after
    #  the error encountered above:
    caughtException = False
    settings = {'n_discrete_bins': 2}
    try:
        mi_estimator = JidtDiscreteMI(settings=settings)
        mi_estimator.estimate(var1, var2)
        print('Subsequent JIDT calculation worked OK')
    except ex.JidtOutOfMemoryError:
        caughtException = True
    assert not(caughtException), 'Unable to instantiate MI calculator with 2 bins'


def test_jidt_kraskov_alg1And2():
    """ Test that JIDT estimate changes properly when we change KSG algorithm """
    n = 100;
    source = [sum(pair) for pair in zip(
                        [y for y in range(n)],
                        [rn.normalvariate(0, 0.000001) for r in range(n)])]
    source = np.array(source)
    target = np.array(source) # Target copies source on purpose
    # We've generated simple data 0:99, plus a little noise to ensure
    #  we only even get K nearest neighbours in each space.
    # So result should be:
    settings = {
        'lag': 0,
        'kraskov_k': 4,
        'noise_level': 0,
        'algorithm_num': 1}
    for k in range(4,16):
        settings['kraskov_k'] = k;
        settings['algorithm_num'] = 1;
        est1 = JidtKraskovMI(settings)
        mi_alg1 = est1.estimate(source, target)
        # Neighbour counts n_x and n_y will be k-1 because they are
        #  *strictly* within the boundary
        expected_alg1 = digamma(k) - 2*digamma((k-1)+1) + digamma(n);
        _compare_result(mi_alg1, expected_alg1, 'JidtDiscreteMI_alg1', 'Analytic',
                    'MI', tol=0.00001)
        settings['algorithm_num'] = 2;
        est2 = JidtKraskovMI(settings)
        mi_alg2 = est2.estimate(source, target)
        expected_alg2 = digamma(k) - 1/k - 2*digamma(k) + digamma(n);
        _compare_result(mi_alg2, expected_alg2, 'JidtDiscreteMI_alg2', 'Analytic',
                    'MI', tol=0.00001)
        # And now check that it doesn't work for algorithm "3"
        settings['algorithm_num'] = 3;
        caughtAssertionError = False;
        try:
            est3 = JidtKraskovMI(settings);
        except AssertionError:
            caughtAssertionError = True;
        assert caughtAssertionError, 'Assertion error not raised for KSG algorithm 3 request'


if __name__ == '__main__':
    test_insufficient_no_points()
    test_lagged_mi()
    # test_discretisation()
    test_invalid_history_parameters()
    test_invalid_settings_input()
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
    test_discrete_mi_memerror()
    test_jidt_kraskov_alg1And2()
