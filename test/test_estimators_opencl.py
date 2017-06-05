"""Test OpenCL estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against JIDT estimators.

Created on Thu Jun 1 2017

@author: patricia
"""
import math
import pytest
import random as rn
import numpy as np
from idtxl.estimators_opencl import OpenCLKraskovMI
from idtxl.estimators_jidt import JidtKraskovMI

package_missing = False
try:
    import pyopencl
except ImportError as err:
    package_missing = True
opencl_missing = pytest.mark.skipif(
    package_missing,
    reason="OpenCl is missing, GPU/OpenCl estimators are not available")
package_missing = False
try:
    import jpype
except ImportError as err:
    package_missing = True
jpype_missing = pytest.mark.skipif(
    package_missing,
    reason="Jpype is missing, JIDT estimators are not available")


@opencl_missing
def test_mi_user_input():

    est = OpenCLKraskovMI(opts={})
    N = 1000

    # Unequal variable dimensions.
    with pytest.raises(AssertionError):
        est.estimate(var1=np.random.randn(N, 1),
                     var2=np.random.randn(N + 1, 1))

    # No. chunks doesn't fit the signal length.
    with pytest.raises(AssertionError):
        est.estimate(var1=np.random.randn(N, 1),
                     var2=np.random.randn(N, 1),
                     n_chunks=7)


@opencl_missing
@jpype_missing
def test_mi_correlated_gaussians():
    """Test estimators on correlated Gaussian data."""
    n_obs = 10000
    covariance = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n_obs)]
    target = [sum(pair) for pair in zip(
                    [covariance * y for y in source[0:n_obs]],
                    [(1-covariance) * y for y in [
                        rn.normalvariate(0, 1) for r in range(n_obs)]])]

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(
                               np.expand_dims(np.array(source), axis=1),
                               np.expand_dims(np.array(target), axis=1))

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(np.expand_dims(np.array(source), axis=1),
                                np.expand_dims(np.array(target), axis=1))

    # Compute effective correlation from finite time series
    cov_effective = np.cov(source, target)[1, 0]
    expected_mi = math.log(1 / (1 - math.pow(cov_effective, 2)))
    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@opencl_missing
def test_mi_uncorrelated_gaussians():
    """Test estimators on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(var1, var2)

    # Compute effective correlation from finite time series
    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@opencl_missing
def test_mi_uncorrelated_gaussians_three_dims():
    """Test estimators on uncorrelated Gaussian data."""
    n_obs = 10000
    dim = 3
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(var1, var2)

    # Compute effective correlation from finite time series
    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@opencl_missing
def test_mi_uncorrelated_gaussians_unequal_dims():
    """Test estimators on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(var1, var2)

    # Compute effective correlation from finite time series
    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@opencl_missing
def test_mi_correlated_gaussians_unequal_dims():
    """Test estimators on correlated Gaussian data."""
    n_obs = 10000
    covariance = 0.4
    source = np.array([rn.normalvariate(0, 1) for r in range(n_obs)])
    target = np.array([sum(pair) for pair in zip(
                        [covariance * y for y in source[0:n_obs]],
                        [(1-covariance) * y for y in [
                            rn.normalvariate(0, 1) for r in range(n_obs)]])])

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(
                                                np.expand_dims(source, axis=1),
                                                np.expand_dims(target, axis=1))

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(np.expand_dims(source, axis=1),
                                np.expand_dims(target, axis=1))

    # Compute effective correlation from finite time series
    cov_effective = np.cov(source, target)[1, 0]
    expected_mi = math.log(1 / (1 - math.pow(cov_effective, 2)))
    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


if __name__ == '__main__':
    test_mi_user_input()
    test_mi_correlated_gaussians()
    test_mi_uncorrelated_gaussians()
    test_mi_uncorrelated_gaussians_three_dims()
    test_mi_uncorrelated_gaussians_unequal_dims()
