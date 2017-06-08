"""Test OpenCL estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against JIDT estimators.

Created on Thu Jun 1 2017

@author: patricia
"""
import math
import pytest
import numpy as np
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI
from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI
from test_estimators_jidt import _get_gauss_data

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
def test_user_input():

    est_mi = OpenCLKraskovMI()
    est_cmi = OpenCLKraskovCMI()
    N = 1000

    # Unequal variable dimensions.
    with pytest.raises(AssertionError):
        est_mi.estimate(var1=np.random.randn(N, 1),
                        var2=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_cmi.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N + 1, 1),
                         conditional=np.random.randn(N, 1))
    with pytest.raises(AssertionError):
        est_cmi.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N, 1),
                         conditional=np.random.randn(N + 1, 1))

    # No. chunks doesn't fit the signal length.
    with pytest.raises(AssertionError):
        est_mi.estimate(var1=np.random.randn(N, 1),
                        var2=np.random.randn(N, 1),
                        n_chunks=7)
    with pytest.raises(AssertionError):
        est_cmi.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N, 1),
                         conditional=np.random.randn(N, 1),
                         n_chunks=7)


@opencl_missing
@jpype_missing
def test_mi_correlated_gaussians():
    """Test estimators on correlated Gaussian data."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(source, target)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(source, target)

    cov_effective = np.cov(np.squeeze(source), np.squeeze(target))[1, 0]
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
@jpype_missing
def test_cmi_no_cond_correlated_gaussians():
    """Test estimators on correlated Gaussian data without conditional."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovCMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(source, target)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(opts={})
    mi_jidt = jidt_est.estimate(source, target)

    cov_effective = np.cov(np.squeeze(source), np.squeeze(target))[1, 0]
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
@jpype_missing
def test_cmi_correlated_gaussians():
    """Test estimators on correlated Gaussian data with conditional."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovCMI(opts=opts)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(source, target,
                                                    source_uncorr)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(opts={})
    mi_jidt = jidt_est.estimate(source, target, source_uncorr)

    cov_effective = np.cov(np.squeeze(source), np.squeeze(target))[1, 0]
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
@jpype_missing
def test_mi_correlated_gaussians_two_chunks():
    """Test estimators on two chunks of correlated Gaussian data."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=20000)
    n_points = source.shape[0]

    # Run OpenCL estimator.
    n_chunks = 2
    opts = {'debug': True}
    ocl_est = OpenCLKraskovMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(
                                                            source, target,
                                                            n_chunks=n_chunks)

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(opts={})
    mi_jidt = jidt_est.estimate(source[0:int(n_points/2), :],
                                target[0:int(n_points/2), :])

    cov_effective = np.cov(np.squeeze(source), np.squeeze(target))[1, 0]
    expected_mi = math.log(1 / (1 - math.pow(cov_effective, 2)))
    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: [{1:.4f}, {2:.4f}] '
          'nats; expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl[0], mi_ocl[1], expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[0], expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[0], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[1], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[0], mi_ocl[1], atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@jpype_missing
@opencl_missing
def test_mi_uncorrelated_gaussians():
    """Test MI estimator on uncorrelated Gaussian data."""
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


@jpype_missing
@opencl_missing
def test_cmi_uncorrelated_gaussians():
    """Test CMI estimator on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovCMI(opts=opts)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(opts={})
    mi_jidt = jidt_est.estimate(var1, var2, var3)

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


@jpype_missing
@opencl_missing
def test_mi_uncorrelated_gaussians_three_dims():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
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


@jpype_missing
@opencl_missing
def test_cmi_uncorrelated_gaussians_three_dims():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""
    n_obs = 10000
    dim = 3
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovCMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(opts={})
    mi_jidt = jidt_est.estimate(var1, var2)

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

    # Run with conditional
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]
    mi_jidt = jidt_est.estimate(var1, var2, var3)

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
def test_cmi_uncorrelated_gaussians_unequal_dims():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    # Run OpenCL estimator.
    opts = {'debug': True}
    ocl_est = OpenCLKraskovCMI(opts=opts)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(opts={})
    mi_jidt = jidt_est.estimate(var1, var2)

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

    # Run estimation with conditionals.
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]
    mi_jidt = jidt_est.estimate(var1, var2, var3)

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
def test_local_values():
    """Test estimation of local MI and CMI using OpenCL estimators."""
    # Get data
    n_chunks = 2
    expec_mi, source, source_uncorr, target = _get_gauss_data(n=20000)
    chunklength = int(source.shape[0] / n_chunks)

    # Estimate local values
    opts = {'local_values': True}
    est_cmi = OpenCLKraskovCMI(opts=opts)
    mi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(opts=opts)
    cmi = est_mi.estimate(source, target, n_chunks=n_chunks)

    mi_ch1 = np.mean(mi[0:chunklength])
    mi_ch2 = np.mean(mi[chunklength:])
    cmi_ch1 = np.mean(cmi[0:chunklength])
    cmi_ch2 = np.mean(cmi[chunklength:])

    # Estimate non-local values for comparison
    opts = {'local_values': False}
    est_cmi = OpenCLKraskovCMI(opts=opts)
    mi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(opts=opts)
    cmi = est_mi.estimate(source, target, n_chunks=n_chunks)

    # Report results
    print('OpenCL MI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2)'
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_ch1, mi_ch2, expec_mi))
    print('OpenCL CMI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2)'
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(cmi_ch1, cmi_ch2, expec_mi))

    assert np.isclose(mi_ch1, expec_mi, atol=0.05)
    assert np.isclose(mi_ch2, expec_mi, atol=0.05)
    assert np.isclose(cmi_ch1, expec_mi, atol=0.05)
    assert np.isclose(cmi_ch2, expec_mi, atol=0.05)
    assert np.isclose(mi_ch1, mi_ch2, atol=0.05)
    assert np.isclose(mi_ch1, mi[0], atol=0.05)
    assert np.isclose(mi_ch2, mi[1], atol=0.05)
    assert np.isclose(cmi_ch1, cmi_ch2, atol=0.05)
    assert np.isclose(cmi_ch1, cmi[0], atol=0.05)
    assert np.isclose(cmi_ch2, cmi[1], atol=0.05)



if __name__ == '__main__':
    test_local_values()
    test_mi_correlated_gaussians_two_chunks()
    test_cmi_uncorrelated_gaussians_unequal_dims()
    test_cmi_uncorrelated_gaussians_three_dims()
    test_cmi_uncorrelated_gaussians()
    test_cmi_no_cond_correlated_gaussians()
    test_cmi_correlated_gaussians()
    test_user_input()
    test_mi_correlated_gaussians()
    test_mi_uncorrelated_gaussians()
    test_mi_uncorrelated_gaussians_three_dims()
