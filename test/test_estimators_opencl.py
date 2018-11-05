"""Test OpenCL estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against JIDT estimators.
"""
import math
import pytest
import numpy as np
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI
from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI
from test_estimators_jidt import _get_gauss_data

# Skip test module if opencl is missing
pytest.importorskip('pyopencl')

package_missing = False
try:
    import jpype
except ImportError as err:
    package_missing = True
jpype_missing = pytest.mark.skipif(
    package_missing,
    reason="Jpype is missing, JIDT estimators are not available")


def test_debug_setting():
    """Test setting of debugging options."""
    settings = {'debug': False, 'return_counts': True}
    # Estimators should raise an error if returning of neighborhood counts is
    # requested without the debugging option being set.
    with pytest.raises(RuntimeError): OpenCLKraskovMI(settings=settings)
    with pytest.raises(RuntimeError): OpenCLKraskovCMI(settings=settings)

    settings['debug'] = True
    est = OpenCLKraskovMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10))
    assert len(res) == 4, (
        'Requesting debugging output from MI estimator did not return the '
        'correct no. values.')
    est = OpenCLKraskovCMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10), np.arange(10))
    assert len(res) == 5, (
        'Requesting debugging output from CMI estimator did not return the '
        'correct no. values.')


def test_amd_data_padding():
    """Test padding necessary for AMD devices."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    settings = {'debug': True, 'return_counts': True}
    est_mi = OpenCLKraskovMI(settings=settings)
    est_cmi = OpenCLKraskovCMI(settings=settings)

    # Run OpenCL estimator for various data sizes.
    for n in [11, 13, 25, 64, 100, 128, 999, 10000, 3781, 50000]:
        for n_chunks in [1, 3, 10, 50, 99]:
            data_run_source = np.tile(source[:n], (n_chunks, 1))
            data_run_target = np.tile(target[:n], (n_chunks, 1))
            mi, dist, n_range_var1, n_range_var2 = est_mi.estimate(
                data_run_source, data_run_target, n_chunks=n_chunks)
            cmi, dist, n_range_var1, n_range_var2 = est_cmi.estimate(
                data_run_source, data_run_target, n_chunks=n_chunks)
    # Run OpenCL esitmator for various no. points and check result for
    # correctness. Note that for smaller sample sizes the error becomes too
    # large.
    n_chunks = 1
    for n in [832, 999, 10000, 3781, 50000]:
        data_run_source = np.tile(source[:n], (n_chunks, 1))
        data_run_target = np.tile(target[:n], (n_chunks, 1))
        mi, dist, n_range_var1, n_range_var2 = est_mi.estimate(
            data_run_source, data_run_target, n_chunks=n_chunks)
        cmi, dist, n_range_var1, n_range_var2 = est_cmi.estimate(
            data_run_source, data_run_target, n_chunks=n_chunks)
        print('{0} points, {1} chunks: OpenCL MI result: {2:.4f} nats; '
              'expected to be close to {3:.4f} nats for correlated '
              'Gaussians.'.format(n, n_chunks, mi[0], expected_mi))
        assert np.isclose(mi[0], expected_mi, atol=0.05), (
            'MI estimation for uncorrelated Gaussians using the OpenCL '
            'estimator failed (error larger 0.05).')
        print('OpenCL CMI result: {0:.4f} nats; expected to be close to '
              '{1:.4f} nats for correlated Gaussians.'.format(
                    cmi[0], expected_mi))
        assert np.isclose(cmi[0], expected_mi, atol=0.05), (
            'CMI estimation for uncorrelated Gaussians using the OpenCL '
            'estimator failed (error larger 0.05).')

    # Test debugging switched off
    settings = {'debug': False, 'return_counts': False}
    est_mi = OpenCLKraskovMI(settings=settings)
    est_cmi = OpenCLKraskovCMI(settings=settings)
    mi = est_mi.estimate(source, target)
    cmi = est_cmi.estimate(source, target)

    settings['local_values'] = True
    est_mi = OpenCLKraskovMI(settings=settings)
    est_cmi = OpenCLKraskovCMI(settings=settings)
    mi = est_mi.estimate(source, target)
    cmi = est_cmi.estimate(source, target)


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


@jpype_missing
def test_mi_correlated_gaussians():
    """Test estimators on correlated Gaussian data."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(source, target)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(source, target)

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


@jpype_missing
def test_cmi_no_cond_correlated_gaussians():
    """Test estimators on correlated Gaussian data without conditional."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(source, target)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
    mi_jidt = jidt_est.estimate(source, target)

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


@jpype_missing
def test_cmi_correlated_gaussians():
    """Test estimators on correlated Gaussian data with conditional."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(source, target,
                                                    source_uncorr)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
    mi_jidt = jidt_est.estimate(source, target, source_uncorr)

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


@jpype_missing
def test_mi_correlated_gaussians_two_chunks():
    """Test estimators on two chunks of correlated Gaussian data."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=20000)
    n_points = source.shape[0]

    # Run OpenCL estimator.
    n_chunks = 2
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(
                                                            source, target,
                                                            n_chunks=n_chunks)

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(source[0:int(n_points/2), :],
                                target[0:int(n_points/2), :])

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
def test_mi_uncorrelated_gaussians():
    """Test MI estimator on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
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
def test_cmi_uncorrelated_gaussians():
    """Test CMI estimator on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
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
def test_mi_uncorrelated_gaussians_three_dims():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
    n_obs = 10000
    dim = 3
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
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
def test_cmi_uncorrelated_gaussians_three_dims():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""
    n_obs = 10000
    dim = 3
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
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


@jpype_missing
def test_cmi_uncorrelated_gaussians_unequal_dims():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
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


def test_local_values():
    """Test estimation of local MI and CMI using OpenCL estimators."""
    # Get data
    n_chunks = 2
    expec_mi, source, source_uncorr, target = _get_gauss_data(n=20000)
    chunklength = int(source.shape[0] / n_chunks)

    # Estimate local values
    settings = {'local_values': True}
    est_cmi = OpenCLKraskovCMI(settings=settings)
    cmi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(settings=settings)
    mi = est_mi.estimate(source, target, n_chunks=n_chunks)

    mi_ch1 = np.mean(mi[0:chunklength])
    mi_ch2 = np.mean(mi[chunklength:])
    cmi_ch1 = np.mean(cmi[0:chunklength])
    cmi_ch2 = np.mean(cmi[chunklength:])

    # Estimate non-local values for comparison
    settings = {'local_values': False}
    est_cmi = OpenCLKraskovCMI(settings=settings)
    mi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(settings=settings)
    cmi = est_mi.estimate(source, target, n_chunks=n_chunks)

    # Report results
    print('OpenCL MI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_ch1, mi_ch2, expec_mi))
    print('OpenCL CMI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
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
    est = OpenCLKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = OpenCLKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

    # Test a second combination with a Theiler-correction != 0
    settings['theiler_t'] = 1
    settings['kraskov_k'] = 2
    est = OpenCLKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = OpenCLKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)


@jpype_missing
def test_multi_gpu():
    """Test use of multiple GPUs."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()
    settings = {'debug': True, 'return_counts': True}

    # Get no. available devices on current platform.
    device_list, _, _ = OpenCLKraskovCMI()._get_device(gpuid=0)
    print(device_list)
    n_devices = len(device_list)

    # Try initialising estimator with unavailable GPU ID
    with pytest.raises(RuntimeError):
        settings['gpuid'] = n_devices + 1
        OpenCLKraskovCMI(settings=settings)

    # Run OpenCL estimator on available device with highest available ID.
    settings['gpuid'] = n_devices - 1
    ocl_est = OpenCLKraskovCMI(settings=settings)

    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(source, target,
                                                    source_uncorr)

    mi_ocl = mi_ocl[0]
    print('Expected MI: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(expected_mi, mi_ocl))
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


if __name__ == '__main__':
    test_multi_gpu()
    test_debug_setting()
    test_local_values()
    test_amd_data_padding()
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
