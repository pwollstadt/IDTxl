"""Test CUDA estimators.

This module provides unit tests for CUDA estimators. Estimators are tested
against JIDT estimators.
"""
import logging
import pytest
import numpy as np
from idtxl.estimators_cuda import CudaKraskovMI, CudaKraskovCMI
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI
from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI
from test_estimators_jidt import _get_gauss_data, jpype_missing
from test_active_information_storage import opencl_missing
from idtxl.idtxl_utils import setup_logging, get_cuda_lib

setup_logging(logging_level=logging.DEBUG)

package_missing = False
try:
    print('getting CUDA libraries')
    get_cuda_lib()
except OSError as err:
    package_missing = True
    print("CUDA is missing, CUDA GPU estimators are not available")
cuda_missing = pytest.mark.skipif(
    package_missing,
    reason="CUDA is missing, CUDA GPU estimators are not available")


def test_debug_setting():
    """Test setting of debugging options."""
    settings = {'debug': False, 'return_counts': True}
    # Estimators should raise an error if returning of neighborhood counts is
    # requested without the debugging option being set.
    with pytest.raises(RuntimeError): CudaKraskovMI(settings=settings)
    with pytest.raises(RuntimeError): CudaKraskovCMI(settings=settings)

    settings['debug'] = True
    est = CudaKraskovMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10))
    assert len(res) == 4, (
        'Requesting debugging output from MI estimator did not return the '
        'correct no. values.')
    est = CudaKraskovCMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10), np.arange(10))
    assert len(res) == 5, (
        'Requesting debugging output from CMI estimator did not return the '
        'correct no. values.')


def test_user_input():

    est_mi = CudaKraskovMI()
    est_cmi = CudaKraskovCMI()
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

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovMI(settings=settings)
    mi_cuda, dist, n_range_var1, n_range_var2 = est_cuda.estimate(source, target)

    mi_cuda = mi_cuda[0]
    # Run JIDT estimator.
    est_jidt = JidtKraskovMI(settings={})
    mi_jidt = est_jidt.estimate(source, target)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_cuda, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_cmi_no_cond_correlated_gaussians():
    """Test estimators on correlated Gaussian data without conditional."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovCMI(settings=settings)
    mi_cuda, dist, n_range_var1, n_range_var2 = est_cuda.estimate(source, target)

    mi_cuda = mi_cuda[0]
    # Run JIDT estimator.
    est_jidt = JidtKraskovCMI(settings={})
    mi_jidt = est_jidt.estimate(source, target)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_cuda, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_cmi_correlated_gaussians():
    """Test estimators on correlated Gaussian data with conditional."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovCMI(settings=settings)
    (mi_cuda, dist, n_range_var1,
     n_range_var2, n_range_cond) = est_cuda.estimate(source, target,
                                                    source_uncorr)

    mi_cuda = mi_cuda[0]
    # Run JIDT estimator.
    est_jidt = JidtKraskovCMI(settings={})
    mi_jidt = est_jidt.estimate(source, target, source_uncorr)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_cuda, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_mi_correlated_gaussians_two_chunks():
    """Test estimators on two chunks of correlated Gaussian data."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=20000)
    n_points = source.shape[0]

    # Run CUDA estimator.
    n_chunks = 2
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovMI(settings=settings)
    mi_cuda, dist, n_range_var1, n_range_var2 = est_cuda.estimate(
                                                            source, target,
                                                            n_chunks=n_chunks)

    # Run JIDT estimator.
    est_jidt = JidtKraskovMI(settings={})
    mi_jidt = est_jidt.estimate(source[0:int(n_points/2), :],
                                target[0:int(n_points/2), :])

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: [{1:.4f}, {2:.4f}] '
          'nats; expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_cuda[0], mi_cuda[1], expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda[0], expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda[0], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda[1], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda[0], mi_cuda[1], atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_mi_uncorrelated_gaussians():
    """Test MI estimator on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovMI(settings=settings)
    mi_cuda, dist, n_range_var1, n_range_var2 = est_cuda.estimate(var1, var2)
    mi_cuda = mi_cuda[0]

    # Run JIDT estimator.
    est_jidt = JidtKraskovMI(settings={})
    mi_jidt = est_jidt.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_cuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_cmi_uncorrelated_gaussians():
    """Test CMI estimator on uncorrelated Gaussian data."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovCMI(settings=settings)
    (mi_cuda, dist, n_range_var1,
     n_range_var2, n_range_var3) = est_cuda.estimate(var1, var2, var3)
    mi_cuda = mi_cuda[0]

    # Run JIDT estimator.
    est_jidt = JidtKraskovCMI(settings={})
    mi_jidt = est_jidt.estimate(var1, var2, var3)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_cuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_mi_uncorrelated_gaussians_three_dims():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
    n_obs = 10000
    dim = 3
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovMI(settings=settings)
    mi_cuda, dist, n_range_var1, n_range_var2 = est_cuda.estimate(var1, var2)
    mi_cuda = mi_cuda[0]

    # Run JIDT estimator.
    est_jidt = JidtKraskovMI(settings={})
    mi_jidt = est_jidt.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_cuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_cmi_uncorrelated_gaussians_three_dims():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""
    n_obs = 10000
    dim = 3
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    # Run CUDA estimator.
    settings = {'debug': True, 'return_counts': True}
    est_cuda = CudaKraskovCMI(settings=settings)
    mi_cuda, dist, n_range_var1, n_range_var2 = est_cuda.estimate(var1, var2)
    mi_cuda = mi_cuda[0]

    # Run JIDT estimator.
    est_jidt = JidtKraskovCMI(settings={})
    mi_jidt = est_jidt.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_cuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')

    # Run with conditional
    (mi_cuda, dist, n_range_var1,
     n_range_var2, n_range_var3) = est_cuda.estimate(var1, var2, var3)
    mi_cuda = mi_cuda[0]
    mi_jidt = est_jidt.estimate(var1, var2, var3)

    print('JIDT MI result: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_cuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@jpype_missing
def test_cmi_uncorrelated_gaussians_unequal_dims():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""
    n_obs = 10000
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    def _print_results(mi_jidt, mi_ocl, mi_cuda):

        logging.debug('Unequal neighbor counts: {}'.format(
            (mi_ocl[2][:n_obs] != mi_cuda[2][:n_obs]).sum()))
        ind = np.where(mi_ocl[2][:n_obs] - mi_cuda[2][:n_obs])[0]
        logging.debug('counts OpenCL: {}'.format(mi_ocl[2][ind]))  # count conditional
        logging.debug('counts CUDA: {}'.format(mi_cuda[2][ind]))
        logging.debug('count diff {}:'.format(mi_ocl[2][ind] - mi_cuda[2][ind]))
        # print(mi_ocl[1][ind])  # distances
        # print(mi_cuda[1][ind])

        print('JIDT result: {0:.4f} nats; OpenCL result: {1:.4f} nats; '
              'CUDA result: {2:.4f} nats;'
              'expected to be close to 0 nats for uncorrelated '
              'Gaussians.'.format(mi_jidt, mi_ocl[0][0], mi_cuda[0][0]))
        assert np.isclose(mi_jidt, 0, atol=0.05), (
            'Estimation for uncorrelated Gaussians using the JIDT estimator '
            'failed (error larger 0.05).')
        assert np.isclose(mi_cuda[0][0], 0, atol=0.05), (
            'Estimation for uncorrelated Gaussians using the CUDA estimator '
            'failed (error larger 0.05).')
        assert np.isclose(mi_cuda[0][0], mi_jidt, atol=0.0001), (
            'Estimation for uncorrelated Gaussians using the CUDA estimator '
            'failed (error larger 0.05).')

    settings = {'debug': True, 'return_counts': True}

    est_cuda = CudaKraskovCMI(settings)
    est_ocl = OpenCLKraskovCMI(settings)
    est_jidt = JidtKraskovCMI(settings={})

    # Run estimators without conditional.
    mi_cuda = est_cuda.estimate(var1, var2)
    mi_ocl = est_ocl.estimate(var1, var2)
    mi_jidt = est_jidt.estimate(var1, var2)
    _print_results(mi_jidt, mi_ocl, mi_cuda)

    # Run estimation with conditionals.
    mi_cuda = est_cuda.estimate(var1, var2, var3)
    mi_ocl = est_ocl.estimate(var1, var2, var3)
    mi_jidt = est_jidt.estimate(var1, var2, var3)
    _print_results(mi_jidt, mi_ocl, mi_cuda)


def test_local_values():
    """Test estimation of local MI and CMI using CUDA estimators."""
    # Get data
    n_chunks = 2
    expec_mi, source, source_uncorr, target = _get_gauss_data(n=20000)
    chunklength = int(source.shape[0] / n_chunks)

    # Estimate local values
    settings = {'local_values': True}
    est_cmi = CudaKraskovCMI(settings=settings)
    cmi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = CudaKraskovMI(settings=settings)
    mi = est_mi.estimate(source, target, n_chunks=n_chunks)

    mi_ch1 = np.mean(mi[0:chunklength])
    mi_ch2 = np.mean(mi[chunklength:])
    cmi_ch1 = np.mean(cmi[0:chunklength])
    cmi_ch2 = np.mean(cmi[chunklength:])

    # Estimate non-local values for comparison
    settings = {'local_values': False}
    est_cmi = CudaKraskovCMI(settings=settings)
    mi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = CudaKraskovMI(settings=settings)
    cmi = est_mi.estimate(source, target, n_chunks=n_chunks)

    # Report results
    print('CUDA MI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_ch1, mi_ch2, expec_mi))
    print('CUDA CMI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
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
    est = CudaKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = CudaKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

    # Test a second combination with a Theiler-correction != 0
    est_mi = CudaKraskovMI(settings)
    est_mi.settings['theiler_t'] = 1
    est_mi.settings['kraskov_k'] = 2
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = CudaKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)


@jpype_missing
def test_multi_gpu():
    """Test use of multiple GPUs."""
    expected_mi, source, source_uncorr, target = _get_gauss_data()
    settings = {'debug': True, 'return_counts': True}

    # Get no. available devices on current platform.
    cmi_est = CudaKraskovCMI(settings=settings)
    n_devices = len(cmi_est.devices)

    # Try initialising estimator with unavailable GPU ID
    with pytest.raises(RuntimeError):
        settings['gpuid'] = n_devices + 1
        CudaKraskovCMI(settings=settings)

    # Run CUDA estimator on available device with highest available ID.
    settings['gpuid'] = n_devices - 1
    est_cuda = CudaKraskovCMI(settings=settings)

    (mi_cuda, dist, n_range_var1,
     n_range_var2, n_range_cond) = est_cuda.estimate(source, target,
                                                    source_uncorr)

    mi_cuda = mi_cuda[0]
    print('Expected MI: {0:.4f} nats; CUDA MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(expected_mi, mi_cuda))
    assert np.isclose(mi_cuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


@opencl_missing
def test_compare_to_cuda():
    # Compare CUDA estimates against OpenCL
    expected_mi, source, source_uncorr, target = _get_gauss_data()

    # Run OpenCL estimator.
    est_ocl = OpenCLKraskovMI(settings={})
    mi_ocl = est_ocl.estimate(source, target)[0]
    # Run CUDA estimator.
    est_cuda = CudaKraskovMI(settings={})
    mi_cuda = est_cuda.estimate(source, target)[0]

    print('CUDA MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_cuda, mi_ocl, expected_mi))
    assert np.isclose(mi_cuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_cuda, mi_ocl, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'CUDA estimator failed (error larger 0.05).')


if __name__ == '__main__':
    test_compare_to_cuda()
    test_cmi_uncorrelated_gaussians_unequal_dims()
    test_multi_gpu()
    test_debug_setting()
    test_local_values()
    test_mi_correlated_gaussians_two_chunks()
    test_cmi_uncorrelated_gaussians_three_dims()
    test_cmi_uncorrelated_gaussians()
    test_cmi_no_cond_correlated_gaussians()
    test_cmi_correlated_gaussians()
    test_user_input()
    test_mi_correlated_gaussians()
    test_mi_uncorrelated_gaussians()
    test_mi_uncorrelated_gaussians_three_dims()
