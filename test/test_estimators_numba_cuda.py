"""Test Numba estimators.

This module provides unit tests for Numba estimators.
Estimators are tested against JIDT and OpenCL estimators.

by Michael Lindner, Uni Göttingen, 2021
"""

import pytest
import numpy as np
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI
from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI
from idtxl.estimators_numba import NumbaCudaKraskovMI, NumbaCudaKraskovCMI
from idtxl.idtxl_utils import calculate_mi
import random as rn

# Skip test module if opencl or numba is missing
pytest.importorskip('pyopencl')
pytest.importorskip('numba')

package_missing = False
try:
    import jpype
except ImportError as err:
    package_missing = True
jpype_missing = pytest.mark.skipif(
    package_missing,
    reason="Jpype is missing, JIDT estimators are not available")

SEED = 0


def _get_gauss_data(n=100000, nrtrials=1, covariance=0.4, expand=True, seed=None):
    """Generate correlated and uncorrelated Gaussian variables.

    Generate two sets of random normal data, where one set has a given
    covariance and the second is uncorrelated.
    """
    np.random.seed(seed)
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

    if nrtrials>1:
        var1 = np.random.rand(0, src_corr.shape[1]).astype(np.float32)
        var2 = np.random.rand(0, src_corr.shape[1]).astype(np.float32)
        var3 = np.random.rand(0, src_corr.shape[1]).astype(np.float32)

        for i in range(nrtrials):
            var1 = np.concatenate((var1, src_corr), axis=0)
            var2 = np.concatenate((var2, src_uncorr), axis=0)
            var3 = np.concatenate((var3, target), axis=0)

        src_corr = var1
        src_uncorr = var2
        target = var3

    return expected_mi, src_corr, src_uncorr, target


@jpype_missing
def test_mi_correlated_gaussians():
    """Test MI estimator on correlated Gaussian data."""

    print('test MI on correlated Gaussian data')

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run NumbaCuda MI estimator
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
     n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(source, target)
    mi_numbaCuda = mi_numbaCuda[0]

    # Run OpenCL MI estimator
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist_ocl, n_range_var1_ocl, n_range_var2_ocl = ocl_est.estimate(source, target)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(source, target)

    print('\tJIDT MI result: {0:.4f} nats\n\tOpenCL MI result: {1:.4f} nats '
          '\n\tNumbaCUDA MI result: {2:.4f} nats '
          '\n\texpected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')

    print("passed")


@jpype_missing
def test_mi_uncorrelated_gaussians():
    """Test estimators on uncorrelated Gaussian data."""

    print('test MI on uncorrelated Gaussian data')

    n_obs = 100000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
    n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(var1, var2)
    mi_numbaCuda = mi_numbaCuda[0]

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    (mi_ocl, dist_ocl, n_range_var1_ocl, n_range_var2_ocl) = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('\tJIDT MI result: {0:.4f} nats\n\tOpenCL MI result: {1:.4f} nats\n\tNumbaCUDA MI result: {2:.4f} nats '
          '\n\texpected to be close to 0 for uncorrelated Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')

    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')

    print("passed")


@jpype_missing
def test_mi_uncorrelated_gaussians_three_dims():
    """Test MI estimator on uncorrelated 3D Gaussian data."""

    print('test MI on uncorrelated Gaussian data with three dimensions')

    n_obs = 20000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
     n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(var1, var2)
    mi_numbaCuda = mi_numbaCuda[0]

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('\tJIDT MI result: {0:.4f} nats\n\tOpenCL MI result: {1:.4f} nats\n\tNumbaCUDA MI result: {2:.4f} nats; '
          '\n\texpected to be close to 0 nats for uncorrelated 3D'
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    print("passed")


@jpype_missing
def test_mi_correlated_gaussians_two_chunks():
    """Test estimators on two chunks of correlated Gaussian data."""

    print('test MI on two chunks of correlated Gaussian data')

    expected_mi, source, source_uncorr, target = _get_gauss_data(
        n=20000, seed=SEED)
    n_points = source.shape[0]

    n_chunks = 2

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
    n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(source, target,
                                                                                      n_chunks = n_chunks)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(
                                                            source, target,
                                                            n_chunks=n_chunks)
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(source[0:int(n_points/2), :],
                                target[0:int(n_points/2), :])

    print('\tJIDT MI result: {0:.4f} nats\n\tOpenCL MI result: [{1:.4f}, {2:.4f}] '
          'nats\n\tNumbaCUDA MI result: [{3:.4f}, {4:.4f}] nats'
          '\n\texpected to be close to {5:.4f} nats for correlated Gaussians '
          'with two chunks.'.format(mi_jidt, mi_ocl[0], mi_ocl[1], mi_numbaCuda[0], mi_numbaCuda[1], expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda[0], expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda[0], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda[1], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda[0], mi_numbaCuda[1], atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')

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
    print('passed')


@jpype_missing
def test_cmi_correlated_gaussians():
    """Test estimators on correlated Gaussian data with conditional."""

    print('test CMI on correlated Gaussian data with conditional')

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
    n_range_var1cond_numbaCuda, n_range_condvar2_numbaCuda, n_range_cond_numbaCuda) = numbaCuda_est.estimate(source, target,
                                                    source_uncorr)
    mi_numbaCuda = mi_numbaCuda[0]

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

    print('\tJIDT CMI result: {0:.4f} nats\n\tOpenCL CMI result: {1:.4f} nats '
          '\n\tNumbaCUDA CMI result: {2:.4f} nats\n\texpected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.005).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.005).')

    print("passed")


@jpype_missing
def test_cmi_uncorrelated_gaussians():
    """Test CMI estimator on uncorrelated Gaussian data with conditional."""

    print('test CMI on uncorrelated Gaussian data with conditional')

    n_obs = 100000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda, n_range_var1cond_numbaCuda, n_range_condvar2_numbaCuda,
     n_range_cond_numbaCuda) = numbaCuda_est.estimate(var1, var2, var3)
    mi_numbaCuda = mi_numbaCuda[0]

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
    mi_jidt = jidt_est.estimate(var1, var2, var3)

    print('\tJIDT CMI result: {0:.4f} nats\n\tOpenCL CMI result: {1:.4f} nats '
          '\n\tNumbaCUDA CMI result: {2:.4f} nats\n\texpected to be close to {3:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda, 0))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.005).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.005).')
    print("passed")


@jpype_missing
def test_cmi_uncorrelated_gaussians_three_dims():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""

    print('test CMI on uncorrelated 3D Gaussian data with and without conditional: ')

    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    print('\twithout conditional')

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
     n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(var1, var2)
    mi_numbaCuda = mi_numbaCuda[0]

    # Run OpenCL estimator.
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('\tJIDT CMI result: {0:.4f} nats\n\tOpenCL CMI result: {2:.4f} nats\n\tNumbaCUDA CMI result: {1:.4f} nats'
          '\n\texpected to be close to 0 nats for uncorrelated 3D'
          'Gaussians.'.format(mi_jidt, mi_numbaCuda, mi_ocl))
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    print("passed")

    print('\twith conditional')
    # Run with conditional
    (mi_numbaCuda, dist_numbaCuda, n_range_var1cond_numbaCuda, n_range_condvar2_numbaCuda,
     n_range_cond_numbaCuda) = numbaCuda_est.estimate(var1, var2, var3)
    mi_numbaCuda = mi_numbaCuda[0]
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]
    mi_jidt = jidt_est.estimate(var1, var2, var3)

    print('\tJIDT MI result: {0:.4f} nats\n\tOpenCL MI result: {2:.4f} nats\n\tNumbaCUDA MI result: {1:.4f} nats '
          '\n\texpected to be close to 0 nats for uncorrelated 3D'
          'Gaussians.'.format(mi_jidt, mi_numbaCuda, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    print("passed")


@jpype_missing
def test_cmi_uncorrelated_gaussians_unequal_dims():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""

    print('test CMI on uncorrelated Gaussian data (unequal dimensions) with and without conditional: ')

    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    print('\twithout conditional')

    # Run Numba Cuda estimator.
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    (mi_numbaCuda, dist_numbaCuda,
     n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(var1, var2)
    mi_numbaCuda = mi_numbaCuda[0]

    # Run OpenCL estimator.
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('\tJIDT MI result: {0:.4f} nats\n\tOpenCL MI result: {2:.4f} nats \n\tNumbaCUDA MI result: {1:.4f} nats'
          '\n\texpected to be close to 0 nats for uncorrelated '
          'Gaussians with unequal dims.'.format(mi_jidt, mi_numbaCuda, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    print('passed')

    print('\twith conditional')
    # Run with conditional
    (mi_numbaCuda, dist_numbaCuda, n_range_var1cond_numbaCuda, n_range_condvar2_numbaCuda,
     n_range_cond_numbaCuda) = numbaCuda_est.estimate(var1, var2, var3)
    mi_numbaCuda = mi_numbaCuda[0]
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]
    mi_jidt = jidt_est.estimate(var1, var2, var3)

    print('\tJIDT CMI result: {0:.4f} nats\n\tNumbaCUDA MI result: {1:.4f} nats\n\tOpenCL MI result: {2:.4f} nats '
          '\n\texpected to be close to 0 nats for uncorrelated '
          'Gaussians with unequal dims.'.format(mi_jidt, mi_numbaCuda, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCUDA estimator failed (error larger 0.005).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.0001).')
    print('passed')


@jpype_missing
def test_local_values():
    """Test estimation of local MI and CMI using NumbaCUDA estimators."""

    print('test local MI and CMI on correlated Gaussian data: ')

    # Get data
    n_chunks = 2
    expec_mi, source, source_uncorr, target = _get_gauss_data(
        n=20000, seed=SEED)
    chunklength = int(source.shape[0] / n_chunks)

    # Estimate local values
    settings = {'local_values': True}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    cmi_numbaCuda = numbaCuda_est.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    mi_numbaCuda = numbaCuda_est.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_cmi = OpenCLKraskovCMI(settings=settings)
    cmi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(settings=settings)
    mi = est_mi.estimate(source, target, n_chunks=n_chunks)

    mi_ch1 = np.mean(mi[0:chunklength])
    mi_ch2 = np.mean(mi[chunklength:])
    cmi_ch1 = np.mean(cmi[0:chunklength])
    cmi_ch2 = np.mean(cmi[chunklength:])
    mi_numbaCuda_ch1 = np.mean(mi_numbaCuda[0:chunklength])
    mi_numbaCuda_ch2 = np.mean(mi_numbaCuda[chunklength:])
    cmi_numbaCuda_ch1 = np.mean(cmi_numbaCuda[0:chunklength])
    cmi_numbaCuda_ch2 = np.mean(cmi_numbaCuda[chunklength:])

    # Estimate non-local values for comparison
    settings = {'local_values': False}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    mi_numbaCuda = numbaCuda_est.estimate(source, target, n_chunks=n_chunks)

    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    cmi_numbaCuda = numbaCuda_est.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_cmi = OpenCLKraskovCMI(settings=settings)
    cmi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(settings=settings)
    mi = est_mi.estimate(source, target, n_chunks=n_chunks)

    # Report results
    print('\tOpenCL MI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
          '\n\tNumbaCUDA MI result: {2:.4f} nats (chunk 1); {3:.4f} nats (chunk 2) '
          '\n\texpected to be close to {4:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_ch1, mi_ch2, mi_numbaCuda_ch1, mi_numbaCuda_ch2, expec_mi))
    print('\tOpenCL CMI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
          '\n\tNumbaCUDA CMI result: {2:.4f} nats (chunk 1); {3:.4f} nats (chunk 2) '
          '\n\texpected to be close to {4:.4f} nats for uncorrelated '
          'Gaussians.'.format(cmi_ch1, cmi_ch2, cmi_numbaCuda_ch1, cmi_numbaCuda_ch2, expec_mi))

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

    assert np.isclose(mi_numbaCuda_ch1, expec_mi, atol=0.05)
    assert np.isclose(mi_numbaCuda_ch2, expec_mi, atol=0.05)
    assert np.isclose(cmi_numbaCuda_ch1, expec_mi, atol=0.05)
    assert np.isclose(cmi_numbaCuda_ch2, expec_mi, atol=0.05)
    assert np.isclose(mi_numbaCuda_ch1, mi_numbaCuda_ch2, atol=0.05)
    assert np.isclose(mi_numbaCuda_ch1, mi_numbaCuda[0], atol=0.05)
    assert np.isclose(mi_numbaCuda_ch2, mi_numbaCuda[1], atol=0.05)
    assert np.isclose(cmi_numbaCuda_ch1, cmi_numbaCuda_ch2, atol=0.05)
    assert np.isclose(cmi_numbaCuda_ch1, cmi_numbaCuda[0], atol=0.05)
    assert np.isclose(cmi_numbaCuda_ch2, cmi_numbaCuda[1], atol=0.05)

    print('passed')


def test_insufficient_no_points():
    """Test if estimation aborts for too few data points."""

    print('test local MI and CMI with insufficient amount of data points ')

    expected_mi, source1, source2, target = _get_gauss_data(n=4, seed=SEED)

    settings = {
        'kraskov_k': 4,
        'theiler_t': 0,
        'history': 1,
        'history_target': 1,
        'lag_mi': 1,
        'source_target_delay': 1}

    # Test first settings combination with k==N
    est = NumbaCudaKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = NumbaCudaKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

    # Test a second combination with a Theiler-correction != 0
    settings['theiler_t'] = 1
    settings['kraskov_k'] = 2
    est = NumbaCudaKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = NumbaCudaKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

    print('passed')


def test_user_input():
    """Test numba CUDA estimator for invalid user input"""

    print('Test invalid user input')

    est_mi = NumbaCudaKraskovMI()
    est_cmi = NumbaCudaKraskovCMI()
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
    print('passed')


def test_debug_setting():
    """Test setting of debugging options."""

    print('Test debug settings')

    settings = {'debug': False, 'return_counts': True}
    # Estimators should raise an error if returning of neighborhood counts is
    # requested without the debugging option being set.
    with pytest.raises(RuntimeError): NumbaCudaKraskovMI(settings=settings)
    with pytest.raises(RuntimeError): NumbaCudaKraskovCMI(settings=settings)

    settings['debug'] = True
    est = NumbaCudaKraskovMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10))
    assert len(res) == 4, (
        'Requesting debugging output from MI estimator did not return the '
        'correct no. values.')
    est = NumbaCudaKraskovCMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10), np.arange(10))
    assert len(res) == 5, (
        'Requesting debugging output from CMI estimator did not return the '
        'correct no. values.')
    print('passed')


def test_knn_one_dim():
    """Test kNN search in 1D."""

    print("test_knn_one_dim")

    settings = {'theiler_t': 0,
                'kraskov_k': 1,
                'noise_level': 0,
                'gpu_id': 0,
                'debug': True,
                'return_counts': True,
                'verbose': True}

    numbaEST_MI = NumbaCudaKraskovMI(settings=settings)

    """Test kNN search in 1D."""
    n_chunks = 16
    pointset1 = np.expand_dims(np.array([-1, -1.2, 1, 1.1]), axis=1)
    pointset2 = np.expand_dims(np.array([99, 99, 99, 99]), axis=1)  # dummy
    pointset1 = np.tile(pointset1, (n_chunks, 1))
    pointset2 = np.tile(pointset2, (n_chunks, 1))
    # Call MI estimator
    mi, dist1, npoints_x, npoints_y = numbaEST_MI.estimate(
        pointset1, pointset2, n_chunks=n_chunks)
    assert np.isclose(dist1[0], 0.2), 'Distance 0 not correct.'
    assert np.isclose(dist1[1], 0.2), 'Distance 1 not correct.'
    assert np.isclose(dist1[2], 0.1), 'Distance 2 not correct.'
    assert np.isclose(dist1[3], 0.1), 'Distance 3 not correct.'

    print("passed")


def test_knn_two_dim():
    """Test kNN search in 2D."""

    print("test_knn_two_dim")

    settings = {'theiler_t': 0,
                'kraskov_k': 1,
                'noise_level': 0,
                'gpu_id': 0,
                'debug': True,
                'return_counts': True,
                'verbose': True}

    numbaEST_MI = NumbaCudaKraskovMI(settings=settings)

    n_chunks = 16
    pointset1 = np.array([
        [-1, -1],
        [0.5, 0.5],
        [1.1, 1.1],
        [2, 2]])
    pointset1 = np.tile(pointset1, (n_chunks, 1))
    pointset2 = np.ones(pointset1.shape) * 99

    # Call MI estimator
    mi, dist1, npoints_x, npoints_y = numbaEST_MI.estimate(
        pointset1, pointset2, n_chunks=n_chunks)
    assert np.isclose(dist1[0], 1.5), 'Distances 0 not correct.'
    assert np.isclose(dist1[1], 0.6), 'Distances 1 not correct.'
    assert np.isclose(dist1[2], 0.6), 'Distances 2 not correct.'
    assert np.isclose(dist1[3], 0.9), 'Distances 3 not correct.'

    print("passed")


if __name__ == '__main__':
    test_user_input()
    test_debug_setting()
    test_knn_one_dim()
    test_knn_two_dim()
    test_mi_uncorrelated_gaussians()
    test_mi_correlated_gaussians()
    test_mi_uncorrelated_gaussians_three_dims()
    test_mi_correlated_gaussians_two_chunks()
    test_cmi_correlated_gaussians()
    test_cmi_uncorrelated_gaussians()
    test_cmi_uncorrelated_gaussians_three_dims()
    test_cmi_uncorrelated_gaussians_unequal_dims()
    test_local_values()
    test_insufficient_no_points()
