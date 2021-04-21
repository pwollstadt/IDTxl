"""Speedtest Numba estimators.
This module provides a huge dataset speed test for Numba estimators against JIDT and OpenCL estimators.
by Michael Lindner, Uni Göttingen, 2021
"""

import pytest
import numpy as np
import time
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


package_missing = False
try:
    import jpype
except ImportError as err:
    package_missing = True
jpype_missing = pytest.mark.skipif(
    package_missing,
    reason="Jpype is missing, JIDT estimators are not available")

SEED = 0
obs = 100000


def _get_gauss_data(n=10000, nrtrials=1, covariance=0.4, expand=True, seed=None):
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


def initialize_numba():
    """precompile numba kernels before speed test"""
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=100, seed=SEED)
    settings = {'debug': False}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    mi = numbaCuda_est.estimate(source, target)

@jpype_missing
def test_mi_correlated_gaussians():
    """Test MI estimator on uncorrelated Gaussian data."""

    print('test MI on correlated Gaussian')

    print('\tCreate test data')
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=obs, seed=SEED)

    # Run NumbaCuda MI estimator
    print('\tNumbaCuda')
    settings = {'debug': True, 'return_counts': True, 'threadsperblock': 160}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    start1 = time.process_time()
    (mi_numbaCuda, dist_numbaCuda,
     n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(source, target)
    print("\t\tcalculation time", time.process_time() - start1)

    # Run OpenCL MI estimator
    print('\tOpenCL')
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    start2 = time.process_time()
    mi_ocl, dist_ocl, n_range_var1_ocl, n_range_var2_ocl = ocl_est.estimate(source, target)
    print("\t\tcalculation time", time.process_time() - start2)

    # Run JIDT estimator.
    print('\tJIDT')
    jidt_est = JidtKraskovMI(settings={})
    start3 = time.process_time()
    mi_jidt = jidt_est.estimate(source, target)
    print("\t\tcalculation time", time.process_time() - start3)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'Numba CUDA MI result: {2:.4f} nats; expected to be close to {3:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl[0], mi_numbaCuda[0], expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.005).')
    print("passed")


@jpype_missing
def test_cmi_correlated_gaussians():
    """Test estimators on correlated Gaussian data with conditional."""

    print('test CMI on correlated Gaussian data with conditional')

    expected_mi, source, source_uncorr, target = _get_gauss_data(n=obs, seed=SEED)

    # Run Numba Cuda estimator.
    print('\tNumbaCuda')
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    start1 = time.process_time()
    (mi_numbaCuda, dist_numbaCuda,
    n_range_var1cond_numbaCuda, n_range_condvar2_numbaCuda, n_range_cond_numbaCuda) = numbaCuda_est.estimate(source, target,
                                                    source_uncorr)
    print("\t\tcalculation time", time.process_time() - start1)

    # Run OpenCL estimator.
    print('\tOpenCL')
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    start2 = time.process_time()
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(source, target,
                                                    source_uncorr)
    print("\t\tcalculation time", time.process_time() - start2)

    # Run JIDT estimator.
    print('\tJIDT')
    jidt_est = JidtKraskovCMI(settings={})
    start3 = time.process_time()
    mi_jidt = jidt_est.estimate(source, target, source_uncorr)
    print("\t\tcalculation time", time.process_time() - start3)

    print('\tJIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'NumbaCuda MI result: {2:.4f} nats; expected to be close to {3:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl[0], mi_numbaCuda[0], expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'numba CUDA estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.005).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'numba CUDA estimator failed (error larger 0.005).')

    print("passed")


if __name__ == '__main__':
    initialize_numba()
    test_mi_correlated_gaussians()
    test_cmi_correlated_gaussians()