"""Test Numba estimators.

This module provides unit tests for Numba estimators.
Estimators are tested against JIDT and OpenCL estimators.
"""

import math
import pytest
import numpy as np
import time
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI
from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI
from idtxl.estimators_numba import NumbaCPUKraskovMI, NumbaCudaKraskovMI
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
obs = 100000
nrtrial = 1


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



@jpype_missing
def test_mi_correlated_gaussians():
    """Test MI estimator on uncorrelated Gaussian data."""

    print("test_mi_correlated_gaussion")

    expected_mi, source, source_uncorr, target = _get_gauss_data(n=obs, nrtrials=nrtrial, seed=SEED)

    # Run NumbaCuda MI estimator
    print('\tNumbaCuda')
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    start1 = time.process_time()
    (mi_numbaCuda, dist_numbaCuda,
     n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(source, target)
    print("\t\tcalculation time", time.process_time() - start1)
    mi_numbaCuda = mi_numbaCuda[0]


    # Run OpenCL MI estimator
    print('\tOpenCL')
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    start2 = time.process_time()
    mi_ocl, dist_ocl, n_range_var1_ocl, n_range_var2_ocl = ocl_est.estimate(source, target)
    print("\t\tcalculation time", time.process_time() - start2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    print('\tJIDT')
    jidt_est = JidtKraskovMI(settings={})
    start3 = time.process_time()
    mi_jidt = jidt_est.estimate(source, target)
    print("\t\tcalculation time", time.process_time() - start3)

    print("\tResults of mi calculations")
    print("\t\t", mi_jidt, " mi JIDT")
    print("\t\t", mi_ocl, " mi ocl",)
    print("\t\t", mi_numbaCuda, " mi numbaCPU CUDA")

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'NumbaCuda MI result: {2:.4f} nats; expected to be close to {3:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.05).')

    print("passed")




@jpype_missing
def test_mi_uncorrelated_gaussians():
    """Test estimators on correlated Gaussian data with conditional."""
    print('test_mi_uncorrelated_gaussians')

    n_obs = 100000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run Numba Cuda estimator.
    print('Numba Cuda')
    settings = {'debug': True, 'return_counts': True}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    start1 = time.process_time()
    (mi_numbaCuda, dist_numbaCuda,
    n_range_var1_numbaCuda, n_range_var2_numbaCuda) = numbaCuda_est.estimate(var1, var2)
    mi_numbaCuda = mi_numbaCuda[0]
    print("\t\tcalculation time", time.process_time() - start1)

    # Run OpenCL estimator.
    print('openCL')
    settings = {'debug': True, 'return_counts': True}
    start2 = time.process_time()
    ocl_est = OpenCLKraskovMI(settings=settings)
    (mi_ocl, dist_ocl, n_range_var1_ocl, n_range_var2_ocl) = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]
    print("\t\tcalculation time", time.process_time() - start2)

    # Run JIDT estimator.
    print('jidt')
    start3 = time.process_time()
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(var1, var2)
    print("\t\tcalculation time", time.process_time() - start3)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; NumbaCuda MI result: {2:.4f} nats; '
          'expected to be close to 0 for uncorrelated Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.05).')

    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.05).')



@jpype_missing
def test_mi_uncorrelated_gaussians_three_dims():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run Numba Cuda estimator.
    print('Numba Cuda')
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

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; Numba Cuda MI result: {2:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_numbaCuda))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'NumbaCuda estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_numbaCuda, mi_jidt, atol=0.005), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


'''
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
'''

if __name__ == '__main__':
    #test_knn_one_dim()
    #test_knn_two_dim()
    test_mi_uncorrelated_gaussians()
    test_mi_correlated_gaussians()
    test_mi_uncorrelated_gaussians_three_dims()
