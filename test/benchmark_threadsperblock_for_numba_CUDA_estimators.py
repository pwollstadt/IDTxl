"""Testing performance of the numba CUDA calculation for the setting threadsperblock
threadsperblock are depending on the GPU hardware and data size and should be tested
in advance for optimal calculation time of the numba CUDA MI and CMI estimators
choosing the appropriate signallength by adjusting values in line 25ff.

by Michael Lindner, Uni Göttingen, 2021
"""

import sys
import numpy as np
import time
from idtxl.estimators_numba import NumbaCudaKraskovMI, NumbaCudaKraskovCMI
from idtxl.idtxl_utils import calculate_mi
import random as rn
import idtxl.idtxl_exceptions as ex
from idtxl.idtxl_utils import DotDict
try:
    from numba import cuda
except ImportError as err:
    ex.package_missing(err, 'Numba is not available on this system. Install '
                            'it using pip or the package manager to use '
                            'the Numba estimators.')

# ---------- Set values for benchmarking here --------------------------
gpuid = 0  # id of the GPU you want to test on
signallength = 100000  # signallength of the data you want to use the numba CUDA estimator on
dimension = 1  # dimension of the data you want to use the numba CUDA estimator on
iterations = 5  # number of iterations for the mean calculation time
maxtpb = 512  # maximum value for TPB that should be tested (round multiples of warp size are test until this value)
# ----------------------------------------------------------------------

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


def _get_warp_size(gpuid):
    # check if cuda driver is available
    if not cuda.is_available():
        raise RuntimeError('No cuda driver available!')

    # detect if supported CUDA device are available
    std_ref = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    if not cuda.detect():
        raise RuntimeError('No cuda devices available!')
    sys.stdout = std_ref

    nr_devices = len(cuda.gpus.lst)
    if gpuid > nr_devices:
        raise RuntimeError(
            'No device with gpuid {0} (available device IDs: {1}).'.format(
                gpuid, np.arange(nr_devices)))

    # list of cuda devices
    gpus = cuda.list_devices()

    my_gpu_devices = {}
    for i in range(nr_devices):
        my_gpu_devices[i] = DotDict()
        name = gpus[i]._device.name
        my_gpu_devices[i].name = name.decode('utf-8')
        my_gpu_devices[i].global_mem_size = cuda.cudadrv.devices.get_context(gpuid).get_memory_info().total
        my_gpu_devices[i].free_mem_size = cuda.cudadrv.devices.get_context(gpuid).get_memory_info().free

    # select device
    cuda.select_device(gpuid)

    # get current device
    device = cuda.get_current_device()

    # get warp size
    warpsize = device.WARP_SIZE

    return warpsize, gpus[i]._device.name


def initialize_numba():
    """precompile numba kernels before benchmarking"""
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=100, seed=0)
    settings = {'debug': False}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    mi = numbaCuda_est.estimate(source, target)


def test_tpb_mi_uncorrelated_gaussians(tpb):
    """Test MI estimator on uncorrelated Gaussian data."""

    np.random.seed(0)
    source = np.random.randn(signallength, dimension)
    target = np.random.randn(signallength, dimension)

    # Run NumbaCuda MI estimator
    settings = {'threadsperblock': tpb}
    numbaCuda_est = NumbaCudaKraskovMI(settings=settings)
    start = time.process_time()
    mi = numbaCuda_est.estimate(source, target)
    calctime = time.process_time() - start

    return calctime


def test_tpb_cmi_uncorrelated_gaussians(tpb):
    """Test MI estimator on uncorrelated Gaussian data."""

    np.random.seed(0)
    source = np.random.randn(signallength, dimension)
    target = np.random.randn(signallength, dimension)
    cond = np.random.randn(signallength, dimension)

    # Run NumbaCuda MI estimator
    settings = {'threadsperblock': tpb}
    numbaCuda_est = NumbaCudaKraskovCMI(settings=settings)
    start = time.process_time()
    cmi = numbaCuda_est.estimate(source, target, cond)
    calctime = time.process_time() - start

    return calctime


def benchmark_mi():

    warpsize, GPU = _get_warp_size(gpuid)

    a = np.arange(warpsize, maxtpb+1, warpsize)

    print('Benchmark numba CUDA MI estimator on {0} with'
          ' signallength {1:d} and {2:d} iterations:'.format(GPU, signallength, iterations))

    tpbtime = np.empty(shape=(len(a), 1), dtype=float)
    idx=0
    for tpb in a:
        print('TPB:', tpb)
        c=np.empty(shape=(iterations, 1), dtype=float)
        for ii in range(iterations):
            c[ii] = test_tpb_mi_uncorrelated_gaussians(tpb)
        tpbtime[idx] = np.mean(c)
        print('mean calculation time: ', tpbtime[idx][0])

        idx += 1

    mincalctime = np.amin(tpbtime)
    optidx = np.where(tpbtime == mincalctime)

    opttpb = a[optidx[0]]

    print('Best TPB for MI estimation on {0} with signallength {1:d} seems to '
          'be: {2:d} with a mean calculation time of: {3:.4f}'
          's'.format(GPU, int(signallength), opttpb[0], mincalctime))


def benchmark_cmi():

    warpsize, GPU = _get_warp_size(gpuid)

    a = np.arange(warpsize, maxtpb+1, warpsize)

    print('Benchmark numba CUDA CMI estimator on {0} with'
          ' signallength {1:d} and {2:d} iterations:'.format(GPU, signallength, iterations))

    tpbtime = np.empty(shape=(len(a), 1), dtype=float)
    idx=0
    for tpb in a:
        print('TPB:', tpb)
        c=np.empty(shape=(iterations, 1), dtype=float)
        for ii in range(iterations):
            c[ii] = test_tpb_cmi_uncorrelated_gaussians(tpb)
        tpbtime[idx] = np.mean(c)
        print('mean calculation time: ', tpbtime[idx][0])

        idx += 1

    mincalctime = np.amin(tpbtime)
    optidx = np.where(tpbtime == mincalctime)

    opttpb = a[optidx[0]]

    print('Best TPB for CMI estimation on {0} with signallength {1:d} seems to '
          'be: {2:d} with a mean calculation time of: {3:.4f}'
          's'.format(GPU, int(signallength), opttpb[0], mincalctime))


if __name__ == '__main__':
    initialize_numba()
    benchmark_mi()
    benchmark_cmi()
