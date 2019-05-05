import logging
import random as rn
import numpy as np
from idtxl.idtxl_utils import calculate_mi
from idtxl.estimators_cuda import CudaKraskovCMI
from idtxl.estimators_opencl import OpenCLKraskovCMI


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


logging.basicConfig(
    format='%(asctime)s - %(levelname)-4s  [%(filename)s:%(funcName)20s():l %(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
logging.info('Started')

expected_mi, source, source_uncorr, target = _get_gauss_data()

settings = {'debug': True, 'return_counts': True}
est_cuda = CudaKraskovCMI(settings)
cmi_cuda, distances, count_var1, count_var2, count_cond = est_cuda.estimate(
    source, target, source)

est_ocl = OpenCLKraskovCMI(settings)
cmi_ocl, distances, count_var1, count_var2, count_cond = est_ocl.estimate(
    source, target, source)

print(cmi_cuda)
print(cmi_ocl)
print('CUDA est: {0} \n OCL est: {1}'.format(cmi_cuda, cmi_ocl))
