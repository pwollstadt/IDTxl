"""Test CMI estimators.

This module provides unit tests for CMI estimators.

Created on Thu Mar 31 11:29:06 2016

@author: patricia
"""
import math
import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_cmi


def test_cmi_estimator_jidt_kraskov():
    """Test CMI estimation on two sets of Gaussian random data.

    The first test set is correlated, the second uncorrelated. This example
    is adapted from the JIDT demo 4:
    https://github.com/jlizier/jidt/raw/master/demos/python/
        example4TeContinuousDataKraskov.py

    Note:
        The actual estimates are compared to the analytical result. I set some
        boundaries for the maximum error based on 500 test runs. The test may
        still fail from time to time, check the results printed to the console
        for these cases.
    """
    # Generate two sets of random normal data, where one set has a given
    # covariance and the second is uncorrelated.
    n = 1000
    cov = 0.4
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    source_2 = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1)
    source_2 = np.expand_dims(np.array(source_2), axis=1)
    target = np.expand_dims(np.array(target), axis=1)
    # Note that the calculation is a random variable (because the generated
    # data is a set of random variables) - the result will be of the order of
    # what we expect, but not exactly equal to it; in fact, there will be a
    # large variance around it.
    opts = {'kraskov_k': 4, 'normalise': True}
    calculator_name = 'jidt_kraskov'
    est = Estimator_cmi(calculator_name)
    res_1 = est.estimate(var1=source_1[1:], var2=target[1:],
                         conditional=target[:-1], opts=opts)
    res_2 = est.estimate(var1=source_2[1:], var2=target[1:],
                         conditional=target[:-1], opts=opts)
    expected_res = math.log(1 / (1 - math.pow(cov, 2)))
    print('Example 1: TE result {0:.4f} nats; expected to be close to {1:.4f} '
          'nats for these correlated Gaussians.'.format(res_1,
                                                        expected_res))
    print('Example 2: TE result {0:.4f} nats; expected to be close to 0 nats '
          'for these uncorrelated Gaussians.'.format(res_2))
    # For 500 repetitions I got mean errors of 0.02097686 and 0.01454073 for
    # examples 1 and 2 respectively. The maximum errors were 0.093841 and
    # 0.05833172 respectively. This inspired the following error boundaries.
    assert (np.abs(res_1 - expected_res) < 0.1), ('CMI calculation for '
                                                  'correlated Gaussians failed'
                                                  '(error larger 0.1).')
    assert (np.abs(res_2) < 0.07), ('CMI calculation for uncorrelated '
                                    'Gaussians failed (error larger 0.07).')


def test_cmi_estimator_ocl():
    """Test CMI estimation on two sets of Gaussian random data.

    The first test set is correlated, the second uncorrelated. This example
    is adapted from the JIDT demo 4:
    https://github.com/jlizier/jidt/raw/master/demos/python/
        example4TeContinuousDataKraskov.py

    Note:
        The actual estimates are compared to the analytical result. I set some
        boundaries for the maximum error based on 500 test runs. The test may
        still fail from time to time, check the results printed to the console
        for these cases.
    """
    # Generate two sets of random normal data, where one set has a given
    # covariance and the second is uncorrelated.
    n = 4001 # This needs to be odd as we loose one sample when shifting signals
    cov = 0.4
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    source_2 = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1)
    source_2 = np.expand_dims(np.array(source_2), axis=1)
    target = np.expand_dims(np.array(target), axis=1)
    # Note that the calculation is a random variable (because the generated
    # data is a set of random variables) - the result will be of the order of
    # what we expect, but not exactly equal to it; in fact, there will be a
    # large variance around it.
    opts = {'kraskov_k': 4, 'normalise': True, 'nchunkspergpu': 2}
    calculator_name = 'opencl_kraskov'
    est = Estimator_cmi(calculator_name)
    res_1 = est.estimate(var1=source_1[1:], var2=target[1:],
                         conditional=target[:-1], opts=opts)
    res_2 = est.estimate(var1=source_2[1:], var2=target[1:],
                         conditional=target[:-1], opts=opts)
    expected_res = math.log(1 / (1 - math.pow(cov, 2)))
    print('Example 1: TE result for second chunk {0:.4f} nats;'
          ' expected to be close to {1:.4f} nats for these correlated'
          ' Gaussians.'.format(res_1[1], expected_res))
    print('Example 2: TE result for second chunk {0:.4f} nats; expected to be'
          ' close to 0 nats for these uncorrelated'
          ' Gaussians.'.format(res_2[1]))
    print('Example 2: TE results for first chunk is {0:.4f} nats,'
          ' and for second chunk {1:.4f} nats.'.format(res_2[0], res_2[1]))

    # For 500 repetitions I got mean errors of 0.02097686 and 0.01454073 for
    # examples 1 and 2 respectively. The maximum errors were 0.093841 and
    # 0.05833172 respectively. This inspired the following error boundaries.
    assert (np.abs(res_1[0] - expected_res) < 0.1), ('CMI calculation for '
                                                     'correlated Gaussians '
                                                     'failed(error > 0.1).')
    assert (np.abs(res_2[0]) < 0.07), ('CMI calculation for uncorrelated '
                                       'Gaussians failed (error > 0.07).')
    # TODO: error bounds here may need tightening
    assert (np.abs(res_2[0]-res_2[1]) < 0.1), ('CMI calculations for first and'
                                               ' second chunk deviate by more'
                                               'than 0.01')
    # check that the separate computation for the individual data chunks is
    # performed, instead of lumping all data together
    assert(res_1[0] != res_1[1]), ('CMI results for chunk 1 and 2 are'
                                   'identical, this  is unlikely for random'
                                   'data.')

def test_cmi_no_c_estimator_ocl():
    """Tests CMI estimation without a conditional variable

    The estimator should fall back to MI estimation and provide the correct result
    """
    n = 4000
    cov = 0.4
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1)
    target = np.expand_dims(np.array(target), axis=1)
    # Note that the calculation is a random variable (because the generated
    # data is a set of random variables) - the result will be of the order of
    # what we expect, but not exactly equal to it; in fact, there will be a
    # large variance around it.
    opts = {'kraskov_k': 4, 'normalise': True, 'nchunkspergpu': 2}
    calculator_name = 'opencl_kraskov'
    est = Estimator_cmi(calculator_name)
    res_1 = est.estimate(var1=source_1[1:], var2=target[1:],
                         conditional=None, opts=opts)
    expected_res = math.log(1 / (1 - math.pow(cov, 2)))
    print('Example 1: TE result for second chunk is {0:.4f} nats;'
          ' expected to be close to {1:.4f} nats for these correlated'
          ' Gaussians.'.format(res_1[0], expected_res))
    assert(res_1[0] != res_1[1]), ('CMI results for chunk 1 and 2 are'
                                   'identical, this  is unlikely for random'
                                   'data.')

# TODO: add assertions for the right values

if __name__ == '__main__':
    test_cmi_estimator_jidt_kraskov()
    test_cmi_estimator_ocl()
    test_cmi_no_c_estimator_ocl()
