"""Test TE estimators.

This module provides unit tests for TE estimators.

Created on Fri Oct 21 11:29:06 2016

@author: jlizier
"""
import random as rn
import numpy as np
import math
from idtxl.set_estimator import Estimator_te
from idtxl.data import Data
import idtxl.idtxl_utils


def test_multivariate_te_corr_gaussian():
    """Test multivariate TE estimation on correlated Gaussians.

    Run the multivariate TE algorithm on two sets of random Gaussian data with
    a given covariance. The second data set is shifted by one sample creating
    a source-target delay of one sample. This example is modeled after the
    JIDT demo 4 for transfer entropy. The resulting TE can be compared to the
    analytical result (but expect some error in the estimate).

    Note:
        This test runs considerably faster than other system tests.
        This produces strange small values for non-coupled sources.  TODO
    """
    n = 1000
    cov = 0.4
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # source_2 = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1).astype(float)
    # source_2 = np.expand_dims(np.array(source_2), axis=1)
    target = np.expand_dims(np.array(target), axis=1).astype(float)

    dat = Data(normalise=True)
    dat.set_data(np.vstack((source_1[1:].T, target[:-1].T)), 'ps')
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': 'false',
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau_target': 1,
        'tau_source': 1,
        'source_target_delay': 1,
        'history_target':1,
        'history_source': 1,
        }
    te_est = Estimator_te('jidt_kraskov')
    te_est.estimate(source_1, target, analysis_opts)


def test_te_estimator_jidt_discrete():
    """Test TE estimation on sets of discrete random data.
    """
    opts = {'num_discrete_bins': 2, 'history_target': 1, 'discretise_method' : 'none'}
    calculator_name = 'jidt_discrete'
    est = Estimator_te(calculator_name)
    n = 1001 # Need this to be an odd number for the test to work
    source_1 = np.zeros(n, np.int_)
    target = np.zeros(n, np.int_)

    # Test 1: target is an independent copy of the source, with no k=1 dependence
    target[0] = 1
    for t in range(n):
        # Create source array: 0,0,1,1,0,0,1,1,...
        source_1[t] = math.floor(t / 2) % 2
        if (t > 0):
            target[t] = source_1[t-1]
    res_1 = est.estimate(source_1, target, opts)
    print('Example 1: TE result {0:.4f} bits; expected to be 1 bit for the copy'.format(res_1))
    assert (np.abs(res_1 - 1) < 0.0001), ('TE calculation for copy failed.')

    # Test 2: target is a copy of the source, but dictacted by k=2 history of source
    opts['history_target'] = 2
    res_2 = est.estimate(source_1, target, opts)
    print('Example 2: TE result {0:.4f} bits; expected to be 0 bit for k={1:d} dependent copy'.format(res_2, opts['history_target']))
    assert (np.abs(res_2 - 0) < 0.0001), ('TE calculation for k=2 dependent copy failed.')

    # Test 3: target is an XOR of source and target past
    opts['history_target'] = 1
    source_2 = np.array([rn.randint(0,1) for t in range(n)])
    for t in range(1,n):
        target[t] = target[t-1] ^ source_2[t-1]
    res_3 = est.estimate(source_2, target, opts)
    print('Example 3: TE result {0:.4f} bits; expected to be 1 bit for k={1:d} dependent XOR'.format(res_3, opts['history_target']))
    assert (np.abs(res_3 - 1) < 0.01), ('TE calculation for XOR failed.')

    # Test 4: target is an XOR of source and delayed target past
    opts['history_target'] = 1
    target[1] = 0
    for t in range(2,n):
        target[t] = target[t-2] ^ source_2[t-1]
    res_4a = est.estimate(source_2, target, opts)
    print('Example 4a: TE result {0:.4f} bits; expected to be 0 bit for delayed XOR beyond k=1'.format(res_4a))
    assert (np.abs(res_4a - 0) < 0.01), ('TE calculation for delayed XOR failed.')
    opts['history_target'] = 2
    res_4b = est.estimate(source_2, target, opts)
    print('Example 4b: TE result {0:.4f} bits; expected to be 1 bit for delayed XOR at k=2'.format(res_4b))
    assert (np.abs(res_4b - 1) < 0.01), ('TE calculation for delayed XOR failed.')

    # Test 5: target is an XOR of source and delayed target past
    opts['history_target'] = 1
    for t in range(2,n):
        target[t] = target[t-1] ^ source_2[t-2]
    res_5a = est.estimate(source_2, target, opts)
    print('Example 5a: TE result {0:.4f} bits; expected to be 0 bit for delayed XOR beyond source delay 1'.format(res_5a))
    assert (np.abs(res_5a - 0) < 0.01), ('TE calculation for delayed XOR failed.')
    opts['history_source'] = 2
    res_5b = est.estimate(source_2, target, opts)
    print('Example 5b: TE result {0:.4f} bits; expected to be 1 bit for delayed XOR at source history length 2'.format(res_5b))
    assert (np.abs(res_5b - 1) < 0.01), ('TE calculation for delayed XOR failed.')
    opts['history_source'] = 1
    opts['source_target_delay'] = 2
    res_5c = est.estimate(source_2, target, opts)
    print('Example 5c: TE result {0:.4f} bits; expected to be 1 bit for delayed XOR at source delay 2'.format(res_5c))
    assert (np.abs(res_5c - 1) < 0.01), ('TE calculation for delayed XOR failed.')

if __name__ == '__main__':
    test_cmi_estimator_jidt_discrete()
    test_multivariate_te_corr_gaussian()
