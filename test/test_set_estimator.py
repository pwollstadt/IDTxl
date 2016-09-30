"""Test the setting of estimators.

This module provides unit tests for estimator setting. The user can choose from
a variety of estimators (depending on data type and measure to be estimated).
This functionality is handled by the estimator class and tested here.

Created on Fri Sep 30 14:39:06 2016

@author: patricia
"""
import math
import random
import numpy as np
from idtxl.set_estimator import Estimator_te
from idtxl.set_estimator import Estimator_cmi
from idtxl.set_estimator import Estimator_mi


def test_estimators_correlated_gauss_data():
    """Test estimators on correlated Gauss data."""
    estimator_name = 'jidt_kraskov'
    te_estimator = Estimator_te(estimator_name)

    n_obs = 10000
    covariance = 0.4
    source = [random.normalvariate(0, 1) for r in range(n_obs)]
    target = [0] + [sum(pair) for pair in zip(
                    [covariance * y for y in source[0:n_obs-1]],
                    [(1-covariance) * y for y in [
                        random.normalvariate(0, 1) for r in range(n_obs-1)]])]
    options = {
        'kraskov_k': 4,
        'history_target': 3
        }

    te = te_estimator.estimate(np.array(source), np.array(target), options)
    expected_te = math.log(1 / (1 - math.pow(covariance, 2)))
    print('TE estimator is {0}'.format(te_estimator.estimator_name))
    print('TE result: {0:.4f} nats; expected to be close to {1:.4f} nats for '
          'correlated Gaussians.'.format(te, expected_te))
    # For 500 repetitions I got a mean error of 0.02097686 for this example.
    # The maximum error was 0.093841. This inspired the following error
    # boundary.
    assert (np.abs(te - expected_te) < 0.1), ('CMI calculation for '
                                              'correlated Gaussians failed'
                                              '(error larger 0.1).')
    assert estimator_name == te_estimator.estimator_name, (
                'The estimator was not set correctly')


def test_estimators_uncorrelated_random_data():
    """Test estimators on uncorrelated Gauss data."""
    estimator_name = 'jidt_kraskov'
    cmi_estimator = Estimator_cmi(estimator_name)

    dim = 5
    n_obs = 10000
    var1 = [[random.normalvariate(0, 1) for x in range(dim)] for
            x in range(n_obs)]
    var2 = [[random.normalvariate(0, 1) for x in range(dim)] for
            x in range(n_obs)]
    conditional = [[random.normalvariate(0, 1) for x in range(dim)] for
                   x in range(n_obs)]
    opts = {'kraskov_k': 4}
    cmi = cmi_estimator.estimate(np.array(var1), np.array(var2),
                                 np.array(conditional), opts=opts)

    # For 500 repetitions I got a mean error of 0.01454073 for this example.
    # The maximum error was 0.05833172. This inspired the following error
    # boundary.
    assert (np.abs(cmi) < 0.07), ('CMI calculation for uncorrelated '
                                  'Gaussians failed (error larger 0.07).')
    assert estimator_name == cmi_estimator.estimator_name, (
                'The estimator was not set correctly')


def test_estimator_change():
    """Test dynamic changing of estimators at runtime."""
    estimator_name_1 = 'jidt_kraskov'
    estimator_name_2 = 'opencl_kraskov'
    cmi_estimator_1 = Estimator_cmi(estimator_name_1)
    cmi_estimator_2 = Estimator_cmi(estimator_name_2)
    mi_estimator_1 = Estimator_mi(estimator_name_1)
    assert cmi_estimator_1.estimator_name == estimator_name_1, (
                'The estimator was not set correctly')
    assert cmi_estimator_2.estimator_name == estimator_name_2, (
                'The estimator was not set correctly')
    assert mi_estimator_1.estimator_name == estimator_name_1, (
                'The estimator was not set correctly')

if __name__ == '__main__':
    test_estimator_change()
    test_estimators_correlated_gauss_data()
    test_estimators_uncorrelated_random_data()
