"""Test the setting of estimators.

This module provides unit tests for estimator setting. The user can choose from
a variety of estimators (depending on data type and measure to be estimated).
This functionality is handled by the estimator class and tested here.

Created on Fri Sep 30 14:39:06 2016

@author: patricia
"""
import inspect
import pytest
from idtxl.estimator import Estimator, find_estimator
from idtxl.multivariate_te import MultivariateTE


class EstimatorTestEstimate(Estimator):

    def __init__(self):
        pass

    def is_parallel(self):
        return True


class EstimatorTestIsParallel(Estimator):

    def __init__(self):
        pass

    def estimate(self):
        return True


def test_base_class_implementation():
    """Test if instantiation fails if abstract methods aren't implemented."""

    with pytest.raises(TypeError):
        EstimatorTestEstimate()
    with pytest.raises(TypeError):
        EstimatorTestIsParallel()


def test_find_estimator():
    """Test dynamic loading of classes."""

    # Test if a class is returned.
    e = find_estimator('JidtKraskovMI')
    assert inspect.isclass(e)
    f = find_estimator(e)
    assert inspect.isclass(f)

    # Try loading non-existent estimator
    with pytest.raises(RuntimeError):
        find_estimator('test')

    # Try using a class without an estimate method
    with pytest.raises(AssertionError):
        find_estimator(MultivariateTE)


if __name__ == '__main__':
    test_find_estimator()
    test_base_class_implementation()
