"""Test the setting of estimators.

This module provides unit tests for estimator setting. The user can choose from
a variety of estimators (depending on data type and measure to be estimated).
This functionality is handled by the estimator class and tested here.
"""
import inspect
import pytest
import numpy as np
from idtxl.estimator import find_estimator
from idtxl.multivariate_te import MultivariateTE
from idtxl.estimators_jidt import JidtKraskovMI
from test_estimators_jidt import jpype_missing, _get_gauss_data

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
    with pytest.raises(RuntimeError):
        find_estimator(MultivariateTE)


@jpype_missing
def test_estimate_parallel():
    """Test estimate_parallel() against estimate()."""
    expected_mi, source1, source2, target = _get_gauss_data()

    source_chunks = np.vstack((source1, source1))
    target_chunks = np.vstack((target, target))

    # Compare MI-estimates from serial and parallel estimator.
    mi_estimator = JidtKraskovMI(settings={'noise_level': 0})
    mi = mi_estimator.estimate(source1, target)
    with pytest.raises(AssertionError):
        mi_estimator.estimate_parallel(
            n_chunks=2,
            var1=source_chunks,
            var2=target)
    mi_parallel1 = mi_estimator.estimate_parallel(
            n_chunks=2,
            re_use=['var2'],
            var1=source_chunks,
            var2=target)
    mi_parallel2 = mi_estimator.estimate_parallel(
            n_chunks=2,
            var1=source_chunks,
            var2=target_chunks)
    assert (mi_parallel1 == mi_parallel2).all(), (
        'Results for stacked ({0}) and re-used ({1}) target differ.'.format(
            mi_parallel1, mi_parallel2))
    assert mi_parallel1[0] == mi, (
        'Results for first chunk differ from serial estimate.')
    assert mi_parallel1[1] == mi, (
        'Results for second chunk differ from serial estimate.')

    # Check if a single chunk is returned if all variables are defined as
    # reusable.
    mi_parallel3 = mi_estimator.estimate_parallel(
            n_chunks=2,
            re_use=['var1', 'var2'],
            var1=source1,
            var2=target)
    assert len(mi_parallel3) == 1, (
        'Single chunk data returned more than one estimate.')

    # Check assertion for incorrect number of samples in data to be reused in
    # parallel estimator.
    with pytest.raises(AssertionError):
        mi_estimator.estimate_parallel(
                n_chunks=2,
                re_use=['var2'],
                var1=source_chunks,
                var2=target[:100])


if __name__ == '__main__':
    test_find_estimator()
    test_estimate_parallel()
