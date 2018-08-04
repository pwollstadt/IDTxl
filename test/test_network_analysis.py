"""Test NetworkAnalysis.

This module provides unit tests for the NetworkAnalysis class.
"""
import pytest
import numpy as np
from idtxl.network_analysis import NetworkAnalysis
from idtxl.data import Data
from test_estimators_jidt import _get_gauss_data
from idtxl.estimators_jidt import JidtKraskovCMI, JidtKraskovMI


def test_calculate_single_link():
    """Test calculation of single link (conditional) MI and TE."""

    expected_mi, source, source_uncorr, target = _get_gauss_data()
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]
    data = Data(np.hstack((source, source_uncorr, target)), dim_order='sp')

    n = NetworkAnalysis()
    n._cmi_estimator = JidtKraskovCMI(settings={})
    n.settings = {
        'local_values': False
    }
    current_value = (2, 1)

    # Test single link estimation for a single and multiple sources for
    # cases: no target vars, source vars/no source vars (tests if the
    # conditioning set is built correctly for conditioning='full').
    source_realisations = data.get_realisations(current_value, [(0, 0)])[0]
    current_value_realisations = data.get_realisations(
        current_value, [current_value])[0]
    expected_mi = n._cmi_estimator.estimate(
        current_value_realisations, source_realisations)
    # cond. on second source
    cond_realisations = data.get_realisations(current_value, [(1, 0)])[0]
    expected_mi_cond1 = n._cmi_estimator.estimate(
        current_value_realisations, source_realisations, cond_realisations)

    for sources in ['all', [0]]:
        for conditioning in ['full', 'target', 'none']:
            for source_vars in [[(0, 0)], [(0, 0), (1, 0)]]:
                mi = n._calculate_single_link(
                    data, current_value, source_vars, target_vars=None,
                    sources=sources, conditioning=conditioning)
                if mi.shape[0] > 1:  # array for source='all'
                    mi = mi[0]

                if source_vars == [(0, 0)]:  # no conditioning
                    assert np.isclose(mi, expected_mi, rtol=0.05), (
                        'Estimated single-link MI ({0}) differs from expected '
                        'MI ({1}).'.format(mi, expected_mi))
                else:
                    if conditioning == 'full':  # cond. on second source
                        assert np.isclose(mi, expected_mi_cond1, rtol=0.05), (
                            'Estimated single-link MI ({0}) differs from '
                            'expected MI ({1}).'.format(mi, expected_mi_cond1))
                    else:  # no conditioning
                        assert np.isclose(mi, expected_mi, rtol=0.05), (
                            'Estimated single-link MI ({0}) differs from '
                            'expected MI ({1}).'.format(mi, expected_mi))

        # Test single link estimation for a single and multiple sources for
        # cases: target vars/no target vars, source vars (tests if the
        # conditioning set is built correctly for conditioning='full').
        cond_realisations = np.hstack((  # cond. on second source and target
            data.get_realisations(current_value, [(1, 0)])[0],
            data.get_realisations(current_value, [(2, 0)])[0]
            ))
        expected_mi_cond2 = n._cmi_estimator.estimate(
            current_value_realisations, source_realisations, cond_realisations)
        # cond. on target
        cond_realisations = data.get_realisations(current_value, [(2, 0)])[0]
        expected_mi_cond3 = n._cmi_estimator.estimate(
            current_value_realisations, source_realisations, cond_realisations)

        for target_vars in [None, [(2, 0)]]:
            for conditioning in ['full', 'target', 'none']:
                mi = n._calculate_single_link(
                    data, current_value, source_vars=[(0, 0), (1, 0)],
                    target_vars=target_vars, sources=sources,
                    conditioning=conditioning)
                if mi.shape[0] > 1:  # array for source='all'
                    mi = mi[0]

                if conditioning == 'none':  # no conditioning
                    assert np.isclose(mi, expected_mi, rtol=0.05), (
                        'Estimated single-link MI ({0}) differs from expected '
                        'MI ({1}).'.format(mi, expected_mi))
                else:
                    # target only
                    if target_vars is not None and conditioning == 'target':
                        assert np.isclose(mi, expected_mi_cond3, rtol=0.05), (
                            'Estimated single-link MI ({0}) differs from '
                            'expected MI ({1}).'.format(mi, expected_mi_cond3))
                    # target and 2nd source
                    if target_vars is not None and conditioning == 'full':
                        assert np.isclose(mi, expected_mi_cond2, rtol=0.05), (
                            'Estimated single-link MI ({0}) differs from '
                            'expected MI ({1}).'.format(mi, expected_mi_cond2))
                    # target is None, condition on second target
                    else:
                        if conditioning == 'full':
                            assert np.isclose(mi, expected_mi_cond1, rtol=0.05), (
                                'Estimated single-link MI ({0}) differs from expected '
                                'MI ({1}).'.format(mi, expected_mi_cond1))

    # Test requested sources not in source vars
    with pytest.raises(RuntimeError):
        mi = n._calculate_single_link(
            data, current_value, source_vars=[(0, 0), (3, 0)],
            target_vars=None, sources=4, conditioning='full')
    # Test source vars not in data/processes
    with pytest.raises(IndexError):
        mi = n._calculate_single_link(
            data, current_value, source_vars=[(0, 0), (10, 0)],
            target_vars=None, sources='all', conditioning='full')
    # Test unknown conditioning
    with pytest.raises(RuntimeError):
        mi = n._calculate_single_link(
            data, current_value, source_vars=[(0, 0)], conditioning='test')


def test_separate_realisations():
    n = NetworkAnalysis()
    r_1 = np.ones((10, 1))
    r_2 = np.ones((10, 1)) * 2
    r_3 = np.ones((10, 1)) * 3
    idx = [(0, 1), (0, 2), (0, 3)]
    n._append_selected_vars_idx(idx)
    n._append_selected_vars_realisations(np.hstack((r_1, r_2, r_3)))
    print(n._selected_vars_realisations)
    [remain, single] = n._separate_realisations(idx, idx[0])
    print(single.shape)
    a = np.empty((10, 1))
    b = np.empty((10, 2))
    a[0:10, ] = single
    b[0:10, ] = remain
    print(a.shape)
    print(b)

    assert np.all(remain == np.hstack((r_2, r_3))), 'Remainder is incorrect.'
    assert np.all(single == r_1), 'Single realisations are incorrect.'
    [remain, single] = n._separate_realisations([idx[0]], idx[0])
    assert remain is None, 'Remainder should be None.'


def test_idx_to_lag():
    n = NetworkAnalysis()
    n.current_value = (0, 5)
    idx_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    reference_list = [(1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]
    lag_list = n._idx_to_lag(idx_list)
    results = [True for i, j in zip(lag_list, reference_list) if i == j]
    assert np.array(results).all(), (
        'Indices were not converted to lags correctly.')

    with pytest.raises(IndexError):
        idx_list = [(1, 7)]
        n._idx_to_lag(idx_list)


def test_lag_to_idx():
    n = NetworkAnalysis()
    n.current_value = (0, 5)
    lag_list = [(1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]
    reference_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    idx_list = n._lag_to_idx(lag_list)
    results = [True for i, j in zip(idx_list, reference_list) if i == j]
    assert np.array(results).all(), (
        'Lags were not converted to indices correctly.')

    with pytest.raises(IndexError):
        idx_list = [(1, 7)]
        n._idx_to_lag(idx_list)


if __name__ == '__main__':
    test_calculate_single_link()
    test_idx_to_lag()
    test_lag_to_idx()
    test_separate_realisations()
