"""Test Network_analysis.

This module provides unit tests for the Network_analysis class.

Created on Tue Aug 02 04:47:01 2016

@author: patricia
"""
import pytest
import numpy as np
from idtxl.network_analysis import Network_analysis

def test_separate_realisations():
    n = Network_analysis()
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
    n = Network_analysis()
    n.current_value = (0, 5)
    idx_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    reference_list = [(1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]
    lag_list = n._idx_to_lag(idx_list)
    res = [True for i, j in zip(lag_list, reference_list) if i == j]
    assert np.array(res).all(), 'Indices were not converted to lags correctly.'

    with pytest.raises(IndexError):
        idx_list = [(1, 7)]
        n._idx_to_lag(idx_list)


def test_lag_to_idx():
    n = Network_analysis()
    n.current_value = (0, 5)
    lag_list = [(1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]
    reference_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    idx_list = n._lag_to_idx(lag_list)
    res = [True for i, j in zip(idx_list, reference_list) if i == j]
    assert np.array(res).all(), 'Lags were not converted to indices correctly.'

    with pytest.raises(IndexError):
        idx_list = [(1, 7)]
        n._idx_to_lag(idx_list)

if __name__ == '__main__':
    test_idx_to_lag()
    test_lag_to_idx()
    test_separate_realisations()
