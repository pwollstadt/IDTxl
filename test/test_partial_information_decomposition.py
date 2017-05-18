"""Provide unit tests for high-level PID estimation.

@author: patricia
"""
import pytest
import numpy as np
from idtxl.partial_information_decomposition import (
                                        Partial_information_decomposition)
from idtxl.data import Data
from test_estimators_pid import optimization_not_available


@optimization_not_available
def test_pid_user_input():
    """Test if user input is handled correctly."""
    # Test missing calculator name
    with pytest.raises(KeyError):
        Partial_information_decomposition(options=dict())

    # Test wrong calculator name
    with pytest.raises(AttributeError):
        analysis_opts = {'pid_calc_name': 'pid_test'}
        Partial_information_decomposition(options=analysis_opts)

    n = 20
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Test two-tailed significance test
    analysis_opts = {'pid_calc_name': 'pid_tartu',
                     'tail': 'two'}
    pid = Partial_information_decomposition(options=analysis_opts)

    with pytest.raises(RuntimeError):  # Test incorrect number of sources
        pid.analyse_single_target(data=dat, target=2, sources=[1, 2, 3],
                                  lags=[0, 0])
    with pytest.raises(RuntimeError):  # Test incorrect number of lags
        pid.analyse_single_target(data=dat, target=2, sources=[1, 3],
                                  lags=[0, 0, 0])
    with pytest.raises(RuntimeError):  # Test target in sources
        pid.analyse_single_target(data=dat, target=2, sources=[2, 3],
                                  lags=[0, 0])
    with pytest.raises(RuntimeError):  # Test lag > no. samples
        pid.analyse_single_target(data=dat, target=2, sources=[0, 1],
                                  lags=[n * 3, 0])
    with pytest.raises(RuntimeError):  # Test lag == no. samples
        pid.analyse_single_target(data=dat, target=2, sources=[0, 1],
                                  lags=[n, 0])
    with pytest.raises(IndexError):  # Test target not in processes
        pid.analyse_single_target(data=dat, target=5, sources=[0, 1],
                                  lags=[0, 0])


@optimization_not_available
def test_network_analysis():
    """Test call to network_analysis method."""
    n = 50
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Run Tartu estimator
    analysis_opts = {'pid_calc_name': 'pid_tartu',
                     'tail': 'two'}
    pid = Partial_information_decomposition(options=analysis_opts)
    est_tartu = pid.analyse_network(data=dat, targets=[0, 2],
                                    sources=[[1, 2], [0, 1]],
                                    lags=[[0, 0], [0, 0]])
    assert 0.9 < est_tartu[2]['syn_s1_s2'] <= 1.1, (
            'Tartu estimator incorrect synergy: {0}, should approx. 1'.format(
                                                    est_tartu[2]['syn_s1_s2']))


@optimization_not_available
def test_analyse_single_target():
    """Test call to network_analysis method."""
    n = 50
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Run Tartu estimator
    analysis_opts = {'pid_calc_name': 'pid_tartu',
                     'tail': 'two'}
    pid = Partial_information_decomposition(options=analysis_opts)
    est_tartu = pid.analyse_single_target(data=dat, target=2, sources=[0, 1],
                                          lags=[0, 0])
    assert 0.9 < est_tartu['syn_s1_s2'] <= 1.1, (
            'Tartu estimator incorrect synergy: {0}, should approx. 1'.format(
                                                    est_tartu['syn_s1_s2']))
    assert est_tartu['unq_s1'] < 0.1, ('Tartu estimator incorrect unique '
                                       's1: {0}, should approx. 0'.format(
                                                    est_tartu['unq_s1']))
    assert est_tartu['unq_s2'] < 0.1, ('Tartu estimator incorrect unique '
                                       's2: {0}, should approx. 0'.format(
                                                    est_tartu['unq_s2']))


if __name__ == '__main__':
    test_pid_user_input()
    test_network_analysis()
    test_pid_xor_data()
