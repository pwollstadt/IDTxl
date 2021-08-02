"""Provides unit test for the Rudelt estimators"""

import numpy as np
import pytest
from idtxl.data_spiketime import Data_spiketime
from idtxl.estimators_Rudelt import RudeltNSBEstimatorSymbolsMI, \
    RudeltPluginEstimatorSymbolsMI, RudeltBBCEstimator, RudeltShufflingEstimator


def test_user_input():

    est_nsb = RudeltNSBEstimatorSymbolsMI()
    est_plugin = RudeltPluginEstimatorSymbolsMI()
    est_bbc = RudeltBBCEstimator()
    est_shuffling = RudeltShufflingEstimator()

    N = 300

    with pytest.raises(AssertionError):
        est_nsb.estimate(np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N + 1, dtype=int),
                         np.random.randint(6, size=N, dtype=int))
    with pytest.raises(AssertionError):
        est_nsb.estimate(np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N + 1, dtype=int))
    with pytest.raises(AssertionError):
        est_nsb.estimate(np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=(N, 2), dtype=int))

    with pytest.raises(AssertionError):
        est_plugin.estimate(np.random.randint(6, size=N, dtype=int),
                            np.random.randint(6, size=N + 1, dtype=int),
                            np.random.randint(6, size=N, dtype=int))
    with pytest.raises(AssertionError):
        est_plugin.estimate(np.random.randint(6, size=N, dtype=int),
                            np.random.randint(6, size=N, dtype=int),
                            np.random.randint(6, size=N + 1, dtype=int))
    with pytest.raises(AssertionError):
        est_plugin.estimate(np.random.randint(6, size=N, dtype=int),
                            np.random.randint(6, size=N, dtype=int),
                            np.random.randint(6, size=(N, 2), dtype=int))

    with pytest.raises(AssertionError):
        est_bbc.estimate(np.random.randint(6, size=N, dtype=int),
                            np.random.randint(6, size=N + 1, dtype=int),
                            np.random.randint(6, size=N, dtype=int))
    with pytest.raises(AssertionError):
        est_bbc.estimate(np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N + 1, dtype=int),
                         0.01)
    with pytest.raises(AssertionError):
        est_bbc.estimate(np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=(N, 2), dtype=int),
                         0.01)
    with pytest.raises(AssertionError):
        est_bbc.estimate(np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N, dtype=int),
                         np.random.randint(6, size=N, dtype=int),
                         'test')

    with pytest.raises(AssertionError):
        est_shuffling.estimate(np.random.randint(6, size=(N, 2), dtype=int))


def test_nsb_and_plugin_estimator():
    data = Data_spiketime()
    data.load_Rudelt_data()

    process_list = [0]
    symbol_array, past_symbol_array, current_symbol_array, symbol_array_length = \
        data.get_realisations_symbols(process_list, 0.005, 1, 0.0, 0.005)

    est_nsb = RudeltNSBEstimatorSymbolsMI()
    I_nsb, R_nsb = est_nsb.estimate(symbol_array[0], past_symbol_array[0], current_symbol_array[0])

    est_plugin = RudeltPluginEstimatorSymbolsMI()
    I_plugin, R_plugin = est_plugin.estimate(symbol_array[0], past_symbol_array[0], current_symbol_array[0])

    print('NSB MI result: {0:.4f}; Plugin MI result: {1:.4f}'.format(I_nsb, I_plugin))
    print('NSB MI / H_uncond result: {0:.4f}; Plugin MI / H_uncond result: {1:.4f}'.format(R_nsb, R_plugin))

    assert np.isclose(I_nsb, I_plugin, atol=0.01), (
        'MI of NSB and Plugin estimators are not close to each other (error larger 0.01).')

    assert np.isclose(R_nsb, R_plugin, atol=0.01), (
        'MI / H_uncond of NSB and Plugin estimators are not close to each other (error larger 0.01).')


def test_bbc_and_shuffling_estimator():
    data = Data_spiketime()
    data.load_Rudelt_data()

    process_list = [0]
    symbol_array, past_symbol_array, current_symbol_array, symbol_array_length = \
        data.get_realisations_symbols(process_list, 0.005, 1, 0.0, 0.005)

    est_bbc = RudeltBBCEstimator()
    I_bbc, R_bbc, bbc_term = est_bbc.estimate(symbol_array[0], past_symbol_array[0], current_symbol_array[0])

    est_shuffling = RudeltShufflingEstimator()
    I_shuffling, R_shuffling = est_shuffling.estimate(symbol_array[0])

    print('BBC MI result: {0:.4f}; Shuffling MI result: {1:.4f}'.format(I_bbc, I_shuffling))
    print('BBC MI / H_uncond result: {0:.4f}; Shuffling MI / H_uncond result: {1:.4f}'.format(R_bbc, R_shuffling))

    assert np.isclose(I_bbc, I_shuffling, atol=0.01), (
        'MI of BBC and Shuffling estimators are not close to each other (error larger 0.01).')

    assert np.isclose(R_bbc, R_shuffling, atol=0.01), (
        'MI / H_uncond of BBC and Shuffling estimators are not close to each other (error larger 0.01).')


if __name__ == '__main__':
    test_user_input()
    test_nsb_and_plugin_estimator()
    test_bbc_and_shuffling_estimator()