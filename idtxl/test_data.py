# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:36:41 2016

@author: patricia
"""
import numpy as np
from data import Data
from . import utils


def test_set_data():
    """Test if data is written correctly into a Data instance."""
    source = np.expand_dims(np.repeat(1, 30), axis=1)
    target = np.expand_dims(np.arange(30), axis=1)

    data = Data(normalise=False)
    data.set_data(np.vstack((source.T, target.T)), 'ps')

    assert (data.data[0, :].T == source.T).all(), ('Class data does not match '
                                                   'input (source).')
    assert (data.data[1, :].T == target.T).all(), ('Class data does not match '
                                                   'input (target).')


def test_data_normalisation():
    """Test if data are normalised correctly when stored in a Data instance."""
    a_1 = 100
    a_2 = 1000
    source = np.random.randint(a_1, size=1000)
    target = np.random.randint(a_2, size=1000)

    data = Data(normalise=True)
    data.set_data(np.vstack((source.T, target.T)), 'ps')

    source_std = utils.standardise(source)
    target_std = utils.standardise(target)
    assert (source_std == data.data[0, :, 0]).all(), ('Standardising the '
                                                      'source did not work.')
    assert (target_std == data.data[1, :, 0]).all(), ('Standardising the '
                                                      'target did not work.')
