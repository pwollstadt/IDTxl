"""Test data class.

Created on Mon Apr  4 16:36:41 2016

@author: patricia
"""
import pytest
import numpy as np
from idtxl.data import Data
import idtxl.idtxl_utils as utils


def test_data_properties():

    n = 10
    d = Data(np.arange(n), 's', normalise=False)
    real_time = d.n_realisations_samples()
    assert (real_time == n), 'Realisations in time are not returned correctly.'
    cv = (0, 8)
    real_time = d.n_realisations_samples(current_value=cv)
    assert (real_time == (n - cv[1])), ('Realisations in time are not '
                                        'returned correctly when current value'
                                        ' is set.')


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

    d = Data()
    dat = np.arange(10000).reshape((2, 1000, 5))  # random data with correct
    d = Data(dat, dim_order='psr')               # order od dimensions
    assert (d.data.shape[0] == 2), ('Class data does not match input, number '
                                    'of processes wrong.')
    assert (d.data.shape[1] == 1000), ('Class data does not match input, '
                                       'number of samples wrong.')
    assert (d.data.shape[2] == 5), ('Class data does not match input, number '
                                    'of replications wrong.')
    dat = np.arange(3000).reshape((3, 1000))  # random data with incorrect
    d = Data(dat, dim_order='ps')            # order of dimensions
    assert (d.data.shape[0] == 3), ('Class data does not match input, number '
                                    'of processes wrong.')
    assert (d.data.shape[1] == 1000), ('Class data does not match input, '
                                       'number of samples wrong.')
    assert (d.data.shape[2] == 1), ('Class data does not match input, number '
                                    'of replications wrong.')
    dat = np.arange(5000)
    d.set_data(dat, 's')
    assert (d.data.shape[0] == 1), ('Class data does not match input, number '
                                    'of processes wrong.')
    assert (d.data.shape[1] == 5000), ('Class data does not match input, '
                                       'number of samples wrong.')
    assert (d.data.shape[2] == 1), ('Class data does not match input, number '
                                    'of replications wrong.')


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


def test_get_data():
    """Test low-level function for data retrieval."""
    dat = Data()
    dat.generate_mute_data()
    idx_list = [(0, 4), (0, 6)]
    current_value = (0, 3)
    with pytest.raises(RuntimeError):
        dat._get_data(idx_list, current_value)

    # Test retrieved data for one/two replications in time (i.e., the current
    # value is equal to the last sample)
    n = 7
    d = Data(np.arange(n + 1), 's', normalise=False)
    current_value = (0, n)
    dat = d._get_data([(0, 1)], current_value)[0]
    assert (dat[0][0] == 1)
    assert (dat.shape == (1, 1))
    d = Data(np.arange(n + 2), 's', normalise=False)
    current_value = (0, n)
    dat = d._get_data([(0, 1)], current_value)[0]
    assert (dat[0][0] == 1)
    assert (dat[1][0] == 2)
    assert (dat.shape == (2, 1))

    # Test retrieval of realisations of the current value.
    n = 7
    d = Data(np.arange(n), 's', normalise=False)
    current_value = (0, n)
    dat = d._get_data([current_value], current_value)[0]


def test_permute_replications():
    """Test surrogate creation by permuting replications."""
    n = 20
    data = Data(np.vstack((np.zeros(n),
                           np.ones(n) * 1,
                           np.ones(n) * 2,
                           np.ones(n) * 3)).astype(int),
                         'rs',
                         normalise=False)
    current_value = (0, n)
    l = [(0, 1), (0, 3), (0, 7)]
    [perm, perm_idx] = data.permute_replications(current_value=current_value,
                                                 idx_list=l)
    assert (np.all(perm[:, 0] == perm_idx)), 'Permutation did not work.'

    # Assert that samples have been swapped within the permutation range for
    # the first replication.
    rng = 3
    current_value = (0, 3)
    l = [(0, 0), (0, 1), (0, 2)]
    #data = Data(np.arange(n), 's', normalise=False)
    data = Data(np.vstack((np.arange(n),
                           np.arange(n))).astype(int),
                         'rs',
                         normalise=False)
    [perm, perm_idx] = data.permute_samples(current_value=current_value,
                                            idx_list=l,
                                            perm_range=rng)
    samples = np.arange(rng)
    i = 0
    n_per_repl = int(data.n_realisations(current_value) / data.n_replications)
    for p in range(n_per_repl // rng):
        assert (np.unique(perm[i:i + rng, 0]) == samples).all(), ('The '
            'permutation range was not respected.')
        samples += rng
        i += rng
    rem = n_per_repl % rng
    if rem > 0:
        assert (np.unique(perm[i:i + rem, 0]) == samples[0:rem]).all(), ('The '
            'remainder did not contain the same realisations.')

    # Test assertions that perm_range is not too low or too high.
    with pytest.raises(AssertionError):
        data.permute_samples(current_value=current_value,
                             idx_list=l,
                             perm_range=1)
    with pytest.raises(AssertionError):
        data.permute_samples(current_value=current_value,
                             idx_list=l,
                             perm_range=np.inf)
    # Test ValueError if a string other than 'max' is given for perm_range.
    with pytest.raises(ValueError):
        data.permute_samples(current_value=current_value,
                             idx_list=l,
                             perm_range='foo')

def test_permute_samples():
    pass

if __name__ == '__main__':
    test_get_data()
    test_data_normalisation()
    test_set_data()
    test_permute_replications()
    test_data_properties()
