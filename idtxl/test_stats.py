# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:46:09 2016

@author: patricia
"""
import pytest
import numpy as np
import stats


def test_omnibus_test():
    print('Write test for omnibus test.')


def test_max_statistic():
    print('Write test for max_statistic.')


def test_min_statistic():
    print('Write test for min_statistic.')


def test_max_statistic_sequential():
    print('Write test for max_statistic_sequential.')


def test_network_fdr():

    target_0 = {
        'conditional_sources': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 0)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                             (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    target_1 = {
        'conditional_sources': [(1, 2), (2, 1), (2, 2)],
        'conditional_full': [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01])
        }
    target_2 = {
        'conditional_sources': [],
        'conditional_full': [(2, 0), (2, 1)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res = {
        0: target_0,
        1: target_1,
        2: target_2
    }
    res_pruned = stats.network_fdr(res)
    assert (not res_pruned[2]['conditional_sources']), ('Target ')

    for k in res_pruned.keys():
        if res_pruned[k]['cond_sources_pval'] is None:
            assert (not res_pruned[k]['conditional_sources'])
        else:
            assert (len(res_pruned[k]['conditional_sources']) ==
                    len(res_pruned[k]['cond_sources_pval'])), ('Source list '
                                                               'and list of p-'
                                                               'values should '
                                                               'have the '
                                                               'same length.')


def test_permute_realisations():
    n = 30
    n_var = 3
    n_per_repl = 5
    real = np.arange(n).reshape(n_var, n / n_var).T
    reps = int(n / n_var / n_per_repl)
    real_idx = np.repeat(np.arange(reps), n_per_repl)
    # real_idx = np.zeros(60).astype(int)
    rng = 3
    perm = stats._permute_realisations(realisations=real,
                                       replication_idx=real_idx,
                                       perm_range=rng)
    # Assert the permutation worked.
    assert (not (perm == real).all()), 'Permutation did not work.'
    assert (np.unique(perm) == np.unique(real)).all(), ('Entries in original '
                                                        'and permuted '
                                                        'realisations are not '
                                                        'the same.')
    # Assert that samples have been swapped within the permutation range for
    # the first replication.
    samples = np.arange(rng)
    i = 0
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
        stats._permute_realisations(realisations=real,
                                    replication_idx=real_idx,
                                    perm_range=1)
    with pytest.raises(AssertionError):
        stats._permute_realisations(realisations=real,
                                    replication_idx=real_idx,
                                    perm_range=np.inf)
    # Test assertion of equal array sizes. replication_idx must hold the
    # replication number for each entry in realisations, i.e., dimensions
    # must be equal.
    with pytest.raises(AssertionError):
        stats._permute_realisations(realisations=real,
                                    replication_idx=real_idx[:1],
                                    perm_range=rng)
    # Test ValueError if a string other than 'max' is given for perm_range.
    with pytest.raises(ValueError):
        stats._permute_realisations(realisations=real,
                                    replication_idx=real_idx,
                                    perm_range='foo')


def test_find_pvalue():
    test_val = 1
    distribution = np.random.rand(500)  # normally distributed floats in [0,1)
    alpha = 0.05
    tail = 'two'  # 'one' or 'two'
    [s, p] = stats._find_pvalue(test_val, distribution, alpha, tail)
    assert(s is True)

    # If the statistic is bigger than the whole distribution, the p-value
    # should be set to the smallest possible value that could have been
    # expected from a test given the number of permutations.
    test_val = np.inf
    [s, p] = stats._find_pvalue(test_val, distribution, alpha, tail)
    n_perm = distribution.shape[0]
    assert(p == (1 / n_perm))

    # Test assertion that the test distributions is a one-dimensional array.
    with pytest.raises(AssertionError):
        stats._find_pvalue(test_val, np.expand_dims(distribution, axis=1),
                           alpha, tail)
    # Test assertion that no. permutations is high enough to theoretically
    # calculate the requested alpha level.
    with pytest.raises(AssertionError):
        stats._find_pvalue(test_val, distribution[:5], alpha, tail)
    # Check if wrong parameter for tail raises a value error.
    with pytest.raises(ValueError):
        stats._find_pvalue(test_val, distribution, alpha, tail='foo')


def test_find_table_max():
    tab = np.array([[0, 2, 1], [3, 4, 5], [10, 8, 1]])
    res = stats._find_table_max(tab)
    assert (res == np.array([10,  8,  5])).all(), ('Function did not return '
                                                   'maximum for each column.')


def test_find_table_min():
    tab = np.array([[0, 2, 1], [3, 4, 5], [10, 8, 1]])
    res = stats._find_table_min(tab)
    assert (res == np.array([0, 2, 1])).all(), ('Function did not return '
                                                'minimum for each column.')

if __name__ == '__main__':
    test_network_fdr()
    test_permute_realisations()
    test_find_pvalue()
    test_find_table_max()
    test_find_table_min()
    test_omnibus_test()
    test_max_statistic()
    test_min_statistic()
    test_max_statistic_sequential()
