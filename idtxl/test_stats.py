# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:46:09 2016

@author: patricia
"""
import numpy as np
import stats


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

    # Assert that samples have been swapped within the permutation range for
    # the first replication.
    samples = np.arange(rng)
    i = 0
    for p in range(n_per_repl // rng):
        assert (np.unique(perm[i:i + rng, 0]) == samples).all(), ('Something '
            'went wrong when permuting realisations')
        samples += rng
        i += rng
    rem = n_per_repl % rng
    if rem > 0:
        assert (np.unique(perm[i:i + rem, 0]) == samples[0:rem]).all(), ('Something'
            ' went wrong when permuting realisations')


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
    test_permute_realisations()
    test_find_pvalue()
    test_find_table_max()
    test_find_table_min()
