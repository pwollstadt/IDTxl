# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:03:14 2016


http://stackoverflow.com/questions/1593019/
    is-there-any-simple-way-to-benchmark-python-script
https://docs.python.org/3.4/library/timeit.html

@author: patricia
"""
import cProfile
import numpy as np
from data import Data
from multivariate_te import Multivariate_te

data = Data(normalise=False)
data.generate_mute_data()
#data.set_data(np.arange(30).reshape(3,10), 'ps')
#n_permutations = 100

idx = [(1, 0), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (1, 0), (1, 2), (1, 3),
       (1, 4), (2, 3), (2, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
cv = (0, 5)

def old(data, idx, cv):
    return data._get_data(idx, cv)


def new(data, idx_list, current_value, shuffle=False):
    n_realisations_time = data.n_samples - current_value[1]
    n_realisations_replications = data.n_replications
    realisations = np.empty((n_realisations_time *
                             n_realisations_replications,
                             len(idx_list)))

    if shuffle:
        replications_order = np.random.permutation(data.n_replications)
    else:
        replications_order = np.arange(data.n_replications)

    i = 0
    for idx in idx_list:  # TODO test this for single trials!
        r = 0
        last_sample = current_value[1] - idx[1]
        if last_sample == 0:
            last_sample = None
        for replication in replications_order:
            realisations[r:r+n_realisations_time, i] = data.data[idx[0],idx[1]:-last_sample, replication]
            if np.isnan(realisations[r, i]):
                print('boom')
            r += n_realisations_time
        i += 1

    return realisations

def new2(data, idx_list, current_value, shuffle=False):
    n_realisations_time = data.n_samples - current_value[1]
    n_realisations_replications = data.n_replications
    realisations = np.empty((n_realisations_time *
                             n_realisations_replications,
                             len(idx_list)))

    if shuffle:
        replications_order = np.random.permutation(data.n_replications)
    else:
        replications_order = np.arange(data.n_replications)

    i = 0
    for idx in idx_list:  # TODO test this for single trials!
        r = 0
        last_sample = current_value[1] - idx[1]
        realisations[:, i] = data.data[idx[0],idx[1]:-last_sample,
                                       replications_order].reshape(realisations.shape[0])
        i += 1

    return realisations

cProfile.run('a=old(data, idx, cv)')
cProfile.run('b=new(data, idx, cv)')
cProfile.run('b=new2(data, idx, cv)')
a=old(data, idx, cv)[0]
b=new(data, idx, cv)
c=new2(data, idx, cv)
assert((a == b).all()), 'Results diverged!'
assert((a == c).all()), 'Results diverged!'