# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:34:31 2016

@author: patricia
"""
import numpy as np
import matplotlib.pyplot as plt
from data import Data
from set_estimator import Estimator_te

d = np.load('/home/patricia/repos/IDTxl/testing/data/lorenz_2_exampledata.npy')
dat = Data()
dat.set_data(d[:, :, :20], 'psr')

# %%
lag = 45
te = []
min_u = 30
max_u = 55
for lag in range(min_u, max_u):
    cur_val = (1, lag)
    source = np.squeeze(dat.get_realisations(cur_val, [(0, 0)])[0])
    target = np.squeeze(dat.get_realisations(cur_val, [(1, lag)])[0])

    opts = {
        'noise_level': 0,
        'history_target': 20,
        'history_source': 1,
        'tau_target': 2,
        'tau_source': 1,
        'source_target_delay': 0,
        }

    est_te = Estimator_te('jidt_kraskov')
    te.append(est_te.estimate(source, target, opts))
    print('TE for lag {0}: {1}'.format(lag, te[-1]))

plt.plot(np.arange(min_u, max_u), te)
