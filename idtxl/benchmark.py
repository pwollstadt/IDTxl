# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:03:14 2016


http://stackoverflow.com/questions/1593019/
    is-there-any-simple-way-to-benchmark-python-script
https://docs.python.org/3.4/library/timeit.html

@author: patricia
"""
import cProfile
import pstats
import numpy as np
from data import Data
from multivariate_te import Multivariate_te

dat = Data()
dat.generate_mute_data(100, 5)
max_lag = 5
min_lag = 4
analysis_opts = {
    'cmi_calc_name': 'jidt_kraskov',
    'n_perm_max_stat': 20,
    'n_perm_min_stat': 20,
    'n_perm_omnibus': 500,
    'n_perm_max_seq': 500,
    }
target = 0
sources = [1, 2, 3]

netw = Multivariate_te(max_lag, analysis_opts, min_lag)

filename = 'profile_stats.stats'
cProfile.run('res = netw.analyse_single_target(dat, target, sources)',
             filename)
stats = pstats.Stats('profile_stats.stats')
stats.strip_dirs()  # clean up filenames for the report
stats.sort_stats('cumulative')
stats.print_stats()

#cProfile.run('a=old(data, idx, cv)')
#cProfile.run('b=new(data, idx, cv)')
#a=old(data, idx, cv)[0]
#b=new(data, idx, cv)
#assert((a == b).all()), 'Results diverged!'