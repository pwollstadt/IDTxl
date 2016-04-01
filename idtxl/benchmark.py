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
import random as rn
from set_estimator import Estimator_cmi

n = 10000
cov = 0.4
source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
target = [sum(pair) for pair in zip(
    [cov * y for y in source_1],
    [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
source_1 = np.expand_dims(np.array(source_1), axis=1)
target = np.expand_dims(np.array(target), axis=1)
opts = {'kraskov_k': 4, 'normalise': True}
expected_res = np.log(1 / (1 - np.power(cov, 2)))


def test_old(n=100):
    calculator_name_1 = 'jidt_kraskov'
    est_1 = Estimator_cmi(calculator_name_1)
    for i in range(n):
        res_1 = est_1.estimate(var1=source_1[1:], var2=target[1:],
                               conditional=target[:-1], opts=opts)
    print('Old result {0:.4f} nats; expected:{1:.4f}'.format(res_1,
                                                             expected_res))


def test_new(n=100):
    calculator_name_2 = 'jidt_kraskov_fast'
    est_2 = Estimator_cmi(calculator_name_2)
    for i in range(n):
        res_2 = est_2.estimate(var1=source_1[1:], var2=target[1:],
                               conditional=target[:-1], opts=opts)
    print('New result {0:.4f} nats; expected:{1:.4f}'.format(res_2,
                                                             expected_res))

filename = 'profile_stats_old.stats'
cProfile.run('test_old()', filename)
filename = 'profile_stats_new.stats'
cProfile.run('test_new()', filename)
stats_old = pstats.Stats('profile_stats_old.stats')
stats_new = pstats.Stats('profile_stats_new.stats')
stats_old.strip_dirs()  # clean up filenames for the report
stats_old.sort_stats('cumulative')
stats_old.print_stats()
stats_new.strip_dirs()  # clean up filenames for the report
stats_new.sort_stats('cumulative')
stats_new.print_stats()
print('Total time old: {0:.5f} s\nTotal time new: {1:.5f} s'.format(
                                    stats_old.total_tt, stats_new.total_tt))


#cProfile.run('a=old(data, idx, cv)')
#cProfile.run('b=new(data, idx, cv)')
#a=old(data, idx, cv)[0]
#b=new(data, idx, cv)
#assert((a == b).all()), 'Results diverged!'