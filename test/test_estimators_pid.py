# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:51:56 2016

@author: wibral
"""

import numpy as np
from idtxl.estimators_pid import pid

def test_logical_xor():

    # logical AND
    n = 1000
    alph = 2
    s1 = np.random.randint(0, alph, n)
    s2 = np.random.randint(0, alph, n)
    target = np.logical_xor(s1, s2).astype(int)
    cfg = {
        'alph_s1': 2,
        'alph_s2': 2,
        'alph_t': 2,
        'jarpath': 'infodynamics.jar',
        'iterations': 1000
    }
    print('Testing PID estimator on binary AND, pointsset size{0}, iterations: {1}'.format(
                                                        n, cfg['iterations']))
    [est, opt] = pid(s1, s2, target, cfg)
    print("----Results: ----")
    print("unq_s1: {0}".format(est['unq_s1']))
    print("unq_s2: {0}".format(est['unq_s2']))
    print("shd_s1s2: {0}".format(est['shd_s1s2']))
    print("syn_s1s2: {0}".format(est['syn_s1s2']))
    assert 0.9 < est['syn_s1s2'] <=1.1, 'incorrect synergy: {0}, expected was {1}'.format(est['syn_s1s2'], 0.98)

if __name__ == '__main__':
    test_logical_xor()