"""Provide unit tests for PID estimators.

Created on Mon Apr 11 21:51:56 2016

@author: wibral
"""
import numpy as np
from idtxl.set_estimator import Estimator_pid


def test_logical_xor():
    """Test PID estimation with Sydney estimator on binary XOR."""
    # logical AND
    n = 1000
    alph = 2
    s1 = np.random.randint(0, alph, n)
    s2 = np.random.randint(0, alph, n)
    target = np.logical_xor(s1, s2).astype(int)
    analysis_opts = {
        'alph_s1': 2,
        'alph_s2': 2,
        'alph_t': 2,
        'max_unsuc_swaps_row_parm': 3,
        'num_reps': 63,
        'max_iters': 10000
    }
    print('Testing PID estimator on binary XOR, pointsset size{0}, iterations:'
          ' {1}'.format(n, analysis_opts['max_iters']))
    pid_est = Estimator_pid('pid_sydney')
    pid_est = pid_est.estimate(s1, s2, target, analysis_opts)
    print("----Results: ----")
    print("unq_s1: {0}".format(pid_est['unq_s1']))
    print("unq_s2: {0}".format(pid_est['unq_s2']))
    print("shd_s1_s2: {0}".format(pid_est['shd_s1_s2']))
    print("syn_s1_s2: {0}".format(pid_est['syn_s1_s2']))
    assert 0.9 < pid_est['syn_s1_s2'] <= 1.1, (
            'incorrect synergy: {0}, expected was {1}'.format(
                    pid_est['syn_s1s2'], 0.98))

if __name__ == '__main__':
    test_logical_xor()
