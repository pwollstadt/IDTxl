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


r1 = {
    'conditional_sources': [(0, 1), (0, 2), (0, 3), (2, 1), (2, 0)],
    'conditional_full': [(0, 1), (0, 2), (0, 3), (2, 1), (2, 0)],
    'omnibus_sign': True,
    'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
    }
r2 = {
    'conditional_sources': [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
    'conditional_full': [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
    'omnibus_sign': True,
    'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
    }
r3 = {
    'conditional_sources': [],
    'conditional_full': [(3, 0), (3, 1)],
    'omnibus_sign': False,
    'cond_sources_pval': None
    }
res = {
    1: r1,
    2: r2,
    3: r3
}


def network_fdr(results, alpha=0.05):
    # Get p-values from results.
    pval = np.arange(0)
    target_idx = np.arange(0).astype(int)
    cands = []
    for target in results.keys():
        if not results[target]['omnibus_sign']:
            continue
        n_sign = results[target]['cond_sources_pval'].size
        pval = np.append(pval, results[target]['cond_sources_pval'])
        target_idx = np.append(target_idx,
                               np.ones(n_sign) * target).astype(int)
        cands = cands + results[target]['conditional_sources']
    sort_idx = np.argsort(pval)
    pval.sort()

    # Calculate threshold (exact or by approximating the harmonic sum).
    n = pval.size
    if n < 1000:
        thresh = ((np.arange(1, n + 1) / n) * alpha /
                  sum(1 / np.arange(1, n + 1)))
    else:
        thresh = ((np.arange(1, n + 1) / n) * alpha /
                  (np.log(n) + np.e))  # aprx. harmonic sum with Euler's number

    # Compare data to threshold and prepare output:
    sign = pval <= thresh
    first_false = np.where(sign == False)[0][0]
    sign[first_false:] = False  # to avoid false positives due to equal pvals
    sign = sign[sort_idx]
    for s in range(sign.shape[0]):
        if sign[s]:
            continue
        else:
            # Remove non-significant candidate and its p-value from results.
            t = target_idx[s]
            cand = cands[s]
            cand_ind = results[t]['conditional_sources'].index(cand)
            results[t]['conditional_sources'].pop(cand_ind)
            np.delete(results[t]['cond_sources_pval'], cand_ind)
            results[t]['conditional_full'].pop(
                                    results[t]['conditional_full'].index(cand))
    return results

res_pruned = network_fdr(res)

#cProfile.run('a=old(data, idx, cv)')
#cProfile.run('b=new(data, idx, cv)')
#a=old(data, idx, cv)[0]
#b=new(data, idx, cv)
#assert((a == b).all()), 'Results diverged!'