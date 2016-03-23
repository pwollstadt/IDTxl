# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:32:46 2016

@author: patricia
"""
import numpy as np
from set_estimator import Estimator_cmi


def compare_networks(network_1, network_2, data_1, data_2, options):
    """Perform statistical comparison between two networks.

    Arguments:
        network_1 : dict
            results from network inference
        network_2 : dict
            results from network inference
        options : dict
            options for statistical comparison of networks

    Returns:

    """
    cmi_calculator = Estimator_cmi(options['cmi_calc_name'])
    try:
        n_perm = options['n_perm_comp']
    except KeyError:
        n_permutations = 10

    # generate union of links
    union = _create_union(network_1, network_2)

    # calculate statistic
    cmi_diff = _calculate_cmi_diff(data_1, data_2, union, cmi_calculator)

    # generate surrogate distribution
    surr_distribution = np.zeros(n_permutations)
    for p in n_permutations:
        surr_distribution[p] = _calculate_cmi_diff(data_1, data_2, union,
                                                   cmi_calculator)




def _create_union(network_1, network_2):
    targets = np.array([i for i in res_0.keys()] + [i for i in res_1.keys()])
    targets = np.unique(targets)
    union = {}
    for t in targets:
        union[t] = {}
        try:
            cond_src_0 = res_0[t]['conditional_sources']
            cond_tgt_0 = res_0[t]['conditional_target']
        except KeyError:
            cond_src_0 = []
            cond_tgt_0 = []
        try:
            cond_src_1 = res_1[t]['conditional_sources']
            cond_tgt_1 = res_1[t]['conditional_target']
        except KeyError:
            cond_src_1 = []
            cond_tgt_1 = []

        try:
            union[t]['conditional_sources'] += cond_src_0
            union[t]['conditional_target'] += cond_tgt_0
        except KeyError:
            union[t]['conditional_sources'] = cond_src_0
            union[t]['conditional_target'] = cond_tgt_0
        for c in cond_src_1:
            try:
                union[t]['conditional_sources'].index(c)
            except ValueError:
                union[t]['conditional_sources'].append(c)
        for c in cond_tgt_1:
            try:
                union[t]['conditional_target'].index(c)
            except ValueError:
                union[t]['conditional_target'].append(c)

    return union


def _calculate_cmi_diff(data_1, data_2, union, cmi_calculator):
    """
    """
    cmi = 0.2
    return cmi

if __name__ == 'main':
    r0 = {
        'conditional_sources': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 0)],
        'conditional_target': [(0, 1), (0, 2), (0, 3)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                             (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    r1 = {
        'conditional_sources': [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
        'conditional_target': [(1, 0), (1, 1)],
        'conditional_full': [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1),
                             (3, 2)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
        }
    r2 = {
        'conditional_sources': [],
        'conditional_target': [(3, 0), (3, 1)],
        'conditional_full': [(3, 0), (3, 1)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res_0 = {
        0: r0,
        1: r1,
        2: r2
    }

    r0 = {
        'conditional_sources': [(2, 1), (2, 0)],
        'conditional_target': [(0, 1), (0, 2), (0, 4)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    r1 = {
        'conditional_sources': [(2, 0), (2, 1), (3, 2), (3, 0)],
        'conditional_target': [(1, 1), (1, 2), (1, 3)],
        'conditional_full': [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (3, 2),
                             (3, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
        }
    r3 = {
        'conditional_sources': [(2, 0), (2, 1), (1, 0), (1, 1), (1, 2)],
        'conditional_target': [(3, 1), (3, 3)],
        'conditional_full': [(3, 1), (3, 3), (2, 0), (2, 1), (1, 0), (1, 1),
                             (1, 2)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res_1 = {
        0: r0,
        1: r1,
        3: r3
    }

    data_1 = Data()
    data_1.generate_mute_data(100, 5)
    data_2 = Data()
    data_2.generate_mute_data(100, 5)

    res = compare_networks(res_1, res_2, data_1, data_2, options)
