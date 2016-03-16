# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import sys
import copy as cp
import numpy as np
import utils


def omnibus_test(analysis_setup, data, n_permutations=3):
    """Perform an omnibus test on identified conditional variables.

    Test the joint information transfer from all identified sources to the
    current value conditional on candidates in the target's past. To test for
    significance, this is repeated for shuffled realisations of the sources.
    The distribution of values from shuffled data is then used as test
    distribution.

    Args:
        analysis_setup : Multivariate_te instance
            instance holding temporary data to be tested
        data: Data instance
            raw data
        n_permutations : int [optional]
            number of permutations for testing

    Returns:
        bool
            True if test significant
        float
            the test's p-value
    """
    print('no. target sourcesces: {0}, no. sources: {1}'.format(
                                    len(analysis_setup.conditional_target),
                                    len(analysis_setup.conditional_sources)))

    # TODO I'm doing this, b/c reals for sources and targets are created on the
    # fly, which is costly, this sucks b/c current_value_realisations are the
    # actual data
    cond_source_realisations = analysis_setup._conditional_sources_realisations
    cond_target_realisations = analysis_setup._conditional_target_realisations
    te_orig = analysis_setup._cmi_estimator.estimate(
                                    cond_source_realisations,
                                    analysis_setup._current_value_realisations,
                                    cond_target_realisations)

    surr_distribution = np.zeros(n_permutations)
    for perm in range(n_permutations):
        surr_conditional_realisations = data.permute_data(
                                        analysis_setup,
                                        analysis_setup.conditional_sources)
        surr_distribution[perm] = analysis_setup._cmi_estimator.estimate(
                                    surr_conditional_realisations,
                                    analysis_setup._current_value_realisations,
                                    cond_target_realisations)
    [significance, pvalue] = _find_pvalue(te_orig, surr_distribution)
    return significance, pvalue


def max_statistic(analysis_setup, data, candidate_set, te_max_candidate,
                  n_permutations=3):
    """Perform maximum statistics for one candidate source.

    Test if a transfer entropy value is significantly bigger than the maximum
    values obtained from surrogates of all remanining candidates.

    Args:
        analysis_setup : Multivariate_te instance
            instance holding temporary data to be tested
        data: Data instance
            raw data
        candidate_set : list of int
            indices of remaning candidates
        te_max_candidate : float
            transfer entropy value to be tested
        n_permutations : int [optional]
            number of permutations for testing

    Returns:
        bool
            True if test significant
        float
            the test's p-value
    """
    test_set = cp.copy(candidate_set)

    if not test_set:  # TODO this is an interim thing -> decide what to do
        return True

    stats_table = _create_surrogate_table(analysis_setup, data, test_set,
                                          n_permutations)
    max_distribution = _find_table_max(stats_table)
    [significance, pvalue] = _find_pvalue(te_max_candidate, max_distribution)
    # return np.random.rand() > 0.5
    # return True
    return significance, pvalue


def max_statistic_sequential(analysis_setup, data, n_permutations=5):
    """Perform sequential maximum statistics for a set of candidate sources.

    Test if sorted transfer entropy (TE) values are significantly bigger than
    their respective counterpart obtained from surrogates of all remanining
    candidates: test if the biggest TE is bigger than the distribution
    of biggest TE surrogate values; test if the 2nd biggest TE is bigger than
    the distribution of 2nd biggest surrogate TE values; ...
    Stop comparison if a TE value is non significant, all smaller values are
    considered non-significant as well.

    Args:
        analysis_setup : Multivariate_te instance
            instance holding temporary data to be tested
        data: Data instance
            raw data
        n_permutations : int [optional]
            number of permutations for testing

    Returns:
        bool
            True if test significant
        float
            the test's p-value
    """
    conditional_te = np.empty(len(analysis_setup.conditional_full))
    i = 0
    for conditional in analysis_setup.conditional_full:  # TODO only test source candidates
        [temp_cond, temp_cand] = analysis_setup._remove_realisation(
                                            analysis_setup.conditional_full,
                                            conditional)
        conditional_te[i] = analysis_setup._cmi_estimator.estimate(
                                    temp_cand,
                                    analysis_setup._current_value_realisations,
                                    temp_cond)
        i += 1
    conditional_order = np.argsort(conditional_te)
    conditional_te_sorted = conditional_te
    conditional_te_sorted.sort()

    # TODO not sure about this, because the surrogate table also contains the
    # candidate we're testing
    stats_table = _create_surrogate_table(analysis_setup, data,
                                          analysis_setup.conditional_full,
                                          n_permutations)
    max_distribution = _sort_table_max(stats_table)

    significance = np.zeros(len(analysis_setup.conditional_full)).astype(bool)
    pvalue = np.zeros(len(analysis_setup.conditional_full))
    for c in range(len(analysis_setup.conditional_full)):
        [s, v] = _find_pvalue(conditional_te_sorted[c],
                              max_distribution[c, ])
        significance[c] = s
        pvalue[c] = v

        if not s:
            break

    # get back original order
    significance = significance[conditional_order]
    pvalue = pvalue[conditional_order]

    # return True
    return significance, pvalue


def min_statistic(analysis_setup, data, candidate_set, te_min_candidate,
                  n_permutations=3):
    """Perform minimum statistics for one candidate source.

    Test if a transfer entropy value is significantly bigger than the minimum
    values obtained from surrogates of all remanining candidates.

    Args:
        analysis_setup : Multivariate_te instance
            instance holding temporary data to be tested
        data: Data instance
            raw data
        candidate_set : list of int
            indices of remaning candidates
        te_min_candidate : float
            transfer entropy value to be tested
        n_permutations : int [optional]
            number of permutations for testing

    Returns:
        bool
            True if test significant
        float
            the test's p-value
    """
    test_set = cp.copy(candidate_set)

    if not test_set:  # TODO this is an interim thing -> decide what to do
        return True

    stats_table = _create_surrogate_table(analysis_setup, data, test_set,
                                          n_permutations)
    min_distribution = _find_table_min(stats_table)
    [significance, pvalue] = _find_pvalue(te_min_candidate, min_distribution)
    # return np.random.rand() > 0.5
    return significance, pvalue


def _create_surrogate_table(analysis_setup, data, idx_test_set,
                            n_permutations):
    """Create a table of surrogate transfer entropy values.

    Calculate transfer entropy between surrogates for each source in the test
    set and the target in the analysis setup using the current conditional in
    the analysis setup.

    Args:
        analysis_setup : Multivariate_te instance
            instance holding temporary data to be tested
        data: Data instance
            raw data
        idx_test_set : list of tuples
            samples to be used as sources
        n_permutations : int [optional]
            number of permutations for testing

    Returns:
        numpy array
            surrogate TE values, has dimensions (len(idx_test_set),
            n_permutations)
    """
    stats_table = np.zeros((len(idx_test_set), n_permutations))
    current_value_realisations = analysis_setup._current_value_realisations
    idx_c = 0
    print('fill surrogates table')
    for candidate in idx_test_set:
        print('\tcand. {0}, n_perm: {1}. Done:   '.format(candidate,
                                                          n_permutations),
              end='')
        for perm in range(n_permutations):
            surr_candidate_realisations = data.permute_data(analysis_setup,
                                                            [candidate])
            stats_table[idx_c, perm] = analysis_setup._cmi_estimator.estimate(
                        surr_candidate_realisations,
                        current_value_realisations,
                        analysis_setup._conditional_realisations)  # TODO remove current candidate from this
            print('\b\b{num:02d}'.format(num=perm + 1), end='')
            sys.stdout.flush()
        print(' ')
        idx_c += 1

    return stats_table


def _find_table_max(table):
    """Find maximum for each column of a table."""
    return np.max(table, axis=0)


def _find_table_min(table):
    """Find minimum for each column of a table."""
    return np.min(table, axis=0)


def _sort_table_max(table):
    """Sort each column in a table in ascending order."""
    for permutation in range(table.shape[1]):
        table[:, permutation].sort()
    return table


def _sort_table_min(table):
    """Sort each column in a table in descending order."""
    table_sorted = np.empty(table.shape)
    for permutation in range(0, table.shape[1]):
        table_sorted[:, permutation] = utils.sort_descending(
                                            table[:, permutation])
    return table_sorted


def _find_pvalue(statistic, distribution, alpha=0.05, tail='one'):
    """Find p-value of a test statistic under some distribution.

    Args:
        statistic : numeric
            value to be tested against distribution
        distribution : numpy array
            distribution under which statistic is tested (1-dimensional)
        alpha : float
            critical alpha level (has to be < 1)
        tail : str
            'one' for one-tailed, 'two' for two-tailed testing

    Returns:
        bool
            True if test significant
        float
            the test's p-value
    """
    assert(distribution.ndim == 1)
    assert(alpha <= 1)
    if tail == 'one':
        pvalue = sum(distribution > statistic) / distribution.shape[0]
    elif tail == 'two':
        p_bigger = sum(distribution > statistic) / distribution.shape[0]
        p_smaller = sum(distribution < statistic) / distribution.shape[0]
        pvalue = min(p_bigger, p_smaller)
    else:
        raise ValueError('Tail of a test has to be set to "one" or "two".')
    significance = pvalue < alpha
    return significance, pvalue
