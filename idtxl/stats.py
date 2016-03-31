# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import sys
import copy as cp
import numpy as np
import utils as utils

VERBOSE = False


def network_fdr(results, alpha=0.05):
    """Perform FDR-correction on results of network inference.

    Perform correction of the false discovery rate (FDR) for all significant
    links obtained from network inference. Reference:

        Genovese, C.R., Lazar, N.A., & Nichols, T. (2002). Thresholding of
        statistical maps in functional neuroimaging using the false discovery
        rate. Neuroimage, 15(4), 870-878.

    Args:
        results : dict
            network inference results where each dict entry represents results
            for one target node
        alpha : float [optional]
            critical alpha value for statistical significance

    Returns:
        dict
            input results structure pruned of non-significant links.
    """
    # Get candidates and their test results from the results dictionary, i.e.,
    # collect results over targets.
    pval = np.arange(0)
    target_idx = np.arange(0).astype(int)
    cands = []
    for target in results.keys():
        if not results[target]['omnibus_sign']:  # skip if not significant
            continue
        n_sign = results[target]['cond_sources_pval'].size
        pval = np.append(pval, results[target]['cond_sources_pval'])
        target_idx = np.append(target_idx,
                               np.ones(n_sign) * target).astype(int)
        cands = cands + results[target]['conditional_sources']

    if pval.size == 0:
        print('No links in final results. Return ...')
        return

    # Sort all p-values in ascending order.
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

    # Compare data to threshold.
    sign = pval <= thresh
    first_false = np.where(np.invert(sign))[0][0]
    sign[first_false:] = False  # to avoid false positives due to equal pvals

    # Go over list of all candidates and remove them from the results dict.
    sign = sign[sort_idx]
    for s in range(sign.shape[0]):
        if sign[s]:
            continue
        else:  # remove non-significant candidate and it's p-value from results
            t = target_idx[s]
            cand = cands[s]
            cand_ind = results[t]['conditional_sources'].index(cand)
            results[t]['conditional_sources'].pop(cand_ind)
            results[t]['cond_sources_pval'] = np.delete(
                                    results[t]['cond_sources_pval'], cand_ind)
            results[t]['conditional_full'].pop(
                                    results[t]['conditional_full'].index(cand))
    return results


def omnibus_test(analysis_setup, data, opts):
    """Perform an omnibus test on identified conditional variables.

    Test the joint information transfer from all identified sources to the
    current value conditional on candidates in the target's past. To test for
    significance, this is repeated for shuffled realisations of the sources.
    The distribution of values from shuffled data is then used as test
    distribution.

    Args:
        analysis_setup : Multivariate_te instance
            information on the current analysis
        data : Data instance
            raw data
        opts : dict [optional]
            parameters for statistical testing, can contain
            'n_perm_omnibus' - number of permutations (default=500)
            'alpha_omnibus' - critical alpha level (default=0.05)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    try:
        n_permutations = opts['n_perm_omnibus']
    except KeyError:
        n_permutations = 3  # 200
    try:
        alpha = opts['alpha_omnibus']
    except KeyError:
        alpha = 0.05
    print('no. target sourcesces: {0}, no. sources: {1}'.format(
                                    len(analysis_setup.conditional_target),
                                    len(analysis_setup.conditional_sources)))

    # Create temporary variables b/c realisations for sources and targets are
    # created on the fly, which is costly, so we want to re-use them after
    # creation. (This does not apply to the current value realisations).
    cond_source_realisations = analysis_setup._conditional_sources_realisations
    cond_target_realisations = analysis_setup._conditional_target_realisations
    te_orig = analysis_setup._cmi_calculator.estimate(
                                    cond_source_realisations,
                                    analysis_setup._current_value_realisations,
                                    cond_target_realisations,
                                    analysis_setup.options)

    # Check if n_replications is high enough to allow for the requested number
    # of permutations.
    if np.math.factorial(data.n_replications) > n_permutations:
        permute_over_replications = True
    else:
        permute_over_replications = False

    # Create the surrogate distribution by permuting the conditional sources.
    surr_distribution = np.zeros(n_permutations)
    for perm in range(n_permutations):
        if permute_over_replications:
            surr_cond_real = data.permute_data(
                                        analysis_setup.current_value,
                                        analysis_setup.conditional_sources)[0]
        else:
            surr_cond_real = _permute_realisations(
                                            cond_source_realisations,
                                            analysis_setup._replication_index)
        surr_distribution[perm] = analysis_setup._cmi_calculator.estimate(
                                    surr_cond_real,
                                    analysis_setup._current_value_realisations,
                                    cond_target_realisations,
                                    analysis_setup.options)
    [significance, pvalue] = _find_pvalue(te_orig, surr_distribution, alpha)
    return significance, pvalue


def max_statistic(analysis_setup, data, candidate_set, te_max_candidate,
                  opts=None):
    """Perform maximum statistics for one candidate source.

    Test if a transfer entropy value is significantly bigger than the maximum
    values obtained from surrogates of all remanining candidates.

    Args:
        analysis_setup : Multivariate_te instance
            information on the current analysis
        data : Data instance
            raw data
        candidate_set : list of tuples
            list of indices of remaning candidates
        te_max_candidate : float
            transfer entropy value to be tested
        opts : dict [optional]
            parameters for statistical testing, can contain
            'n_perm_max_stat' - number of permutations (default=500)
            'alpha_max_stat' - critical alpha level (default=0.05)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    try:
        n_perm = opts['n_perm_max_stat']
    except KeyError:
        n_perm = 3  # 200
    try:
        alpha = opts['alpha_max_stat']
    except KeyError:
        alpha = 0.05

    test_set = cp.copy(candidate_set)
    assert(test_set), 'The test set is empty.'

    surr_table = _create_surrogate_table(analysis_setup, data, test_set,
                                         n_perm)
    max_distribution = _find_table_max(surr_table)
    [significance, pvalue] = _find_pvalue(te_max_candidate, max_distribution,
                                          alpha)
    return significance, pvalue


def max_statistic_sequential(analysis_setup, data, opts=None):
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
            information on the current analysis
        data : Data instance
            raw data
        opts : dict [optional]
            parameters for statistical testing, can contain
            'n_perm_max_seq' - number of permutations (default=500)
            'alpha_max_seq' - critical alpha level (default=0.05)

    Returns:
        numpy array, bool
            statistical significance of each source
        numpy array, float
            the test's p-values for each source
        numpy array, float
            TE values for individual sources
    """
    try:
        n_permutations = opts['n_perm_max_seq']
    except KeyError:
        n_permutations = 3  # 200
    try:
        alpha = opts['alpha_max_seq']
    except KeyError:
        alpha = 0.05

    # Calculate TE for each candidate in the conditional source set and sort
    # TE values.
    individual_te = np.empty(len(analysis_setup.conditional_sources))
    i = 0
    for conditional in analysis_setup.conditional_sources:
        [temp_cond, temp_cand] = analysis_setup._separate_realisations(
                                            analysis_setup.conditional_sources,
                                            conditional)
        individual_te[i] = analysis_setup._cmi_calculator.estimate(
                                    temp_cand,
                                    analysis_setup._current_value_realisations,
                                    temp_cond)
        i += 1
    conditional_order = np.argsort(individual_te)
    individual_te_sorted = individual_te
    individual_te_sorted.sort()

    # Create a surrogate table and sort it.
    surr_table = _create_surrogate_table(analysis_setup, data,
                                         analysis_setup.conditional_sources,
                                         n_permutations)
    max_distribution = _sort_table_max(surr_table)

    # Compare each TE value with the distribution of the same rank.
    significance = np.zeros(individual_te.shape[0]).astype(bool)
    pvalue = np.zeros(individual_te.shape[0])
    for c in range(individual_te.shape[0]):
        [s, v] = _find_pvalue(individual_te_sorted[c],
                              max_distribution[c, ], alpha)
        significance[c] = s
        pvalue[c] = v
        if not s:  # break as soon as a candidate is no longer significant
            break

    # Get back original order and return results.
    significance = significance[conditional_order]
    pvalue = pvalue[conditional_order]
    return significance, pvalue, individual_te


def min_statistic(analysis_setup, data, candidate_set, te_min_candidate,
                  opts=None):
    """Perform minimum statistics for one candidate source.

    Test if a transfer entropy value is significantly bigger than the minimum
    values obtained from surrogates of all remanining candidates.

    Args:
        analysis_setup : Multivariate_te instance
            information on the current analysis
        data : Data instance
            raw data
        candidate_set : list of tuples
            list of indices of remaning candidates
        te_min_candidate : float
            transfer entropy value to be tested
        opts : dict [optional]
            parameters for statistical testing, can contain
            'n_perm_min_stat' - number of permutations (default=500)
            'alpha_min_stat' - critical alpha level (default=0.05)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    try:
        n_perm = opts['n_perm_min_stat']
    except KeyError:
        n_perm = 3  # 200
    try:
        alpha = opts['alpha_min_stat']
    except KeyError:
        alpha = 0.05

    test_set = cp.copy(candidate_set)
    assert(test_set), 'The test set is empty.'

    surr_table = _create_surrogate_table(analysis_setup, data, test_set,
                                         n_perm)
    min_distribution = _find_table_min(surr_table)
    [significance, pvalue] = _find_pvalue(te_min_candidate, min_distribution,
                                          alpha)
    return significance, pvalue


def _create_surrogate_table(analysis_setup, data, idx_test_set, n_perm):
    """Create a table of surrogate transfer entropy values.

    Calculate transfer entropy between surrogates for each source in the test
    set and the target in the analysis setup using the current conditional in
    the analysis setup.

    Args:
        analysis_setup : Multivariate_te instance
            information on the current analysis
        data : Data instance
            raw data
        idx_test_set : list of tuples
            list od indices indicating samples to be used as sources
        n_perm : int [optional]
            number of permutations for testing (default=500)

    Returns:
        numpy array
            surrogate TE values, dimensions: (length test set, number of
            surrogates)
    """
    # Check if n_replications is high enough to allow for the requested number
    # of permutations. If not permute samples over time
    if np.math.factorial(data.n_replications) > n_perm:
        permute_over_replications = True
    else:
        permute_over_replications = False
        try:
            perm_range = analysis_setup.options['perm_range']
        except KeyError:
            perm_range = 'max'

    # Create surrogate table.
    if VERBOSE:
        print('create surrogates table')
    surr_table = np.zeros((len(idx_test_set), n_perm))
    current_value_realisations = analysis_setup._current_value_realisations
    idx_c = 0
    for candidate in idx_test_set:
        if VERBOSE:
            print('\tcand. {0}, n_perm: {1} -    '.format(candidate, n_perm),
                  end='')
        for perm in range(n_perm):
            # Check the permutation type for the current candidate.
            if permute_over_replications:
                surr_candidate_realisations = data.permute_data(
                                                analysis_setup.current_value,
                                                [candidate])[0]
            else:
                [real, repl_idx] = data.get_realisations(
                                                analysis_setup.current_value,
                                                [candidate])
                surr_candidate_realisations = _permute_realisations(real,
                                                                    repl_idx,
                                                                    perm_range)
            surr_table[idx_c, perm] = analysis_setup._cmi_calculator.estimate(
                        surr_candidate_realisations,
                        current_value_realisations,
                        analysis_setup._conditional_realisations,
                        analysis_setup.options)
            if VERBOSE:
                print('\b\b\b{num:03d}'.format(num=perm + 1), end='')
                sys.stdout.flush()
        if VERBOSE:
            print(' ')
        idx_c += 1

    return surr_table


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


def _permute_realisations(realisations, replication_idx, perm_range='max'):
    """Permute realisations in time within each replication.

    Permute realisations in time but within each replication. This is the
    fall-back option if the number of replications is too small to allow a
    sufficient number of permutations for the generation of surrogate data. If
    no permutation range is given, samples are randomly permuted over the
    whole replication, i.e., over all time indices in the replication. If a
    permutation range is given, samples are shuffled within blocks of length
    permutation range.

    Args:
        realisations : numpy array
            shape[0] realisations of shape[1] variables
        replication_idx : numpy array
            index of replication a realisation came from
        perm_range : int or 'max'
            range over which realisations are permuted, if 'max' realisations
            are permuted over the whole replication, otherwise realisations are
            permuted over blocks of length perm_range

    Returns:
        numpy array
            realisations permuted over time
    """
    realisations_perm = cp.copy(realisations)
    n_per_repl = sum(replication_idx == 0)
    if type(perm_range) is not str:
        assert (perm_range > 1), ('Permutation range has to be larger than 1',
                                  'otherwise there is nothing to permute.')
    else:
        if perm_range == 'max':
            perm_range = n_per_repl
        else:
            raise ValueError('Unkown value for "perm_range": {0}'.format(
                perm_range))
    assert (replication_idx.shape[0] == realisations.shape[0]), (
            'Array "replication" index must have as many entries as the first '
            'dimension of array "realisations".')

    assert (n_per_repl >= perm_range), ('Not enough realisations per '
                                        'replication ({0}) to allow for the '
                                        'requested "perm_range" of {1}.'
                                        .format(n_per_repl, perm_range))

    # Create a permutation of the data that respects the requested permutation
    # range and can be applied to the realisations from each replication in
    # turn.
    if perm_range == n_per_repl:  # permute all realisations in one replication
        perm = np.random.permutation(n_per_repl)
    else:  # build a permutation that permutes only within the perm_range
        perm = np.empty(n_per_repl, dtype=int)
        remainder = n_per_repl % perm_range
        i = 0
        for p in range(n_per_repl // perm_range):
            perm[i:i + perm_range] = np.random.permutation(perm_range) + i
            i += perm_range
        if remainder > 0:
            perm[-remainder:] = np.random.permutation(remainder) + i

    # Apply the permutation to data from each replication, individually.
    for replication in range(max(replication_idx) + 1):
        mask = replication_idx == replication
        d = realisations_perm[mask, :]
        realisations_perm[mask, :] = d[perm, :]

    return realisations_perm


def _find_pvalue(statistic, distribution, alpha=0.05, tail='one'):
    """Find p-value of a test statistic under some distribution.

    Args:
        statistic: numeric
            value to be tested against distribution
        distribution: numpy array
            1-dimensional distribution of values, test distribution
        alpha: float
            critical alpha level for statistical significance
        tail: str
            'one' or 'two' for one-/two-tailed testing

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    assert(distribution.ndim == 1)
    assert(1.0 / distribution.shape[0] < alpha), ('The numper of permutations '
                                                  'is to small ({0}) to test '
                                                  'the requested alpha level '
                                                  '({1}).'.format(
                                                       distribution.shape[0],
                                                       alpha))
#    if (1.0 / distribution.shape[0] >= alpha):
#        print('The numper of permutations is to small ({0}) to test the '
#              'requested alpha level ({1}).'.format(distribution.shape[0],
#                                                    alpha))
    if tail == 'one':
        pvalue = sum(distribution > statistic) / distribution.shape[0]
    elif tail == 'two':
        p_bigger = sum(distribution > statistic) / distribution.shape[0]
        p_smaller = sum(distribution < statistic) / distribution.shape[0]
        pvalue = min(p_bigger, p_smaller)
        alpha = alpha / 2
    else:
        raise ValueError(('Unkown value for "tail" (should be "one" or "two"):'
                          ' {0}'.format(tail)))

    # If the statistic is larger than all values in the test distribution, set
    # the p-value to the smallest possible value 1/n_perm.
    if pvalue == 0:
        pvalue = 1.0 / distribution.shape[0]
    significance = pvalue < alpha

    return significance, pvalue
