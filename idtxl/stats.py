"""Provide statistics functions.

Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import copy as cp
import numpy as np
from . import idtxl_utils as utils

VERBOSE = True


def network_fdr(results, alpha=0.05, correct_by_target=True):
    """Perform FDR-correction on results of network inference.

    Perform correction of the false discovery rate (FDR) after network
    analysis. FDR correction can either be applied at the target level
    (by correcting omnibus p-values) or at the single-link level (by correcting
    p-values of individual links between single samples and the target).
    Reference for FDR correction:

    Genovese, C.R., Lazar, N.A., & Nichols, T. (2002). Thresholding of
    statistical maps in functional neuroimaging using the false discovery
    rate. Neuroimage, 15(4), 870-878.

    Args:
        results : dict
            network inference results where each dict entry represents results
            for one target node
        alpha : float [optional]
            critical alpha value for statistical significance (default=0.05)
        correct_by_target : bool
            if true p-values are corrected on the target level and on the
            single-link level otherwise (default=True)

    Returns:
        dict
            input results structure pruned of non-significant links.
    """
    # Make a copy that can be changed and returned after testing.
    res = cp.copy(results)

    # Get candidates and their test results from the results dictionary, i.e.,
    # collect results over targets.
    pval = np.arange(0)
    target_idx = np.arange(0).astype(int)
    cands = []
    if correct_by_target:  # correct p-value of whole target (all candidates)
        for target in res.keys():
            if not res[target]['omnibus_sign']:  # skip if not significant
                continue
            pval = np.append(pval, res[target]['omnibus_pval'])
            target_idx = np.append(target_idx, target)
    else:   # correct p-value of single candidates
        for target in res.keys():
            if not res[target]['omnibus_sign']:  # skip if not significant
                continue
            n_sign = res[target]['cond_sources_pval'].size
            pval = np.append(pval, res[target]['cond_sources_pval'])
            target_idx = np.append(target_idx,
                                   np.ones(n_sign) * target).astype(int)
            cands = cands + res[target]['selected_vars_sources']

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
    if np.invert(sign).any():
        first_false = np.where(np.invert(sign))[0][0]
        sign[first_false:] = False  # avoids false positives due to equal pvals

    # Go over list of all candidates and remove them from the results dict.
    sign = sign[sort_idx]
    for s in range(sign.shape[0]):
        if sign[s]:
            continue
        else:  # remove non-significant candidate and it's p-value from results
            if correct_by_target:
                t = target_idx[s]
                res[t]['selected_vars_full'] = res[t]['selected_vars_target']
                res[t]['cond_sources_te'] = None
                res[t]['cond_sources_pval'] = None
                res[t]['selected_vars_sources'] = []
                res[t]['omnibus_pval'] = 1
                res[t]['omnibus_sign'] = False
            else:
                t = target_idx[s]
                cand = cands[s]
                cand_ind = res[t]['selected_vars_sources'].index(cand)
                res[t]['selected_vars_sources'].pop(cand_ind)
                res[t]['cond_sources_pval'] = np.delete(
                                    res[t]['cond_sources_pval'], cand_ind)
                res[t]['cond_sources_te'] = np.delete(
                                    res[t]['cond_sources_te'], cand_ind)
                res[t]['selected_vars_full'].pop(
                                    res[t]['selected_vars_full'].index(cand))
    return res


def omnibus_test(analysis_setup, data, opts=None):
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

            - 'n_perm_omnibus' - number of permutations (default=500)
            - 'alpha_omnibus' - critical alpha level (default=0.05)
            - 'perm_range' - permutation range if permutation over samples is
              used to create surrogates (default='max')

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    if opts is None:
        opts = {}
    n_permutations = opts.get('n_perm_omnibus', 21)
    alpha = opts.get('alpha_omnibus', 0.05)
    perm_range = opts.get('perm_range_omnibus', 'max')
    print('no. target sources: {0}, no. sources: {1}'.format(
                                    len(analysis_setup.selected_vars_target),
                                    len(analysis_setup.selected_vars_sources)))

    # Create temporary variables b/c realisations for sources and targets are
    # created on the fly, which is costly, so we want to re-use them after
    # creation. (This does not apply to the current value realisations).
    cond_source_realisations = (analysis_setup
                                ._selected_vars_sources_realisations)
    cond_target_realisations = (analysis_setup
                                ._selected_vars_target_realisations)
    te_orig = analysis_setup._cmi_calculator.estimate(
                            var1=cond_source_realisations,
                            var2=analysis_setup._current_value_realisations,
                            conditional=cond_target_realisations,
                            opts=analysis_setup.options)

    # Create the surrogate distribution by permuting the conditional sources.
    if VERBOSE:
        print('omnibus test, n_perm: {0}'.format(n_permutations))
    '''
    # Calculate TE in parallel for all permutations
    surr_cond_real = np.empty(
        (n_permutations * data.n_realisations(analysis_setup.current_value),
         len(analysis_setup.selected_vars_sources)))
    i_1 = 0
    i_2 = data.n_realisations(analysis_setup.current_value)
    for perm in range(n_permutations):
        if permute_over_replications:
            surr_cond_real[i_1:i_2, ] = data.permute_data(
                                    analysis_setup.current_value,
                                    analysis_setup.selected_vars_sources)[0]
        else:
            surr_cond_real[i_1:i_2, ] = _permute_realisations(
                                            cond_source_realisations,
                                            analysis_setup._replication_index)
        i_1 = i_2
        i_2 += data.n_realisations(analysis_setup.current_value)
        '''
    surr_cond_real = _generate_surrogates(data,
                                          analysis_setup.current_value,
                                          analysis_setup.selected_vars_sources,
                                          n_permutations,
                                          perm_range)

    surr_distribution = analysis_setup._cmi_calculator.estimate_mult(
                            n_chunks=n_permutations,
                            options=analysis_setup.options,
                            re_use=['var2', 'conditional'],
                            var1=surr_cond_real,
                            var2=analysis_setup._current_value_realisations,
                            conditional=cond_target_realisations)
    [significance, pvalue] = _find_pvalue(te_orig, surr_distribution, alpha)
    if VERBOSE:
        if significance:
            print(' -- significant')
        else:
            print(' -- not significant')
    return significance, pvalue, te_orig


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
            parameters for statistical testing, can contain:

            - 'n_perm_max_stat' - number of permutations (default=500)
            - 'alpha_max_stat' - critical alpha level (default=0.05)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
        numpy array
            surrogate table
    """
    if opts is None:
        opts = {}
    n_perm = opts.get('n_perm_max_stat', 21)
    alpha = opts.get('alpha_max_stat', 0.05)
    assert(candidate_set), 'The candidate set is empty.'

    surr_table = _create_surrogate_table(analysis_setup, data, candidate_set,
                                         n_perm)
    max_distribution = _find_table_max(surr_table)
    [significance, pvalue] = _find_pvalue(te_max_candidate, max_distribution,
                                          alpha)
    return significance, pvalue, surr_table


def max_statistic_sequential(analysis_setup, data, opts=None):
    """Perform sequential maximum statistics for a set of candidate sources.

    Test if sorted transfer entropy (TE) values are significantly bigger than
    their respective counterpart obtained from surrogates of all remanining
    candidates: test if the biggest TE is bigger than the distribution
    of biggest TE surrogate values; test if the 2nd biggest TE is bigger than
    the distribution of 2nd biggest surrogate TE values; ...
    Stop comparison if a TE value is non significant, all smaller values are
    considered non-significant as well.

    This function will re-use the surrogate table created in the last min-stats
    round if that table is in the analysis_setup. This saves the complete
    calculation of surrogates for this statistic.

    Args:
        analysis_setup : Multivariate_te instance
            information on the current analysis
        data : Data instance
            raw data
        opts : dict [optional]
            parameters for statistical testing, can contain:

            - 'n_perm_max_seq' - number of permutations
              (default='n_perm_min_stat'|500)
            - 'alpha_max_seq' - critical alpha level (default=0.05)

    Returns:
        numpy array, bool
            statistical significance of each source
        numpy array, float
            the test's p-values for each source
        numpy array, float
            TE values for individual sources
    """
    if opts is None:
        opts = {}
    try:
        n_permutations = opts['n_perm_max_seq']
    except KeyError:
        try:  # use the same n_perm as for min_stats if surr table is reused
            n_permutations = analysis_setup._min_stats_surr_table.shape[1]
        except TypeError:  # is surr table is None, use default
            n_permutations = 500
    alpha = opts.get('alpha_max_seq', 0.05)

    # Calculate TE for each candidate in the conditional source set and sort
    # TE values.
    candidate_realisations = np.empty(
                        (data.n_realisations(analysis_setup.current_value) *
                         len(analysis_setup.selected_vars_sources), 1))
    conditional_realisations = np.empty(
                        (data.n_realisations(analysis_setup.current_value) *
                            len(analysis_setup.selected_vars_sources),
                         len(analysis_setup.selected_vars_full) - 1))
    i_1 = 0
    i_2 = data.n_realisations(analysis_setup.current_value)
    for conditional in analysis_setup.selected_vars_sources:
        [temp_cond, temp_cand] = analysis_setup._separate_realisations(
                                            analysis_setup.selected_vars_full,
                                            conditional)
        if temp_cond is None:
            conditional_realisations = None
        else:
            conditional_realisations[i_1:i_2, ] = temp_cond
        candidate_realisations[i_1:i_2, ] = temp_cand
        i_1 = i_2
        i_2 += data.n_realisations(analysis_setup.current_value)

    individual_te = analysis_setup._cmi_calculator.estimate_mult(
                            n_chunks=len(analysis_setup.selected_vars_sources),
                            options=opts,
                            re_use=['var2'],
                            var1=candidate_realisations,
                            var2=analysis_setup._current_value_realisations,
                            conditional=conditional_realisations)

    selected_vars_order = utils.argsort_descending(individual_te)
    individual_te_sorted = utils.sort_descending(individual_te)

    # Re-use or create surrogate table and sort it, this saves some time
    if (analysis_setup._min_stats_surr_table is not None and
            n_permutations <= analysis_setup._min_stats_surr_table.shape[1]):
        surr_table = analysis_setup._min_stats_surr_table[:, :n_permutations]
        assert len(analysis_setup.selected_vars_sources) == surr_table.shape[0]
    else:
        surr_table = _create_surrogate_table(
                                        analysis_setup,
                                        data,
                                        analysis_setup.selected_vars_sources,
                                        n_permutations)
    max_distribution = _sort_table_max(surr_table)

    # Compare each TE value with the distribution of the same rank, starting
    # with the highest TE.
    significance = np.zeros(individual_te.shape[0]).astype(bool)
    pvalue = np.ones(individual_te.shape[0])
    for c in range(individual_te.shape[0]):
        [s, p] = _find_pvalue(individual_te_sorted[c],
                              max_distribution[c, ], alpha)
        significance[c] = s
        pvalue[c] = p
        if not s:  # break as soon as a candidate is no longer significant
            if VERBOSE:
                print('Stopping sequential max stats at candidate with rank '
                      '{0}.'.format(c))
            break

    # Get back original order and return results.
    significance = significance[selected_vars_order]
    pvalue = pvalue[selected_vars_order]
    return significance, pvalue, individual_te


# TODO opts is part of analysis setup, see mi_stats below
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
            parameters for statistical testing, can contain:

            - 'n_perm_min_stat' - number of permutations (default=500)
            - 'alpha_min_stat' - critical alpha level (default=0.05)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
        numpy array
            surrogate table
    """
    if opts is None:
        opts = {}
    n_perm = opts.get('n_perm_min_stat', 21)
    alpha = opts.get('alpha_min_stat', 0.05)
    assert(candidate_set), 'The candidate set is empty.'

    surr_table = _create_surrogate_table(analysis_setup, data, candidate_set,
                                         n_perm)
    min_distribution = _find_table_min(surr_table)
    [significance, pvalue] = _find_pvalue(te_min_candidate, min_distribution,
                                          alpha)
    return significance, pvalue, surr_table


def mi_against_surrogates(analysis_setup, data, opts=None):
    """Test estimaed mutual information for significance against surrogate data.

    Shuffle realisations of the current value (point to be predicted) and re-
    calculate mutual information (MI) for shuffled data. The actual estimated
    MI is then compared against this distribution of MI values from surrogate
    data.

    Args:
        analysis_setup : Multivariate_te instance
            information on the current analysis
        data : Data instance
            raw data
        opts : dict [optional]
            parameters for statistical testing, can contain:

            - 'n_perm_mi' - number of permutations (default=500)
            - 'alpha_mi' - critical alpha level (default=0.05)
            - 'tail_mi' - tail for testing, can be 'one' or 'two'
              (default='one')
            - 'perm_range' - permutation range if permutation over samples is
              used to create surrogates (default='max')

    Returns:
        float
            estimated MI value
        bool
            statistical significance
        float
            p_value for estimated MI value
    """
    if opts is None:
        opts = {}
    n_perm = analysis_setup.options.get('n_perm_mi', 500)
    alpha = analysis_setup.options.get('alpha_mi', 0.05)
    tail = analysis_setup.options.get('tail_mi', 'one')
    perm_range = analysis_setup.options.get('perm_range', 'max')
    '''
    surr_realisations = np.empty(
                        (data.n_realisations(analysis_setup.current_value) *
                         (n_perm + 1), 1))
    i_1 = 0
    i_2 = data.n_realisations(analysis_setup.current_value)
    # The first chunk holds the original data
    surr_realisations[i_1:i_2, ] = analysis_setup._current_value_realisations
    # Create surrogate data by shuffling the realisations of the current value.
    for perm in range(n_perm):
        i_1 = i_2
        i_2 += data.n_realisations(analysis_setup.current_value)
        # Check the permutation type for the current candidate.
        if permute_over_replications:
            surr_temp = data.permute_data(analysis_setup.current_value,
                                          [analysis_setup.current_value])[0]
        else:
            [real, repl_idx] = data.get_realisations(
                                            analysis_setup.current_value,
                                            [analysis_setup.current_value])
            surr_temp = _permute_realisations(real, repl_idx, perm_range)
        # Add current shuffled realisation to the array of all realisations for
        # parallel MI estimation.
        # surr_realisations[i_1:i_2, ] = surr_temp
        [real, repl_idx] = data.get_realisations(
                                            analysis_setup.current_value,
                                            [analysis_setup.current_value])
        '''
    surr_realisations = _generate_surrogates(data,
                                             analysis_setup.current_value,
                                             [analysis_setup.current_value],
                                             n_perm,
                                             perm_range)

    surr_dist = analysis_setup._cmi_calculator.estimate_mult(
                            n_chunks=n_perm,
                            options=analysis_setup.options,
                            re_use=['var2'],
                            var1=surr_realisations,
                            var2=analysis_setup._selected_vars_realisations,
                            conditional=None)
    orig_mi = analysis_setup._cmi_calculator.estimate(
                            opts=analysis_setup.options,
                            var1=analysis_setup._current_value_realisations,
                            var2=analysis_setup._selected_vars_realisations,
                            conditional=None
                            )
    [significance, p_value] = _find_pvalue(statistic=orig_mi,
                                           distribution=surr_dist,
                                           alpha=alpha,
                                           tail=tail)
    return [orig_mi, significance, p_value]


def check_n_perm(n_perm, alpha):
    """Check if no. permutations is big enough to obtain the requested alpha.

    Note:
        The no. permutations must be big enough to theoretically allow for the
        detection of a p-value that is smaller than the critical alpha level.
        Otherwise the permutation test is pointless. The smalles possible
        p-value is 1/n_perm.
    """
    if not 1.0 / n_perm < alpha:
        raise RuntimeError('The number of permutations {0} is to small to test'
                           ' the requested alpha level {1}. The number of '
                           'permutations must be greater than (1/alpha).'.
                               format(n_perm, alpha))

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
            list of indices indicating samples to be used as sources
        n_perm : int
            number of permutations for testing

    Returns:
        numpy array
            surrogate TE values, dimensions: (length test set, number of
            surrogates)
    """
    # Check if n_replications is high enough to allow for the requested number
    # of permutations. If not permute samples over time

    # permute_over_replications = _permute_over_replications(data, n_perm)
    perm_range = analysis_setup.options.get('perm_range', 'max')

    # Create surrogate table.
    if VERBOSE:
        print('\ncreate surrogates table with {0} permutations'.format(n_perm))
    surr_table = np.zeros((len(idx_test_set), n_perm))  # surrogate TE values
    current_value_realisations = analysis_setup._current_value_realisations
    idx_c = 0
    for candidate in idx_test_set:
        if VERBOSE:
            print('\tcand. {0}'.format(
                                analysis_setup._idx_to_lag([candidate])[0]))
        '''
        surr_candidate_realisations = np.empty(
                (data.n_realisations(analysis_setup.current_value) * n_perm,
                1))
        i_1 = 0
        i_2 = data.n_realisations(analysis_setup.current_value)
        for perm in range(n_perm):
            # Check the permutation type for the current candidate.
            if permute_over_replications:
                sur_temp = data.permute_data(analysis_setup.current_value,
                                             [candidate])[0]
            else:
                [real, repl_idx] = data.get_realisations(
                                                analysis_setup.current_value,
                                                [candidate])
                sur_temp = _permute_realisations(real, repl_idx, perm_range)

            surr_candidate_realisations[i_1:i_2, ] = sur_temp
            i_1 = i_2
            i_2 += data.n_realisations(analysis_setup.current_value)
        '''
        surr_candidate_realisations = _generate_surrogates(
                                                 data,
                                                 analysis_setup.current_value,
                                                 [candidate],
                                                 n_perm,
                                                 perm_range)
        surr_table[idx_c, :] = analysis_setup._cmi_calculator.estimate_mult(
                    n_chunks=n_perm,
                    options=analysis_setup.options,
                    re_use=['var2', 'conditional'],
                    var1=surr_candidate_realisations,  # too long
                    var2=current_value_realisations,
                    conditional=analysis_setup._selected_vars_realisations)
        idx_c += 1

    return surr_table


def _find_table_max(table):
    """Find maximum for each column of a table."""
    return np.max(table, axis=0)


def _find_table_min(table):
    """Find minimum for each column of a table."""
    return np.min(table, axis=0)


def _sort_table_min(table):
    """Sort each column in a table in ascending order."""
    for permutation in range(table.shape[1]):
        table[:, permutation].sort()
    return table


def _sort_table_max(table):
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
            1-dimensional distribution of values, test distribution
        alpha : float [optional]
            critical alpha level for statistical significance (default=0.05)
        tail : str [optional]
            'one' or 'two' for one-/two-tailed testing (default='one')

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    assert(distribution.ndim == 1)
    check_n_perm(distribution.shape[0], alpha)
#    if (1.0 / distribution.shape[0] >= alpha):
#        print('The number of permutations is to small ({0}) to test the '
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


def _sufficient_replications(data, n_perm):
    """Test if no. replications is high enough for surrogate creation.

    Test if the number of replications is high enough to allow for the required
    number of permutations.
    """
    if np.math.factorial(data.n_replications) > n_perm:
        return True
    else:
        return False


def _generate_surrogates(data, current_value, idx_list, n_perm,
                         perm_range='max'):
    """Generate surrogate data for statistical testing.

    The method for data generation depends on whether sufficient replications
    of the data exists. If the number of replications is high enough
    (reps! > n_permutations), surrogates are created by shuffling data over
    replications (while keeping the temporal order of samples intact). If the
    number of replications is too low, samples are shuffled over time (while
    keeping the order of replications intact).

    Args:
        data : Data instance
            raw data for analysis
        current_value : tuple
            index of the current value in current analysis, has to have the
            form (idx process, idx sample)
        idx_list : list of tuples
            list of variables, for which surrogates have to be created
        n_perm : int
            number of permutations
        perm_range : int [optional]
            permutation range if permutation over samples is used to create
            surrogates (default='max')

    Returns:
        numpy array
            surrogate data with dimensions
            (realisations * n_perm) x len(idx_list)
    """
    # Allocate memory for surrogates
    n_realisations = data.n_realisations(current_value)
    surrogates = np.empty((n_realisations * n_perm, len(idx_list)))

    # Generate surrogates by permuting over replications if possible (no.
    # replications needs to be sufficient); else permute samples over time.
    i_1 = 0
    i_2 = n_realisations
    if _sufficient_replications(data, n_perm):  # permute replications
        for perm in range(n_perm):
            surrogates[i_1:i_2, ] = data.permute_replications(current_value,
                                                              idx_list)[0]
            i_1 = i_2
            i_2 += n_realisations
    else:  # permute samples
        for perm in range(n_perm):
            surrogates[i_1:i_2, ] = data.permute_samples(current_value,
                                                         idx_list,
                                                         perm_range)[0]
            i_1 = i_2
            i_2 += n_realisations
    return surrogates
