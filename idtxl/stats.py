"""Provide statistics functions."""
import copy as cp
import numpy as np
from . import idtxl_utils as utils
from . import idtxl_exceptions as ex


def ais_fdr(settings=None, *results):
    """Perform FDR-correction on results of network AIS estimation.

    Perform correction of the false discovery rate (FDR) after estimation of
    active information storage (AIS) for all processes in the network. FDR
    correction is applied by correcting the AIS estimate's omnibus p-values for
    individual processes/nodes in the network.

    Input can be a list of partial results to combine results from parallel
    analysis.

    References:

    - Genovese, C.R., Lazar, N.A., & Nichols, T. (2002). Thresholding of
      statistical maps in functional neuroimaging using the false discovery
      rate. Neuroimage, 15(4), 870-878.

    Args:
        settings : dict [optional]
            parameters for statistical testing with entries:

            - alpha_fdr : float [optional] - critical alpha level
              (default=0.05)
            - fdr_constant : int [optional] - choose one of two constants used
              for calculating the FDR-thresholds according to Genovese (2002):
              1 will divide alpha by 1, 2 will divide alpha by the sum_i(1/i);
              see the paper for details on the assumptions (default=2)

        results : instances of ResultsSingleProcessAnalysis
            results of network AIS estimation, see documentation of
            ResultsSingleProcessAnalysis()

    Returns:
        ResultsSingleProcessAnalysis instance
            input results objects pruned of non-significant estimates
    """
    if settings is None:
        settings = {}
    # Set defaults and get parameters from settings dictionary
    alpha = settings.get('alpha_fdr', 0.05)
    constant = settings.get('fdr_constant', 2)

    # Combine results into single results dict.
    if len(results) > 1:
        results_comb = cp.deepcopy(results[0])
        results_comb.combine_results(*results[1:])
    else:
        results_comb = cp.deepcopy(results[0])

    # Collect p-values of whole processes (determined by the omnibus test).
    pval = np.arange(0)
    process_idx = np.arange(0).astype(int)
    n_perm = np.arange(0).astype(int)
    for process in results_comb.processes_analysed:
        if results_comb._single_process[process].ais_sign:
            pval = np.append(
                pval, results_comb._single_process[process].ais_pval)
            process_idx = np.append(process_idx, process)
            n_perm = np.append(
                    n_perm, results_comb.settings.n_perm_mi)

    if pval.size == 0:
        print('FDR correction: no links in final results ...\n')
        results_comb._add_fdr(fdr=None, alpha=alpha, constant=constant)
        return results_comb

    sign, thresh = _perform_fdr_corretion(pval, constant, alpha)

    # If the number of permutations for calculating p-values for individual
    # variables is too low, return without performing any correction.
    if (1 / min(n_perm)) > thresh[0]:
        print('WARNING: Number of permutations (''n_perm_max_seq'') for at '
              'least one target is too low to allow for FDR correction '
              '(FDR-threshold: {0:.4f}, min. theoretically possible p-value: '
              '{1}).'.format(thresh[0], 1 / min(n_perm)))
        results_comb._add_fdr(fdr=None, alpha=alpha, constant=constant)
        return results_comb

    # Go over list of all candidates and remove non-significant results from
    # the results object. Create a copy of the results object to leave the
    # original intact.
    fdr = cp.deepcopy(results_comb._single_process)
    for s in range(sign.shape[0]):
        if not sign[s]:
            t = process_idx[s]
            fdr[t].selected_vars = []
            fdr[t].ais_pval = 1
            fdr[t].ais_sign = False
    results_comb._add_fdr(fdr, alpha, constant)
    return results_comb


def network_fdr(settings=None, *results):
    """Perform FDR-correction on results of network inference.

    Perform correction of the false discovery rate (FDR) after network
    analysis. FDR correction can either be applied at the target level
    (by correcting omnibus p-values) or at the single-link level (by correcting
    p-values of individual links between single samples and the target).

    Input can be a list of partial results to combine results from parallel
    analysis.

    References:

    - Genovese, C.R., Lazar, N.A., & Nichols, T. (2002). Thresholding of
      statistical maps in functional neuroimaging using the false discovery
      rate. Neuroimage, 15(4), 870-878.

    Args:
        settings : dict [optional]
            parameters for statistical testing with entries:

            - alpha_fdr : float [optional] - critical alpha level
              (default=0.05)
            - correct_by_target : bool [optional] - if true correct p-values on
              on the target level (omnibus test p-values), otherwise correct
              p_values for individual variables (sequential max stats p-values)
              (default=True)
            - fdr_constant : int [optional] - choose one of two constants used
              for calculating the FDR-thresholds according to Genovese (2002):
              1 will divide alpha by 1, 2 will divide alpha by the sum_i(1/i);
              see the paper for details on the assumptions (default=2)

        results : instances of ResultsNetworkInference
            results of network inference, see documentation of
            ResultsNetworkInference()

    Returns:
        ResultsNetworkInference instance
            input object pruned of non-significant links
    """
    if settings is None:
        settings = {}
    # Set defaults and get parameters from settings dictionary
    alpha = settings.get('alpha_fdr', 0.05)
    correct_by_target = settings.get('correct_by_target', True)
    constant = settings.get('fdr_constant', 2)

    # Combine results into single results dict.
    if len(results) > 1:
        results_comb = cp.deepcopy(results[0])
        results_comb.combine_results(*results[1:])
    else:
        results_comb = cp.deepcopy(results[0])

    # Collect significant source variables for all targets. Either correct
    # p-value of whole target (all candidates), or correct p-value of
    # individual source variables. Use targets with significant input only
    # (determined by the omnibus test).
    pval = np.arange(0)
    target_idx = np.arange(0).astype(int)
    n_perm = np.arange(0).astype(int)
    cands = []
    if correct_by_target:  # whole target
        for target in results_comb.targets_analysed:
            if results_comb._single_target[target].omnibus_sign:
                pval = np.append(
                    pval, results_comb._single_target[target].omnibus_pval)
                target_idx = np.append(target_idx, target)
                n_perm = np.append(
                        n_perm, results_comb.settings.n_perm_omnibus)
    else:  # individual variables
        for target in results_comb.targets_analysed:
            if results_comb._single_target[target].omnibus_sign:
                n_sign = (results_comb._single_target[target].
                          selected_sources_pval.size)
                pval = np.append(
                    pval, (results_comb._single_target[target].
                           selected_sources_pval))
                target_idx = np.append(target_idx,
                                       np.ones(n_sign) * target).astype(int)
                cands = (cands +
                         (results_comb._single_target[target].
                          selected_vars_sources))
                n_perm = np.append(
                    n_perm, results_comb.settings.n_perm_max_seq)

    if pval.size == 0:
        print('No links in final results ...')
        results_comb._add_fdr(
            fdr=None, alpha=alpha, correct_by_target=correct_by_target,
            constant=constant)
        return results_comb

    sign, thresh = _perform_fdr_corretion(pval, constant, alpha)

    # If the number of permutations for calculating p-values for individual
    # variables is too low, return without performing any correction.
    if (1 / min(n_perm)) > thresh[0]:
        print('WARNING: Number of permutations (''n_perm_max_seq'') for at '
              'least one target is too low to allow for FDR correction '
              '(FDR-threshold: {0:.4f}, min. theoretically possible p-value: '
              '{1}).'.format(thresh[0], 1 / min(n_perm)))
        results_comb._add_fdr(
            fdr=None, alpha=alpha, correct_by_target=correct_by_target,
            constant=constant)
        return results_comb

    # Go over list of all candidates and remove non-significant results from
    # the results object. Create a copy of the results object to leave the
    # original intact.
    fdr = cp.deepcopy(results_comb._single_target)
    for s in range(sign.shape[0]):
        if not sign[s]:
            if correct_by_target:
                t = target_idx[s]
                fdr[t].selected_vars_full = cp.deepcopy(
                    results_comb._single_target[t].selected_vars_target)
                fdr[t].selected_sources_te = None
                fdr[t].selected_sources_pval = None
                fdr[t].selected_vars_sources = []
                fdr[t].omnibus_pval = 1
                fdr[t].omnibus_sign = False
            else:
                t = target_idx[s]
                cand = cands[s]
                cand_ind = (fdr[t].selected_vars_sources.index(cand))
                fdr[t].selected_vars_sources.pop(cand_ind)
                fdr[t].selected_sources_pval = np.delete(
                    fdr[t].selected_sources_pval, cand_ind)
                fdr[t].selected_sources_te = np.delete(
                    fdr[t].selected_sources_te, cand_ind)
                fdr[t].selected_vars_full.pop(
                    fdr[t].selected_vars_full.index(cand))
    results_comb._add_fdr(fdr, alpha, correct_by_target, constant)
    return results_comb


def _perform_fdr_corretion(pval, constant, alpha):
    """Calculate sequential threshold for FDR-correction.

    Calculate sequential thresholds for FDR-correction of p-values. The
    constant defines how the threshold is calculated. See Genovese (2002) for
    details.

    References:

    - Genovese, C.R., Lazar, N.A., & Nichols, T. (2002). Thresholding of
      statistical maps in functional neuroimaging using the false discovery
      rate. Neuroimage, 15(4), 870-878.

    Args:
        pval : numpy array
            p-values to be corrected
        alpha : float
            critical alpha level
        fdr_constant : int
            one of two constants used for calculating the FDR-thresholds
            according to Genovese (2002): 1 will divide alpha by 1, 2 will
            divide alpha by the sum_i(1/i); see the paper for details on the
            assumptions (default=2)

    Returns:
        array of bools
            significance of p-values
        array of floats
            FDR-thresholds for each p-value
    """
    # Sort all p-values in ascending order.
    sort_idx = np.argsort(pval)
    pval.sort()

    # Calculate threshold
    n = pval.size
    if constant == 2:  # pick the requested constant (see Genovese, p.872)
        if n < 1000:
            const = sum(1 / np.arange(1, n + 1))
        else:
            const = np.log(n) + np.e  # aprx. harmonic sum with Euler's number
    elif constant == 1:
        # This is less strict than the other one and corresponds to a
        # Bonoferroni-correction for the first p-value, however, it makes more
        # strict assumptions on the distribution of p-values, while constant 2
        # works for any joint distribution of the p-values.
        const = 1
    thresh = (np.arange(1, n + 1) / n) * alpha / const

    # Compare data to threshold.
    sign = pval <= thresh
    if np.invert(sign).any():
        first_false = np.where(np.invert(sign))[0][0]
        sign[first_false:] = False  # avoids false positives due to equal pvals
    sign = sign[sort_idx]  # restore original ordering of significance values
    return sign, thresh


def omnibus_test(analysis_setup, data):
    """Perform an omnibus test on identified conditional variables.

    Test the joint information transfer from all identified sources to the
    current value conditional on candidates in the target's past. To test for
    significance, this is repeated for shuffled realisations of the sources.
    The distribution of values from shuffled data is then used as test
    distribution.

    Args:
        analysis_setup : MultivariateTE instance
            information on the current analysis, can have an optional attribute
            'settings', a dictionary with parameters for statistical testing:

            - n_perm_omnibus : int [optional] - number of permutations
              (default=500)
            - alpha_omnibus : float [optional] - critical alpha level
              (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data

    Returns:
        bool
            statistical significance
        float
            the test's p-value
        float
            the estimated test statisic, i.e., the information transfer from
            all sources into the target

    Raises:
        ex.AlgorithmExhaustedError
            Raised from estimate() calls when calculation cannot be made
    """
    # Set defaults and get parameters from settings dictionary
    analysis_setup.settings.setdefault('n_perm_omnibus', 500)
    n_permutations = analysis_setup.settings['n_perm_omnibus']
    analysis_setup.settings.setdefault('alpha_omnibus', 0.05)
    alpha = analysis_setup.settings['alpha_omnibus']
    permute_in_time = _check_permute_in_time(analysis_setup, data,
                                             n_permutations)
    assert analysis_setup.selected_vars_sources, 'No sources to test.'

    # Create temporary variables b/c realisations for sources and targets are
    # created on the fly, which is costly, so we want to re-use them after
    # creation. (This does not apply to the current value realisations).
    # If there was no target variable selected (e.g., if MI is used for network
    # inference), set conditional to None such that the MI instead of the CMI
    # estimator is used when calculating the statistic.
    cond_source_realisations = (analysis_setup
                                ._selected_vars_sources_realisations)
    if analysis_setup._selected_vars_target:
        cond_target_realisations = (analysis_setup
                                    ._selected_vars_target_realisations)
    else:
        cond_target_realisations = None
    statistic = analysis_setup._cmi_estimator.estimate(
                            var1=cond_source_realisations,
                            var2=analysis_setup._current_value_realisations,
                            conditional=cond_target_realisations)

    # Create the surrogate distribution by permuting the conditional sources.
    if analysis_setup.settings['verbose']:
        print('omnibus test, n_perm: {0}'.format(n_permutations))
    if (analysis_setup._cmi_estimator.is_analytic_null_estimator() and
            permute_in_time):
        # Generate the surrogates analytically
        analysis_setup.settings['analytical_surrogates'] = True
        surr_distribution = (analysis_setup._cmi_estimator.
                             estimate_surrogates_analytic(
                               n_perm=n_permutations,
                               var1=cond_source_realisations,
                               var2=analysis_setup._current_value_realisations,
                               conditional=cond_target_realisations))
    else:
        analysis_setup.settings['analytical_surrogates'] = False
        surr_cond_real = _get_surrogates(data,
                                         analysis_setup.current_value,
                                         analysis_setup.selected_vars_sources,
                                         n_permutations,
                                         analysis_setup.settings)

        surr_distribution = analysis_setup._cmi_estimator.estimate_parallel(
                            n_chunks=n_permutations,
                            re_use=['var2', 'conditional'],
                            var1=surr_cond_real,
                            var2=analysis_setup._current_value_realisations,
                            conditional=cond_target_realisations)
    [significance, pvalue] = _find_pvalue(statistic, surr_distribution,
                                          alpha, 'one_bigger')
    if analysis_setup.settings['verbose']:
        if significance:
            print(' -- significant\n')
        else:
            print(' -- not significant\n')
    return significance, pvalue, statistic


def max_statistic(analysis_setup, data, candidate_set, te_max_candidate,
                  conditional=None):
    """Perform maximum statistics for one candidate source.

    Test if a transfer entropy value is significantly bigger than the maximum
    values obtained from surrogates of all remanining candidates.

    Args:
        analysis_setup : MultivariateTE instance
            information on the current analysis, can have an optional attribute
            'settings', a dictionary with parameters for statistical testing:

            - n_perm_max_stat : int [optional] - number of permutations
              (default=200)
            - alpha_max_stat : float [optional] - critical alpha level
              (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data
        candidate_set : list of tuples
            list of indices of remaning candidates
        te_max_candidate : float
            transfer entropy value to be tested
        conditional : numpy array [optional]
            realisations of conditional, 2D numpy array where array dimensions
            represent [realisations x variable dimension] (per default all
            already selected source and target variables from the
            analysis_setup are used)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
        numpy array
            surrogate table

    Raises:
        ex.AlgorithmExhaustedError
            Raised from _create_surrogate_table() when calculation cannot be made
    """
    # Set defaults and get parameters from settings dictionary
    analysis_setup.settings.setdefault('n_perm_max_stat', 200)
    n_perm = analysis_setup.settings['n_perm_max_stat']
    analysis_setup.settings.setdefault('alpha_max_stat', 0.05)
    alpha = analysis_setup.settings['alpha_max_stat']
    _check_permute_in_time(analysis_setup, data, n_perm)
    assert(candidate_set), 'The candidate set is empty.'
    if analysis_setup.settings['verbose']:
        print('maximum statistic, n_perm: {0}'.format(
                            analysis_setup.settings['n_perm_max_stat']))

    surr_table = _create_surrogate_table(analysis_setup, data, candidate_set,
                                         n_perm, conditional)
    max_distribution = _find_table_max(surr_table)
    [significance, pvalue] = _find_pvalue(statistic=te_max_candidate,
                                          distribution=max_distribution,
                                          alpha=alpha,
                                          tail='one_bigger')
    return significance, pvalue, surr_table


def max_statistic_sequential(analysis_setup, data):
    """Perform sequential maximum statistics for a set of candidate sources.

    Test multivariate/bivariate MI/TE values against surrogates. Test highest
    TE/MI value against distribution of highest surrogate values, second
    highest against distribution of second highest, and so forth. Surrogates
    are created from each candidate in the candidate set, including the
    candidate that is currently tested. Surrogates are then sorted over
    candidates. This is repeated n_perm_max_seq times. Stop comparison if a
    TE/MI value is not significant compared to the distribution of surrogate
    values of the same rank. All smaller values are considered non-significant
    as well.

    The conditional for estimation of MI/TE is taken from the current set of
    conditional variables in the analysis setup. For multivariate MI or TE
    surrogate creation, the full set of conditional variables is used. For
    bivariate MI or TE surrogate creation, the conditioning set has to be
    restricted to a subset of the current set of conditional variables: for
    bivariate MI no conditioning set is required, for bivariate TE only the
    past variables from the target are required (not the variables selected
    from other relevant sources).

    This function will re-use the surrogate table created in the last min-stats
    round if that table is in the analysis_setup. This saves the complete
    calculation of surrogates for this statistic.

    Args:

        analysis_setup : MultivariateTE instance
            information on the current analysis, can have an optional attribute
            'settings', a dictionary with parameters for statistical testing:

            - n_perm_max_seq : int [optional] - number of permutations
              (default='n_perm_min_stat'|500)
            - alpha_max_seq : float [optional] - critical alpha level
              (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data

    Returns:
        numpy array, bool
            statistical significance of each source
        numpy array, float
            the test's p-values for each source
        numpy array, float
            TE values for individual sources
    """
    try:
        n_permutations = analysis_setup.settings['n_perm_max_seq']
    except KeyError:
        try:  # use the same n_perm as for min_stats if surr table is reused
            n_permutations = analysis_setup._min_stats_surr_table.shape[1]
            analysis_setup.settings['n_perm_max_seq'] = n_permutations
        except AttributeError:  # is surr table is None, use default
            analysis_setup.settings['n_perm_max_seq'] = 500
            n_permutations = analysis_setup.settings['n_perm_max_seq']
    analysis_setup.settings.setdefault('alpha_max_seq', 0.05)
    alpha = analysis_setup.settings['alpha_max_seq']
    _check_permute_in_time(analysis_setup, data, n_permutations)
    if analysis_setup.settings['verbose']:
        print('sequential maximum statistic, n_perm: {0}'.format(
            n_permutations))

    assert analysis_setup.selected_vars_sources, 'No sources to test.'

    idx_conditional = analysis_setup.selected_vars_full
    conditional_realisations = np.empty(
        (data.n_realisations(analysis_setup.current_value) *
            len(analysis_setup.selected_vars_sources),
            len(idx_conditional) - 1)).astype(data.data_type)
    candidate_realisations = np.empty(
        (data.n_realisations(analysis_setup.current_value) *
         len(analysis_setup.selected_vars_sources), 1)).astype(data.data_type)

    # Calculate TE for each candidate in the conditional source set, i.e.,
    # calculate the conditional MI between each candidate and the current
    # value, conditional on all selected variables in the conditioning set.
    # Then sort the estimated TE values.
    i_1 = 0
    i_2 = data.n_realisations(analysis_setup.current_value)
    # Collect data for each candidate and the corresponding conditioning set.
    for candidate in analysis_setup.selected_vars_sources:
        [temp_cond, temp_cand] = analysis_setup._separate_realisations(
                                            idx_conditional,
                                            candidate)

        # The following may happen if either the requested conditing is 'none'
        # or if the conditiong set that is tested consists only of a single
        # candidate.
        if temp_cond is None:
            conditional_realisations = None
            re_use = ['var2', 'conditional']
        else:
            conditional_realisations[i_1:i_2, ] = temp_cond
            re_use = ['var2']
        candidate_realisations[i_1:i_2, ] = temp_cand
        i_1 = i_2
        i_2 += data.n_realisations(analysis_setup.current_value)

    # Calculate original statistic (multivariate/bivariate TE/MI)
    try:
        individual_stat = analysis_setup._cmi_estimator.estimate_parallel(
                            n_chunks=len(analysis_setup.selected_vars_sources),
                            re_use=re_use,
                            var1=candidate_realisations,
                            var2=analysis_setup._current_value_realisations,
                            conditional=conditional_realisations)
    except ex.AlgorithmExhaustedError as aee:
        # The aglorithm cannot continue here, so
        #  we'll terminate the max sequential stats test,
        #  and declare all not significant
        print('AlgorithmExhaustedError encountered in '
            'estimations: ' + aee.message)
        print('Stopping sequential max stats at candidate with rank 0')
        # For now we don't need a stack trace:
        # traceback.print_tb(aee.__traceback__)
        # Return (signficance, pvalue, TEs):
        return \
            (np.zeros(len(analysis_setup.selected_vars_sources)).astype(bool),
            np.ones(len(analysis_setup.selected_vars_sources)),
            np.zeros(len(analysis_setup.selected_vars_sources)))

    selected_vars_order = utils.argsort_descending(individual_stat)
    individual_stat_sorted = utils.sort_descending(individual_stat)

    # Re-use surrogate table from previous pruning using min stats, if it
    # already exists. This saves some time. Otherwise create surrogate table.
    # Sort surrogate table.
    if (analysis_setup._min_stats_surr_table is not None and
            n_permutations <= analysis_setup._min_stats_surr_table.shape[1]):
        surr_table = analysis_setup._min_stats_surr_table[:, :n_permutations]
        assert len(analysis_setup.selected_vars_sources) == surr_table.shape[0]
    else:
        try:
            surr_table = _create_surrogate_table(
                            analysis_setup=analysis_setup,
                            data=data,
                            idx_test_set=analysis_setup.selected_vars_sources,
                            n_perm=n_permutations)
        except ex.AlgorithmExhaustedError as aee:
            # The aglorithm cannot continue here, so
            #  we'll terminate the max sequential stats test,
            #  and declare all not significant
            print('AlgorithmExhaustedError encountered in '
                'estimations: ' + aee.message)
            print('Stopping sequential max stats at candidate with rank 0')
            # For now we don't need a stack trace:
            # traceback.print_tb(aee.__traceback__)
            # Return (signficance, pvalue, TEs):
            return \
                (np.zeros(len(analysis_setup.selected_vars_sources)).astype(bool),
                np.ones(len(analysis_setup.selected_vars_sources)),
                np.zeros(len(analysis_setup.selected_vars_sources)))
    max_distribution = _sort_table_max(surr_table)

    # Compare each original value with the distribution of the same rank,
    # starting with the highest value.
    significance = np.zeros(individual_stat.shape[0]).astype(bool)
    pvalue = np.ones(individual_stat.shape[0])
    for c in range(individual_stat.shape[0]):
        [s, p] = _find_pvalue(individual_stat_sorted[c],
                              max_distribution[c, ], alpha, tail='one_bigger')
        significance[c] = s
        pvalue[c] = p
        if not s:  # break as soon as a candidate is no longer significant
            if analysis_setup.settings['verbose']:
                print('\nStopping sequential max stats at candidate with rank '
                      '{0}.'.format(c))
            break

    # Get back original order and return results.
    significance = significance[selected_vars_order]
    pvalue = pvalue[selected_vars_order]
    return significance, pvalue, individual_stat


def max_statistic_sequential_bivariate(analysis_setup, data):
    """Perform sequential maximum statistics for a set of candidate sources.

    Test multivariate/bivariate MI/TE values against surrogates. Test highest
    TE/MI value against distribution of highest surrogate values, second
    highest against distribution of second highest, and so forth. Surrogates
    are created from each candidate in the candidate set, including the
    candidate that is currently tested. Surrogates are then sorted over
    candidates. This is repeated n_perm_max_seq times. Stop comparison if a
    TE/MI value is not significant compared to the distribution of surrogate
    values of the same rank. All smaller values are considered non-significant
    as well.

    The conditional for estimation of MI/TE is taken from the current set of
    conditional variables in the analysis setup. For multivariate MI or TE
    surrogate creation, the full set of conditional variables is used. For
    bivariate MI or TE surrogate creation, the conditioning set has to be
    restricted to a subset of the current set of conditional variables: for
    bivariate MI no conditioning set is required, for bivariate TE only the
    past variables from the target are required (not the variables selected
    from other relevant sources).

    This function will re-use the surrogate table created in the last min-stats
    round if that table is in the analysis_setup. This saves the complete
    calculation of surrogates for this statistic.

    Args:

        analysis_setup : MultivariateTE instance
            information on the current analysis, can have an optional attribute
            'settings', a dictionary with parameters for statistical testing:

            - n_perm_max_seq : int [optional] - number of permutations
              (default='n_perm_min_stat'|500)
            - alpha_max_seq : float [optional] - critical alpha level
              (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data

    Returns:
        numpy array, bool
            statistical significance of each source
        numpy array, float
            the test's p-values for each source
        numpy array, float
            TE values for individual sources
    """
    try:
        n_permutations = analysis_setup.settings['n_perm_max_seq']
    except KeyError:
        try:  # use the same n_perm as for min_stats if surr table is reused
            n_permutations = analysis_setup._min_stats_surr_table.shape[1]
            analysis_setup.settings['n_perm_max_seq'] = n_permutations
        except AttributeError:  # is surr table is None, use default
            analysis_setup.settings['n_perm_max_seq'] = 500
            n_permutations = analysis_setup.settings['n_perm_max_seq']
    analysis_setup.settings.setdefault('alpha_max_seq', 0.05)
    alpha = analysis_setup.settings['alpha_max_seq']
    _check_permute_in_time(analysis_setup, data, n_permutations)
    if analysis_setup.settings['verbose']:
        print('sequential maximum statistic, n_perm: {0}'.format(
            n_permutations))

    assert analysis_setup.selected_vars_sources, 'No sources to test.'

    # Check if target variables were selected to distinguish between TE and MI
    # analysis.
    if len(analysis_setup._selected_vars_target) == 0:
        conditional_realisations_target = None
    else:
        conditional_realisations_target = analysis_setup._selected_vars_target_realisations

    # Test all selected sources separately. This way, the conditioning
    # uses past variables from the current source only (opposed to past
    # variables from all sources as in multivariate network inference).
    significant_sources = np.unique(
            [s[0] for s in analysis_setup.selected_vars_sources])
    significance = np.zeros(
        len(analysis_setup.selected_vars_sources)).astype(bool)
    pvalue = np.ones(len(analysis_setup.selected_vars_sources))
    stat = np.zeros(len(analysis_setup.selected_vars_sources))
    for source in significant_sources:
        # Find selected past variables for current source
        source_vars = [s for s in analysis_setup.selected_vars_sources if
                       s[0] == source]

        # Determine length of conditioning set and allocate memory.
        idx_conditional = source_vars.copy()
        if conditional_realisations_target is not None:
            idx_conditional += analysis_setup.selected_vars_target
        conditional_realisations = np.empty(
            (data.n_realisations(analysis_setup.current_value) *
                len(source_vars),
                len(idx_conditional) - 1)).astype(data.data_type)
        candidate_realisations = np.empty(
            (data.n_realisations(analysis_setup.current_value) *
                len(source_vars), 1)).astype(data.data_type)

        # Calculate TE/MI for each candidate in the conditional source set,
        # i.e., calculate the conditional MI between each candidate and the
        # current value, conditional on all selected variables in the
        # conditioning set. Then sort the estimated TE/MI values.
        i_1 = 0
        i_2 = data.n_realisations(analysis_setup.current_value)
        # Collect data for each candidate and the corresponding conditioning set.
        for candidate in source_vars:
            temp_cond = data.get_realisations(
                        analysis_setup.current_value,
                        set(source_vars).difference(set([candidate])))[0]
            temp_cand = data.get_realisations(
                        analysis_setup.current_value, [candidate])[0]
            # The following may happen if either the requested conditing is 'none'
            # or if the conditiong set that is tested consists only of a single
            # candidate.
            if temp_cond is None:
                conditional_realisations = conditional_realisations_target
                re_use = ['var2', 'conditional']
            else:
                re_use = ['var2']
                if conditional_realisations_target is None:
                    conditional_realisations[i_1:i_2, ] = temp_cond
                else:
                    conditional_realisations[i_1:i_2, ] = np.hstack((
                        temp_cond, conditional_realisations_target))
            candidate_realisations[i_1:i_2, ] = temp_cand
            i_1 = i_2
            i_2 += data.n_realisations(analysis_setup.current_value)

        # Calculate original statistic (multivariate/bivariate TE/MI)
        try:
            individual_stat = analysis_setup._cmi_estimator.estimate_parallel(
                            n_chunks=len(source_vars),
                            re_use=re_use,
                            var1=candidate_realisations,
                            var2=analysis_setup._current_value_realisations,
                            conditional=conditional_realisations)
        except ex.AlgorithmExhaustedError as aee:
            # The aglorithm cannot continue here, so
            #  we'll terminate the max sequential stats test,
            #  and declare all not significant
            print('AlgorithmExhaustedError encountered in '
                'estimations: ' + aee.message)
            print('Stopping sequential max stats at candidate with rank 0')
            # For now we don't need a stack trace:
            # traceback.print_tb(aee.__traceback__)
            # Return (signficance, pvalue, TEs):
            return \
                (np.zeros(len(analysis_setup.selected_vars_sources)).astype(bool),
                np.ones(len(analysis_setup.selected_vars_sources)),
                np.zeros(len(analysis_setup.selected_vars_sources)))

        selected_vars_order = utils.argsort_descending(individual_stat)
        individual_stat_sorted = utils.sort_descending(individual_stat)

        # Don't re-use surrogate table from previous pruning using min stats
        # like for the multivariate algorithm. There is no longer a global
        # min_stats including all sources variables, but a separate table per
        # source.
        conditional_realisations_sources = data.get_realisations(
                    analysis_setup.current_value, source_vars)[0]
        if conditional_realisations_target is None:
            conditional_realisations = conditional_realisations_sources
        else:
            conditional_realisations = np.hstack((
                conditional_realisations_sources,
                conditional_realisations_target))
        try:
            surr_table = _create_surrogate_table(
                        analysis_setup=analysis_setup,
                        data=data,
                        idx_test_set=analysis_setup.selected_vars_sources,
                        n_perm=n_permutations,
                        conditional=conditional_realisations)
        except ex.AlgorithmExhaustedError as aee:
            # The algorithm cannot continue here, so
            #  we'll terminate the max sequential stats test,
            #  and declare all not significant
            print('AlgorithmExhaustedError encountered in '
                'estimations: ' + aee.message)
            print('Stopping sequential max stats at candidate with rank 0')
            # For now we don't need a stack trace:
            # traceback.print_tb(aee.__traceback__)
            # Return (signficance, pvalue, TEs):
            return \
                (np.zeros(len(analysis_setup.selected_vars_sources)).astype(bool),
                np.ones(len(analysis_setup.selected_vars_sources)),
                np.zeros(len(analysis_setup.selected_vars_sources)))
        max_distribution = _sort_table_max(surr_table)

        # Compare each original value with the distribution of the same rank,
        # starting with the highest value.
        for c in range(individual_stat.shape[0]):
            [s, p] = _find_pvalue(individual_stat_sorted[c],
                                  max_distribution[c, ],
                                  alpha, tail='one_bigger')
            # Write results into an array with the same order as the set of
            # selected sources from all process. Find the currently tested
            # variable and its index in the list of all selected variables.
            current_var = source_vars[selected_vars_order[c]]
            for ind, v in enumerate(analysis_setup.selected_vars_sources):
                if v == current_var:
                    break
            significance[ind] = s
            pvalue[ind] = p
            stat[ind] = individual_stat_sorted[c]
            if not s:  # break as soon as a candidate is no longer significant
                if analysis_setup.settings['verbose']:
                    print('\nStopping sequential max stats at candidate with '
                          'rank {0}.'.format(c))
                break

    return significance, pvalue, stat


def min_statistic(analysis_setup, data, candidate_set, te_min_candidate,
                  conditional=None):
    """Perform minimum statistics for one candidate source.

    Test if a transfer entropy value is significantly bigger than the minimum
    values obtained from surrogates of all remanining candidates.

    Args:
        analysis_setup : MultivariateTE instance
            information on the current analysis, can have an optional attribute
            'settings', a dictionary with parameters for statistical testing:

            - n_perm_min_stat : int [optional] - number of permutations
              (default=500)
            - alpha_min_stat : float [optional] - critical alpha level
              (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data
        candidate_set : list of tuples
            list of indices of remaning candidates
        te_min_candidate : float
            transfer entropy value to be tested
        conditional : numpy array [optional]
            realisations of conditional, 2D numpy array where array dimensions
            represent [realisations x variable dimension] (per default all
            already selected source and target variables from the
            analysis_setup are used)

    Returns:
        bool
            statistical significance
        float
            the test's p-value
        numpy array
            surrogate table

    Raises:
        ex.AlgorithmExhaustedError
            Raised from _create_surrogate_table() when calculation cannot be made
    """
    # Set defaults and get parameters from settings dictionary
    analysis_setup.settings.setdefault('n_perm_min_stat', 500)
    n_perm = analysis_setup.settings['n_perm_min_stat']
    analysis_setup.settings.setdefault('alpha_min_stat', 0.05)
    alpha = analysis_setup.settings['alpha_min_stat']
    _check_permute_in_time(analysis_setup, data, n_perm)
    if analysis_setup.settings['verbose']:
        print('minimum statistic, n_perm: {0}'.format(
            analysis_setup.settings['n_perm_min_stat']))

    assert(candidate_set), 'The candidate set is empty.'

    surr_table = _create_surrogate_table(analysis_setup, data, candidate_set,
                                         n_perm, conditional)
    min_distribution = _find_table_min(surr_table)
    [significance, pvalue] = _find_pvalue(statistic=te_min_candidate,
                                          distribution=min_distribution,
                                          alpha=alpha,
                                          tail='one_bigger')
    return significance, pvalue, surr_table


def mi_against_surrogates(analysis_setup, data):
    """Test estimated mutual information for significance against surrogate data.

    Shuffle realisations of the current value (point to be predicted) and re-
    calculate mutual information (MI) for shuffled data. The actual estimated
    MI is then compared against this distribution of MI values from surrogate
    data.

    Args:
        analysis_setup : MultivariateTE instance
            information on the current analysis, can have an optional attribute
            'settings', a dictionary with parameters for statistical testing:

            - n_perm_mi : int [optional] - number of permutations
              (default=500)
            - alpha_mi : float [optional] - critical alpha level
              (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data

    Returns:
        float
            estimated MI value
        bool
            statistical significance
        float
            p_value for estimated MI value

    Raises:
        ex.AlgorithmExhaustedError
            Raised from estimate() methods when calculation cannot be made
    """
    analysis_setup.settings.setdefault('n_perm_mi', 500)
    n_perm = analysis_setup.settings['n_perm_mi']
    analysis_setup.settings.setdefault('alpha_mi', 0.05)
    alpha = analysis_setup.settings['alpha_mi']
    permute_in_time = _check_permute_in_time(analysis_setup, data, n_perm)
    if analysis_setup.settings['verbose']:
        print('mi permutation test against surrogates, n_perm: {0}'.format(
            analysis_setup.settings['n_perm_mi']))
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
    if (analysis_setup._cmi_estimator.is_analytic_null_estimator() and
            permute_in_time):
        # Generate the surrogates analytically
        analysis_setup.settings['analytical_surrogates'] = True
        surr_dist = (analysis_setup._cmi_estimator.
                     estimate_surrogates_analytic(
                            n_perm=n_perm,
                            var1=analysis_setup._current_value_realisations,
                            var2=analysis_setup._selected_vars_realisations,
                            conditional=None))
    else:
        analysis_setup.settings['analytical_surrogates'] = False
        surr_realisations = _get_surrogates(data,
                                            analysis_setup.current_value,
                                            [analysis_setup.current_value],
                                            n_perm,
                                            analysis_setup.settings)

        surr_dist = analysis_setup._cmi_estimator.estimate_parallel(
                            n_chunks=n_perm,
                            re_use=['var2', 'conditional'],
                            var1=surr_realisations,
                            var2=analysis_setup._selected_vars_realisations,
                            conditional=None)
    orig_mi = analysis_setup._cmi_estimator.estimate(
                            var1=analysis_setup._current_value_realisations,
                            var2=analysis_setup._selected_vars_realisations,
                            conditional=None
                            )
    [significance, p_value] = _find_pvalue(statistic=orig_mi,
                                           distribution=surr_dist,
                                           alpha=alpha,
                                           tail='one_bigger')
    return [orig_mi, significance, p_value]


def unq_against_surrogates(analysis_setup, data):
    """Test the unique information in the PID estimate against surrogate data.

    Shuffle realisations of both sources individually and re-calculate PID,
    in particular the unique information from shuffled data. The original
    unique information is then compared against the distribution of values
    calculated from surrogate data.

    Args:
        analysis_setup : Partial_information_decomposition instance
            information on the current analysis, should have an Attribute
            'settings', a dict with optional fields

            - n_perm : int [optional] - number of permutations (default=500)
            - alpha : float [optional] - critical alpha level (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data

    Returns:
        dict
            PID estimate from original data
        bool
            statistical significance of the unique information in source 1
        float
            p-value of the unique information in source 1
        bool
            statistical significance of the unique information in source 2
        float
            p-value of the unique information in source 2
    """
    # Get analysis settings and defaults.
    analysis_setup.settings.setdefault('n_perm', 500)
    n_perm = analysis_setup.settings['n_perm']
    analysis_setup.settings.setdefault('alpha', 0.05)
    alpha = analysis_setup.settings['alpha']
    _check_permute_in_time(analysis_setup, data, n_perm)

    # Get realisations and estimate PID for orginal data
    target_realisations = data.get_realisations(
                                            analysis_setup.current_value,
                                            [analysis_setup.current_value])[0]
    source_1_realisations = data.get_realisations(
                                        analysis_setup.current_value,
                                        [analysis_setup.sources[0]])[0]
    source_2_realisations = data.get_realisations(
                                        analysis_setup.current_value,
                                        [analysis_setup.sources[1]])[0]
    orig_pid = analysis_setup._pid_estimator.estimate(
                            settings=analysis_setup.settings,
                            s1=source_1_realisations,
                            s2=source_2_realisations,
                            t=target_realisations)

    # Test unique information from source 1
    surr_realisations = _get_surrogates(data,
                                        analysis_setup.current_value,
                                        [analysis_setup.sources[0]],
                                        n_perm,
                                        analysis_setup.settings)
    # Calculate surrogate distribution for unique information of source 1.
    # Note: calling  .estimate_parallel does not work here because the PID
    # estimator returns a dictionary not a single value. We have to get the
    # unique from the dictionary manually.
    surr_dist_s1 = np.empty(n_perm)
    chunk_size = int(surr_realisations.shape[0] / n_perm)
    i_1 = 0
    i_2 = chunk_size
    if analysis_setup.settings['verbose']:
            print('\nTesting unq information in s1')
    for p in range(n_perm):
        if analysis_setup.settings['verbose']:
            print('\tperm {0} of {1}'.format(p, n_perm))
        pid_est = analysis_setup._pid_estimator.estimate(
                                settings=analysis_setup.settings,
                                s1=surr_realisations[i_1:i_2, :],
                                s2=source_2_realisations,
                                t=target_realisations
                                )
        surr_dist_s1[p] = pid_est['unq_s1']
        i_1 = i_2
        i_2 += chunk_size

    # Test unique information from source 2
    surr_realisations = _get_surrogates(data,
                                        analysis_setup.current_value,
                                        [analysis_setup.sources[1]],
                                        n_perm,
                                        analysis_setup.settings)
    # Calculate surrogate distribution for unique information of source 2.
    surr_dist_s2 = np.empty(n_perm)
    chunk_size = int(surr_realisations.shape[0] / n_perm)
    i_1 = 0
    i_2 = chunk_size
    if analysis_setup.settings['verbose']:
            print('\nTesting unq information in s2')
    for p in range(n_perm):
        if analysis_setup.settings['verbose']:
            print('\tperm {0} of {1}'.format(p, n_perm))
        pid_est = analysis_setup._pid_estimator.estimate(
                                settings=analysis_setup.settings,
                                s1=source_1_realisations,
                                s2=surr_realisations[i_1:i_2, :],
                                t=target_realisations)
        surr_dist_s2[p] = pid_est['unq_s2']
        i_1 = i_2
        i_2 += chunk_size
    [sign_1, p_val_1] = _find_pvalue(statistic=orig_pid['unq_s1'],
                                     distribution=surr_dist_s1,
                                     alpha=alpha,
                                     tail='one_bigger')
    [sign_2, p_val_2] = _find_pvalue(statistic=orig_pid['unq_s2'],
                                     distribution=surr_dist_s2,
                                     alpha=alpha,
                                     tail='one_bigger')
    return [orig_pid, sign_1, p_val_1, sign_2, p_val_2]


def syn_shd_against_surrogates(analysis_setup, data):
    """Test the shared/synergistic information in the PID estimate.

    Shuffle realisations of the target and re-calculate PID, in particular the
    synergistic and shared information from shuffled data. The original
    shared and synergistic information are then compared against the
    distribution of values calculated from surrogate data.

    Args:
        analysis_setup : Partial_information_decomposition instance
            information on the current analysis, should have an Attribute
            'settings', a dict with optional fields

            - n_perm : int [optional] - number of permutations (default=500)
            - alpha : float [optional] - critical alpha level (default=0.05)
            - permute_in_time : bool [optional] - generate surrogates by
              shuffling samples in time instead of shuffling whole replications
              (default=False)

        data : Data instance
            raw data

    Returns:
        dict
            PID estimate from original data
        bool
            statistical significance of the shared information
        float
            p-value of the shared information
        bool
            statistical significance of the synergistic information
        float
            p-value of the synergistic information
    """
    # Get analysis settings and defaults.
    analysis_setup.settings.setdefault('n_perm', 500)
    n_perm = analysis_setup.settings['n_perm']
    analysis_setup.settings.setdefault('alpha', 0.05)
    alpha = analysis_setup.settings['alpha']
    _check_permute_in_time(analysis_setup, data, n_perm)

    # Get realisations and estimate PID for original data
    target_realisations = data.get_realisations(
                                            analysis_setup.current_value,
                                            [analysis_setup.current_value])[0]
    source_1_realisations = data.get_realisations(
                                        analysis_setup.current_value,
                                        [analysis_setup.sources[0]])[0]
    source_2_realisations = data.get_realisations(
                                        analysis_setup.current_value,
                                        [analysis_setup.sources[1]])[0]
    orig_pid = analysis_setup._pid_estimator.estimate(
                            settings=analysis_setup.settings,
                            s1=source_1_realisations,
                            s2=source_2_realisations,
                            t=target_realisations)

    # Test shared and synergistic information from both sources
    surr_realisations = _get_surrogates(data,
                                        analysis_setup.current_value,
                                        [analysis_setup.current_value],
                                        n_perm,
                                        analysis_setup.settings)
    # Calculate surrogate distribution for shd/syn information of both sources.
    # Note: calling  .estimate_parallel does not work here because the PID
    # estimator returns a dictionary not a single value. We have to get the
    # shared info and synergy from the dictionary manually.
    surr_dist_shd = np.empty(n_perm)
    surr_dist_syn = np.empty(n_perm)
    chunk_size = int(surr_realisations.shape[0] / n_perm)
    i_1 = 0
    i_2 = chunk_size
    if analysis_setup.settings['verbose']:
            print('\nTesting shd and syn information in both sources')
    for p in range(n_perm):
        if analysis_setup.settings['verbose']:
            print('\tperm {0} of {1}'.format(p, n_perm))
        pid_est = analysis_setup._pid_estimator.estimate(
                                settings=analysis_setup.settings,
                                s1=source_1_realisations,
                                s2=source_2_realisations,
                                t=surr_realisations[i_1:i_2, :])
        surr_dist_shd[p] = pid_est['shd_s1_s2']
        surr_dist_syn[p] = pid_est['syn_s1_s2']
        i_1 = i_2
        i_2 += chunk_size
    [sign_shd, p_val_shd] = _find_pvalue(statistic=orig_pid['shd_s1_s2'],
                                         distribution=surr_dist_shd,
                                         alpha=alpha,
                                         tail='one_bigger')
    [sign_syn, p_val_syn] = _find_pvalue(statistic=orig_pid['syn_s1_s2'],
                                         distribution=surr_dist_syn,
                                         alpha=alpha,
                                         tail='one_bigger')
    return [orig_pid, sign_shd, p_val_shd, sign_syn, p_val_syn]


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
                           'permutations must be greater than 1/alpha.'
                           .format(n_perm, alpha))


def _create_surrogate_table(analysis_setup, data, idx_test_set, n_perm,
                            conditional=None):
    """Create a table of surrogate MI/CMI/TE values.

    Calculate MI/CMI/TE between surrogates for each source variable in the test
    set and the target in the analysis setup. The conditional is taken from the
    current set of conditional variables in the analysis setup. For
    multivariate MI or TE surrogate creation, the full set of conditional
    variables is used. For bivariate MI or TE surrogate creation, the
    conditioning set has to be restricted to a subset of the current set of
    conditional variables: for bivariate MI no conditioning set is required,
    for bivariate TE only the past variables from the target are required (not
    the variables selected from other relevant sources).

    Args:
        analysis_setup : instance of NetworkAnalysis or child class
            information on the current analysis, must contain an attribute
            settings with entry 'permute_in_time'
        data : Data instance
            raw data
        idx_test_set : list of tuples
            list of indices indicating samples to be used as sources
        n_perm : int
            number of permutations for testing
        conditional : numpy array [optional]
            realisations of conditional, 2D numpy array where array dimensions
            represent [realisations x variable dimension] (per default all
            already selected source and target variables from the
            analysis_setup are used)
    Returns:
        numpy array
            surrogate MI/CMI/TE values, dimensions: (length test set, number of
            surrogates)

    Raises:
        ex.AlgorithmExhaustedError
            Raised from estimate_parallel() when calculation cannot be made
    """
    # Check which permutation type is requested by the calling function.
    permute_in_time = analysis_setup.settings['permute_in_time']

    # Check what type of conditioning is requested.
    if conditional is None:
        conditional = analysis_setup._selected_vars_realisations

    # Create surrogate table.
    # if analysis_setup.settings['verbose']:
    #     print('\ncreating surrogate table with {0} permutations:'.format(
    #                                                                 n_perm))
    #     print('\tcand.', end='')
    surr_table = np.zeros((len(idx_test_set), n_perm))
    current_value_realisations = analysis_setup._current_value_realisations
    idx_c = 0
    for candidate in idx_test_set:
        # if analysis_setup.settings['verbose']:
        #     print('\t{0}'.format(analysis_setup._idx_to_lag([candidate])[0]),
        #           end='')
        if (analysis_setup._cmi_estimator.is_analytic_null_estimator() and
                permute_in_time):
            # Generate the surrogates analytically
            analysis_setup.settings['analytical_surrogates'] = True
            surr_table[idx_c, :] = (
                analysis_setup._cmi_estimator.estimate_surrogates_analytic(
                    n_perm=n_perm,
                    var1=data.get_realisations(analysis_setup.current_value,
                                               [candidate])[0],
                    var2=current_value_realisations,
                    conditional=conditional))
        else:
            analysis_setup.settings['analytical_surrogates'] = False
            surr_candidate_realisations = _get_surrogates(
                                                 data,
                                                 analysis_setup.current_value,
                                                 [candidate],
                                                 n_perm,
                                                 analysis_setup.settings)
            surr_table[idx_c, :] = (
                analysis_setup._cmi_estimator.estimate_parallel(
                    n_chunks=n_perm,
                    re_use=['var2', 'conditional'],
                    var1=surr_candidate_realisations,
                    var2=current_value_realisations,
                    conditional=conditional))
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


def _find_pvalue(statistic, distribution, alpha, tail):
    """Find p-value of a test statistic under some distribution.

    Args:
        statistic : numeric
            value to be tested against distribution
        distribution : numpy array
            1-dimensional distribution of values, test distribution
        alpha : float
            critical alpha level for statistical significance
        tail : str
            'one' or 'one_bigger' for one-tailed testing H1 > H0,
            'one_smaller' for one- tailed testing H1 < H0, or 'two' for two-
            tailed testing

    Returns:
        bool
            statistical significance
        float
            the test's p-value
    """
    assert alpha <= 1.0, 'Critical alpha levels needs to be smaller than 1.'
    assert distribution.ndim == 1, 'Test distribution must be 1D.'
    check_n_perm(distribution.shape[0], alpha)

    if tail == 'one_bigger' or tail == 'one':
        pvalue = sum(distribution >= statistic) / distribution.shape[0]
    elif tail == 'one_smaller':
        pvalue = sum(distribution <= statistic) / distribution.shape[0]
    elif tail == 'two':
        p_bigger = sum(distribution >= statistic) / distribution.shape[0]
        p_smaller = sum(distribution <= statistic) / distribution.shape[0]
        pvalue = min(p_bigger, p_smaller)
        alpha = alpha / 2
    else:
        raise ValueError(
            ('Unkown value for ''tail'', should be ''one'', ''one_bigger'','
             ' ''one_smaller'', or ''two''): {0}.'.format(tail)))

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


def _get_surrogates(data, current_value, idx_list, n_perm, perm_settings):
    """Return surrogate data for statistical testing.

    Calls surrogate generation methods of the data instance. The method for
    surrogate generation depends on whether sufficient replications of the data
    exists. If the number of replications is high enough (reps! >
    n_permutations), surrogates are created by shuffling data over replications
    (while keeping the temporal order of samples intact). If the number of
    replications is too low, samples are shuffled over time (while keeping the
    order of replications intact). The latter method can be forced by setting
    'permute_in_time' to True in 'perm_settings'.

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
        perm_settings : dict
            settings for surrogate creation by shuffling samples over time, set
            'permute_in_time' to True to create surrogates by shuffling data
            over time. See Data.permute_samples() for settings for surrogate
            creation.

    Returns:
        numpy array
            surrogate data with dimensions
            (realisations * n_perm) x len(idx_list)
    """
    # Allocate memory for surrogates
    n_realisations = data.n_realisations(current_value)
    surrogates = np.empty((n_realisations * n_perm,
                           len(idx_list))).astype(data.data_type)

    # Check if the user requested to permute samples in time and not over
    # replications
    permute_in_time = perm_settings['permute_in_time']

    # Generate surrogates by permuting over replications if possible (no.
    # replications needs to be sufficient); else permute samples over time.
    i_1 = 0
    i_2 = n_realisations
    # permute samples
    if permute_in_time:
        for perm in range(n_perm):
            surrogates[i_1:i_2, ] = data.permute_samples(current_value,
                                                         idx_list,
                                                         perm_settings)[0]
            i_1 = i_2
            i_2 += n_realisations

    else:  # permute replications
        assert _sufficient_replications(data, n_perm), (
                'Not enough replications for surrogate creation.')
        for perm in range(n_perm):
            surrogates[i_1:i_2, ] = data.permute_replications(current_value,
                                                              idx_list)[0]
            i_1 = i_2
            i_2 += n_realisations
    return surrogates


def _generate_spectral_surrogates(data, scale, n_perm, perm_settings):
    """Generate surrogate data for statistical testing of spectral TE.

    The method for surrogate generation depends on whether sufficient
    replications of the data exists. If the number of replications is high
    enough (reps! > n_permutations), surrogates are created by shuffling data
    over replications (while keeping the temporal order of samples intact). If
    the number of replications is too low, samples are shuffled over time
    (while keeping the order of replications intact).

    Args:
        data : Data instance
            raw data for analysis
        scale : int
            index of the scale to be shuffled
        n_perm : int
            number of permutations
        perm_settings : dict
            settings for surrogate creation by shuffling samples over time

    Returns:
        numpy array
            surrogate data with dimensions
            (realisations * n_perm) x len(idx_list)
    """
    # Allocate memory for surrogates
    surrogates = np.empty((data.n_samples, data.n_replications,
                           n_perm)).astype(data.data_type)
    permute_in_time = perm_settings['permute_in_time']
    # Generate surrogates by permuting over replications if possible (no.
    # replications needs to be sufficient); else permute samples over time.
    if permute_in_time:
        for perm in range(n_perm):
            surrogates[:, :, perm] = data.slice_permute_samples(
                                                    scale, perm_settings)[0]
    else:
        assert(_sufficient_replications(data, n_perm))
        for perm in range(n_perm):
            surrogates[:, :, perm] = data.slice_permute_replications(scale)[0]
    return surrogates


def _check_permute_in_time(analysis_setup, data, n_perm):
    """Set defaults for permuting samples in time.

    The default for creating surrogate data is the permutation of original data
    over replications (such that the temporal ordering of samples stays
    intact). The function checks if this default can be used given the
    requested number of permutations and the number of replications in the
    data.

    The function tries to set the setting 'permute_in_time' to its default
    'False' if no value for 'permute_in_time' was provided by the user. If the
    number of replications is insufficient to generate the requested number of
    permutations, the function sets 'permute_in_time' to true such that
    surrogates are created by permuting samples in time (if not requested
    otherwise the 'perm_type' is set to 'random', see documentation of
    Data().permute_samples() for further settings).
    """
    analysis_setup.settings.setdefault('permute_in_time', False)

    if (not analysis_setup.settings['permute_in_time'] and
            not _sufficient_replications(data, n_perm)):
        print('\nWARNING: Number of replications is not sufficient to '
              'generate the desired number of surrogates. Permuting samples '
              'in time instead.')
        analysis_setup.settings['permute_in_time'] = True

    if analysis_setup.settings['permute_in_time']:
        analysis_setup.settings.setdefault('perm_type', 'random')
    return analysis_setup.settings['permute_in_time']
