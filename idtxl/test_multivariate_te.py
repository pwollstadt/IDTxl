# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:22:14 2016

@author: patricia
"""
import random as rn
import pytest
import numpy as np
from multivariate_te import Multivariate_te
from data import Data


def test_multivariate_te_corr_gaussian():
    """Test multivariate TE estimation on correlated Gaussians.

    Run the multivariate TE algorithm on two sets of random Gaussian data with
    a given covariance. The second data set is shifted by one sample creating
    a source-target delay of one sample. This example is modeled after the
    JIDT demo 4 for transfer entropy. The resulting TE can be compared to the
    analytical result (but expect some error in the estimate).

    Note:
        This test runs considerably faster than other system tests
    """
    n = 1000
    cov = 0.4
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # source_2 = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1)
    # source_2 = np.expand_dims(np.array(source_2), axis=1)
    target = np.expand_dims(np.array(target), axis=1)

    dat = Data()
    dat.set_data(np.vstack((source_1[1:].T, target[:-1].T)), 'ps')
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 100,
        'n_perm_max_seq': 100,
        }
    random_analysis = Multivariate_te(max_lag_sources=5, options=analysis_opts)
    res_1 = random_analysis.analyse_single_target(dat, 1)
    # Assert that there are significant conditionals from the source for target
    # 1. For 500 repetitions I got mean errors of 0.02097686 and 0.01454073 for
    # examples 1 and 2 respectively. The maximum errors were 0.093841 and
    # 0.05833172 repectively. This inspired the following error boundaries.
    expected_res = np.log(1 / (1 - np.power(cov, 2)))
    diff = np.abs(max(res_1['cond_sources_te']) - expected_res)
    assert (diff < 0.1), ('Multivariate TE calculation for correlated '
                          'Gaussians failed (error larger 0.1).')


def test_multivariate_te_lagged_copies():
    """Test multivariate TE estimation on a lagged copy of random data.

    Run the multivariate TE algorithm on two sets of random data, where the
    second set is a lagged copy of the first. This test should find no
    significant conditionals at all (neither in the target's nor in the
    source's past).

    Note:
        This test takes several hours and may take one to two days on some
        machines.
    """
    lag = 3
    d_0 = np.random.rand(1, 1000, 20)
    d_1 = np.hstack((np.random.rand(1, lag, 20), d_0[:, lag:, :]))

    dat = Data()
    dat.set_data(np.vstack((d_0, d_1)), 'psr')
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        }
    random_analysis = Multivariate_te(max_lag_sources=5, options=analysis_opts)
    # Assert that there are no significant conditionals in either direction
    # other than the mandatory single sample in the target's past (which
    # ensures that we calculate a proper TE at any time in the algorithm).
    for target in range(2):
        res = random_analysis.analyse_single_target(dat, target)
        assert (len(res['conditional_full']) == 1), ('Conditional contains '
                                                     'more/less than 1 '
                                                     'variables.')
        assert (not res['conditional_sources']), ('Conditional sources is not '
                                                  'empty.')
        assert (len(res['conditional_target']) == 1), ('Conditional target '
                                                       'contains more/less '
                                                       'than 1 variable.')
        assert (res['cond_sources_pval'] is None), ('Conditional p-value is '
                                                    'not None.')
        assert (res['omnibus_pval'] is None), ('Omnibus p-value is not None.')
        assert (res['omnibus_sign'] is None), ('Omnibus significance is not '
                                               'None.')
        assert (res['conditional_sources_te'] is None), ('Conditional TE '
                                                         'values is not None.')


def test_multivariate_te_random():
    """Test multivariate TE estimation on two random data sets.

    Run the multivariate TE algorithm on two sets of random data with no
    coupling. This test should find no significant conditionals at all (neither
    in the target's nor in the source's past).

    Note:
        This test takes several hours and may take one to two days on some
        machines.
    """
    d = np.random.rand(2, 1000, 20)
    dat = Data()
    dat.set_data(d, 'psr')
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        }
    random_analysis = Multivariate_te(max_lag_sources=5, options=analysis_opts)
    # Assert that there are no significant conditionals in either direction
    # other than the mandatory single sample in the target's past (which
    # ensures that we calculate a proper TE at any time in the algorithm).
    for target in range(2):
        res = random_analysis.analyse_single_target(dat, target)
        assert (len(res['conditional_full']) == 1), ('Conditional contains '
                                                     'more/less than 1 '
                                                     'variables.')
        assert (not res['conditional_sources']), ('Conditional sources is not '
                                                  'empty.')
        assert (len(res['conditional_target']) == 1), ('Conditional target '
                                                       'contains more/less '
                                                       'than 1 variable.')
        assert (res['cond_sources_pval'] is None), ('Conditional p-value is '
                                                    'not None.')
        assert (res['omnibus_pval'] is None), ('Omnibus p-value is not None.')
        assert (res['omnibus_sign'] is None), ('Omnibus significance is not '
                                               'None.')
        assert (res['conditional_sources_te'] is None), ('Conditional TE '
                                                         'values is not None.')


def test_multivariate_te_lorenz_2():
    """Test multivariate TE estimation on bivariately couled Lorenz systems.

    Run the multivariate TE algorithm on two Lorenz systems with a coupling
    from first to second system with delay u = 45 samples. Both directions are
    analyzed, the algorithm should not find a coupling from system two to one.

    Note:
        This test takes several hours and may take one to two days on some
        machines.
    """
    d = np.load('/home/patricia/repos/IDTxl/testing/data/'
                'lorenz_2_exampledata.npy')
    dat = Data()
    dat.set_data(d[:, :, :50], 'psr')
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 21,  # 200
        'n_perm_min_stat': 21,  # 200
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        }
    lorenz_analysis = Multivariate_te(max_lag_sources=50, min_lag_sources=40,
                                      max_lag_target=20, options=analysis_opts)
    # res = lorenz_analysis.analyse_network(dat)
    res_0 = lorenz_analysis.analyse_single_target(dat, 0)
    res_1 = lorenz_analysis.analyse_single_target(dat, 1)


def test_multivariate_te():
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    max_lag_target = 5
    max_lag_sources = 7
    min_lag_sources = 4
    target = 0
    sources = [2, 3, 4]
    dat = Data()
    dat.generate_mute_data(100, 5)
    nw_0 = Multivariate_te(max_lag_sources, analysis_opts, min_lag_sources,
                           max_lag_target)
    nw_0.analyse_single_target(dat, target, sources)

    # Test what happens if the target max lag is bigger than the source max
    # lag
    max_lag_sources = 5
    max_lag_target = 7
    nw_1 = Multivariate_te(max_lag_sources, analysis_opts, min_lag_sources,
                           max_lag_target)
    res_1 = nw_1.analyse_single_target(dat, target, sources)

    # The following should crash: min lag bigger than max lag
    max_lag_sources = 5
    min_lag_sources = 7
    nw_2 = Multivariate_te(max_lag_sources, analysis_opts, min_lag_sources,
                           max_lag_target)
    with pytest.raises(AssertionError):
        nw_2.analyse_single_target(dat, target, sources)


def test_multivariate_te_initialise():
    """Test if all values are set correctly in _initialise()."""
    # Create a data set where one pattern fits into the time series exactly
    # once, this way, we get one realisation per replication for each variable.
    # This is easyer to assert/verify later. We also test data.get_realisations
    # this way.
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    max_lag_target = 5
    max_lag_sources = max_lag_target
    min_lag_sources = 4
    target = 1
    dat = Data(normalise=False)
    n_repl = 30
    n_procs = 2
    n_points = n_procs * (max_lag_sources + 1) * n_repl
    dat.set_data(np.arange(n_points).reshape(n_procs, max_lag_sources + 1,
                                             n_repl), 'psr')
    nw_0 = Multivariate_te(max_lag_sources, analysis_opts, min_lag_sources,
                           max_lag_target)
    nw_0._initialise(dat, 'all', target)
    assert (not nw_0.conditional_full)
    assert (not nw_0.conditional_sources)
    assert (not nw_0.conditional_target)
    assert ((nw_0._replication_index == np.arange(n_repl)).all())
    assert (nw_0._current_value == (target, max(max_lag_sources,
                                                max_lag_target)))
    assert ((nw_0._current_value_realisations ==
             np.arange(n_points - n_repl, n_points).reshape(n_repl, 1)).all())


def test_check_source_set():
    """Test the method _check_source_set.

    This method sets the list of source processes from which candidates are
    taken for multivariate TE estimation.
    """

    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag_sources = 7
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    nw_0 = Multivariate_te(max_lag_sources, analysis_opts)
    sources = [1, 2, 3]
    nw_0._check_source_set(sources, dat.n_processes)
    sources = [0, 1, 2, 3]
    nw_0 = Multivariate_te(max_lag_sources, analysis_opts)
    nw_0.target = 0
    failed_on_target_in_source = False
    try:
        nw_0._check_source_set(sources, dat.n_processes)
    except RuntimeError:
        failed_on_target_in_source = True
    assert (failed_on_target_in_source)
    sources = 1
    nw_0 = Multivariate_te(max_lag_sources, analysis_opts)
    nw_0._check_source_set(sources, dat.n_processes)
    assert (type(nw_0.source_set) is list)


def test_include_source_candidates():
    pass


def test_include_target_candidates():
    pass


def test_test_final_conditional():
    pass


def test_include_candidates():
    pass


def test_prune_candidates():
    pass


def test_separate_realisations():
    pass


def test_indices_to_lags():
    pass


if __name__ == '__main__':
    # test_multivariate_te_lorenz_2()
    # test_multivariate_te_random()
    test_multivariate_te_corr_gaussian()
    # test_multivariate_te_initialise()
    # test_multivariate_te()
    # test_check_source_set()
