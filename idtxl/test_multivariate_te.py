# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:22:14 2016

@author: patricia
"""
import numpy as np
from multivariate_te import Multivariate_te
from data import Data


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
    try:
        nw_2.analyse_single_target(dat, target, sources)
    except AssertionError:
        print('Test correctly failed due to wrong user input.')


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
    test_multivariate_te_initialise()
    # test_multivariate_te()
    test_check_source_set()
