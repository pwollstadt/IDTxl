"""Provide unit tests for multivariate TE estimation.

Created on Fri Mar 25 12:22:14 2016

@author: patricia
"""
import pytest
import itertools as it
import numpy as np
from idtxl.multivariate_te import Multivariate_te
from idtxl.data import Data


def test_multivariate_te_init():
    """Test instance creation for Multivariate_te class."""
    # Test error on missing estimator
    with pytest.raises(KeyError):
        Multivariate_te(max_lag_sources=5,
                        min_lag_sources=3,
                        max_lag_target=7,
                        options={})

    # Test setting of min and max lags
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    dat = Data()
    dat.generate_mute_data(100, 5)

    # Valid: max lag sources bigger than max lag target
    Multivariate_te(max_lag_sources=5,
                    min_lag_sources=3,
                    max_lag_target=7,
                    options=analysis_opts)

    # Valid: max lag sources smaller than max lag target
    Multivariate_te(max_lag_sources=7,
                    min_lag_sources=3,
                    max_lag_target=5,
                    options=analysis_opts)

    # Invladid: min lag sources bigger than max lag
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=7,
                             min_lag_sources=8,
                             max_lag_target=5,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')

    # Invalid: negative lags or taus
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=-7,
                             min_lag_sources=-4,
                             max_lag_target=-1,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=1,
                             min_lag_sources=-4,
                             max_lag_target=-1,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=1,
                             min_lag_sources=1,
                             max_lag_target=-1,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=1,
                             min_lag_sources=1,
                             max_lag_target=1,
                             tau_sources=-1,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=1,
                             min_lag_sources=1,
                             max_lag_target=1,
                             tau_target=-1,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')

    # Invalid: lags or taus are no integers
    with pytest.raises(AssertionError):
        nw = Multivariate_te(max_lag_sources=3,
                             min_lag_sources=1.5,
                             max_lag_target=1,
                             options=analysis_opts)
        nw.analyse_single_target(data=dat, target=0, sources='all')

    # Invalid: sources or target is no int
    nw = Multivariate_te(max_lag_sources=3,
                         min_lag_sources=1.5,
                         max_lag_target=1,
                         options=analysis_opts)
    with pytest.raises(AssertionError):
        nw.analyse_single_target(data=dat, target=1.5, sources='all')
    with pytest.raises(AssertionError):
        nw.analyse_single_target(data=dat, target=-1, sources='all')
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(data=dat, target=10, sources='all')
    with pytest.raises(AssertionError):
        nw.analyse_single_target(data=dat, target={}, sources='all')
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(data=dat, target=0, sources=-1)
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(data=dat, target=0, sources=[-1])
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(data=dat, target=0, sources=20)
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(data=dat, target=0, sources=[20])

    # Force conditionals
    analysis_opts['add_conditionals'] = [(0, 1), (1, 3)]
    nw = Multivariate_te(max_lag_sources=3,
                         min_lag_sources=1.5,
                         max_lag_target=1,
                         options=analysis_opts)
    analysis_opts['add_conditionals'] = (8, 0)
    nw = Multivariate_te(max_lag_sources=3,
                         min_lag_sources=1.5,
                         max_lag_target=1,
                         options=analysis_opts)


def test_multivariate_te_one_realisation_per_replication():
    """Test boundary case of one realisation per replication."""
    # Create a data set where one pattern fits into the time series exactly
    # once, this way, we get one realisation per replication for each variable.
    # This is easyer to assert/verify later. We also test data.get_realisations
    # this way.
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    max_lag_target = 5
    max_lag_sources = max_lag_target
    min_lag_sources = 4
    target = 0
    dat = Data(normalise=False)
    n_repl = 10
    n_procs = 2
    n_points = n_procs * (max_lag_sources + 1) * n_repl
    dat.set_data(np.arange(n_points).reshape(n_procs, max_lag_sources + 1,
                                             n_repl), 'psr')
    nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, analysis_opts,
                           max_lag_target)
    nw_0._initialise(dat, 'all', target)
    assert (not nw_0.selected_vars_full)
    assert (not nw_0.selected_vars_sources)
    assert (not nw_0.selected_vars_target)
    assert ((nw_0._replication_index == np.arange(n_repl)).all())
    assert (nw_0._current_value == (target, max(max_lag_sources,
                                                max_lag_target)))
    assert (nw_0._current_value_realisations[:, 0] ==
            dat.data[target, -1, :]).all()


def test_faes_method():
    """Check if the Faes method is working."""
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov',
                     'add_conditionals': 'faes'}
    nw_1 = Multivariate_te(max_lag_sources=5,
                           min_lag_sources=3,
                           max_lag_target=7,
                           options=analysis_opts)
    dat = Data()
    dat.generate_mute_data()
    sources = [1, 2, 3]
    target = 0
    nw_1._initialise(dat, sources, target)
    assert (nw_1._selected_vars_sources ==
            [i for i in it.product(sources, [nw_1.current_value[1]])]), (
                'Did not add correct additional conditioning vars.')


def test_add_conditional_manually():
    """Adda variable that is not in the data set."""
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov',
                     'add_conditionals': (8, 0)}
    nw_1 = Multivariate_te(max_lag_sources=5,
                           min_lag_sources=3,
                           options=analysis_opts,
                           max_lag_target=7)
    dat = Data()
    dat.generate_mute_data()
    sources = [1, 2, 3]
    target = 0
    with pytest.raises(IndexError):
        nw_1._initialise(dat, sources, target)


def test_check_source_set():
    """Test the method _check_source_set.

    This method sets the list of source processes from which candidates are
    taken for multivariate TE estimation.
    """
    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag_sources = 7
    min_lag_sources = 5
    max_lag_target = 5
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, analysis_opts,
                           max_lag_target)
    # Add list of sources.
    sources = [1, 2, 3]
    nw_0._check_source_set(sources, dat.n_processes)
    assert nw_0.source_set == sources, 'Sources were not added correctly.'

    # Assert that initialisation fails if the target is also in the source list
    sources = [0, 1, 2, 3]
    nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, analysis_opts,
                           max_lag_target)
    nw_0.target = 0
    with pytest.raises(RuntimeError):
        nw_0._check_source_set(sources=[0, 1, 2, 3],
                               n_processes=dat.n_processes)

    # Test if a single source, no list is added correctly.
    sources = 1
    nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, analysis_opts,
                           max_lag_target)
    nw_0._check_source_set(sources, dat.n_processes)
    assert (type(nw_0.source_set) is list)

    # Test if 'all' is handled correctly
    nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, analysis_opts,
                           max_lag_target)
    nw_0.target = 0
    nw_0._check_source_set('all', dat.n_processes)
    assert nw_0.source_set == [1, 2, 3, 4], 'Sources were not added correctly.'

    # Test invalid inputs.
    nw_0 = Multivariate_te(max_lag_sources, min_lag_sources, analysis_opts,
                           max_lag_target)
    with pytest.raises(RuntimeError):   # sources greater than no. procs
        nw_0._check_source_set(8, dat.n_processes)
    with pytest.raises(RuntimeError):  # negative value as source
        nw_0._check_source_set(-3, dat.n_processes)


def test_define_candidates():
    """Test candidate definition from a list of procs and a list of samples."""
    analysis_opts = {'cmi_calc_name': 'jidt_kraskov'}
    target = 1
    tau_target = 3
    max_lag_target = 10
    current_val = (target, 10)
    procs = [target]
    samples = np.arange(current_val[1] - 1, current_val[1] - max_lag_target,
                        -tau_target)
    nw = Multivariate_te(5, 1, analysis_opts, 5)
    candidates = nw._define_candidates(procs, samples)
    assert (1, 9) in candidates, 'Sample missing from candidates: (1, 9).'
    assert (1, 6) in candidates, 'Sample missing from candidates: (1, 6).'
    assert (1, 3) in candidates, 'Sample missing from candidates: (1, 3).'


def test_analyse_network():
    """Test method for full network analysis."""
    n_processes = 5  # the MuTE network has 5 nodes
    dat = Data()
    dat.generate_mute_data(10, 5)
    nw_0 = Multivariate_te(max_lag_sources=5,
                           min_lag_sources=4,
                           options={'cmi_calc_name': 'jidt_kraskov'},
                           max_lag_target=5)

    # Test all to all analysis
    r = nw_0.analyse_network(dat, targets='all', sources='all')
    try:
        del r['fdr']
    except:
        pass
    k = list(r.keys())
    sources = np.arange(n_processes)
    assert all(np.array(k) == np.arange(n_processes)), (
                'Network analysis did not run on all targets.')
    for t in r.keys():
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(r[t]['sources_tested']) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))
    # Test analysis for subset of targets
    target_list = [1, 2, 3]
    r = nw_0.analyse_network(dat, targets=target_list, sources='all')
    try:
        del r['fdr']
    except:
        pass
    k = list(r.keys())
    assert all(np.array(k) == np.array(target_list)), (
                'Network analysis did not run on correct subset of targets.')
    for t in r.keys():
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(r[t]['sources_tested']) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))

    # Test analysis for subset of sources
    source_list = [1, 2, 3]
    target_list = [0, 4]
    r = nw_0.analyse_network(dat, targets=target_list, sources=source_list)
    try:
        del r['fdr']
    except:
        pass
    k = list(r.keys())
    assert all(np.array(k) == np.array(target_list)), (
                'Network analysis did not run for all targets.')
    for t in r.keys():
        assert all(r[t]['sources_tested'] == np.array(source_list)), (
            'Network analysis did not run on the correct subset of sources '
            'for target {0}'.format(t))


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
    test_analyse_network()
    test_check_source_set()
    test_multivariate_te_init()  # test init function of the Class
    test_multivariate_te_one_realisation_per_replication()
    test_faes_method()
    test_add_conditional_manually()
    test_check_source_set()
    test_define_candidates()
