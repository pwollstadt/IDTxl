"""Provide unit tests for multivariate MI estimation.

Created on Fri Mar 25 12:22:14 2016

@author: patricia
"""
import pytest
import itertools as it
import numpy as np
from idtxl.multivariate_mi import MultivariateMI
from idtxl.data import Data
from idtxl.estimators_jidt import JidtDiscreteCMI
from test_estimators_jidt import jpype_missing
from idtxl.idtxl_utils import calculate_mi


@jpype_missing
def test_return_local_values():
    """Test estimation of local values."""
    max_lag = 5
    data = Data()
    data.generate_mute_data(500, 5)
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'local_values': True,  # request calculation of local values
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_max_seq': 21,
        'n_perm_omnibus': 21,
        'max_lag_sources': max_lag,
        'min_lag_sources': 4,
        'max_lag_target': max_lag}
    target = 1
    mi = MultivariateMI()
    results = mi.analyse_network(settings, data, targets=[target])

    # Test if any sources were inferred. If not, return (this may happen
    # sometimes due to too few samples, however, a higher no. samples is not
    # feasible for a unit test).
    if results.get_single_target(target, fdr=False)['mi'] is None:
        return

    lmi = results.get_single_target(target, fdr=False)['mi']
    n_sources = len(results.get_target_sources(target, fdr=False))
    assert type(lmi) is np.ndarray, (
        'LMI estimation did not return an array of values')
    assert lmi.shape[0] == n_sources, (
        'Wrong dim (no. sources) in LMI estimate')
    assert lmi.shape[1] == data.n_realisations_samples((0, max_lag)), (
        'Wrong dim (no. samples) in LMI estimate')
    assert lmi.shape[2] == data.n_replications, (
        'Wrong dim (no. replications) in LMI estimate')

    # Test for correctnes of single link MI estimation by comparing it to the
    # omnibus MI. In this case (single source), the two should be the same.
    settings['local_values'] = False
    results_avg = mi.analyse_network(settings, data, targets=[target])
    if results_avg.get_single_target(target, fdr=False)['mi'] is None:
        return
    mi_single_link = results_avg.get_single_target(target, fdr=False)['mi'][0]
    mi_omnibus = results_avg.get_single_target(target, fdr=False)['omnibus_mi']
    assert np.isclose(mi_single_link, mi_omnibus, rtol=0.00005), (
        'Single link MI is not equal to omnibus information.')
    # Compare mean local MI to average MI.
    assert np.isclose(mi_single_link, np.mean(lmi), rtol=0.00005), (
        'Single link average MI and mean LMI deviate.')


@jpype_missing
def test_multivariate_mi_init():
    """Test instance creation for MultivariateMI class."""
    # Test error on missing estimator
    settings = {
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'max_lag_sources': 7,
        'min_lag_sources': 2}
    nw = MultivariateMI()
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=Data(), target=1)

    # Test setting of min and max lags
    settings['cmi_estimator'] = 'JidtKraskovCMI'
    data = Data()
    data.generate_mute_data(n_samples=10, n_replications=5)

    # Invalid: min lag sources bigger than max lag
    settings['min_lag_sources'] = 8
    settings['max_lag_sources'] = 7
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: tau bigger than lags
    settings['min_lag_sources'] = 2
    settings['max_lag_sources'] = 4
    settings['tau_sources'] = 10
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: negative lags or taus
    settings['tau_sources'] = 1
    settings['min_lag_sources'] = 1
    settings['max_lag_sources'] = -7
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['max_lag_sources'] = 7
    settings['min_lag_sources'] = -4
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['min_lag_sources'] = 4
    settings['tau_sources'] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: lags or taus are no integers
    settings['min_lag_sources'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['min_lag_sources'] = 1
    settings['max_lag_sources'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['max_lag_sources'] = 7
    settings['tau_sources'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: sources or target is no int
    with pytest.raises(RuntimeError):  # no int
        nw.analyse_single_target(settings=settings, data=data, target=1.5)
    with pytest.raises(RuntimeError):  # negative
        nw.analyse_single_target(settings=settings, data=data, target=-1)
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=10)
    with pytest.raises(RuntimeError):  # wrong type
        nw.analyse_single_target(settings=settings, data=data, target={})
    with pytest.raises(RuntimeError):  # negative
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=-1)
    with pytest.raises(RuntimeError):   # negative
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=[-1])
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=20)
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=[20])


@jpype_missing
def test_multivariate_mi_one_realisation_per_replication():
    """Test boundary case of one realisation per replication."""
    # Create a data set where one pattern fits into the time series exactly
    # once, this way, we get one realisation per replication for each variable.
    # This is easyer to assert/verify later. We also test data.get_realisations
    # this way.
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 21,
        'max_lag_sources': 5,
        'min_lag_sources': 4}
    target = 0
    data = Data(normalise=False)
    n_repl = 10
    n_procs = 2
    n_points = n_procs * (settings['max_lag_sources'] + 1) * n_repl
    data.set_data(np.arange(n_points).reshape(
                                        n_procs,
                                        settings['max_lag_sources'] + 1,
                                        n_repl), 'psr')
    nw_0 = MultivariateMI()
    nw_0._initialise(settings, data, 'all', target)
    assert (not nw_0.selected_vars_full)
    assert (not nw_0.selected_vars_sources)
    assert (not nw_0.selected_vars_target)
    assert ((nw_0._replication_index == np.arange(n_repl)).all())
    assert (nw_0._current_value == (target, settings['max_lag_sources']))
    assert (nw_0._current_value_realisations[:, 0] ==
            data.data[target, -1, :]).all()


@jpype_missing
def test_faes_method():
    """Check if the Faes method is working."""
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'add_conditionals': 'faes',
                'max_lag_sources': 5,
                'min_lag_sources': 3}
    nw_1 = MultivariateMI()
    data = Data()
    data.generate_mute_data()
    sources = [1, 2, 3]
    target = 0
    nw_1._initialise(settings, data, sources, target)
    assert (nw_1._selected_vars_sources ==
            [i for i in it.product(sources, [nw_1.current_value[1]])]), (
                'Did not add correct additional conditioning vars.')


@jpype_missing
def test_add_conditional_manually():
    """Enforce the conditioning on additional variables."""
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 3}
    nw = MultivariateMI()
    data = Data()
    data.generate_mute_data()

    # Add a conditional with a lag bigger than the max_lag requested above
    settings['add_conditionals'] = (8, 0)
    with pytest.raises(IndexError):
        nw._initialise(settings, data, sources=[1, 2], target=0)

    # Add valid conditionals and test if they were added
    settings['add_conditionals'] = [(0, 1), (1, 3)]
    nw._initialise(settings=settings, data=data, target=0, sources=[1, 2])
    # Get list of conditionals after intialisation and convert absolute samples
    # back to lags for comparison.
    cond_list = nw._idx_to_lag(nw.selected_vars_full)
    assert settings['add_conditionals'][0] in cond_list, (
        'First enforced conditional is missing from results.')
    assert settings['add_conditionals'][1] in cond_list, (
        'Second enforced conditional is missing from results.')


@jpype_missing
def test_check_source_set():
    """Test the method _check_source_set.

    This method sets the list of source processes from which candidates are
    taken for multivariate MI estimation.
    """
    data = Data()
    data.generate_mute_data(100, 5)
    nw_0 = MultivariateMI()
    nw_0.settings = {'verbose': True}
    # Add list of sources.
    sources = [1, 2, 3]
    nw_0._check_source_set(sources, data.n_processes)
    assert nw_0.source_set == sources, 'Sources were not added correctly.'

    # Assert that initialisation fails if the target is also in the source list
    sources = [0, 1, 2, 3]
    nw_0.target = 0
    with pytest.raises(RuntimeError):
        nw_0._check_source_set(sources=[0, 1, 2, 3],
                               n_processes=data.n_processes)

    # Test if a single source, no list is added correctly.
    sources = 1
    nw_0._check_source_set(sources, data.n_processes)
    assert (type(nw_0.source_set) is list)

    # Test if 'all' is handled correctly
    nw_0.target = 0
    nw_0._check_source_set('all', data.n_processes)
    assert nw_0.source_set == [1, 2, 3, 4], 'Sources were not added correctly.'

    # Test invalid inputs.
    with pytest.raises(RuntimeError):   # sources greater than no. procs
        nw_0._check_source_set(8, data.n_processes)
    with pytest.raises(RuntimeError):  # negative value as source
        nw_0._check_source_set(-3, data.n_processes)


@jpype_missing
def test_define_candidates():
    """Test candidate definition from a list of procs and a list of samples."""
    target = 1
    tau_sources = 3
    max_lag_sources = 10
    current_val = (target, 10)
    procs = [target]
    samples = np.arange(current_val[1] - 1, current_val[1] - max_lag_sources,
                        -tau_sources)
    nw = MultivariateMI()
    candidates = nw._define_candidates(procs, samples)
    assert (1, 9) in candidates, 'Sample missing from candidates: (1, 9).'
    assert (1, 6) in candidates, 'Sample missing from candidates: (1, 6).'
    assert (1, 3) in candidates, 'Sample missing from candidates: (1, 3).'


@jpype_missing
def test_analyse_network():
    """Test method for full network analysis."""
    n_processes = 5  # the MuTE network has 5 nodes
    data = Data()
    data.generate_mute_data(10, 5)
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_max_seq': 21,
        'n_perm_omnibus': 21,
        'max_lag_sources': 5,
        'min_lag_sources': 4}
    nw_0 = MultivariateMI()

    # Test all to all analysis
    results = nw_0.analyse_network(
        settings, data, targets='all', sources='all')
    targets_analysed = results.targets_analysed
    sources = np.arange(n_processes)
    assert all(np.array(targets_analysed) == np.arange(n_processes)), (
                'Network analysis did not run on all targets.')
    for t in targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(results._single_target[t].sources_tested) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))
    # Test analysis for subset of targets
    target_list = [1, 2, 3]
    results = nw_0.analyse_network(
        settings, data, targets=target_list, sources='all')
    targets_analysed = results.targets_analysed
    assert all(np.array(targets_analysed) == np.array(target_list)), (
                'Network analysis did not run on correct subset of targets.')
    for t in targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(results._single_target[t].sources_tested) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))

    # Test analysis for subset of sources
    source_list = [1, 2, 3]
    target_list = [0, 4]
    results = nw_0.analyse_network(
        settings, data, targets=target_list, sources=source_list)
    targets_analysed = results.targets_analysed
    assert all(np.array(targets_analysed) == np.array(target_list)), (
                'Network analysis did not run for all targets.')
    for t in targets_analysed:
        assert all(results._single_target[t].sources_tested ==
                   np.array(source_list)), (
                        'Network analysis did not run on the correct subset '
                        'of sources for target {0}'.format(t))


@jpype_missing
def test_permute_time():
    """Create surrogates by permuting data in time instead of over replic."""
    # Test if perm type is set to default
    default = 'random'
    data = Data()
    data.generate_mute_data(10, 5)
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 21,
        'n_perm_max_seq': 21,
        'n_perm_omnibus': 21,
        'max_lag_sources': 5,
        'min_lag_sources': 4,
        'permute_in_time': True}
    nw_0 = MultivariateMI()
    results = nw_0.analyse_network(
        settings, data, targets='all', sources='all')
    assert results.settings.perm_type == default, (
        'Perm type was not set to default.')


def test_discrete_input():
    """Test multivariate MI estimation from discrete data."""
    # Generate Gaussian test data
    covariance = 0.4
    n = 10000
    delay = 1
    source = np.random.normal(0, 1, size=n)
    target = (covariance * source + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    corr_expected = covariance / (
        1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    source = source[delay:]
    target = target[:-delay]

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    data = Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=False)
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,  # alphabet size of the variables analysed
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 1,
        'max_lag_sources': 2}
    nw = MultivariateMI()
    res = nw.analyse_single_target(settings=settings, data=data, target=1)
    assert np.isclose(
        res._single_target[1].omnibus_mi, expected_mi, atol=0.05), (
            'Estimated MI for discrete variables is not correct. Expected: '
            '{0}, Actual results: {1}.'.format(
                expected_mi, res['selected_sources_te'][0]))


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
    test_return_local_values()
    test_analyse_network()
    test_discrete_input()
    test_check_source_set()
    test_multivariate_mi_init()  # test init function of the Class
    test_multivariate_mi_one_realisation_per_replication()
    test_faes_method()
    test_add_conditional_manually()
    test_check_source_set()
    test_define_candidates()
