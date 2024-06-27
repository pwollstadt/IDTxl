"""Test bivariate TE analysis class.

This module provides unit tests for the bivariate TE analysis class.
"""
import pytest
import itertools as it
import numpy as np
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.estimators_jidt import JidtKraskovTE
from test_estimators_jidt import jpype_missing
from test_results import _get_discrete_gauss_data
from idtxl.idtxl_utils import calculate_mi
from test_estimators_jidt import _get_gauss_data

SEED = 0


@jpype_missing
def test_gauss_data():
    """Test bivariate TE estimation from correlated Gaussians."""
    # Generate data and add a delay one one sample.
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]
    data = Data(
        np.hstack((source, source_uncorr, target)),
        dim_order="sp",
        normalise=False,
        seed=SEED,
    )
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": 1,
        "min_lag_sources": 1,
        "max_lag_target": 1,
    }
    nw = BivariateTE()
    results = nw.analyse_single_target(settings, data, target=2, sources=[0, 1])
    te = results.get_single_target(2, fdr=False)["te"][0]
    sources = results.get_target_sources(2, fdr=False)

    # Assert that only the correlated source was detected.
    assert len(sources) == 1, "Wrong no. inferred sources: {0}.".format(len(sources))
    assert sources[0] == 0, "Wrong inferred source: {0}.".format(sources[0])
    # Compare BivarateTE() estimate to JIDT estimate.
    est = JidtKraskovTE(
        {
            "history_target": 1,
            "history_source": 1,
            "source_target_delay": 1,
            "normalise": False,
        }
    )
    jidt_cmi = est.estimate(source=source, target=target)
    print(
        "Estimated TE: {0:0.6f}, estimated TE using JIDT core estimator: "
        "{1:0.6f} (expected: ~ {2:0.6f}).".format(te, jidt_cmi, expected_mi)
    )
    assert np.isclose(te, jidt_cmi, atol=0.005), (
        "Estimated TE {0:0.6f} differs from JIDT estimate {1:0.6f} (expected: "
        "TE {2:0.6f}).".format(te, jidt_cmi, expected_mi)
    )
    assert np.isclose(
        te, expected_mi, atol=0.05
    ), "Estimated TE {0:0.6f} differs from expected TE {1:0.6f}.".format(
        te, expected_mi
    )


@jpype_missing
def test_return_local_values():
    """Test estimation of local values."""
    max_lag = 5
    data = Data(seed=SEED)
    data.generate_mute_data(200, 5)
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "local_values": True,  # request calculation of local values
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": max_lag,
        "min_lag_sources": max_lag,
        "max_lag_target": max_lag,
    }
    target = 2
    te = BivariateTE()
    results_local = te.analyse_network(settings, data, targets=[target])

    lte = results_local.get_single_target(target, fdr=False)["te"]
    if lte is None:
        return
    n_sources = len(results_local.get_target_sources(target, fdr=False))
    assert (
        type(lte) is np.ndarray
    ), "LTE estimation did not return an array of values: {0}".format(lte)
    assert (
        lte.shape[0] == n_sources
    ), "Wrong dim (no. sources) in LTE estimate: {0}".format(lte.shape)
    assert lte.shape[1] == data.n_realisations_samples(
        (0, max_lag)
    ), "Wrong dim (no. samples) in LTE estimate: {0}".format(lte.shape)
    assert (
        lte.shape[2] == data.n_replications
    ), "Wrong dim (no. replications) in LTE estimate: {0}".format(lte.shape)

    # Test for correctnes of single link TE estimation by comparing it to the
    # TE between single variables and the target. For this test case where we
    # find only one significant past variable per source, the two should be the
    # same. Also compare single link average TE to mean local TE for each
    # link.
    settings["local_values"] = False
    results_avg = te.analyse_network(settings, data, targets=[target])
    te_single_link = results_avg.get_single_target(target, fdr=False)["te"]
    te_selected_sources = results_avg.get_single_target(target, fdr=False)[
        "selected_sources_te"
    ]
    sources_local = results_local.get_target_sources(target, fdr=False)
    sources_avg = results_avg.get_target_sources(target, fdr=False)
    assert np.isclose(
        te_single_link, te_selected_sources, atol=0.005
    ).all(), "Single link average TE {0} and single source TE {1} deviate.".format(
        te_single_link, te_selected_sources
    )
    # Check if average and local values are the same. Make sure target pasts
    # are the same and test each source separately. Inferred source and target
    # may differ between the two calls to analyse_network() due to random data
    # and low number of surrogates used in unit testing. Different no. inferred
    # past variables will also lead to differences in estimates.
    if (
        results_avg.get_single_target(target, fdr=False).selected_vars_target
        == results_local.get_single_target(target, fdr=False).selected_vars_target
    ):
        print("Compare average and local values.")
        for s in list(set(sources_avg).intersection(sources_local)):
            i1 = np.where(sources_avg == s)[0][0]
            i2 = np.where(sources_local == s)[0][0]
            assert np.isclose(te_single_link[i1], np.mean(lte[i2, :, :]), atol=0.005), (
                "Single link average TE {0:.6f} and mean LTE {1:.6f} deviate for "
                "source {2}.".format(te_single_link[i1], np.mean(lte[i2, :, :]), s)
            )
            assert np.isclose(
                te_single_link[i1], te_selected_sources[i1], atol=0.005
            ), (
                "Single link average TE {0:.6f} and single source TE {1:.6f} "
                "deviate.".format(te_single_link[i1], te_selected_sources[i1])
            )


@jpype_missing
def test_bivariate_te_init():
    """Test instance creation for BivariateTE class."""
    # Test error on missing estimator
    settings = {
        "n_perm_max_stat": 21,
        "n_perm_omnibus": 30,
        "n_perm_max_seq": 30,
        "max_lag_sources": 7,
        "min_lag_sources": 2,
        "max_lag_target": 5,
    }
    nw = BivariateTE()
    with pytest.raises(AssertionError):
        nw.analyse_single_target(settings=settings, data=Data(seed=SEED), target=1)

    # Test setting of min and max lags
    settings["cmi_estimator"] = "JidtKraskovCMI"
    data = Data(seed=SEED)
    data.generate_mute_data(n_samples=10, n_replications=5)

    # Valid: max lag sources bigger than max lag target
    nw.analyse_single_target(settings=settings, data=data, target=1)

    # Valid: max lag sources smaller than max lag target
    settings["max_lag_sources"] = 3
    nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: min lag sources bigger than max lag
    settings["min_lag_sources"] = 8
    settings["max_lag_sources"] = 7
    settings["max_lag_target"] = 5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: taus bigger than lags
    settings["min_lag_sources"] = 2
    settings["max_lag_sources"] = 4
    settings["max_lag_target"] = 5
    settings["tau_sources"] = 10
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["tau_sources"] = 1
    settings["tau_target"] = 10
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: negative lags or taus
    settings["min_lag_sources"] = 1
    settings["max_lag_target"] = 5
    settings["max_lag_sources"] = -7
    settings["tau_target"] = 1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["max_lag_sources"] = 7
    settings["min_lag_sources"] = -4
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["min_lag_sources"] = 4
    settings["max_lag_target"] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["max_lag_target"] = 5
    settings["tau_sources"] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["tau_sources"] = 1
    settings["tau_target"] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: lags or taus are no integers
    settings["tau_target"] = 1
    settings["min_lag_sources"] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["min_lag_sources"] = 1
    settings["max_lag_sources"] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["max_lag_sources"] = 7
    settings["tau_sources"] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["tau_sources"] = 1
    settings["tau_target"] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings["tau_target"] = 1

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
        nw.analyse_single_target(settings=settings, data=data, target=0, sources=-1)
    with pytest.raises(RuntimeError):  # negative
        nw.analyse_single_target(settings=settings, data=data, target=0, sources=[-1])
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=0, sources=20)
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=0, sources=[20])


@jpype_missing
def test_bivariate_te_one_realisation_per_replication():
    """Test boundary case of one realisation per replication."""
    # Create a data set where one pattern fits into the time series exactly
    # once, this way, we get one realisation per replication for each variable.
    # This is easyer to assert/verify later. We also test data.get_realisations
    # this way.
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "n_perm_max_stat": 21,
        "max_lag_target": 5,
        "max_lag_sources": 5,
        "min_lag_sources": 4,
    }
    target = 0
    data = Data(normalise=False)
    n_repl = 10
    n_procs = 2
    n_points = n_procs * (settings["max_lag_sources"] + 1) * n_repl
    data.set_data(
        np.arange(n_points).reshape(n_procs, settings["max_lag_sources"] + 1, n_repl),
        "psr",
    )
    nw = BivariateTE()
    nw._initialise(settings, data, "all", target)
    assert not nw.selected_vars_full
    assert not nw.selected_vars_sources
    assert not nw.selected_vars_target
    assert nw._current_value == (
        target,
        max(settings["max_lag_sources"], settings["max_lag_target"]),
    )
    assert (nw._current_value_realisations[:, 0] == data.data[target, -1, :]).all()


@jpype_missing
def test_faes_method():
    """Check if the Faes method is working."""
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "add_conditionals": "faes",
        "max_lag_sources": 5,
        "min_lag_sources": 3,
        "max_lag_target": 7,
    }
    nw_1 = BivariateTE()
    data = Data(seed=SEED)
    data.generate_mute_data()
    sources = [1, 2, 3]
    target = 0
    nw_1._initialise(settings, data, sources, target)
    assert nw_1._selected_vars_sources == [
        i for i in it.product(sources, [nw_1.current_value[1]])
    ], "Did not add correct additional conditioning vars."


@jpype_missing
def test_add_conditional_manually():
    """Enforce the conditioning on additional variables."""
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 3,
        "max_lag_target": 7,
    }
    nw = BivariateTE()
    data = Data(seed=SEED)
    data.generate_mute_data()

    # Add a conditional with a lag bigger than the max_lag requested above
    settings["add_conditionals"] = (8, 0)
    with pytest.raises(IndexError):
        nw.analyse_single_target(settings=settings, data=data, target=0)

    # Add valid conditionals and test if they were added
    settings["add_conditionals"] = [(0, 1), (1, 3)]
    nw._initialise(settings=settings, data=data, target=0, sources=[1, 2])
    # Get list of conditionals after intialisation and convert absolute samples
    # back to lags for comparison.
    cond_list = nw._idx_to_lag(nw.selected_vars_full)
    assert (
        settings["add_conditionals"][0] in cond_list
    ), "First enforced conditional is missing from results."
    assert (
        settings["add_conditionals"][1] in cond_list
    ), "Second enforced conditional is missing from results."


@jpype_missing
def test_check_source_set():
    """Test the method _check_source_set.

    This method sets the list of source processes from which candidates are
    taken for multivariate TE estimation.
    """
    data = Data(seed=SEED)
    data.generate_mute_data(100, 5)
    nw = BivariateTE()
    nw.settings = {"verbose": True}
    # Add list of sources.
    sources = [1, 2, 3]
    nw._check_source_set(sources, data.n_processes)
    assert nw.source_set == sources, "Sources were not added correctly."

    # Assert that initialisation fails if the target is also in the source list
    sources = [0, 1, 2, 3]
    nw.target = 0
    with pytest.raises(RuntimeError):
        nw._check_source_set(sources=[0, 1, 2, 3], n_processes=data.n_processes)

    # Test if a single source, no list is added correctly.
    sources = 1
    nw._check_source_set(sources, data.n_processes)
    assert type(nw.source_set) is list

    # Test if 'all' is handled correctly
    nw.target = 0
    nw._check_source_set("all", data.n_processes)
    assert nw.source_set == [1, 2, 3, 4], "Sources were not added correctly."

    # Test invalid inputs.
    with pytest.raises(RuntimeError):  # sources greater than no. procs
        nw._check_source_set(8, data.n_processes)
    with pytest.raises(RuntimeError):  # negative value as source
        nw._check_source_set(-3, data.n_processes)


@jpype_missing
def test_define_candidates():
    """Test candidate definition from a list of procs and a list of samples."""
    target = 1
    tau_sources = 3
    max_lag_sources = 10
    current_val = (target, 10)
    procs = [target]
    samples = np.arange(
        current_val[1] - 1, current_val[1] - max_lag_sources, -tau_sources
    )
    # Test if candidates that are added manually to the conditioning set are
    # removed from the candidate set.
    nw = BivariateTE()
    nw.current_value = current_val
    settings = [
        {"add_conditionals": None},
        {"add_conditionals": (2, 3)},
        {"add_conditionals": [(2, 3), (4, 1)]},
        {"add_conditionals": [(1, 9)]},
        {"add_conditionals": [(1, 9), (2, 3), (4, 1)]},
    ]
    for s in settings:
        nw.settings = s
        candidates = nw._define_candidates(procs, samples)
        assert (1, 9) in candidates, "Sample missing from candidates: (1, 9)."
        assert (1, 6) in candidates, "Sample missing from candidates: (1, 6)."
        assert (1, 3) in candidates, "Sample missing from candidates: (1, 3)."
        if s["add_conditionals"] is not None:
            if type(s["add_conditionals"]) is tuple:
                cond_ind = nw._lag_to_idx([s["add_conditionals"]])
            else:
                cond_ind = nw._lag_to_idx(s["add_conditionals"])
            for c in cond_ind:
                assert (
                    c not in candidates
                ), "Sample added erronously to candidates: {}.".format(c)


@jpype_missing
def test_analyse_network():
    """Test method for full network analysis."""
    n_processes = 5  # the MuTE network has 5 nodes
    data = Data(seed=SEED)
    data.generate_mute_data(10, 5)
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 30,
        "max_lag_sources": 5,
        "min_lag_sources": 4,
        "max_lag_target": 5,
    }
    nw = BivariateTE()

    # Test all to all analysis
    results = nw.analyse_network(settings, data, targets="all", sources="all")
    targets_analysed = results.targets_analysed
    sources = np.arange(n_processes)
    assert all(
        np.array(targets_analysed) == np.arange(n_processes)
    ), "Network analysis did not run on all targets."
    for t in targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(
            np.array(results._single_target[t].sources_tested) == s
        ), "Network analysis did not run on all sources for target " "{0}".format(t)
    # Test analysis for subset of targets
    target_list = [1, 2, 3]
    results = nw.analyse_network(settings, data, targets=target_list, sources="all")
    targets_analysed = results.targets_analysed
    assert all(
        np.array(targets_analysed) == np.array(target_list)
    ), "Network analysis did not run on correct subset of targets."
    for t in targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(
            np.array(results._single_target[t].sources_tested) == s
        ), "Network analysis did not run on all sources for target " "{0}".format(t)

    # Test analysis for subset of sources
    source_list = [1, 2, 3]
    target_list = [0, 4]
    results = nw.analyse_network(
        settings, data, targets=target_list, sources=source_list
    )
    targets_analysed = results.targets_analysed
    assert all(
        np.array(targets_analysed) == np.array(target_list)
    ), "Network analysis did not run for all targets."
    for t in results.targets_analysed:
        assert all(results._single_target[t].sources_tested == np.array(source_list)), (
            "Network analysis did not run on the correct subset "
            "of sources for target {0}".format(t)
        )


def test_discrete_input():
    """Test bivariate TE estimation from discrete data."""
    # Generate Gaussian test data
    covariance = 0.4
    data = _get_discrete_gauss_data(
        covariance=covariance, n=10000, delay=1, normalise=False, seed=SEED
    )
    corr_expected = covariance / (1 * np.sqrt(covariance**2 + (1 - covariance) ** 2))
    expected_mi = calculate_mi(corr_expected)
    settings = {
        "cmi_estimator": "JidtDiscreteCMI",
        "discretise_method": "none",
        "n_discrete_bins": 5,  # alphabet size of the variables analysed
        "n_perm_max_stat": 21,
        "n_perm_omnibus": 30,
        "n_perm_max_seq": 30,
        "min_lag_sources": 1,
        "max_lag_sources": 2,
        "max_lag_target": 1,
    }
    nw = BivariateTE()
    res = nw.analyse_single_target(settings=settings, data=data, target=1)
    assert np.isclose(res._single_target[1].omnibus_te, expected_mi, atol=0.05), (
        "Estimated TE for discrete variables is not correct. Expected: "
        "{0}, Actual results: {1}.".format(
            expected_mi, res._single_target[1].omnibus_te
        )
    )


@jpype_missing
def test_mute_data():
    """Test estimation from MuTE data."""
    max_lag = 5
    data = Data(seed=SEED)
    data.generate_mute_data(200, 5)
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": max_lag,
        "min_lag_sources": 1,
        "add_conditionals": [(1, 3), (1, 2)],
        "max_lag_target": max_lag,
    }
    target = 2
    te = BivariateTE()
    te.analyse_network(settings, data, targets=[target])


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


def test_compare_bivariate_and_multivariate_te():
    """Compare bivariate to multivariate TE estimation."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]
    data = Data(
        np.hstack((source, source_uncorr, target)),
        dim_order="sp",
        normalise=False,
        seed=SEED,
    )
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": 1,
        "min_lag_sources": 1,
        "max_lag_target": 1,
    }
    nw_bivar = BivariateTE()
    results = nw_bivar.analyse_single_target(settings, data, target=2, sources=[0, 1])
    te_bivar = results.get_single_target(2, fdr=False)["te"][0]

    nw_multivar = MultivariateTE()
    results = nw_multivar.analyse_single_target(
        settings, data, target=2, sources=[0, 1]
    )
    te_multivar = results.get_single_target(2, fdr=False)["te"][0]

    print(
        "Estimated TE: {0:0.6f}, estimated TE using multivariate algorithm: "
        "{1:0.6f} (expected: ~ {2:0.6f}).".format(te_bivar, te_multivar, expected_mi)
    )
    assert np.isclose(te_bivar, te_multivar, atol=0.005), (
        "Estimated TE {0:0.6f} differs from multivariate estimate {1:0.6f} "
        "(expected: TE {2:0.6f}).".format(te_bivar, te_multivar, expected_mi)
    )


if __name__ == "__main__":
    test_compare_bivariate_and_multivariate_te()
    test_mute_data()
    test_return_local_values()
    test_gauss_data()
    test_discrete_input()
    test_analyse_network()
    test_check_source_set()
    test_bivariate_te_init()
    test_bivariate_te_one_realisation_per_replication()
    test_faes_method()
    test_add_conditional_manually()
    test_check_source_set()
    test_define_candidates()
