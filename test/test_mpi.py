"""
Test the MPI parallelization.

To run these tests, use:

mpirun -n 8 python -m mpi4py.futures -m pytest --with-mpi test/test_mpi.py
"""
import os
import time

import pytest
import numpy as np

from idtxl.estimator import get_estimator
from idtxl.estimators_jidt import JidtKraskov
from idtxl.estimators_mpi import MPIEstimator, _get_worker_estimator
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

# Skip test module if opencl is missing
pytest.importorskip("mpi4py")

from mpi4py import (  # pylint: disable=import-error,wrong-import-position,wrong-import-order
    MPI,
)


N_PERM = (
    21  # this no. permutations is usually sufficient to find links in the MUTE data
)
N_SAMPLES = 1000
SEED = 0
MAX_WORKERS = 4


def test_mpi_estimator_creation():
    """Check whether a JIDT Kraskov estimator with MPI is correctly initialized"""

    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "MPI": True,
        "max_workers": MAX_WORKERS,
    }

    estimator = get_estimator(settings["cmi_estimator"], settings)

    assert isinstance(estimator, MPIEstimator), "MPIEstimator has wrong class!"
    assert isinstance(
        _get_worker_estimator(estimator._id, None, None), JidtKraskov
    ), "MPIEstimator wraps the wrong estimator!"


@pytest.mark.mpi
def test_mpi_installation():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert rank == 0, "Test not run on main rank"
    assert (
        size >= MAX_WORKERS + 1
    ), f"Insufficient MPI workers for testing. Please have at least {MAX_WORKERS + 1} threads available."


def _wait_and_get_rank(arg):
    time.sleep(3)
    return MPI.COMM_WORLD.Get_rank()


@pytest.mark.mpi
@pytest.mark.xfail
def test_mpi_ranks():
    """
    Test whether the number of workers is restricted to max_workers, even if more threads are available.
    Currently, this does not work, which is most likely a bug in mpi4py.
    """
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "MPI": True,
        "max_workers": MAX_WORKERS,
    }

    estimator = get_estimator(settings["cmi_estimator"], settings)

    ranks = list(estimator._executor.map(_wait_and_get_rank, range(2 * MAX_WORKERS)))
    print(ranks)

    assert np.array_equal(
        np.unique(ranks), np.arange(1, MAX_WORKERS + 1)
    ), "The actual number of worker ranks is not equal to MAX_RANKS."


@pytest.mark.mpi
def test_mpi_JidtGaussianCMI_MTE():
    """Compute Multivariate transfer entropy for test data with and without MPI and compare results"""

    # a) Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)

    # b) Initialise analysis object with and without MPI
    network_analysis1 = MultivariateTE()
    settings1 = {
        "cmi_estimator": "JidtGaussianCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "MPI": True,
        "max_workers": MAX_WORKERS,
    }
    network_analysis2 = MultivariateTE()
    settings2 = {
        "cmi_estimator": "JidtGaussianCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 1,
    }

    # c) Run analysis
    print("Running MTE analysis with MPI")
    results1 = network_analysis1.analyse_network(settings=settings1, data=data)
    print("Running MTE analysis without MPI")
    results2 = network_analysis2.analyse_network(settings=settings2, data=data)

    # d) Compare results
    adj_matrix1 = results1.get_adjacency_matrix(weights="binary", fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights="binary", fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()

    print(f"{result1=}")
    print(f"{result2=}")

    assert np.array_equal(result1, result2), "Results with and without MPI differ!"


def _clear_ckp(filename):
    # Delete created checkpoint from disk.
    os.remove(f"{filename}.ckp")
    try:
        os.remove(f"{filename}.ckp.old")
    except FileNotFoundError:
        # If algorithm ran for only one iteration, no old checkpoint exists.
        print(f"No file {filename}.ckp.old")
    os.remove(f"{filename}.dat")
    os.remove(f"{filename}.json")


@pytest.mark.mpi
def test_mpi_checkpoint_resume():
    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data(seed=SEED)
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    filename_ckp = os.path.join(os.path.dirname(__file__), "data", "my_checkpoint")
    network_analysis = MultivariateTE()
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "max_lag_sources": 3,
        "min_lag_sources": 2,
        "write_ckp": True,
        "filename_ckp": filename_ckp,
        "MPI": True,
        "max_workers": MAX_WORKERS,
    }

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but
    # in all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._set_checkpointing_defaults(settings, data, sources, targets)
    for i in range(len(targets)):
        network_analysis._initialise(settings, data, sources, targets[i])
        network_analysis.selected_vars_full = add_vars[i]
        network_analysis._update_checkpoint("{}.ckp".format(filename_ckp))

    # Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp
    )

    # Test if variables were added correctly
    for i in range(len(targets)):
        network_analysis_res._initialise(settings, data, sources, targets[i])
        assert (
            network_analysis_res.selected_vars_full == add_vars[i]
        ), "Variables were not added correctly for target {0}.".format(targets[i])

    # Test if analysis runs from resumed checkpoint.
    network_analysis_res.analyse_network(
        settings=settings, data=data, targets=targets, sources=sources
    )

    _clear_ckp(filename_ckp)
