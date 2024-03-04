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
from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.estimators_mpi import MPIEstimator
import idtxl.estimators_mpi as estimators_mpi
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
MAX_WORKERS = 2

# Decorator to skip tests if MPI is not available
def skip_if_not_mpi(func):
    if MPI.COMM_WORLD.Get_size() < 2:
        return pytest.mark.skip(reason="MPI not available")(func)
    return func

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
        estimators_mpi._worker_estimator, JidtKraskovCMI
    ), "MPIEstimator wraps the wrong estimator!"

@skip_if_not_mpi
def test_mpi_installation():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert rank == 0, "Test not run on main rank"
    assert (
        size >= MAX_WORKERS + 1
    ), f"Insufficient MPI workers for testing. Please have at least {MAX_WORKERS + 1} threads available."

@skip_if_not_mpi
def test_mpi_estimation():
    """Test whether the MPI parallelization works correctly"""

    # Create random data
    data_array = np.random.rand(2, N_SAMPLES)

    # Create Data object
    data = Data(data_array, dim_order="ps")

    # Create estimator
    jidtEstimator = JidtKraskovCMI()
    mpiEstimator = MPIEstimator("JidtKraskovCMI", dict(max_workers=MAX_WORKERS))

    # Estimate TE on main rank
    jidt_te = jidtEstimator.estimate(var1=data_array[0], var2=data_array[1])
    mpi_te = mpiEstimator.estimate(var1=data_array[0], var2=data_array[1])

    # Estimate TE on workers
    mpi_te_workers = mpiEstimator.estimate_parallel(var1=[data_array[0]] * MAX_WORKERS, var2=[data_array[1]] * MAX_WORKERS)

    # Check whether the results are the same
    assert np.array_equal(jidt_te, mpi_te), "MPI parallelization does not work correctly!"
    assert np.array_equal(jidt_te, mpi_te_workers[0]), "MPI parallelization does not work correctly!"
    assert all(np.array_equal(mpi_te, mpi_te_) for mpi_te_ in mpi_te_workers), "MPI parallelization does not work correctly!"


def test_lazy_array_estimation():
    """Test whether the MPI parallelization works correctly with lazy arrays"""

    # Create random data
    data_array = np.random.rand(2, N_SAMPLES)

    # Create Data object
    data = Data(data_array, dim_order="ps")
    var1 = data.get_realisations((0, 1), [(0, 1)])
    var2 = data.get_realisations((0, 1), [(1, 1)])

    # Create estimator
    jidtEstimator = JidtKraskovCMI()
    mpiEstimator = MPIEstimator("JidtKraskovCMI", dict(max_workers=MAX_WORKERS))

    # Estimate TE on main rank
    jidt_te = jidtEstimator.estimate(var1=var1, var2=var2)
    mpi_te = mpiEstimator.estimate(var1=var1, var2=var2)

    # Estimate TE on workers
    mpi_te_workers = mpiEstimator.estimate_parallel(var1=[var1] * MAX_WORKERS, var2=[var2] * MAX_WORKERS)

    # Check whether the results are the same
    assert np.array_equal(jidt_te, mpi_te), "MPI parallelization does not work correctly!"
    assert np.array_equal(jidt_te, mpi_te_workers[0]), "MPI parallelization does not work correctly!"
    assert all(np.array_equal(mpi_te, mpi_te_) for mpi_te_ in mpi_te_workers), "MPI parallelization does not work correctly!"

if __name__ == "__main__":
    pytest.main([__file__])