"""
Test the MPI parallelization.

To run these tests, use:

mpirun -n 4 python test/test_mpi.py
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

from testutils import mpi_missing

N_PERM = (
    21  # this no. permutations is usually sufficient to find links in the MUTE data
)
N_SAMPLES = 1000
SEED = 0
MAX_WORKERS = 2

@mpi_missing
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

@mpi_missing
def test_mpi_installation():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert rank == 0, "Test not run on main rank"
    assert (
        size >= MAX_WORKERS + 1
    ), f"Insufficient MPI workers for testing. Please have at least {MAX_WORKERS + 1} threads available."

@mpi_missing
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

    # Estimate TE on workers and check if performance warning is raised
    with pytest.warns(RuntimeWarning, match="Performance warning:.*"):
        mpi_te_workers = mpiEstimator.estimate_parallel(var1=[data_array[0]] * MAX_WORKERS, var2=[data_array[1]] * MAX_WORKERS)

    # Check whether the results are the same
    assert np.array_equal(jidt_te, mpi_te), "MPI parallelization does not work correctly!"
    assert np.array_equal(jidt_te, mpi_te_workers[0]), "MPI parallelization does not work correctly!"
    assert all(np.array_equal(mpi_te, mpi_te_) for mpi_te_ in mpi_te_workers), "MPI parallelization does not work correctly!"

@mpi_missing
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

@mpi_missing
def test_lazy_array_base_array_error():

    # Create random data
    data_array = np.random.rand(2, N_SAMPLES)

    # Create Data object
    data = Data(data_array, dim_order="ps")
    var1 = data.get_realisations((0, 1), [(0, 1)])
    var2 = data.get_realisations((0, 1), [(1, 1)])

    # Create estimator
    mpiEstimator = MPIEstimator("JidtKraskovCMI", dict(max_workers=MAX_WORKERS))

    # Create second Data object with different base array
    data = Data(np.array(data_array), dim_order="ps")
    var1_other_base_array = data.get_realisations((0, 1), [(0, 1)])

    # Estimate TE on worker ranks with different base array for var1 and var2
    with pytest.raises(ValueError, match="LazyArrays must have the same base array"):
        _ = mpiEstimator.estimate_parallel(var1=[var1_other_base_array] * MAX_WORKERS, var2=[var2] * MAX_WORKERS)

    # Estimate TE on worker ranks with different base array for chunks of var1
    with pytest.raises(ValueError, match="LazyArrays must have the same base array"):
        _ = mpiEstimator.estimate_parallel(var1=[var1, var1_other_base_array], var2=[var2, var2])

@mpi_missing
def test_error_unequal_number_of_chunks():

    # Create random data
    data_array = np.random.rand(2, N_SAMPLES)

    # Create Data object
    data = Data(data_array, dim_order="ps")
    var1 = data.get_realisations((0, 1), [(0, 1)])
    var2 = data.get_realisations((0, 1), [(1, 1)])

    # Create estimator
    mpiEstimator = MPIEstimator("JidtKraskovCMI", dict(max_workers=MAX_WORKERS))

    # Estimate TE on worker ranks with different base array for var1 and var2
    with pytest.raises(ValueError, match="All variables must have the same number of chunks"):
        _ = mpiEstimator.estimate_parallel(var1=[var1] * 4, var2=[var2] * 5)

@mpi_missing
def test_caching():

    # Create random data
    data_array = np.random.rand(2, N_SAMPLES)

    # Create Data object
    data = Data(data_array, dim_order="ps")
    var1 = data.get_realisations((0, 1), [(0, 1)])
    var2 = data.get_realisations((0, 1), [(1, 1)])
    cond = data.get_realisations((0, 1), [(0, 1)])
        
    # Create estimator
    mpiEstimator = MPIEstimator("JidtKraskovCMI", dict(max_workers=MAX_WORKERS, noise_level=0))

    # Estimate TE on workers
    mpi_te_workers = mpiEstimator.estimate_parallel(var1=[var1] * 4 + [var2] * 4, var2=[var2] * 8, conditional=[cond, None] * 4)
    
    # Compare equal results
    assert mpi_te_workers[0] == mpi_te_workers[2]
    assert mpi_te_workers[1] == mpi_te_workers[3]
    assert mpi_te_workers[4] == mpi_te_workers[6]
    assert mpi_te_workers[5] == mpi_te_workers[7]

    # Run again
    mpi_te_workers2 = mpiEstimator.estimate_parallel(var1=[var2] * 4 + [var1] * 4, var2=[var2] * 8, conditional=[None, cond] * 4)
    
    # Compare equal results
    assert np.array_equal(mpi_te_workers, mpi_te_workers2[::-1])

if __name__ == "__main__":
    pytest.main([__file__])
