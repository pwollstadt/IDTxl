from .estimator import Estimator
from .estimator import get_estimator
from . import idtxl_exceptions as ex
from .idtxl_utils import timeout

import numpy as np
import itertools
from uuid import uuid4

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError as err:
    ex.package_missing(
        err,
        "MPI is not available on this system. Install it"
        "from https://pypi.org/project/mpi4py/ to use"
        "MPI parallelization.",
    )

_worker_estimators = {}
"""Estimator instances on worker ranks

Used so that Estimators do not have to be created new for each task given to them.
Estimators are indexed by the ID of the corresponding MPIEstimator instance on the MPI main rank.
"""


def _get_worker_estimator(id_, est, settings):
    """Return Estimator instance on worker rank.
    If no Estimator for the given MPIEstimator id exists, create a new one
    """

    # Create new estimator if necessary
    if id_ not in _worker_estimators:
        # There is currently no good way to delete Estimators from _worker_estimators
        # caches when the corresponding MPIEstimator ceases to exist.
        # To avoid memory leaks, we currently allow only a single cached estimator that is replaced for new MPIEstimators.
        _worker_estimators.clear()

        _worker_estimators[id_] = get_estimator(est, settings)

    return _worker_estimators[id_]


def _dispatch_task(id_, est, settings, data):
    """Estimates a single chunk of data on an MPI worker rank.
    Calls the estimate function of the base Estimator
    """

    estimator = _get_worker_estimator(id_, est, settings)

    if estimator.is_parallel():
        return estimator.estimate(n_chunks=1, **data)
    else:
        return estimator.estimate(**data)


class MPIEstimator(Estimator):
    """MPI Wrapper for arbitrary Estimator implementations

    Make sure to have an "if __name__=='__main__':" guard in your main script
    to avoid infinite recursion!

    To use MPI, add MPI=True to the Estimator settings dictionary and
    optionally provide max_workers

    Call using mpiexec:
        >>> mpiexec -n 1 -usize <max workers + 1> python <python script>

    or, if MPI does not support spawning new workers (i.e. MPI version < 2)
        >>> mpiexec -n <max workers + 1> python -m mpi4py.futures <python script>

    Call using slurm:
        >>> srun -n $SLURM_NTASKS --mpi=pmi2 python -m mpi4py.futures <python script>
    """

    def __init__(self, est, settings):
        """Creates new MPIEstimator instance

        Immediately creates instances of est on each MPI worker.

        Args:
            est : str | Callable[[dict], Estimator]
                Name of of or callable returning an instance of the base
                Estimator
            settings : dict
                settings for the base Estimator.
            max_workers : int (optional)
                Number of MPI workers, default=MPI_UNIVERSE_SIZE
        """

        self._est = est
        self._settings = self._check_settings(settings).copy()

        # Create unique id for this instance to access cached estimators
        self._id = uuid4().int

        # Create the MPIPoolExecutor and initialize Estimators on worker ranks
        self._executor = MPIPoolExecutor(max_workers=settings.get("max_workers", None))

        # Boot up the executor with timeout
        with timeout(
            timeout_duration=settings.get("mpi_bootup_timeout", 10),
            exception_message="Bootup of MPI workers timed out.\n\
                Make sure the script was started in an MPI enrivonment using mpiexec, mpirun, srun (SLURM) or equivalent.\n\
                If necessary, increase the timeout in the settings dictionary using the key mpi_bootup_timeout.",
        ):
            self._executor.bootup(wait=True)

        # Create Estimator for rank 0.
        _get_worker_estimator(self._id, est, settings)

    def __del__(self):
        """Shut down MPIPoolExecutor upon deletion of MPIEstimator"""
        self._executor.shutdown()

    def _chunk_data(self, data, chunksize, n_chunks):
        """
        Iterator chopping data dictionary into n_chunks chunks of size chunksize
        """
        for i in range(n_chunks):
            yield {
                var: (
                    None
                    if data[var] is None
                    else data[var][i * chunksize : (i + 1) * chunksize]
                )
                for var in data
            }

    def estimate(self, *, n_chunks=1, **data):
        """Distributes the given chunks of a task to Estimators on worker ranks using MPI.

        Needs to be called with kwargs only.

        Args:
            n_chunks : int  [optional]
                Number of chunks to split the data into, default=1.
            data : dict[str, Sequence]
                Dictionary of random variable realizations
        Returns:
            numpy array
                Estimates of information-theoretic quantities as np.double
                values
        """

        assert n_chunks > 0, "Number of chunks must be at least one."

        samplesize = len(next(iter(data.values())))

        assert all(
            var is None or len(var) == samplesize for var in data.values()
        ), "All variables must have the same number of realizations."

        assert (
            samplesize % n_chunks == 0
        ), "Number of realizations must be divisible by number of chunks!"

        # Split the data into chunks
        chunksize = samplesize // n_chunks

        chunked_data = self._chunk_data(data, chunksize, n_chunks)

        result_generator = self._executor.map(
            _dispatch_task,
            itertools.repeat(self._id),
            itertools.repeat(self._est),
            itertools.repeat(self._settings),
            chunked_data,
        )

        return np.fromiter(result_generator, dtype=np.double)

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        """Test if the base Estimator is an analytic null estimator."""

        return _get_worker_estimator(
            self._id, self._est, self._settings
        ).is_analytic_null_estimator()

    def estimate_surrogates_analytic(self, **data):
        """Forward analytic estimation to the base Estimator.

        Analytic estimation is assumed to have shorter runtime and is thus
        performed on rank 0 alone for now.
        """

        return _get_worker_estimator(
            self._id, self._est, self._settings
        ).estimate_surrogates_analytic(**data)
