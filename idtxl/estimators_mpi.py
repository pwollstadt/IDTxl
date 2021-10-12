from .estimator import Estimator
from .estimator import get_estimator
from . import idtxl_exceptions as ex
import numpy as np

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError as err:
    ex.package_missing(err, 'MPI is not available on this system. Install it'
                       'from https://pypi.org/project/mpi4py/ to use'
                       'MPI parallelization.')
    raise err


class MPIEstimator(Estimator):
    """MPI Wrapper for arbitrary Estimator implementations

    Upon calling estimate, one instance of the base Estimator is created
    on each worker rank.

    Requires MPI 2 or later as MPIPoolExecutor makes use of the spawn command.

    Make sure to have an "if __name__=='__main__':" guard in your script to avoid
    infinite recursion!

    Call using mpiexec:
    mpiexec -n 1 -usize <max workers + 1> python <python script>

    Call using slurm:
    srun -n $SLURM_NTASKS --mpi=pmi2 python -m mpi4py.futures <python script> 

    """

    def __init__(self, est, settings):
        """Creates new MPIEstimator instance

        Args:
            est (str): Name of the base Estimator
            settings (dict): settings for the base Estimator.
                max_workers (optional): Number of MPI workers. Default: MPI_UNIVERSE_SIZE
        """

        self._settings = self._check_settings(settings).copy()
        self._settings['estimator'] = est

        # Create the MPIPoolExecutor
        self._executor = MPIPoolExecutor(
            max_workers=self._settings.get('max_workers', None), path=['./test'])

        # Create Estimator for rank 0.
        self._estimator = self._create_estimator()

    def __del__(self):
        """De-initializer

        """

        # If this is rank 0, clean up MPIPoolExecutor
        if self._executor is not None:
            self._executor.shutdown()

    def __getstate__(self):
        """Returns the state dict of the MPIEstimator instance for pickling.

        Called on rank 0 to broadcast self to the workers.
        Exclude both self._executor and self._estimator from pickling.

        Returns:
            dict: [description]
        """

        # Do not pickle _executor or _estimator.
        state = self.__dict__.copy()
        del state['_executor']
        del state['_estimator']
        return state

    def __setstate__(self, state):
        """Restores the instance from a pickle state dict.

        Called on worker ranks.
        self._executor is not necessar on the worker ranks, but a static _estimator is created.

        Args:
            state (dict): state dictionary
        """

        # Restore instance attributes (exept for self._executor and self._estimator).
        self.__dict__.update(state)
        self._executor = None

        # Create separate Estimator instances on the worker ranks
        self._estimator = self._create_estimator()

    def _create_estimator(self):
        """Creates the Estimator

        """

        return get_estimator(self._settings['estimator'], self._settings)

    def estimate(self, n_chunks=1, **data):
        """Distributes the given chunks of a task to Estimators on worker ranks using MPI.

        Needs to be called with kwargs only.

        Args:
            n_chunks (int, optional): Number of chunks to split the data into. Defaults to 1.

        Returns:
            numpy array: Estimates of information-theoretic quantities
        """

        assert n_chunks > 0, 'Number of chunks must be at least one.'

        samplesize = len(list(data.values())[0])

        assert all(var is None or len(var) == samplesize for var in data.values(
        )), 'All variables must have the same number of realizations.'

        # Split the data into chunks
        chunksize = samplesize // n_chunks

        chunked_data = ({var: (None if data[var] is None else data[var][i*chunksize:min(
            (i+1)*chunksize, samplesize)]) for var in data} for i in range(n_chunks))

        # This call implicitly pickles self and sends it to the worker ranks along with the chunked data
        return np.fromiter(self._executor.map(self._estimate_single_chunk, chunked_data), dtype=np.double)

    def _estimate_single_chunk(self, data):
        """Estimates a single chunk of data on an MPI worker rank.

        Calls the estimate function of the base Estimator

        """

        if self._estimator.is_parallel():
            return self._estimator.estimate(n_chunks=1, **data)
        else:
            return self._estimator.estimate(**data)

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        """Test if the base Estimator is an analytic null estimator.

        """

        return self._estimator.is_analytic_null_estimator()

    def estimate_surrogates_analytic(self, **data):
        """Forward analytic estimation to the base Estimator.

        Analytic estimation is assumed to have negligible runtime and is thus performed on rank 0 alone.
        """

        return self._estimator.estimate_surrogates_analytic(**data)
