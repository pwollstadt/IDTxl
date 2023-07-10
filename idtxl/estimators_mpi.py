from .estimator import Estimator
from .estimator import get_estimator
from . import idtxl_exceptions as ex
from .idtxl_utils import timeout

import numpy as np

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError as err:
    ex.package_missing(err, 'MPI is not available on this system. Install it'
                       'from https://pypi.org/project/mpi4py/ to use'
                       'MPI parallelization.')
    raise err

def worker_estimate(tokens):
    return worker_estimate._estimator.access_data_and_estimate(**tokens)

def _init_worker(estimator_name, data, settings):
    worker_estimate._estimator = get_estimator(estimator_name, data, settings)

def unzip_dict(d):
    """Unzip a dictionary of lists into a list of dictionaries"""
    return [{k: v[i] if isinstance(v, list) else v for k, v in d.items()} for i in range(max([1] + [len(v) for v in d.values() if isinstance(v, list)]))]

class MPIEstimator(Estimator):
    """MPI Wrapper for arbitrary Estimator implementations

    Make sure to have an "if __name__=='__main__':" guard in your main script to avoid
    infinite recursion!

    To use MPI, add MPI=True to the Estimator settings dictionary and optionally provide max_workers

    Call using mpiexec:
    mpiexec -n 1 -usize <max workers + 1> python <python script>

    or, if MPI does not support spawning new workers (i.e. MPI version < 2)
    mpiexec -n <max workers + 1> python -m mpi4py.futures <python script>

    Call using slurm:
    srun -n $SLURM_NTASKS --mpi=pmi2 python -m mpi4py.futures <python script> 

    """

    def __init__(self, est, data, settings):
        
        _init_worker(est, data, settings)

        # Create the MPIPoolExecutor and initialize Estimators on worker ranks
        self._executor = MPIPoolExecutor(max_workers=settings.get('max_workers', None), initializer=_init_worker, initargs=(est, data, settings))

        # Boot up the executor with timeout
        with timeout(timeout_duration=settings.get('mpi_bootup_timeout', 10), exception_message='Bootup of MPI workers timed out.\n\
                Make sure the script was started in an MPI enrivonment using mpiexec, mpirun, srun (SLURM) or equivalent.\n\
                If necessary, increase the timeout in the settings dictionary using the key mpi_bootup_timeout.'):
            self._executor.bootup(wait=True)

    def __del__(self):
        """
        Shut down MPIPoolExecutor upon deletion of MPIEstimator
        """

        self._executor.shutdown()

    def estimate(self, **vars):
        raise NotImplementedError("This method is not implemented for this estimator.")

    def access_data_and_estimate_parallel(self, **tokens):
        """Distribute the tokens among the workers"""

        results_generator = self._executor.map(worker_estimate, unzip_dict(tokens))

        return np.fromiter(results_generator, dtype=np.double)

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        return worker_estimate._estimator.is_analytic_null_estimator()
    
    def estimate_surrogates_analytic(self, *args, **kwargs):
        return worker_estimate._estimator.estimate_surrogates_analytic(*args, **kwargs)
