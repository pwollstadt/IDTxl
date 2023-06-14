from multiprocessing import get_context, Pool

import numpy as np

from .estimator import Estimator, get_estimator

def worker_estimate(tokens):
    return worker_estimate._estimator.access_data_and_estimate(**tokens)

def _init_worker(estimator_name, data, settings):
    worker_estimate._estimator = get_estimator(estimator_name, data, settings)

def unzip_dict(d):
    """Unzip a dictionary of lists into a list of dictionaries"""
    return [{k: v[i] if isinstance(v, list) else v for k, v in d.items()} for i in range(max([1] + [len(v) for v in d.values() if isinstance(v, list)]))]

class MultiprocessingEstimator(Estimator):

    def __init__(self, estimator_name, data, settings):

        # Create worker for the main process.
        _init_worker(estimator_name, data, settings)

        # If the start method is fork, we can use shared memory and do not need to re-initialize the worker.
        if get_context().get_start_method() == 'fork':
            self._pool = Pool(processes=settings.get('n_processes', None))
        else:
            # Otherwise, we need to re-initialize the worker in each process.
            self._pool = Pool(processes=settings.get('n_processes', None), initializer=_init_worker, initargs=(estimator_name, data, settings))
    
    def access_data_and_estimate_parallel(self, **tokens):
        """Distribute the tokens among the workers"""

        results_generator = self._pool.map(worker_estimate, unzip_dict(tokens))

        return np.fromiter(results_generator, dtype=np.double)
    
    def estimate(self, **vars):
        raise NotImplementedError("This method is not implemented for this estimator.")

    def is_parallel(self):
        return True
    
    def is_analytic_null_estimator(self):
        return worker_estimate._estimator.is_analytic_null_estimator()
    
    def estimate_surrogates_analytic(self, *args, **kwargs):
        return worker_estimate._estimator.estimate_surrogates_analytic(*args, **kwargs)