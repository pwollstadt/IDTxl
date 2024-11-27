from functools import reduce
import warnings
import atexit

import numpy as np

from idtxl.lazy_array import LazyArray
from idtxl.estimator import get_estimator
from idtxl import idtxl_exceptions as ex

try:
    from mpi4py import MPI
except ImportError as err:
    MPI = None

class tags:
    SET_ESTIMATOR = 0
    SET_DATA = 1
    ESTIMATE = 2
    DONE = 3

if MPI is not None:

    # Get MPI world communicator
    _comm_world = MPI.COMM_WORLD
    _size_world = _comm_world.Get_size()
    _rank_world = _comm_world.Get_rank()

    # Split comm into node comms with shared memory
    _comm_node = _comm_world.Split_type(MPI.COMM_TYPE_SHARED)
    _size_node = _comm_node.Get_size()
    _rank_node = _comm_node.Get_rank()

    # Split comm into rank_node == 0 and rank_node > 0
    _internode_color = int(_rank_node != 0)
    _comm_internode = _comm_world.Split(_internode_color, _rank_world)
    _size_internode = _comm_internode.Get_size()
    _rank_internode = _comm_internode.Get_rank()

class MPIEstimator():

    def __init__(self, est, settings=None):

        if MPI is None:
            ex.package_missing(
                err,
                "MPI is not available on this system. Install it"
                "from https://pypi.org/project/mpi4py/ to use"
                "MPI parallelization.",
            )
        
        # Check that at least two ranks are available
        if _size_world < 2:
            raise ValueError('MPIEstimator requires at least two MPI ranks. Make sure to run your script with mpirun -n <n_ranks> python <script.py>')
        elif _size_world == 2:
            warnings.warn('Only two ranks available. MPI reserves one main rank, thus no speedup is expected.', RuntimeWarning)
        
        self._mpi_batch_size = settings.get('mpi_batch_size', 1)
        
        _bcast_estimator(est, settings)

    def estimate_parallel(self, **data):

        n_tasks = len(data[list(data.keys())[0]])

        # Ensure that all variables have the same number of chunks
        if not all(len(var) == n_tasks for var in data.values()):
            raise ValueError('All variables must have the same number of chunks')
        
        # If only one task, estimate on rank 0
        if n_tasks == 1:
            return _worker_estimator.estimate_parallel(**data)
        
        # If lazy arrays are used, make sure they all have the same base array.
        lazyArrays = [var for vars in data.values() for var in vars if isinstance(var, LazyArray)]

        if lazyArrays:
            base_array_id = lazyArrays[0]._original_base_array_id

            if not all(var._original_base_array_id == base_array_id for var in lazyArrays):
                raise ValueError('LazyArrays must have the same base array')

        # Broadcast shared data to all workers if necessary
        if lazyArrays and base_array_id != _worker_data_id:
            _bcast_shared_data(lazyArrays[0]._base_array)

        n_workers = _size_world - 1

        # Chunk data
        batch_sizes = self.compute_batch_sizes(n_tasks, self._mpi_batch_size, n_workers)
            
        return _estimate(self._chunk_data(data, batch_sizes=batch_sizes), n_batches=len(batch_sizes))
    
    def compute_batch_sizes(self, n_tasks, max_batch_size, n_workers):

        n_full_batches = n_tasks // (max_batch_size * n_workers) * n_workers
        remaining_tasks = n_tasks - n_full_batches * max_batch_size

        n_partial_batches = min(n_workers, remaining_tasks)

        batch_sizes = np.empty(n_full_batches + n_partial_batches, dtype=np.int32)

        batch_sizes[:n_full_batches] = max_batch_size

        if n_partial_batches:
            size_partial_batches = remaining_tasks // n_partial_batches
            batch_sizes[n_full_batches:] = size_partial_batches
            batch_sizes[n_full_batches:n_full_batches + remaining_tasks % n_partial_batches] += 1

        assert batch_sizes.sum() == n_tasks

        return batch_sizes
    
    def estimate(self, **data):
        return _worker_estimator.estimate(**data)
    
    def _chunk_data(self, data, batch_sizes):
        def get_batch(batch_idx, previous):
            """

            Careful: previous is modified inplace!
            """
            n_tasks = batch_sizes[batch_idx]
            
            dicts = [None] * n_tasks
            offset = batch_sizes[:batch_idx].sum()
            for i in range(n_tasks):
                idx = offset + i

                if previous is None:
                    d = {key: var[idx] for key, var in data.items()}
                else:
                    d = {key: var[idx] for key, var in data.items() if var[idx] is not previous.get(key, 'somevalue')}

                # Remove base_array from lazy arrays
                for key, var in d.items():
                    if isinstance(var, LazyArray):
                        if id(var._base_array) != _worker_data_id:
                            d[key] = var[:]
                            warnings.warn('Performance warning: LazyArray was copied to worker.', RuntimeWarning)
                    elif var is not None:
                        warnings.warn('Performance warning: Non-Lazy Array was copied to worker.', RuntimeWarning)
                
                dicts[i] = d
                previous.update(d) # previous is modified inplace!

            return dicts
        return get_batch

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        """Test if the base Estimator is an analytic null estimator.

        """
        return _worker_estimator.is_analytic_null_estimator()

    def estimate_surrogates_analytic(self, **data):
        """Forward analytic estimation to the base Estimator.
        Analytic estimation is assumed to have shorter runtime and is thus performed on rank 0 alone for now.
        """
        return _worker_estimator.estimate_surrogates_analytic(**data)
        
_worker_estimator = None

def _bcast_estimator(est=None, settings=None):
    """Broadcasts an estimator and its settings to all nodes.
    """
    global _worker_estimator

    if _rank_world == 0:
        _comm_world.bcast(tags.SET_ESTIMATOR, root=0)

    est, settings = _comm_world.bcast((est, settings), root=0)
    _worker_estimator = get_estimator(est, settings)

_worker_data = None
_worker_data_id = None

def _bcast_shared_data(data=None):
    global _worker_data, _worker_data_id

    if _rank_world == 0:
        _comm_world.bcast(tags.SET_DATA, root=0)

    # First broadcast dtype and shape
    if _rank_world == 0:
        dtype, shape, id_ = data.dtype, data.shape, id(data)
    else:
        dtype, shape, id_ = None, None, None

    dtype, shape, id_ = _comm_world.bcast((dtype, shape, id_), root=0)

    # Set _worker_data_id
    _worker_data_id = id_

    # Create shared memory array on each node
    size = dtype.itemsize * reduce(lambda x, y: x * y, shape)
    win = MPI.Win.Allocate_shared(size=size if _rank_node == 0 else 0, disp_unit=dtype.itemsize, comm=_comm_node)
    buf, itemsize = win.Shared_query(rank=0)
    shared_data = np.ndarray(buffer=buf, dtype=dtype, shape=shape)

    #  Set shared data on world rank 0
    if _rank_world == 0:
        shared_data[:] = data

    _comm_world.Barrier()

    # Broadcast data to rank 0 of each node
    if _internode_color == 0:
        _comm_internode.Bcast(shared_data, root=0)

    _comm_world.Barrier()
    
    # Set read-only flag
    shared_data.flags.writeable = False

    _worker_data = shared_data
    
    return shared_data

def _stop_workers():
    """Stops all workers when rank 0 terminates.
    """

    if _rank_world == 0:
        _comm_world.bcast(tags.DONE, root=0)

if MPI is not None and _rank_world == 0:
    atexit.register(_stop_workers)

def _estimate(batch_gen, n_batches):
    n_workers = _size_world - 1

    # Set workers to estimation mode
    _comm_world.bcast(tags.ESTIMATE, root=0)

    results = [None] * n_batches

    # Initial synchronisation barrier
    _comm_world.Barrier()

    # Cache last lazy arrays
    previous = [{} for _ in range(n_workers)]

    # Send initial tasks
    for i in range(min(n_workers, n_batches)):
        chunk = batch_gen(i, previous[i]) # previous is modified as a side effect
        _comm_world.send((i, chunk), dest=i+1)

    status = MPI.Status()

    # Receive results and send new tasks until all tasks are done
    for i in range(min(n_workers, n_batches), n_batches):
        result_idx, result = _comm_world.recv(status=status)
        worker_rank = status.Get_source()

        results[result_idx] = result

        # Send new task
        chunk = batch_gen(i, previous[worker_rank-1]) # previous is modified as a side effect
        _comm_world.send((i, chunk), dest=worker_rank)

    # Receive remaining results
    for _ in range(min(n_workers, n_batches)):
        result_idx, result = _comm_world.recv(status=status)
        worker_rank = status.Get_source()
        results[result_idx] = result

    # Send stop signal
    for i in range(n_workers):
        _comm_world.send((None, None), dest=i+1)

    # Synchronize all workers
    _comm_world.Barrier()

    # Turn results into array
    results = np.concatenate(results)

    return results

def _worker_loop():

    while True:

        tag = _comm_world.bcast(None, root=0)

        if tag == tags.SET_ESTIMATOR:
            _bcast_estimator()
        elif tag == tags.SET_DATA:
            _bcast_shared_data()
        elif tag == tags.ESTIMATE:
            _worker_estimate()
        elif tag == tags.DONE:
            exit(0)

def _worker_estimate():

    # Wait for first barrier
    _comm_world.Barrier()

    previous = {}

    # Start estimation loop
    while True:
        # Receive data from rank 0
        batch_idx, batches = _comm_world.recv(source=0)

        # If batch is None, exit loop
        if batch_idx is None:
            break

        # Create array for results
        results = np.ndarray(len(batches), dtype=np.double)

        for i, data in enumerate(batches):

            # Otherwise, estimate
            for varname, var in data.items():
                if isinstance(var, LazyArray):
                    if _worker_data is None:
                        raise ValueError('_worker_data must not be None')
                    if var._original_base_array_id != _worker_data_id:
                        raise ValueError(f'LazyArray base array for vareiable {varname} must be shared with worker')
                    var.set_base_array(_worker_data)

            # Merge data with previous data
            previous.update(data)
            data = previous


            if _worker_estimator.is_parallel():
                result =  _worker_estimator.estimate(n_chunks=1, **data)
            else:
                result =  _worker_estimator.estimate(**data)

            results[i] = result

        # Send result to rank 0
        _comm_world.send((batch_idx, results), dest=0, tag=tags.ESTIMATE)

    # Synchronize all workers
    _comm_world.Barrier()

if MPI is not None:
    if _rank_world != 0:
        _worker_loop()