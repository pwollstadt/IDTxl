from functools import reduce
import warnings
import atexit

import numpy as np

from idtxl.lazy_array import LazyArray

from idtxl.estimator import get_estimator

try:
    from mpi4py import MPI
except ImportError:
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
            raise ImportError('mpi4py is not installed')
        
        _bcast_estimator(est, settings)

    def estimate_parallel(self, **data):
        
        # If lazy arrays are used, broadcast base array to all nodes
        lazyArray = None
        for var in data.values():
            if isinstance(var[0], LazyArray):
                lazyArray = var[0]
                break

        if lazyArray is not None and lazyArray._base_array_id != _worker_data_id:
            _bcast_shared_data(lazyArray._base_array)

        # Chunk data
        n_chunks = len(data[list(data.keys())[0]])
        return _estimate(self._chunk_data(data), n_chunks)
    
    def estimate(self, **data):
        return _worker_estimator.estimate(**data)
    
    def _chunk_data(self, data):
        """
        Iterator chopping data dictionary into n_chunks chunks of size chunksize
        """
        def get_chunk(idx, previous=None):
            if previous is None:
                d = {key: var[idx] for key, var in data.items()}
            else:
                d = {key: var[idx] for key, var in data.items() if var[idx] is not previous[key]}
            

            # Remove base_array from lazy arrays
            for key, var in d.items():
                if isinstance(var, LazyArray):
                    if id(var._base_array) != _worker_data_id:
                        d[key] = var[:]
                        warnings.warn('Performance warning: LazyArray was copied to worker.', RuntimeWarning)
                elif var is not None:
                    warnings.warn('Performance warning: Non-Lazy Array was copied to worker.', RuntimeWarning)
            
            return d
        return get_chunk

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        """Test if the base Estimator is an analytic null estimator.

        """
        return False

    def estimate_surrogates_analytic(self, **data):
        """Forward analytic estimation to the base Estimator.

        Analytic estimation is assumed to have shorter runtime and is thus
        performed on rank 0 alone for now.
        """
        return False
        
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

if _rank_world == 0:
    atexit.register(_stop_workers)

def _estimate(chunk_gen, n_chunks):

    # Set workers to estimation mode
    _comm_world.bcast(tags.ESTIMATE, root=0)

    results = np.empty(n_chunks, dtype=np.double)

    # Initial synchronisation barrier
    _comm_world.Barrier()

    # Cache last lazy arrays
    previous = [None] * (_size_world - 1)

    # Send initial tasks
    for i in range(min(_size_world - 1, n_chunks)):
        chunk = chunk_gen(i)
        previous[i] = chunk
        _comm_world.send((i, chunk), dest=i+1)

    status = MPI.Status()

    # Receive results and send new tasks until all tasks are done
    for i in range(i, n_chunks):
        result_idx, result = _comm_world.recv(status=status)
        worker_rank = status.Get_source()

        results[result_idx] = result

        # Send new task
        chunk = chunk_gen(i, previous=previous[worker_rank-1])
        previous[worker_rank - 1].update(chunk)

        _comm_world.send((i, chunk), dest=worker_rank)

    # Receive remaining results
    for _ in range(min(_size_world - 1, n_chunks)):
        result_idx, result = _comm_world.recv(status=status)
        worker_rank = status.Get_source()
        results[result_idx] = result

    # Send stop signal
    for i in range(1, _size_world):
        _comm_world.send((None, None), dest=i)

    # Synchronize all workers
    _comm_world.Barrier()

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
        data_idx, data = _comm_world.recv(source=0)

        # If tag is 0, exit loop
        if data_idx is None:
            break

        # Otherwise, estimate
        for key, var in data.items():
            if isinstance(var, LazyArray):
                if var._base_array is not None:
                    raise ValueError('LazyArray base array must be None')
                if _worker_data is None:
                    raise ValueError('_worker_data must not be None')
                if var._base_array_id != _worker_data_id:
                    raise ValueError('LazyArray base array must be shared with worker')
                var.set_base_array(_worker_data)

        # Merge data with previous data
        data.update(previous)
        previous = data

        idletime += time.perf_counter() - last
        last = time.perf_counter()

        if _worker_estimator.is_parallel():
            result =  _worker_estimator.estimate(n_chunks=1, **data)
        else:
            result =  _worker_estimator.estimate(**data)

        # Send result to rank 0
        _comm_world.send((data_idx, result), dest=0, tag=tags.ESTIMATE)

    # Synchronize all workers
    _comm_world.Barrier()

if MPI is not None:
    if _rank_world != 0:
        _worker_loop()