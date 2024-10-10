import copy

import numpy as np

def lazy(method):
    """
    Decorator that makes a method lazy.

    """

    def lazy_method(self, *args, **kwargs):

        new_array = copy.copy(self)
        new_array._op_queue.append(method.__name__)
        new_array._op_args_queue.append(args)
        new_array._op_kwargs_queue.append(kwargs)

        return new_array
    
    lazy_method.eager = method
    
    return lazy_method

def eager(method):
    """
    Decorator that makes a method eager.
    """

    def eager_method(self, *args, **kwargs):

        if self._array is None or self._op_idx < len(self._op_queue):
            self.evaluate()

        return method(self, *args, **kwargs)

    return eager_method

class LazyArray():
    """ Provides a lazy wrapper around a numpy array.

    Operations on the array are cached and only executed when the array is evaluated.
    Operations are saved in the _op_queue list with arguments and keyword arguments saved in the _op_args_queue and _op_kwargs_queue lists, respectively.
    
    The base array can be set with the set_base_array method. The _op_idx attribute is used to keep track of the operations that have already been executed
    on the _array.
    
    """
    
    def __init__(self, base_array=None):
        self._original_base_array_id = None
        self.set_base_array(base_array)
        self._op_queue = []
        self._op_args_queue = []
        self._op_kwargs_queue = []

    def set_base_array(self, base_array):
        self._base_array = base_array
        self._array = None
        self._op_idx = 0

        if self._original_base_array_id is None and base_array is not None:
            self._original_base_array_id = id(base_array)

    def __getattr__(self, name):

        if name.startswith('_'):
            raise AttributeError(f'LazyArray has no attribute {name}')

        # Check if the attribute is a method of the base np.ndarray
        try:
            _ = getattr(self._base_array, name)
        except AttributeError:
            raise AttributeError(f'LazyArray has no attribute {name}')
        
        # If it is, make it lazy
        # Caching the lazy methods is not possible because
        # the base array might change
        def lazy_attr(*args, **kwargs):
            new_array = copy.copy(self)
            new_array._op_queue.append(name)
            new_array._op_args_queue.append(args)
            new_array._op_kwargs_queue.append(kwargs)

            return new_array

        def eager_attr(self, *args, **kwargs):
            # Get the method from the current array
            self._array = getattr(self._array, name)(*args, **kwargs)

        lazy_attr.eager = eager_attr

        return lazy_attr

    def __copy__(self):

        # Create new LazyArray, but don't copy the base array
        new_array = type(self)(self._base_array)

        # Copy the cached array if it is not None
        new_array._array = None if self._array is None else np.copy(self._array)
        new_array._op_idx = self._op_idx

        # Copy the op queues
        new_array._op_queue = copy.deepcopy(self._op_queue)
        new_array._op_args_queue = copy.deepcopy(self._op_args_queue)
        new_array._op_kwargs_queue = copy.deepcopy(self._op_kwargs_queue)
        return new_array

    def evaluate(self):
        """ Evaluates the lazy operations on the array.
        
        Saves the results to the _array attribute.
        If no base array is set, raises a ValueError.

        """

        if self._base_array is None:
            raise ValueError('No base array set')
        
        if self._op_idx == 0 and self._array is None:
            self._array = np.copy(self._base_array)

        for i in range(self._op_idx, len(self._op_queue)):
            operation = self._op_queue[i]
            op_args = self._op_args_queue[i]
            op_kwargs = self._op_kwargs_queue[i]

            try:
                method = getattr(self, operation).eager
            except AttributeError:
                raise AttributeError(f'Unknown LazyArray operation {operation}')
                
            method(self, *op_args, **op_kwargs)

        self._op_idx = len(self._op_queue)

    def __getstate__(self):
        return self._op_queue, self._op_args_queue, self._op_kwargs_queue, self._original_base_array_id
    
    def __setstate__(self, state):
        self._op_queue, self._op_args_queue, self._op_kwargs_queue, self._original_base_array_id = state
        self._base_array = None
        self._array = None
        self._op_idx = 0

    def __repr__(self):
        return f'LazyArray(base_array_id={id(self._base_array)}, op_queue={self._op_queue}, op_args_queue={self._op_args_queue}, op_kwargs_queue={self._op_kwargs_queue})'
    
    def __str__(self):
        return f'LazyArray(base_array_id={id(self._base_array)}, op_queue={self._op_queue}, op_args_queue={self._op_args_queue}), op_kwargs_queue={self._op_kwargs_queue})'
    

    @eager
    def __eq__(self, other):
        return self._array == other
    
    @eager
    def __ne__(self, other):
        return self._array != other

    @eager
    def __getitem__(self, key):            
        return self._array[key]
    
    @eager
    def __iter__(self):
        return iter(self._array)

    @eager
    def __add__(self, other):
        return self._array + other
    
    @eager
    def __radd__(self, other):
        return other + self._array
    
    @eager
    def __sub__(self, other):
        return self._array - other
    
    @eager
    def __rsub__(self, other):
        return other - self._array
    
    @eager
    def __mul__(self, other):
        return self._array * other
    
    @eager
    def __rmul__(self, other):
        return other * self._array
    
    @eager
    def __truediv__(self, other):
        return self._array / other
    
    @eager
    def __rtruediv__(self, other):
        return other / self._array
    
    @property
    @eager
    def shape(self):
        return self._array.shape
    
    @property
    @eager
    def size(self):
        return self._array.size
    
    @property
    @eager
    def dtype(self):
        return self._array.dtype
    
    @eager
    def __len__(self):
        return len(self._array)
    
    @eager
    def min(self, *args, **kwargs):
        return self._array.min(*args, **kwargs)
    
    @eager
    def max(self, *args, **kwargs):
        return self._array.max(*args, **kwargs)
    

    ##### IDTxl operations #####
    
    @lazy
    def sliced(self, start, end, step=1, axis=0):
        """
        Slices the array along the first axis.
        """
        self._array = self._array.take(np.arange(start, end, step), axis=axis)
    
    @lazy
    def shifted(self, coords, shifts, length=None):
        """
        Shifts the array along the given axis by the given shifts at the given coordinates.
        """
        if length is None:
            length = self._array.shape[1] - np.max(shifts)

        if len(coords) == 1:
            self._array = self._array[coords[0], shifts[0]:length+shifts[0]][np.newaxis, :]
        else:
            self._array = np.stack([self._array[coord, shift:length+shift] for coord, shift in zip(coords, shifts)])

    @lazy
    def rolled(self, shift, axis):
        """
        Rolls the array along the given axis by the given shift.
        """
        self._array = np.roll(self._array, shift, axis=axis)

    @lazy
    def shuffled(self, axis, philox_key, philox_counter):
        """
        Shuffles the array along the given axis.

        The order of the sub-arrays is changed but the elements within the sub-arrays are not changed.
        """
        rbg = np.random.Philox(key=philox_key, counter=philox_counter)
        rng = np.random.Generator(rbg)

        rng.shuffle(self._array, axis=axis)

    @lazy
    def block_shuffled(self, axis, block_size, perm_range, philox_key, philox_counter):
        """
        Permute blocks of samples in a time series within a given range.

        Permute n samples by swapping blocks of samples within a given range.

        Args:
            block_size : int
                number of samples in a block
            perm_range : int
                range over which blocks can be swapped

        Returns:
            numpy array
                permuted indices with length n
        """

        self._array = np.swapaxes(self._array, 0, axis)

        rbg = np.random.Philox(key=philox_key, counter=philox_counter)
        rng = np.random.Generator(rbg)

        blocks = batched(self._array, block_size)
        range_blocks = batched(blocks, perm_range)

        for range_block in range_blocks:
            rng.shuffle(range_block, axis=0)

        self._array = np.concatenate([block for blocks in range_blocks for block in blocks], axis=0)
        self._array = np.swapaxes(self._array, 0, axis)

    @lazy
    def local_shuffled(self, axis, perm_range, philox_key, philox_counter):
        """
        Shuffles the array along the given axis in blocks of size block_size.
        """
        rbg = np.random.Philox(key=philox_key, counter=philox_counter)
        rng = np.random.Generator(rbg)

        self._array = np.swapaxes(self._array, 0, axis)

        blocks = batched(self._array, perm_range)

        for block in blocks:
            rng.shuffle(block, axis=0)

        self._array = np.concatenate(blocks, axis=0)
        self._array = np.swapaxes(self._array, 0, axis)
    
def batched(array_like, batch_size):
    """
    Splits the array_like into batches of size batch_size.

    The last batch may be smaller than batch_size.
    """
    length = len(array_like)
    return [array_like[i:i + batch_size] for i in range(0, length, batch_size)]