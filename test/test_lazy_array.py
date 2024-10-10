import copy

import pytest
import numpy as np

from idtxl.lazy_array import LazyArray, batched

def test_lazy_array_init():
    """Check whether a LazyArray is correctly initialized"""

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))
    lazy_array = LazyArray(base_array)

    assert lazy_array._base_array is base_array, "LazyArray has wrong base array!"
    assert lazy_array._base_array_id == id(base_array), "LazyArray has wrong base array id!"
    assert lazy_array._op_queue == [], "LazyArray has wrong operation queue!"
    assert lazy_array._op_args_queue == [], "LazyArray has wrong operation arguments queue!"
    assert lazy_array._op_kwargs_queue == [], "LazyArray has wrong operation keyword arguments queue!"

def test_lazy_array_set_base_array():
    """Check whether a LazyArray is correctly initialized"""

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))
    lazy_array = LazyArray(base_array)

    lazy_array = lazy_array.sliced(0, 2)
    lazy_array = lazy_array.transpose()
    lazy_array.evaluate()

    new_base_array = rng.random(size=(10, 10))
    lazy_array.set_base_array(new_base_array)

    compare_array = new_base_array[0:2].T

    assert lazy_array._base_array is new_base_array, "LazyArray has wrong base array!"
    assert lazy_array._base_array_id == id(new_base_array), "LazyArray has wrong base array id!"
    assert lazy_array._op_queue == ["sliced", "transpose"], "LazyArray has wrong operation queue!"
    assert lazy_array._op_args_queue == [(0, 2), ()], "LazyArray has wrong operation arguments queue!"
    assert lazy_array._op_kwargs_queue == [{}, {}], "LazyArray has wrong operation keyword arguments queue!"
    assert lazy_array._array is None, "LazyArray has not evaluated array!"
    assert np.all(lazy_array == compare_array), "LazyArray has wrong evaluated array!"


def test_lazy_array_copy():
    """Check whether a LazyArray is correctly copied"""

    # Test copying of LazyArray with no operations
    rng = np.random.default_rng(42)
    base_array = rng.random(size=(5, 10))

    lazy_array = LazyArray(base_array)
    lazy_array_copy = copy.copy(lazy_array)

    assert lazy_array._base_array is lazy_array_copy._base_array, "LazyArray copy has wrong base array!"
    assert lazy_array._array is None and lazy_array_copy._array is None, "LazyArray copy must not have evaluated array!"
    assert lazy_array._op_queue == lazy_array_copy._op_queue, "LazyArray copy has wrong operation queue!"
    assert lazy_array._op_args_queue == lazy_array_copy._op_args_queue, "LazyArray copy has wrong operation arguments queue!"
    assert lazy_array._op_kwargs_queue == lazy_array_copy._op_kwargs_queue, "LazyArray copy has wrong operation keyword arguments queue!"
    assert lazy_array._base_array_id == lazy_array_copy._base_array_id, "LazyArray copy has wrong base array id!"

    # Test copying of LazyArray with operations
    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.sliced(0, 2)
    lazy_array = lazy_array.transpose()

    lazy_array_copy = copy.copy(lazy_array)

    assert lazy_array._base_array_id == lazy_array_copy._base_array_id, "LazyArray copy has wrong base array id!"
    assert lazy_array._base_array is lazy_array_copy._base_array, "LazyArray copy has wrong base array!"
    assert lazy_array._array is None and lazy_array_copy._array is None, "LazyArray copy must not have evaluated array!"

    assert lazy_array._op_queue is not lazy_array_copy._op_queue, "LazyArray copy must not have the identical operation queue!"
    assert lazy_array._op_queue == lazy_array_copy._op_queue, "LazyArray copy has wrong operation queue!"
    assert lazy_array._op_args_queue is not lazy_array_copy._op_args_queue, "LazyArray copy must not have the identical operation arguments queue!"
    assert lazy_array._op_args_queue == lazy_array_copy._op_args_queue, "LazyArray copy has wrong operation arguments queue!"
    assert lazy_array._op_kwargs_queue is not lazy_array_copy._op_kwargs_queue, "LazyArray copy must not have the identical operation keyword arguments queue!"
    assert lazy_array._op_kwargs_queue == lazy_array_copy._op_kwargs_queue, "LazyArray copy has wrong operation keyword arguments queue!"

    # Evaluate the both arrays
    lazy_array.evaluate()
    lazy_array_copy.evaluate()

    assert lazy_array._base_array is lazy_array_copy._base_array, "LazyArray copy has wrong base array!"
    assert lazy_array._array is not lazy_array_copy._array, "LazyArray copy must not have the identical evaluated array!"
    assert np.all(lazy_array._array == lazy_array_copy._array), "LazyArray copy has wrong evaluated array!"

def test_lazy_array_evaluate():
    """Check whether a LazyArray is correctly evaluated"""

    # Test evaluation of LazyArray
    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.sliced(0, 2)
    lazy_array = lazy_array.transpose()
    lazy_array.evaluate()

    compare_array = base_array[0:2].T

    assert lazy_array._base_array is base_array, "LazyArray has wrong base array!"
    assert lazy_array._op_queue == ["sliced", "transpose"], "LazyArray has wrong operation queue!"
    assert lazy_array._op_args_queue == [(0, 2), ()], "LazyArray has wrong operation arguments queue!"
    assert lazy_array._op_kwargs_queue == [{}, {}], "LazyArray has wrong operation keyword arguments queue!"
    assert lazy_array._op_idx == 2, "LazyArray has wrong operation index!"

    assert lazy_array._array is not None, "LazyArray has not evaluated array!"
    assert np.all(lazy_array._array == compare_array), "LazyArray has wrong evaluated array!"

def test_lazy_array_evaluate_eager():
    """ Test evaluation due to a call of an eager method """

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.sliced(0, 2)
    lazy_array = lazy_array.transpose()
    result = 2.0 * lazy_array # This is an eager method

    compare_array = 2.0 * base_array[0:2].T

    assert isinstance(result, np.ndarray), "Eager method did not return a numpy array!"
    assert np.all(result == compare_array), "Eager method returned wrong array!"

def test_lazy_array_pickling():
    """ Test pickling and unpickling of a LazyArray """

    import pickle

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.sliced(0, 2)
    lazy_array = lazy_array.transpose()
    lazy_array.evaluate()

    pickled_lazy_array = pickle.dumps(lazy_array)
    unpickled_lazy_array = pickle.loads(pickled_lazy_array)

    assert unpickled_lazy_array._base_array is None, "Unpickled LazyArray has wrong base array!"
    assert unpickled_lazy_array._base_array_id == id(base_array), "Unpickled LazyArray has wrong base array id!"
    assert unpickled_lazy_array._array is None, "Unpickled LazyArray should not have evaluated array!"
    assert unpickled_lazy_array._op_idx == 0, "Unpickled LazyArray has wrong operation index!"

    assert lazy_array._op_queue == unpickled_lazy_array._op_queue, "Unpickled LazyArray has wrong operation queue!"
    assert lazy_array._op_args_queue == unpickled_lazy_array._op_args_queue, "Unpickled LazyArray has wrong operation arguments queue!"
    assert lazy_array._op_kwargs_queue == unpickled_lazy_array._op_kwargs_queue, "Unpickled LazyArray has wrong operation keyword arguments queue!"

    # Assert error if base array is not set
    with pytest.raises(ValueError):
        unpickled_lazy_array[:]

    # Set base array and evaluate the unpickled LazyArray
    unpickled_lazy_array.set_base_array(base_array)
    assert np.all(lazy_array[:] == unpickled_lazy_array[:]), "Unpickled LazyArray has wrong evaluated array!"

def test_lazy_array_np_ndarray_methods():
    """ lazy_arrays should support all non-inplace numpy ndarray methods that return another ndarray.
    Internally, this is done using the custom getattr method
    """

    rng = np.random.default_rng(42)
    base_array = 10 * rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)

    # Test a selection of ndarray methods that return another ndarray
    method_names_and_args = [('argsort', ()), ('astype', ('int32',)), ('clip', (0.2, 0.8)), ('flatten', ()),
                             ('repeat', (3,)), ('reshape', (5, 10, 50)), ('swapaxes', (0, 1)),
                             ('take', ([0, 1],)), ('transpose', ()), ('round', (1,)), ('cumsum', (0,))]

    for method_name, method_args in method_names_and_args:
        array_method = getattr(base_array, method_name)
        array_result = array_method(*method_args)

        lazy_array_method = getattr(lazy_array, method_name)
        lazy_array_result = lazy_array_method(*method_args)

        assert np.all(array_result == lazy_array_result), f"LazyArray method {method_name} returned wrong array!"

def test_lazy_array_eager_methods():
    """ Test explicitly defined eager methods of LazyArray """

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)

    # Test a selection of explicitly defined eager methods that return arrays
    method_names_and_args = [('__getitem__', ((slice(0, 2), slice(0, 2)),)),
                             ('__add__', (0.12345,)), ('__radd__', (0.12345,)), ('__sub__', (0.12345,)),
                             ('__rsub__', (0.12345,)), ('__mul__', (0.12345,)), ('__rmul__', (0.12345,)),
                             ('__truediv__', (0.12345,)), ('__rtruediv__', (0.12345,))]
    
    for method_name, method_args in method_names_and_args:
        array_method = getattr(base_array, method_name)
        array_result = array_method(*method_args)

        lazy_array_method = getattr(copy.copy(lazy_array), method_name)
        lazy_array_result = lazy_array_method(*method_args)

        assert np.all(array_result == lazy_array_result), f"LazyArray method {method_name} returned wrong array!"

    # Test a selection of explicitly defined eager methods that return non-arrays
    assert copy.copy(lazy_array).shape == base_array.shape, "LazyArray method shape returned wrong results!"
    assert copy.copy(lazy_array).size == base_array.size, "LazyArray method size returned wrong results!"
    assert len(copy.copy(lazy_array)) == len(base_array), "LazyArray method len returned wrong results!"

    # Test __iter__ method
    assert all(np.all(x == y) for x, y in zip(copy.copy(lazy_array), base_array)), "LazyArray method __iter__ returned wrong iterable!"

def test_lazy_array_sliced():
    """ Test slicing of a LazyArray """

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.sliced(3, 100, step=-2, axis=1)

    compare_array = base_array[:, 3:100:-2]
    
    assert np.all(lazy_array == compare_array), "LazyArray has wrong evaluated array!"

def test_lazy_array_shifted():
    """ Test shifting of a LazyArray """

    base_array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    # Test shifted with no length parameter
    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.shifted(coords=[1, 0], shifts=[0, 2])

    compare_array = np.array([[6, 7, 8], [3, 4, 5]])
    
    assert np.all(lazy_array == compare_array), "LazyArray has wrong evaluated array!"

    # Test shifted with length parameter
    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.shifted(coords=[1, 0], shifts=[0, 2], length=2)

    compare_array = np.array([[6, 7], [3, 4]])

    assert np.all(lazy_array == compare_array), "LazyArray has wrong evaluated array!"

def test_lazy_array_rolled():
    """ Test rolling of a LazyArray """

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.rolled(shift=3, axis=1)

    compare_array = np.roll(base_array, shift=3, axis=1)

    assert np.all(lazy_array == compare_array), "LazyArray has wrong evaluated array!"

def test_lazy_array_shuffled():
    """ Test shuffling of a LazyArray """

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    # Check that the same shuffled array is returned when using the same key and counter
    lazy_array1 = lazy_array.shuffled(axis=0, philox_key=42, philox_counter=42)
    lazy_array2 = lazy_array.shuffled(axis=0, philox_key=42, philox_counter=42)

    assert np.all(lazy_array1 == lazy_array2), "Shuffled LazyArray has wrong evaluated array!"

    # Check that only the order of the sub-arrays is chanbed but sub-arrays remain same
    base_array = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.shuffled(axis=1, philox_key=42, philox_counter=42)

    assert np.all(lazy_array[0] == lazy_array[1]), "Shuffled LazyArray has wrong evaluated array!"

def test_lazy_array_block_shuffled():

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(1000, 25))

    lazy_array = LazyArray(base_array)
    # Check that the same shuffled array is returned when using the same key and counter
    lazy_array1 = lazy_array.block_shuffled(axis=0, block_size=5, perm_range=5, philox_key=42, philox_counter=42)
    lazy_array2 = lazy_array.block_shuffled(axis=0, block_size=5, perm_range=5, philox_key=42, philox_counter=42)

    assert np.all(lazy_array1 == lazy_array2), "Shuffled LazyArray has wrong evaluated array!"

    # Check that only the order of the sub-arrays is chanbed but sub-arrays remain same
    base_array = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.block_shuffled(axis=0, block_size=3, perm_range=3, philox_key=42, philox_counter=42)

    assert np.all(lazy_array[0] == lazy_array[1]), "Shuffled LazyArray has wrong evaluated array!"

    # Check if blocks work correctly
    base_array = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.block_shuffled(axis=0, block_size=2, perm_range=2, philox_key=42, philox_counter=42)
    assert np.all(lazy_array[0:12:2] == lazy_array[1:12:2]), "Shuffled LazyArray has wrong evaluated array!"

    # check perm_range
    assert set(lazy_array[0:4:2]) == set([0, 1]), "Shuffled LazyArray has wrong evaluated array!"
    assert set(lazy_array[4:8:2]) == set([2, 3]), "Shuffled LazyArray has wrong evaluated array!"
    assert set(lazy_array[8:12:2]) == set([4, 5]), "Shuffled LazyArray has wrong evaluated array!"

def test_lazy_array_local_shuffled():
    """ Test local shuffling of a LazyArray """

    rng = np.random.default_rng(42)
    base_array = rng.random(size=(25, 100))

    lazy_array = LazyArray(base_array)
    lazy_array1 = lazy_array.local_shuffled(axis=0, perm_range=5, philox_key=42, philox_counter=42)
    lazy_array2 = lazy_array.local_shuffled(axis=0, perm_range=5, philox_key=42, philox_counter=42)

    assert np.all(lazy_array1 == lazy_array2), "Shuffled LazyArray has wrong evaluated array!"

    # Check that only the order of the sub-arrays is chanbed but sub-arrays remain same
    base_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]).T
    
    lazy_array = LazyArray(base_array)
    lazy_array = lazy_array.local_shuffled(axis=0, perm_range=3, philox_key=42, philox_counter=42)

    assert np.all(lazy_array[:, 0] == lazy_array[:, 1]), "Shuffled LazyArray has wrong evaluated array!"

    # Check if blocks work correctly
    assert set(lazy_array[0:3, 0]) == set([1, 2, 3]), "local shuffle with perm_range=3 has wrong evaluated array!"
    assert set(lazy_array[3:6, 0]) == set([4, 5, 6]), "local shuffle with perm_range=3 has wrong evaluated array!"
    assert set(lazy_array[6:9, 0]) == set([7, 8, 9]), "local shuffle with perm_range=3 has wrong evaluated array!"

def test_lazy_array_batched():
    """ Test batched method of a LazyArray """

    rng = np.random.default_rng(42)
    a = rng.random(size=(25, 100))

    a_batched = batched(a, 3)
    a_concat = np.concatenate(a_batched, axis=0)

    assert np.all(a == a_concat), "Batched array has wrong evaluated array!"


if __name__ == "__main__":
    pytest.main([__file__])