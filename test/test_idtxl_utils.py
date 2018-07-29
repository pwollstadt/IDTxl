"""Unit tests for IDTxl utilities module."""
import numpy as np
from idtxl import idtxl_utils as utils


def test_swap_chars():
    """Test swapping of characters in a string."""
    s = utils.swap_chars('psr', 0, 1)
    assert s == 'spr', 'Chars were not swapped correctly.'


def test_combine_discrete_dimensions():
    combined = utils.combine_discrete_dimensions(
        np.array([[1, 0, 1], [0, 1, 0]]), 2)
    assert type(combined) == np.ndarray
    assert (type(combined[0]) == np.int32) or (type(combined[0]) == np.int64)
    assert len(combined.shape) == 1
    assert combined.shape[0] == 2
    assert combined[0] == 5
    assert combined[1] == 2
    combined = utils.combine_discrete_dimensions(
        np.array([[1, 0, 1], [0, 1, 0]]), 3)
    assert combined[0] == 10
    assert combined[1] == 3


def test_discretise():
    # Test 1D discretisation
    discretised = utils.discretise(np.array([1.1, 0.55, 0]), 2)
    assert type(discretised) == np.ndarray
    assert len(discretised.shape) == 1
    assert discretised.shape[0] == 3
    assert (
        type(discretised[0]) == np.int32) or (type(discretised[0]) == np.int64)
    assert check_all_bools_true(discretised == np.array([1, 1, 0]))
    # Now test where one value drops below the threshold
    discretised = utils.discretise(np.array([1.1, 0.54, 0]), 2)
    assert check_all_bools_true(discretised == np.array([1, 0, 0]))
    # and test another example that will contrast with the max_ent
    # discretisation:
    discretised = utils.discretise(np.array([1, 0.9, 0.8, 0]), 2)
    assert type(discretised) == np.ndarray
    assert len(discretised.shape) == 1
    assert discretised.shape[0] == 4
    assert check_all_bools_true(discretised == np.array([1, 1, 1, 0]))
    # Test 2D discretisation
    discretised = utils.discretise(
        np.array([[1.1, 1.1], [1.1, 1.1], [0.55, 1.5], [0, 1.9], [0, 2]]), 2)
    assert type(discretised) == np.ndarray
    assert len(discretised.shape) == 2
    assert discretised.shape[0] == 5
    assert discretised.shape[1] == 2
    assert ((type(discretised[0, 0]) == np.int32) or
            (type(discretised[0, 0]) == np.int64))
    assert check_all_bools_true_2d(
        discretised == np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]))


def test_discretise_max_ent():
    # Test 1D discretisation, on an example to contrast with the even bins
    # discretisation:
    discretised = utils.discretise_max_ent(np.array([1, 0.9, 0.8, 0]), 2)
    assert type(discretised) == np.ndarray
    assert len(discretised.shape) == 1
    assert discretised.shape[0] == 4
    assert (type(discretised[0]) == np.int32) or (
        type(discretised[0]) == np.int64)
    assert check_all_bools_true(discretised == np.array([1, 1, 0, 0]))
    # Now test where our boundary has several values on it (all values on it
    # go into lower bin):
    discretised = utils.discretise_max_ent(np.array([1, 0.8, 0.8, 0]), 2)
    assert check_all_bools_true(discretised == np.array([1, 0, 0, 0]))
    # Test 2D discretisation
    discretised = utils.discretise_max_ent(
        np.array([[1, 0], [0.9, 0.8], [0.8, 0.8], [0, 1]]), 2)
    assert type(discretised) == np.ndarray
    assert len(discretised.shape) == 2
    assert discretised.shape[0] == 4
    assert discretised.shape[1] == 2
    assert (type(discretised[0, 0]) == np.int32) or (
        type(discretised[0, 0]) == np.int64)
    assert check_all_bools_true_2d(
        discretised == np.array([[1, 0], [1, 0], [0, 0], [0, 1]]))


def check_all_bools_true(bool_array):
    for ind in range(bool_array.shape[0]):
        if not(bool_array[ind]):
            return False
    return True


def check_all_bools_true_2d(bool_array):
    for ind1 in range(bool_array.shape[0]):
        for ind2 in range(bool_array.shape[1]):
            if not(bool_array[ind1, ind2]):
                return False
    return True


if __name__ == '__main__':
    test_swap_chars()
    test_combine_discrete_dimensions()
    test_discretise()
    test_discretise_max_ent()
