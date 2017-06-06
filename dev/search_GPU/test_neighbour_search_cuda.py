"""Provide unit tests for neighbour searches using CUDA GPU-code.

Tests are based on unit tests by Pedro Mediano

https://github.com/pmediano/jidt/tree/master/java/source/infodynamics/
measures/continuous/kraskov/cuda
"""
import pytest
import numpy as np
from idtxl.neighbour_search_cuda import cudaFindKnnSetGPU, knn_search

# TODO pass 'float64' to high-level functions


def test_knn_one_dim():
    """Test kNN search in 1D."""
    theiler_t = 0
    n_points = 4
    n_dims = 1
    knn_k = 1
    n_chunks = 1
    pointset = np.array([-1, -1.2, 1, 1.1]).astype('float32')
    gpu_id = 0

    # Return arrays.
    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, n_points, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 0, 'Index 1 not correct.'
    assert indexes[0][2] == 3, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances[0][0], 0.2), 'Distance 0 not correct.'
    assert np.isclose(distances[0][1], 0.2), 'Distance 1 not correct.'
    assert np.isclose(distances[0][2], 0.1), 'Distance 2 not correct.'
    assert np.isclose(distances[0][3], 0.1), 'Distance 3 not correct.'

    # Call high-level function.
    (indexes2, distances2) = knn_search(np.expand_dims(pointset, axis=1),
                                        np.expand_dims(pointset, axis=1),
                                        knn_k, theiler_t, n_chunks, gpu_id)

    assert indexes2[0][0] == 1, 'Index 0 not correct.'
    assert indexes2[0][1] == 0, 'Index 1 not correct.'
    assert indexes2[0][2] == 3, 'Index 2 not correct.'
    assert indexes2[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances2[0][0], 0.2), 'Distance 0 not correct.'
    assert np.isclose(distances2[0][1], 0.2), 'Distance 1 not correct.'
    assert np.isclose(distances2[0][2], 0.1), 'Distance 2 not correct.'
    assert np.isclose(distances2[0][3], 0.1), 'Distance 3 not correct.'


def test_knn_two_dim():
    """Test kNN search in 2D."""
    theiler_t = 0
    n_points = 4
    n_dims = 2
    knn_k = 1
    n_chunks = 1
    pointset = np.array([-1, 0.5, 1.1, 2,
                         -1, 0.5, 1.1, 2]).astype('float32')
    gpu_id = 0

    # Return arrays.
    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, n_points, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 2, 'Index 1 not correct.'
    assert indexes[0][2] == 1, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances[0][0], 1.5), 'Distances 0 not correct.'
    assert np.isclose(distances[0][1], 0.6), 'Distances 1 not correct.'
    assert np.isclose(distances[0][2], 0.6), 'Distances 2 not correct.'
    assert np.isclose(distances[0][3], 0.9), 'Distances 3 not correct.'

    # Call high-level function.
    pointset2 = pointset.reshape((n_points, n_dims))
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert indexes2[0][0] == 1, 'Index 0 not correct.'
    assert indexes2[0][1] == 2, 'Index 1 not correct.'
    assert indexes2[0][2] == 1, 'Index 2 not correct.'
    assert indexes2[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances2[0][0], 1.5), 'Distances 0 not correct.'
    assert np.isclose(distances2[0][1], 0.6), 'Distances 1 not correct.'
    assert np.isclose(distances2[0][2], 0.6), 'Distances 2 not correct.'
    assert np.isclose(distances2[0][3], 0.9), 'Distances 3 not correct.'


def test_one_dim_longer_sequence():
    """Test kNN search in 1D."""
    theiler_t = 0
    n_points = 4
    n_dims = 1
    knn_k = 1
    n_chunks = 1
    pointset = np.array([-1, -1.2, 1, 1.1, 10, 11, 10.5, -100, -50, 666]).astype('float32')
    gpu_id = 0

    # Return arrays.
    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, n_points, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 0, 'Index 1 not correct.'
    assert indexes[0][2] == 3, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances[0][0], 0.2), 'Distance 0 not correct.'
    assert np.isclose(distances[0][1], 0.2), 'Distance 1 not correct.'
    assert np.isclose(distances[0][2], 0.1), 'Distance 2 not correct.'
    assert np.isclose(distances[0][3], 0.1), 'Distance 3 not correct.'

    # Call high-level function.
    (indexes2, distances2) = knn_search(np.expand_dims(pointset, axis=1),
                                        np.expand_dims(pointset, axis=1),
                                        knn_k, theiler_t, n_chunks, gpu_id)

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 0, 'Index 1 not correct.'
    assert indexes[0][2] == 3, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances[0][0], 0.2), 'Distance 0 not correct.'
    assert np.isclose(distances[0][1], 0.2), 'Distance 1 not correct.'
    assert np.isclose(distances[0][2], 0.1), 'Distance 2 not correct.'
    assert np.isclose(distances[0][3], 0.1), 'Distance 3 not correct.'


def test_two_dim_longer_sequence():
    """Test kNN with longer sequences.

    Note:
        The expected results differ from the C++ unit tests because we use the
        maximum norm when searching for neighbours.
    """
    theiler_t = 0
    n_points = 10
    n_dims = 2
    knn_k = 1
    n_chunks = 1
    gpu_id = 0
    # This is the same sequence as in the previous test case, padded with a
    # bunch of points very far away.
    pointset = np.array([-1, 0.5, 1.1, 2, 10, 11, 10.5, -100, -50, 666,
                         -1, 0.5, 1.1, 2, 98, -9, -200, 45.3, -53, 0.1]).astype('float32')

    # Return arrays.
    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, n_points, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 2, 'Index 1 not correct.'
    assert indexes[0][2] == 1, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances[0][0], 1.5), 'Distances 0 not correct.'
    assert np.isclose(distances[0][1], 0.6), 'Distances 1 not correct.'
    assert np.isclose(distances[0][2], 0.6), 'Distances 2 not correct.'
    assert np.isclose(distances[0][3], 0.9), 'Distances 3 not correct.'

    # Call high-level function.
    pointset2 = pointset.reshape((n_points, n_dims))
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 2, 'Index 1 not correct.'
    assert indexes[0][2] == 1, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'

    assert np.isclose(distances[0][0], 1.5), 'Distances 0 not correct.'
    assert np.isclose(distances[0][1], 0.6), 'Distances 1 not correct.'
    assert np.isclose(distances[0][2], 0.6), 'Distances 2 not correct.'
    assert np.isclose(distances[0][3], 0.9), 'Distances 3 not correct.'


def test_random_data():
    """Smoke kNN test with big random dataset"""
    theiler_t = 0
    n_points = 1000
    n_dims = 5
    knn_k = 4
    n_chunks = 1
    gpu_id = 0

    # Return arrays.
    indexes = np.zeros((knn_k, n_points), dtype=np.int32)
    distances = np.zeros((knn_k, n_points), dtype=np.float32)

    pointset = np.random.randn(n_points, n_dims).astype('float32')

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, n_points, gpu_id)

    # Call high-level function.
    pointset2 = pointset.reshape((n_points, n_dims))
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'
    assert np.all(indexes == indexes2), ('High- and low-level calls returned '
                                         'different indices.')
    assert np.all(distances == distances2), ('High- and low-level calls '
                                             'returned different distances.')


def test_two_chunks():
    """Run knn search for two chunks."""
    theiler_t = 0
    n_points = 4
    n_dims = 1
    knn_k = 1
    n_chunks = 2
    signal_length = n_points * n_chunks
    gpu_id = 0

    # Return arrays.
    indexes = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances = np.zeros((knn_k, signal_length), dtype=np.float32)

    pointset = np.array([5,   6, -5,  -7,
                         50, -50, 60, -70]).astype('float32')

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, signal_length, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 0, 'Index 1 not correct.'
    assert indexes[0][2] == 3, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'
    assert indexes[0][4] == 2, 'Index 4 not correct.'
    assert indexes[0][5] == 3, 'Index 5 not correct.'
    assert indexes[0][6] == 0, 'Index 6 not correct.'
    assert indexes[0][7] == 1, 'Index 7 not correct.'

    assert np.isclose(distances[0][0], 1), 'Distance 0 not correct.'
    assert np.isclose(distances[0][1], 1), 'Distance 1 not correct.'
    assert np.isclose(distances[0][2], 2), 'Distance 2 not correct.'
    assert np.isclose(distances[0][3], 2), 'Distance 3 not correct.'
    assert np.isclose(distances[0][4], 10), 'Distance 4 not correct.'
    assert np.isclose(distances[0][5], 20), 'Distance 5 not correct.'
    assert np.isclose(distances[0][6], 10), 'Distance 6 not correct.'
    assert np.isclose(distances[0][7], 20), 'Distance 7 not correct.'

    # Call high-level function.
    pointset2 = np.expand_dims(pointset, axis=1)
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert indexes2[0][0] == 1, 'Index 0 not correct.'
    assert indexes2[0][1] == 0, 'Index 1 not correct.'
    assert indexes2[0][2] == 3, 'Index 2 not correct.'
    assert indexes2[0][3] == 2, 'Index 3 not correct.'
    assert indexes2[0][4] == 2, 'Index 4 not correct.'
    assert indexes2[0][5] == 3, 'Index 5 not correct.'
    assert indexes2[0][6] == 0, 'Index 6 not correct.'
    assert indexes2[0][7] == 1, 'Index 7 not correct.'

    assert np.isclose(distances2[0][0], 1), 'Distance 0 not correct.'
    assert np.isclose(distances2[0][1], 1), 'Distance 1 not correct.'
    assert np.isclose(distances2[0][2], 2), 'Distance 2 not correct.'
    assert np.isclose(distances2[0][3], 2), 'Distance 3 not correct.'
    assert np.isclose(distances2[0][4], 10), 'Distance 4 not correct.'
    assert np.isclose(distances2[0][5], 20), 'Distance 5 not correct.'
    assert np.isclose(distances2[0][6], 10), 'Distance 6 not correct.'
    assert np.isclose(distances2[0][7], 20), 'Distance 7 not correct.'


def test_three_chunks():
    """Run knn search for three chunks."""
    theiler_t = 0
    n_points = 4
    n_dims = 1
    knn_k = 1
    n_chunks = 3
    signal_length = n_points*n_chunks
    gpu_id = 0

    # Return arrays.
    indexes = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances = np.zeros((knn_k, signal_length), dtype=np.float32)

    pointset = np.array([5,    6,  -5,   -7,
                         50,  -50,  60,  -70,
                         500, -500, 600, -700]).astype('float32')

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, signal_length, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 nor correct.'
    assert indexes[0][1] == 0, 'Index 1 nor correct.'
    assert indexes[0][2] == 3, 'Index 2 nor correct.'
    assert indexes[0][3] == 2, 'Index 3 nor correct.'
    assert indexes[0][4] == 2, 'Index 4 nor correct.'
    assert indexes[0][5] == 3, 'Index 5 nor correct.'
    assert indexes[0][6] == 0, 'Index 6 nor correct.'
    assert indexes[0][7] == 1, 'Index 7 nor correct.'
    assert indexes[0][8] == 2, 'Index 8 nor correct.'
    assert indexes[0][9] == 3, 'Index 9 nor correct.'
    assert indexes[0][10] == 0, 'Index 10 nor correct.'
    assert indexes[0][11] == 1, 'Index 11 nor correct.'

    assert np.isclose(distances[0][0], 1), 'Distance 0 is not correct.'
    assert np.isclose(distances[0][1], 1), 'Distance 1 is not correct.'
    assert np.isclose(distances[0][2], 2), 'Distance 2 is not correct.'
    assert np.isclose(distances[0][3], 2), 'Distance 3 is not correct.'
    assert np.isclose(distances[0][4], 10), 'Distance 4 is not correct.'
    assert np.isclose(distances[0][5], 20), 'Distance 5 is not correct.'
    assert np.isclose(distances[0][6], 10), 'Distance 6 is not correct.'
    assert np.isclose(distances[0][7], 20), 'Distance 7 is not correct.'
    assert np.isclose(distances[0][8], 100), 'Distance 8 is not correct.'
    assert np.isclose(distances[0][9], 200), 'Distance 9 is not correct.'
    assert np.isclose(distances[0][10], 100), 'Distance 10 is not correct.'
    assert np.isclose(distances[0][11], 200), 'Distance 11 is not correct.'

    # Call high-level function.
    pointset2 = np.expand_dims(pointset, axis=1)
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert indexes2[0][0] == 1, 'Index 0 nor correct.'
    assert indexes2[0][1] == 0, 'Index 1 nor correct.'
    assert indexes2[0][2] == 3, 'Index 2 nor correct.'
    assert indexes2[0][3] == 2, 'Index 3 nor correct.'
    assert indexes2[0][4] == 2, 'Index 4 nor correct.'
    assert indexes2[0][5] == 3, 'Index 5 nor correct.'
    assert indexes2[0][6] == 0, 'Index 6 nor correct.'
    assert indexes2[0][7] == 1, 'Index 7 nor correct.'
    assert indexes2[0][8] == 2, 'Index 8 nor correct.'
    assert indexes2[0][9] == 3, 'Index 9 nor correct.'
    assert indexes2[0][10] == 0, 'Index 10 nor correct.'
    assert indexes2[0][11] == 1, 'Index 11 nor correct.'

    assert np.isclose(distances2[0][0], 1), 'Distance 0 is not correct.'
    assert np.isclose(distances2[0][1], 1), 'Distance 1 is not correct.'
    assert np.isclose(distances2[0][2], 2), 'Distance 2 is not correct.'
    assert np.isclose(distances2[0][3], 2), 'Distance 3 is not correct.'
    assert np.isclose(distances2[0][4], 10), 'Distance 4 is not correct.'
    assert np.isclose(distances2[0][5], 20), 'Distance 5 is not correct.'
    assert np.isclose(distances2[0][6], 10), 'Distance 6 is not correct.'
    assert np.isclose(distances2[0][7], 20), 'Distance 7 is not correct.'
    assert np.isclose(distances2[0][8], 100), 'Distance 8 is not correct.'
    assert np.isclose(distances2[0][9], 200), 'Distance 9 is not correct.'
    assert np.isclose(distances2[0][10], 100), 'Distance 10 is not correct.'
    assert np.isclose(distances2[0][11], 200), 'Distance 11 is not correct.'


def test_two_chunks_two_dim():
    """Test kNN with two chunks of 2D data in the same call."""
    theiler_t = 0
    n_points = 4
    n_dims = 2
    knn_k = 1
    n_chunks = 2
    gpu_id = 0
    signal_length = n_points * n_chunks

    # Return arrays.
    indexes = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances = np.zeros((knn_k, signal_length), dtype=np.float32)

    # Points:       X    Y                   y
    #               1    1                   |  o o
    #             1.1    1                   |
    #              -1   -1               ----+----x
    #            -1.2   -1                   |
    #                                  o  o  |

    pointset = np.array([1, 1.1, -1, -1.2, 1, 1.1, -1, -1.2,
                         1, 1, -1, -1, 1, 1, -1, -1]).astype('float32')

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, signal_length, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 not correct.'
    assert indexes[0][1] == 0, 'Index 1 not correct.'
    assert indexes[0][2] == 3, 'Index 2 not correct.'
    assert indexes[0][3] == 2, 'Index 3 not correct.'
    assert indexes[0][4] == 1, 'Index 4 not correct.'
    assert indexes[0][5] == 0, 'Index 5 not correct.'
    assert indexes[0][6] == 3, 'Index 6 not correct.'
    assert indexes[0][7] == 2, 'Index 7 not correct.'

    assert np.isclose(distances[0][0], 0.1), 'Distance 0 not correct.'
    assert np.isclose(distances[0][1], 0.1), 'Distance 1 not correct.'
    assert np.isclose(distances[0][2], 0.2), 'Distance 2 not correct.'
    assert np.isclose(distances[0][3], 0.2), 'Distance 3 not correct.'
    assert np.isclose(distances[0][4], 0.1), 'Distance 4 not correct.'
    assert np.isclose(distances[0][5], 0.1), 'Distance 5 not correct.'
    assert np.isclose(distances[0][6], 0.2), 'Distance 6 not correct.'
    assert np.isclose(distances[0][7], 0.2), 'Distance 7 not correct.'

    # Call high-level function.
    pointset2 = pointset.reshape((signal_length, n_dims))
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert indexes2[0][0] == 1, 'Index 0 not correct.'
    assert indexes2[0][1] == 0, 'Index 1 not correct.'
    assert indexes2[0][2] == 3, 'Index 2 not correct.'
    assert indexes2[0][3] == 2, 'Index 3 not correct.'
    assert indexes2[0][4] == 1, 'Index 4 not correct.'
    assert indexes2[0][5] == 0, 'Index 5 not correct.'
    assert indexes2[0][6] == 3, 'Index 6 not correct.'
    assert indexes2[0][7] == 2, 'Index 7 not correct.'

    assert np.isclose(distances2[0][0], 0.1), 'Distance 0 not correct.'
    assert np.isclose(distances2[0][1], 0.1), 'Distance 1 not correct.'
    assert np.isclose(distances2[0][2], 0.2), 'Distance 2 not correct.'
    assert np.isclose(distances2[0][3], 0.2), 'Distance 3 not correct.'
    assert np.isclose(distances2[0][4], 0.1), 'Distance 4 not correct.'
    assert np.isclose(distances2[0][5], 0.1), 'Distance 5 not correct.'
    assert np.isclose(distances2[0][6], 0.2), 'Distance 6 not correct.'
    assert np.isclose(distances2[0][7], 0.2), 'Distance 7 not correct.'


def test_two_chunks_odd_dim():
    """Test kNN with two chunks of data with odd dimension."""
    theiler_t = 0
    n_points = 4
    n_dims = 3
    knn_k = 1
    n_chunks = 2
    gpu_id = 0
    signal_length = n_points * n_chunks

    # Return arrays.
    indexes = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances = np.zeros((knn_k, signal_length), dtype=np.float32)

    # Points:       X    Y      Z             y
    #               1    1   1.02            |  o o
    #             1.1    1   1.03            |
    #              -1   -1  -1.04        ----+----x
    #            -1.2   -1  -1.05            |
    #                                  o  o  |

    pointset = np.array([1,  1.1,   -1, -1.2,    1,  1.1,   -1, -1.2,
                         1,    1,   -1,   -1,    1,    1,   -1,   -1,
                         1.02, 1.03, 1.04, 1.05, 1.02, 1.03, 1.04, 1.05]).astype('float32')

    # Call low-level function.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, signal_length, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 is not correct.'
    assert indexes[0][1] == 0, 'Index 1 is not correct.'
    assert indexes[0][2] == 3, 'Index 2 is not correct.'
    assert indexes[0][3] == 2, 'Index 3 is not correct.'
    assert indexes[0][4] == 1, 'Index 4 is not correct.'
    assert indexes[0][5] == 0, 'Index 5 is not correct.'
    assert indexes[0][6] == 3, 'Index 6 is not correct.'
    assert indexes[0][7] == 2, 'Index 7 is not correct.'

    assert np.isclose(distances[0][0], 0.1), 'Distance 0 ist not correct.'
    assert np.isclose(distances[0][1], 0.1), 'Distance 1 ist not correct.'
    assert np.isclose(distances[0][2], 0.2), 'Distance 2 ist not correct.'
    assert np.isclose(distances[0][3], 0.2), 'Distance 3 ist not correct.'
    assert np.isclose(distances[0][4], 0.1), 'Distance 4 ist not correct.'
    assert np.isclose(distances[0][5], 0.1), 'Distance 5 ist not correct.'
    assert np.isclose(distances[0][6], 0.2), 'Distance 6 ist not correct.'
    assert np.isclose(distances[0][7], 0.2), 'Distance 7 ist not correct.'

    # Call high-level function.
    pointset2 = pointset.reshape((signal_length, n_dims))
    (indexes2, distances2) = knn_search(pointset2, pointset2, knn_k, theiler_t,
                                        n_chunks, gpu_id)

    assert indexes2[0][0] == 1, 'Index 0 is not correct.'
    assert indexes2[0][1] == 0, 'Index 1 is not correct.'
    assert indexes2[0][2] == 3, 'Index 2 is not correct.'
    assert indexes2[0][3] == 2, 'Index 3 is not correct.'
    assert indexes2[0][4] == 1, 'Index 4 is not correct.'
    assert indexes2[0][5] == 0, 'Index 5 is not correct.'
    assert indexes2[0][6] == 3, 'Index 6 is not correct.'
    assert indexes2[0][7] == 2, 'Index 7 is not correct.'

    assert np.isclose(distances2[0][0], 0.1), 'Distance 0 ist not correct.'
    assert np.isclose(distances2[0][1], 0.1), 'Distance 1 ist not correct.'
    assert np.isclose(distances2[0][2], 0.2), 'Distance 2 ist not correct.'
    assert np.isclose(distances2[0][3], 0.2), 'Distance 3 ist not correct.'
    assert np.isclose(distances2[0][4], 0.1), 'Distance 4 ist not correct.'
    assert np.isclose(distances2[0][5], 0.1), 'Distance 5 ist not correct.'
    assert np.isclose(distances2[0][6], 0.2), 'Distance 6 ist not correct.'
    assert np.isclose(distances2[0][7], 0.2), 'Distance 7 ist not correct.'


def test_one_dim_two_dim_arg():
    """Test kNN with two chunks of data with odd dimension."""
    theiler_t = 0
    n_points = 4
    n_dims = 3
    knn_k = 1
    n_chunks = 2
    gpu_id = 0
    signal_length = n_points * n_chunks

    # Return arrays.
    indexes = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances = np.zeros((knn_k, signal_length), dtype=np.float32)

    # Points:       X    Y      Z             y
    #               1    1   1.02            |  o o
    #             1.1    1   1.03            |
    #              -1   -1  -1.04        ----+----x
    #            -1.2   -1  -1.05            |
    #                                  o  o  |

    pointset = np.array([1,  1.1,   -1, -1.2,    1,  1.1,   -1, -1.2,
                         1,    1,   -1,   -1,    1,    1,   -1,   -1,
                         1.02, 1.03, 1.04, 1.05, 1.02, 1.03, 1.04, 1.05]).astype('float32')

    # Call low-level function with 1D numpy array. Numpy arranges data in
    # C-order (row major) by default. This is what's expected by CUDA/pyopencl.
    err = cudaFindKnnSetGPU(indexes, distances, pointset, pointset, knn_k,
                            theiler_t, n_chunks, n_dims, signal_length, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes[0][0] == 1, 'Index 0 is not correct.'
    assert indexes[0][1] == 0, 'Index 1 is not correct.'
    assert indexes[0][2] == 3, 'Index 2 is not correct.'
    assert indexes[0][3] == 2, 'Index 3 is not correct.'
    assert indexes[0][4] == 1, 'Index 4 is not correct.'
    assert indexes[0][5] == 0, 'Index 5 is not correct.'
    assert indexes[0][6] == 3, 'Index 6 is not correct.'
    assert indexes[0][7] == 2, 'Index 7 is not correct.'

    assert np.isclose(distances[0][0], 0.1), 'Distance 0 ist not correct.'
    assert np.isclose(distances[0][1], 0.1), 'Distance 1 ist not correct.'
    assert np.isclose(distances[0][2], 0.2), 'Distance 2 ist not correct.'
    assert np.isclose(distances[0][3], 0.2), 'Distance 3 ist not correct.'
    assert np.isclose(distances[0][4], 0.1), 'Distance 4 ist not correct.'
    assert np.isclose(distances[0][5], 0.1), 'Distance 5 ist not correct.'
    assert np.isclose(distances[0][6], 0.2), 'Distance 6 ist not correct.'
    assert np.isclose(distances[0][7], 0.2), 'Distance 7 ist not correct.'

    # Call low-level function with 2D numpy array. Transposing doesn't change
    # anything about the memory layout.
    indexes2 = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances2 = np.zeros((knn_k, signal_length), dtype=np.float32)
    pointset2 = pointset.reshape((signal_length, n_dims)).copy()
    err = cudaFindKnnSetGPU(indexes2, distances2, pointset2, pointset2, knn_k,
                            theiler_t, n_chunks, n_dims, signal_length, gpu_id)

    assert err == 1, 'There was an error during the GPU-call.'

    assert indexes2[0][0] == 1, 'Index 0 is not correct.'
    assert indexes2[0][1] == 0, 'Index 1 is not correct.'
    assert indexes2[0][2] == 3, 'Index 2 is not correct.'
    assert indexes2[0][3] == 2, 'Index 3 is not correct.'
    assert indexes2[0][4] == 1, 'Index 4 is not correct.'
    assert indexes2[0][5] == 0, 'Index 5 is not correct.'
    assert indexes2[0][6] == 3, 'Index 6 is not correct.'
    assert indexes2[0][7] == 2, 'Index 7 is not correct.'

    assert np.isclose(distances2[0][0], 0.1), 'Distance 0 ist not correct.'
    assert np.isclose(distances2[0][1], 0.1), 'Distance 1 ist not correct.'
    assert np.isclose(distances2[0][2], 0.2), 'Distance 2 ist not correct.'
    assert np.isclose(distances2[0][3], 0.2), 'Distance 3 ist not correct.'
    assert np.isclose(distances2[0][4], 0.1), 'Distance 4 ist not correct.'
    assert np.isclose(distances2[0][5], 0.1), 'Distance 5 ist not correct.'
    assert np.isclose(distances2[0][6], 0.2), 'Distance 6 ist not correct.'
    assert np.isclose(distances2[0][7], 0.2), 'Distance 7 ist not correct.'

    # Call low-level function with 2D numpy array in Fortran order.
    indexes3 = np.zeros((knn_k, signal_length), dtype=np.int32)
    distances3 = np.zeros((knn_k, signal_length), dtype=np.float32)
    pointset3 = np.asfortranarray(pointset2)
    print(pointset3.flags['C_CONTIGUOUS'])
    with pytest.raises(AssertionError):
        cudaFindKnnSetGPU(indexes3, distances3, pointset3, pointset3, knn_k,
                          theiler_t, n_chunks, n_dims, signal_length, gpu_id)


if __name__ == '__main__':
    test_one_dim_two_dim_arg()
    test_one_dim_two_dim_arg()
    test_two_chunks_odd_dim()
    test_two_chunks_odd_dim()
    test_two_chunks_two_dim()
    test_two_chunks()
    test_three_chunks()
    test_random_data()
    test_one_dim_longer_sequence
    test_two_dim_longer_sequence()
    test_knn_one_dim()
    test_knn_two_dim()
