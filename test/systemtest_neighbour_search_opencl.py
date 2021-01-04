"""Provide in-depth tests for neighbour searches using OpenCl GPU-code.

Tests are based on unit tests by Pedro Mediano. Test if neighbours are
correctly identified for very large input sizes, in particular at the end of
individual chunks.
"""
import pytest
import numpy as np
from idtxl.estimators_opencl import OpenCLKraskovMI, OpenCLKraskovCMI

# Skip test module if pyopencl is not installed
pytest.importorskip('pyopencl')

settings = {'theiler_t': 0,
            'kraskov_k': 1,
            'noise_level': 0,
            'gpu_id': 0,
            'debug': True,
            'return_counts': True,
            'verbose': True}

EST_MI = OpenCLKraskovMI(settings)
EST_CMI = OpenCLKraskovCMI(settings)

CHUNK_LENGTH = 500000  # add noise to beginning of chunks to achieve this


def test_three_large_chunks():
    """Test kNN with three large chunks, put test points at chunk end."""
    # Data for three individual chunks
    n_chunks = 3
    chunk1 = np.expand_dims(
        np.hstack((np.ones(CHUNK_LENGTH-4)*9999, [5, 6, -5, -7])), axis=1)
    chunk2 = np.expand_dims(
        np.hstack((np.ones(CHUNK_LENGTH-4)*9999, [50, -50, 60, -70])), axis=1)
    chunk3 = np.expand_dims(
        np.hstack((np.ones(CHUNK_LENGTH-4)*9999, [500, -500, 600, -700])), axis=1)
    pointset1 = np.vstack([chunk1, chunk2, chunk3])  # multiply chunk
    pointset2 = np.ones(pointset1.shape) * 9999

    # Call MI estimator
    mi, dist1, npoints_x, npoints_y = EST_MI.estimate(
        pointset1, pointset2, n_chunks=n_chunks)
    assert np.isclose(dist1[CHUNK_LENGTH-4], 1), 'Distance 0 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH-3], 1), 'Distance 1 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH-2], 2), 'Distance 2 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH-1], 2), 'Distance 3 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH*2-4], 10), 'Distance 4 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH*2-3], 20), 'Distance 5 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH*2-2], 10), 'Distance 6 is not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH*2-1], 20), 'Distance 7 is not correct.'
    assert np.isclose(dist1[-4], 100), 'Distance 8 is not correct.'
    assert np.isclose(dist1[-3], 200), 'Distance 9 is not correct.'
    assert np.isclose(dist1[-2], 100), 'Distance 10 is not correct.'
    assert np.isclose(dist1[-1], 200), 'Distance 11 is not correct.'

    # Call CMI estimator with pointset2 as conditional (otherwise the MI
    # estimator is called internally and the CMI estimator is never tested).
    cmi, dist2, npoints_x, npoints_y, npoints_c = EST_CMI.estimate(
        pointset1, pointset2, pointset2, n_chunks=n_chunks)
    assert np.isclose(dist2[CHUNK_LENGTH-4], 1), 'Distance 0 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH-3], 1), 'Distance 1 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH-2], 2), 'Distance 2 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH-1], 2), 'Distance 3 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH*2-4], 10), 'Distance 4 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH*2-3], 20), 'Distance 5 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH*2-2], 10), 'Distance 6 is not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH*2-1], 20), 'Distance 7 is not correct.'
    assert np.isclose(dist2[-4], 100), 'Distance 8 is not correct.'
    assert np.isclose(dist2[-3], 200), 'Distance 9 is not correct.'
    assert np.isclose(dist2[-2], 100), 'Distance 10 is not correct.'
    assert np.isclose(dist2[-1], 200), 'Distance 11 is not correct.'
    print('-- passed')


def test_two_large_chunks_two_dim():
    """Test kNN with two large chunks of 2D data in the same call, put test points at chunk end."""
    n_chunks = 2
    chunk = np.array(  # this is data for a single chunk
        [np.hstack((np.ones(CHUNK_LENGTH-4)*9999, [1, 1.1, -1, -1.2])),
         np.hstack((np.ones(CHUNK_LENGTH-4)*9999, [1, 1, -1, -1]))]).T.copy()
    pointset1 = np.tile(chunk, (n_chunks, 1))  # multiply chunk
    pointset2 = np.ones(pointset1.shape) * 9999
    # Points:       X    Y                   y
    #               1    1                   |  o o
    #             1.1    1                   |
    #              -1   -1               ----+----x
    #            -1.2   -1                   |
    #                                  o  o  |

    # Call MI estimator
    mi, dist1, npoints_x, npoints_y = EST_MI.estimate(
        pointset1, pointset2, n_chunks=n_chunks)
    assert np.isclose(dist1[CHUNK_LENGTH-4], 0.1), 'Distance 0 not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH-3], 0.1), 'Distance 1 not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH-2], 0.2), 'Distance 2 not correct.'
    assert np.isclose(dist1[CHUNK_LENGTH-1], 0.2), 'Distance 3 not correct.'
    assert np.isclose(dist1[-4], 0.1), 'Distance 4 not correct.'
    assert np.isclose(dist1[-3], 0.1), 'Distance 5 not correct.'
    assert np.isclose(dist1[-2], 0.2), 'Distance 6 not correct.'
    assert np.isclose(dist1[-1], 0.2), 'Distance 7 not correct.'

    # Call CMI estimator with pointset2 as conditional (otherwise the MI
    # estimator is called internally and the CMI estimator is never tested).
    cmi, dist2, npoints_x, npoints_y, npoints_c = EST_CMI.estimate(
        pointset1, pointset2, pointset2, n_chunks)
    assert np.isclose(dist2[CHUNK_LENGTH-4], 0.1), 'Distance 0 not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH-3], 0.1), 'Distance 1 not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH-2], 0.2), 'Distance 2 not correct.'
    assert np.isclose(dist2[CHUNK_LENGTH-1], 0.2), 'Distance 3 not correct.'
    assert np.isclose(dist2[-4], 0.1), 'Distance 4 not correct.'
    assert np.isclose(dist2[-3], 0.1), 'Distance 5 not correct.'
    assert np.isclose(dist2[-2], 0.2), 'Distance 6 not correct.'
    assert np.isclose(dist2[-1], 0.2), 'Distance 7 not correct.'
    print('-- passed')


if __name__ == '__main__':
    test_three_large_chunks()
    test_two_large_chunks_two_dim()
