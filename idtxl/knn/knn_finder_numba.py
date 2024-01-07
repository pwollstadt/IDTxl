from numba import njit
import numpy as np
from numpy import ndarray

from idtxl.knn.knn_finder import KnnFinder

class NumbaBruteForceKNNFinder(KnnFinder):
    """Brute force implementation of KNNFinder.
    
    Uses numba to naively compute all distances to find neighbors.
    
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        
        assert self._metric == 'chebyshev', 'Only chebyshev metric is supported by BruteForceKNNFinder'
    
    def find_all_dists_to_kth_neighbor(self, k: int) -> ndarray:
        return _find_all_dists_to_kth_neighbor(self._data, k)
    
    def count_all_neighbors(self, r: float) -> np.ndarray:
        return _count_all_neighbors(self._data, r)

@njit
def _find_all_dists_to_kth_neighbor(data: np.ndarray, k: int) -> np.ndarray:

    distances = np.empty(data.shape[0])
    distances_i = np.empty(k)

    # loop over query points
    for i in range(data.shape[0]):

        for l in range(k):
            distances_i[l] = np.inf
        distances_idx_max = 0

        # loop over data points
        for j in range(data.shape[0]):

            if j == i:
                continue

            dist = _max(data, i, data, j)
            
            if dist < distances_i[distances_idx_max]:
                distances_i[distances_idx_max] = dist
                
                # find new max
                distances_idx_max = _argmax(distances_i)
            
        distances[i] = distances_i[distances_idx_max]

    return distances

@njit(inline='always')
def _max(a: np.ndarray, i,  b: np.ndarray, j) -> float:
    """Calculate the chebychev distance between two points.

    This is faster than np.max(np.abs(a[i] - b[j])) for small arrays.
    """
    dist = 0
    for l in range(a.shape[1]):
        d = abs(a[i, l] - b[j, l])
        if d > dist:
            dist = d
    return dist

@njit(inline='always')
def _argmax(a: np.ndarray) -> int:
    """Get the index of the maximum value in a.

    This is faster than np.argmax(a) for small arrays.
    """
    max_idx = 0
    max_val = a[0]

    for i in range(1, a.shape[0]):
        if a[i] > max_val:
            max_idx = i
            max_val = a[i]

    return max_idx

@njit
def _count_all_neighbors(data: np.ndarray, r: np.ndarray) -> np.ndarray:
    
    counts = np.zeros(data.shape[0], dtype=np.int64)

    # loop over query points
    for i in range(data.shape[0]):

        # loop over data points
        for j in range(data.shape[0]):

            if i == j:
                continue

            counts[i] += _is_inside(data, i, data, j, threshold=r[i])

    return counts

@njit(inline='always')
def _is_inside(a: np.ndarray, i,  b: np.ndarray, j, threshold: float) -> float:
    """Calculate the chebychev distance between two points.
    if the distance is strictly within the threshold, return 1, else 0.
    """
    for l in range(a.shape[1]):
        d = abs(a[i, l] - b[j, l])
        if d >= threshold:
            return 0
    return 1
